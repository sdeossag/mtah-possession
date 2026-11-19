"""Per-frame features computed from an :class:`Observation` (team-level, identity-free).

Design choices worth knowing:

* **Home-attacking frame.** Teams swap ends at half-time and from match to match, so every x
  coordinate is flipped per period so that the home team always attacks towards x = 1. That makes
  "the ball is deep in the away half" mean the same thing in every game.
* **Offline, non-causal.** A submission is produced for a whole match at once, so features may look
  both backwards and forwards (time since the ball was last seen *and* until it is seen again).
* **Missing is information.** Gradient-boosted trees handle NaN natively, so a missing ball is left
  as NaN rather than imputed, and its context (gap length, last/next known state) is made explicit.
* Everything is in metres and seconds.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .match import Observation

FEATURE_VERSION = 3  # bump to invalidate cached features
MAX_PLAYER_SPEED = 12.0  # m/s; above this a slot has jumped to another person
WINDOWS_S = (1, 3, 8)

# Ball cleaning and touch detection
MAX_FILL_GAP = 8  # frames; shorter ball gaps are interpolated
# The four values below were chosen by grid search for touch-detection F1 against the event log
# of open game 1 only (a training game), on its degraded tracking: F1 0.33 -> 0.42.
SMOOTH_WIN = 9  # frames; centred moving average on the ball track
TOUCH_RADIUS = 2.0  # m; a kick needs a player this close to count as a touch
TOUCH_ACC = 40.0  # m/s^2; acceleration peak that marks a kick or a reception
TOUCH_TURN = 90.0  # degrees; direction change that marks a deflection or a pass
TOUCH_MIN_SEP = 6  # frames; peaks closer than this are one touch


def _fill_short_gaps(values: np.ndarray, max_gap: int) -> np.ndarray:
    """Linear interpolation across NaN gaps of at most ``max_gap`` frames, per column."""
    df = pd.DataFrame(values)
    filled = df.interpolate(limit_area="inside")
    missing = df.isna().all(axis=1).to_numpy()
    run_id = np.cumsum(np.concatenate(([True], missing[1:] != missing[:-1])))
    run_len = pd.Series(run_id).map(pd.Series(run_id).value_counts()).to_numpy()
    too_long = missing & (run_len > max_gap)
    out = filled.to_numpy(np.float32)
    out[too_long] = np.nan
    return out


def clean_ball(ball_m: np.ndarray, fps: int) -> dict[str, np.ndarray]:
    """Short gaps filled, jitter smoothed, then velocity / acceleration / turning angle.

    Differencing raw detections turns 0.25 m of positional noise into ~9 m/s of speed noise,
    which buries every kick; smoothing first is what makes touches visible at all.
    """
    pos = _fill_short_gaps(ball_m, MAX_FILL_GAP)
    pos = pd.DataFrame(pos).rolling(SMOOTH_WIN, center=True, min_periods=3).mean().to_numpy(
        np.float32)
    pos[np.isnan(_fill_short_gaps(ball_m, MAX_FILL_GAP)[:, 0])] = np.nan
    vel = np.full_like(pos, np.nan)
    vel[1:-1] = (pos[2:] - pos[:-2]) * fps / 2
    speed = np.linalg.norm(vel, axis=1)
    acc = np.full(len(pos), np.nan, dtype=np.float32)
    acc[1:-1] = np.linalg.norm(vel[2:] - vel[:-2], axis=1) * fps / 2
    lag = 3
    v0, v1 = np.roll(vel, lag, axis=0), np.roll(vel, -lag, axis=0)
    cos = np.sum(v0 * v1, axis=1) / (np.linalg.norm(v0, axis=1) * np.linalg.norm(v1, axis=1)
                                     + 1e-6)
    turn = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    # A ball at rest on both sides has no direction: no turn. A ball at rest on one side only
    # (a kick from a standstill, or a reception that stops it) reads as ~90 degrees, which is
    # deliberate: those are touches too.
    turn[(np.linalg.norm(v0, axis=1) < 1.5) & (np.linalg.norm(v1, axis=1) < 1.5)] = 0.0
    turn[:lag] = np.nan  # np.roll wraps around: the first/last `lag` frames have no past/future
    turn[-lag:] = np.nan
    return {"pos": pos, "speed": speed, "acc": acc, "turn": turn.astype(np.float32)}


def detect_touches(acc: np.ndarray, turn: np.ndarray, d_home: np.ndarray, d_away: np.ndarray):
    """Frames where the ball is kicked, received or deflected near a player, and by which team.

    Returns (touch_frames, touch_team). A touch is a local peak of acceleration (or a sharp turn)
    with a player within TOUCH_RADIUS; its team is the nearer team's.
    """
    strength = np.nan_to_num(acc / TOUCH_ACC) + np.nan_to_num(turn / TOUCH_TURN)
    near = np.fmin(d_home, d_away) < TOUCH_RADIUS
    candidate = (strength >= 1.0) & near
    peak = strength == pd.Series(strength).rolling(2 * TOUCH_MIN_SEP + 1, center=True,
                                                    min_periods=1).max().to_numpy()
    frames = np.flatnonzero(candidate & peak)
    if len(frames):
        keep = np.concatenate(([True], np.diff(frames) >= TOUCH_MIN_SEP))
        frames = frames[keep]
    team = np.where(np.nan_to_num(d_home, nan=np.inf)[frames] <= np.nan_to_num(
        d_away, nan=np.inf)[frames], 0, 1).astype(np.int8)
    return frames, team


def _touch_context(n: int, frames: np.ndarray, team: np.ndarray, fps: int) -> dict:
    """Last / next touch team and timing, plus same-team chain lengths, for every frame."""
    out = {}
    idx = np.arange(n)
    if len(frames) == 0:
        for k in ("last_touch_team", "next_touch_team", "since_touch_s", "until_touch_s",
                  "chain_last", "chain_next"):
            out[k] = np.full(n, np.nan, dtype=np.float32)
        return out
    pos_last = np.searchsorted(frames, idx, side="right") - 1  # touch at or before i
    pos_next = np.searchsorted(frames, idx, side="left")  # touch at or after i
    has_last = pos_last >= 0
    has_next = pos_next < len(frames)
    pl, pn = np.clip(pos_last, 0, len(frames) - 1), np.clip(pos_next, 0, len(frames) - 1)

    chain = np.ones(len(frames), dtype=np.float32)  # consecutive same-team touches ending here
    for k in range(1, len(frames)):
        if team[k] == team[k - 1]:
            chain[k] = chain[k - 1] + 1
    chain_fwd = np.ones(len(frames), dtype=np.float32)  # ... starting here
    for k in range(len(frames) - 2, -1, -1):
        if team[k] == team[k + 1]:
            chain_fwd[k] = chain_fwd[k + 1] + 1

    out["last_touch_team"] = np.where(has_last, team[pl], np.nan).astype(np.float32)
    out["next_touch_team"] = np.where(has_next, team[pn], np.nan).astype(np.float32)
    out["since_touch_s"] = np.where(has_last, (idx - frames[pl]) / fps, np.nan).astype(np.float32)
    out["until_touch_s"] = np.where(has_next, (frames[pn] - idx) / fps, np.nan).astype(np.float32)
    out["chain_last"] = np.where(has_last, chain[pl], np.nan).astype(np.float32)
    out["chain_next"] = np.where(has_next, chain_fwd[pn], np.nan).astype(np.float32)
    # signed versions: positive = home, so one tree split reads "home chain of length k"
    out["chain_last_signed"] = np.where(out["last_touch_team"] == 0, 1, -1) * out["chain_last"]
    out["chain_next_signed"] = np.where(out["next_touch_team"] == 0, 1, -1) * out["chain_next"]
    return out


def home_attacks_positive(obs: Observation) -> dict[int, bool]:
    """Per period: does the home team attack towards x = 1? Decided by team centroids."""
    out = {}
    for p in np.unique(obs.period[obs.period > 0]):
        rows = obs.period == p
        xs = obs.slots[rows, :, 0]
        home = np.nanmean(xs[:, obs.slot_team == 0])
        away = np.nanmean(xs[:, obs.slot_team == 1])
        out[int(p)] = bool(home < away)  # home defends the low-x goal, so attacks +x
    return out


def _flip_x(xy: np.ndarray, period: np.ndarray, direction: dict[int, bool]) -> np.ndarray:
    out = xy.copy()
    for p, positive in direction.items():
        if not positive:
            rows = period == p
            out[rows, ..., 0] = 1.0 - out[rows, ..., 0]
    return out


def _gap_lengths(missing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frames since the last observed frame, and until the next one (0 when observed)."""
    n = len(missing)
    idx = np.arange(n)
    last = np.where(~missing, idx, -1)
    last = np.maximum.accumulate(last)
    since = np.where(last >= 0, idx - last, n).astype(np.float32)
    nxt = np.where(~missing, idx, n)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    until = np.where(nxt < n, nxt - idx, n).astype(np.float32)
    return since, until


def _centred(series: pd.Series, frames: int, how: str = "mean") -> np.ndarray:
    roll = series.rolling(2 * frames + 1, center=True, min_periods=1)
    return getattr(roll, how)().to_numpy(np.float32)


def compute_features(obs: Observation) -> pd.DataFrame:
    # Empty frames (no ball, no players of one team) are expected and become NaN on purpose.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _compute_features(obs)


def _compute_features(obs: Observation) -> pd.DataFrame:
    fps = obs.fps
    pitch = np.asarray(obs.pitch, dtype=np.float32)
    direction = home_attacks_positive(obs)
    ball = _flip_x(obs.ball, obs.period, direction) * pitch
    slots = _flip_x(obs.slots, obs.period, direction) * pitch
    home = obs.slot_team == 0
    away = ~home

    f = {}
    vis = ~np.isnan(ball[:, 0])  # raw detection, before any filling
    f["ball_vis"] = vis.astype(np.float32)
    cb = clean_ball(ball, fps)
    ball = cb["pos"]  # from here on: short gaps filled, jitter smoothed
    vis_c = ~np.isnan(ball[:, 0])
    f["ball_vis_clean"] = vis_c.astype(np.float32)
    f["ball_speed_c"], f["ball_acc_c"], f["ball_turn"] = cb["speed"], cb["acc"], cb["turn"]
    f["ball_x"], f["ball_y"] = ball[:, 0], ball[:, 1]
    f["ball_out"] = ((ball[:, 0] < -0.5) | (ball[:, 0] > pitch[0] + 0.5) |
                     (ball[:, 1] < -0.5) | (ball[:, 1] > pitch[1] + 0.5)).astype(np.float32)
    f["ball_out"][~vis] = np.nan
    f["ball_edge_dist"] = np.minimum.reduce([ball[:, 0], pitch[0] - ball[:, 0],
                                             ball[:, 1], pitch[1] - ball[:, 1]])

    bspeed = np.full(len(ball), np.nan, dtype=np.float32)
    bspeed[1:] = np.linalg.norm(np.diff(ball, axis=0), axis=1) * fps
    bspeed[bspeed > 45] = np.nan
    f["ball_speed"] = bspeed
    f["ball_speed_s1"] = _centred(pd.Series(bspeed), fps)

    dist = np.linalg.norm(slots - ball[:, None, :], axis=2)  # (N, S), NaN where unknown
    with np.errstate(all="ignore"):
        d_home = np.nanmin(np.where(home, dist, np.nan), axis=1)
        d_away = np.nanmin(np.where(away, dist, np.nan), axis=1)
    f["d_home"], f["d_away"] = d_home, d_away
    f["d_diff"] = d_away - d_home  # > 0 means home is closer
    f["n_home_5"] = np.sum((dist < 5) & home, axis=1).astype(np.float32)
    f["n_away_5"] = np.sum((dist < 5) & away, axis=1).astype(np.float32)
    f["n_home_5"][~vis_c] = np.nan
    f["n_away_5"][~vis_c] = np.nan

    # touches: who kicked / received / deflected the ball, and the chains they form
    t_frames, t_team = detect_touches(cb["acc"], cb["turn"], d_home, d_away)
    f.update(_touch_context(len(ball), t_frames, t_team, fps))
    touch_h = np.zeros(len(ball), dtype=np.float32)
    touch_a = np.zeros(len(ball), dtype=np.float32)
    touch_h[t_frames[t_team == 0]] = 1
    touch_a[t_frames[t_team == 1]] = 1

    # close control: a player within 1.5 m of a slow ball (dribbling, shielding, receiving)
    slow = np.nan_to_num(cb["speed"], nan=99) < 6
    ctrl_h = ((np.nan_to_num(d_home, nan=99) < 1.5) & slow).astype(np.float32)
    ctrl_a = ((np.nan_to_num(d_away, nan=99) < 1.5) & slow).astype(np.float32)

    since, until = _gap_lengths(~vis)
    f["ball_since_s"] = np.minimum(since, 30 * fps) / fps
    f["ball_until_s"] = np.minimum(until, 30 * fps) / fps
    dd = pd.Series(f["d_diff"])
    f["d_diff_last"] = dd.ffill().to_numpy(np.float32)
    f["d_diff_next"] = dd.bfill().to_numpy(np.float32)
    bx = pd.Series(f["ball_x"])
    f["ball_x_last"] = bx.ffill().to_numpy(np.float32)
    f["ball_x_next"] = bx.bfill().to_numpy(np.float32)

    # player kinematics
    step = np.full(slots.shape[:2], np.nan, dtype=np.float32)
    step[1:] = np.linalg.norm(np.diff(slots, axis=0), axis=2) * fps
    step[step > MAX_PLAYER_SPEED] = np.nan
    with np.errstate(all="ignore"):
        f["speed_all"] = np.nanmean(step, axis=1)
        f["speed_home"] = np.nanmean(np.where(home, step, np.nan), axis=1)
        f["speed_away"] = np.nanmean(np.where(away, step, np.nan), axis=1)
        f["speed_p90"] = np.nanpercentile(step, 90, axis=1)
        f["n_tracked"] = np.sum(~np.isnan(slots[:, :, 0]), axis=1).astype(np.float32)

        # team shape, in the home-attacking frame
        for name, mask in (("home", home), ("away", away)):
            xs = np.where(mask, slots[:, :, 0], np.nan)
            ys = np.where(mask, slots[:, :, 1], np.nan)
            f[f"cx_{name}"] = np.nanmean(xs, axis=1)
            f[f"sx_{name}"] = np.nanstd(xs, axis=1)
            f[f"sy_{name}"] = np.nanstd(ys, axis=1)
    f["cx_gap"] = f["cx_home"] - f["cx_away"]
    f["width_ratio"] = f["sy_home"] / (f["sy_away"] + 1e-3)

    df = pd.DataFrame(f)
    home_closer = pd.Series(np.where(vis, (d_home < d_away).astype(np.float32), np.nan))
    for w in WINDOWS_S:
        k = w * fps
        df[f"vis_m{w}"] = _centred(df["ball_vis"], k)
        df[f"speed_all_m{w}"] = _centred(df["speed_all"], k)
        df[f"d_diff_m{w}"] = _centred(df["d_diff"], k)
        df[f"home_closer_m{w}"] = _centred(home_closer, k)
        df[f"ball_speed_m{w}"] = _centred(df["ball_speed"], k)
        df[f"cx_gap_m{w}"] = _centred(df["cx_gap"], k)
        th = _centred(pd.Series(touch_h), k, "sum")
        ta = _centred(pd.Series(touch_a), k, "sum")
        df[f"touch_bal_m{w}"] = th - ta
        df[f"touch_n_m{w}"] = th + ta
        df[f"ctrl_bal_m{w}"] = _centred(pd.Series(ctrl_h - ctrl_a), k)
    # one-sided windows: what happened just before vs just after this frame
    for side, shift in (("past", 1), ("future", -1)):
        k = 2 * fps
        th = pd.Series(touch_h).rolling(k, min_periods=1).sum()
        ta = pd.Series(touch_a).rolling(k, min_periods=1).sum()
        bal = (th - ta).to_numpy(np.float32)
        if shift < 0:
            bal = (pd.Series(touch_h[::-1]).rolling(k, min_periods=1).sum()
                   - pd.Series(touch_a[::-1]).rolling(k, min_periods=1).sum()).to_numpy(
                np.float32)[::-1]
        df[f"touch_bal_{side}2"] = bal
    df["speed_all_trend"] = df["speed_all_m3"] - df["speed_all_m8"]
    df["period"] = obs.period.astype(np.float32)
    return df


def swap_teams(obs: Observation) -> Observation:
    """The same match seen with home and away exchanged -- used for symmetric augmentation."""
    return Observation(name=obs.name + "_swapped", fps=obs.fps, pitch=obs.pitch,
                       period=obs.period, ball=obs.ball, slots=obs.slots,
                       slot_team=(1 - obs.slot_team).astype(np.int8), meta=dict(obs.meta))
