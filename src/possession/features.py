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

MAX_PLAYER_SPEED = 12.0  # m/s; above this a slot has jumped to another person
WINDOWS_S = (1, 3, 8)


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
    vis = ~np.isnan(ball[:, 0])
    f["ball_vis"] = vis.astype(np.float32)
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
    f["n_home_5"][~vis] = np.nan
    f["n_away_5"][~vis] = np.nan

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
    df["speed_all_trend"] = df["speed_all_m3"] - df["speed_all_m8"]
    df["period"] = obs.period.astype(np.float32)
    return df


def swap_teams(obs: Observation) -> Observation:
    """The same match seen with home and away exchanged -- used for symmetric augmentation."""
    return Observation(name=obs.name + "_swapped", fps=obs.fps, pitch=obs.pitch,
                       period=obs.period, ball=obs.ball, slots=obs.slots,
                       slot_team=(1 - obs.slot_team).astype(np.int8), meta=dict(obs.meta))
