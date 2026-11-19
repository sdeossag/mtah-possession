"""Turn clean open tracking into hackathon-like automatic tracking.

The hackathon ATD is automatic and uncorrected. The Metrica open games are not: their ball track
is curated, and it is removed almost exactly when the ball is dead (84-99% of dead frames). A
model trained on that would learn "no ball means dead ball", which is false on the hackathon
data, where the ball also vanishes during live play. So this module does two things to the ball
and three to the players, each calibrated to a number published in the challenge's DATA.md:

Ball
  1. *Re-materialise* it during stoppages: it rolls on from where it was last seen and is carried
     to where play restarts -- what an automatic detector would actually see.
  2. *Drop* it with bursty gaps whose rate rises in crowds and at high ball speed (balls in the
     air), scaled until it is missing on ``ball_missing`` (41%) of in-period frames.

Players
  3. Break every player's track into tracklets with short gaps, more often in crowds, scaled until
     ``mean_tracked`` (20.5) objects are visible per frame.
  4. Give every tracklet a recycled anonymous slot from its team's pool (46 slots in total), so
     identities are lost exactly as in the ATD; a small fraction lands in the wrong team's pool.
  5. Add positional jitter.

Everything is driven by one seeded generator, so a (match, seed) pair always gives the same file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .match import Match, Observation


@dataclass
class NoiseConfig:
    ball_missing: float = 0.41  # DATA.md: ball absent on 41% of graded frames
    mean_tracked: float = 20.5  # DATA.md: mean tracked objects over graded frames
    n_slots: int = 46  # DATA.md: 46 track slots
    ball_gap_mean: float = 30.0  # frames; mean length of a ball dropout
    player_gap_mean: float = 15.0  # frames; mean length of a player dropout
    tracklet_mean_s: float = 45.0  # seconds; mean life of a tracklet before a break
    wrong_team_prob: float = 0.03
    ball_jitter_m: float = 0.25
    player_jitter_m: float = 0.15
    seed: int = 0


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, stop) runs where mask is True."""
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[::2], edges[1::2]))


def _metres(xy: np.ndarray, pitch) -> np.ndarray:
    return xy * np.asarray(pitch, dtype=np.float32)


def rematerialise_ball(ball: np.ndarray, period: np.ndarray, rng) -> np.ndarray:
    """Fill dead-ball gaps with a plausible path: roll out, then carried to the restart spot."""
    out = ball.copy()
    for start, stop in _runs(np.isnan(ball[:, 0]) & (period > 0)):
        n = stop - start
        before = ball[start - 1] if start > 0 and not np.isnan(ball[start - 1, 0]) else None
        after = ball[stop] if stop < len(ball) and not np.isnan(ball[stop, 0]) else None
        if before is None and after is None:
            continue
        p0 = before if before is not None else after
        p1 = after if after is not None else before
        # roll on for up to 1 s along the last direction of travel, slowing down
        vel = (ball[start - 1] - ball[start - 2]) if start > 1 and before is not None else 0 * p0
        vel = np.nan_to_num(vel)
        roll = min(n // 3, 25)
        t = np.arange(1, roll + 1)[:, None]
        rolled = p0 + vel * (t - t ** 2 / (2 * max(roll, 1)))
        p_roll = rolled[-1] if roll else p0
        carry = max(n - roll, 0)
        alpha = np.clip(np.linspace(0, 1.4, carry), 0, 1)[:, None]  # arrives early, then waits
        carried = p_roll + (p1 - p_roll) * alpha
        path = np.concatenate([rolled, carried])[:n]
        out[start:stop] = path
    return out


def _bursty_dropout(rate: np.ndarray, gap_mean: float, rng) -> np.ndarray:
    """Mask of dropped frames: gaps start with per-frame probability ``rate``, geometric length."""
    n = len(rate)
    starts = np.flatnonzero(rng.random(n) < rate)
    lengths = rng.geometric(1.0 / gap_mean, size=len(starts))
    delta = np.zeros(n + 1, dtype=np.int32)
    np.add.at(delta, starts, 1)
    np.add.at(delta, np.minimum(starts + lengths, n), -1)
    return np.cumsum(delta[:-1]) > 0


def _calibrate(rate_shape: np.ndarray, gap_mean: float, target: float, valid: np.ndarray,
               seed: int) -> np.ndarray:
    """Scale ``rate_shape`` by bisection so the dropped fraction on ``valid`` hits ``target``."""
    lo, hi = 0.0, 1.0 / rate_shape[valid].mean()
    for _ in range(30):
        mid = (lo + hi) / 2
        mask = _bursty_dropout(np.clip(rate_shape * mid, 0, 1), gap_mean,
                               np.random.default_rng(seed))
        if mask[valid].mean() < target:
            lo = mid
        else:
            hi = mid
    return _bursty_dropout(np.clip(rate_shape * (lo + hi) / 2, 0, 1), gap_mean,
                           np.random.default_rng(seed))


def degrade(match: Match, config: NoiseConfig | None = None) -> Observation:
    cfg = config or NoiseConfig()
    rng = np.random.default_rng(cfg.seed)
    n, n_players = match.players.shape[:2]
    in_period = match.period > 0
    pitch = match.pitch

    # -- ball --------------------------------------------------------------------------------
    clean_dead = np.isnan(match.ball[:, 0])
    ball = rematerialise_ball(match.ball, match.period, rng)
    ball_m = _metres(ball, pitch)
    players_m = _metres(match.players, pitch)
    dist = np.linalg.norm(players_m - ball_m[:, None, :], axis=2)
    crowd = np.nansum(dist < 3.0, axis=1)
    speed = np.zeros(n, dtype=np.float32)
    speed[1:] = np.linalg.norm(np.diff(ball_m, axis=0), axis=1) * match.fps
    speed = np.nan_to_num(np.clip(speed, 0, 40))
    logit = -1.0 + 0.6 * crowd + 0.10 * speed + 0.8 * clean_dead
    shape = 1.0 / (1.0 + np.exp(-logit))
    dropped = _calibrate(shape, cfg.ball_gap_mean, cfg.ball_missing, in_period, cfg.seed + 1)
    ball = ball + rng.normal(0, cfg.ball_jitter_m, ball.shape).astype(np.float32) / np.asarray(
        pitch, dtype=np.float32)
    ball[dropped | np.isnan(ball[:, 0])] = np.nan

    # -- players -----------------------------------------------------------------------------
    present = ~np.isnan(match.players[:, :, 0])
    crowd_p = np.nansum(
        np.linalg.norm(players_m[:, :, None, :] - players_m[:, None, :, :], axis=3) < 2.0,
        axis=2) - 1  # (N, P) neighbours within 2 m
    shape_p = (1.0 + crowd_p.clip(0)).astype(np.float64)
    target_missing = 1.0 - cfg.mean_tracked / max(present[in_period].sum(1).mean(), 1e-9)
    target_missing = float(np.clip(target_missing, 0.0, 0.9))
    flat_shape = shape_p.T.reshape(-1)  # player-major so gaps stay within one player
    flat_valid = (present & in_period[:, None]).T.reshape(-1)
    flat_drop = _calibrate(flat_shape, cfg.player_gap_mean, target_missing, flat_valid,
                           cfg.seed + 2) if target_missing > 0 else np.zeros_like(flat_valid)
    visible = present & ~flat_drop.reshape(n_players, n).T

    # breaks: a tracklet ends at every gap, plus spontaneous re-identifications
    spontaneous = rng.random((n, n_players)) < 1.0 / (cfg.tracklet_mean_s * match.fps)

    per_team = cfg.n_slots // 2
    slot_team = np.array([0] * per_team + [1] * (cfg.n_slots - per_team), dtype=np.int8)
    free_at = np.zeros(cfg.n_slots, dtype=np.int64)  # frame from which a slot may be reused
    slots = np.full((n, cfg.n_slots, 2), np.nan, dtype=np.float32)
    jitter = cfg.player_jitter_m / np.asarray(pitch, dtype=np.float32)

    tracklets = []
    for p in range(n_players):
        vis = visible[:, p].copy()
        cut = spontaneous[:, p] & vis
        for start, stop in _runs(vis):
            cuts = [start] + [int(c) for c in np.flatnonzero(cut[start:stop]) + start
                              if c > start] + [stop]
            tracklets.extend((a, b, p) for a, b in zip(cuts[:-1], cuts[1:]) if b > a)
    tracklets.sort()

    for start, stop, p in tracklets:
        team = int(match.team[p])
        if rng.random() < cfg.wrong_team_prob:
            team = 1 - team
        pool = np.flatnonzero((slot_team == team) & (free_at <= start))
        if len(pool) == 0:  # pool exhausted: take the slot that frees soonest
            pool = np.flatnonzero(slot_team == team)
            pool = pool[[np.argmin(free_at[pool])]]
        s = int(pool[0])
        noise = rng.normal(0, 1, (stop - start, 2)).astype(np.float32) * jitter
        slots[start:stop, s] = match.players[start:stop, p] + noise
        free_at[s] = stop + int(rng.integers(1, 2 * match.fps))

    return Observation(
        name=f"{match.name}_noisy{cfg.seed}",
        fps=match.fps,
        pitch=pitch,
        period=match.period.copy(),
        ball=ball,
        slots=slots,
        slot_team=slot_team,
        meta={"source": "degraded", "config": cfg.__dict__.copy(),
              "n_tracklets": len(tracklets)},
    )
