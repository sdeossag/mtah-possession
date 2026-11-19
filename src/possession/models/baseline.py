"""The official reference baseline, reimplemented on an :class:`Observation`.

Nearest tracked slot to the ball gives its team the ball; no ball means D; then a centred
majority vote over +/- ``window`` frames. With ``window=50`` this is ``baseline_smooth50``, the
published floor (spellF1 0.2433 on the hackathon match). Reproducing it here is what makes our
numbers on the open games comparable to the competition's.
"""

from __future__ import annotations

import numpy as np

from ..match import Observation

NOBODY = -1


def nearest_team(obs: Observation) -> np.ndarray:
    pitch = np.asarray(obs.pitch, dtype=np.float32)
    ball = obs.ball * pitch
    slots = obs.slots * pitch
    dist = np.linalg.norm(slots - ball[:, None, :], axis=2)
    dist = np.where(np.isnan(dist), np.inf, dist)
    best = np.argmin(dist, axis=1)
    codes = obs.slot_team[best].astype(np.int8)
    codes[np.isinf(dist.min(axis=1))] = NOBODY
    return codes


def smooth(codes: np.ndarray, window: int) -> np.ndarray:
    """Label codes 0=H, 1=A, 2=D after a centred majority vote (ties to home, like the original)."""
    if window <= 0:
        return np.where(codes == NOBODY, 2, codes).astype(np.int8)
    ch = np.concatenate(([0], np.cumsum(codes == 0)))
    ca = np.concatenate(([0], np.cumsum(codes == 1)))
    i = np.arange(len(codes))
    lo, hi = np.maximum(0, i - window), np.minimum(len(codes), i + window + 1)
    h, a = ch[hi] - ch[lo], ca[hi] - ca[lo]
    out = np.full(len(codes), 2, dtype=np.int8)
    seen = (h + a) > 0
    out[seen] = np.where(h[seen] >= a[seen], 0, 1)
    return out


def predict(obs: Observation, window: int = 50) -> np.ndarray:
    labels = smooth(nearest_team(obs), window)
    labels[obs.period == 0] = 3
    return labels
