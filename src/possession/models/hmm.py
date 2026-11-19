"""Hybrid HMM: a per-frame classifier for the evidence, Viterbi for the structure.

Why this shape. The ranking metric (spellF1) scores whole possessions, so the decisive property of
a system is how many possessions it reports and where their boundaries fall -- not how many frames
it gets right. A frame classifier on its own flickers and invents possessions. An HMM with three
states {H, A, D} and explicit transition costs turns "how sticky is possession" into two or three
interpretable numbers, and Viterbi returns the single most likely *segmentation* of the match.

Emissions are classifier posteriors divided by class priors (the standard hybrid NN/HMM trick),
raised to a temperature ``tau`` < 1 because neighbouring frames are far from independent and
would otherwise be counted as many times as there are frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

H, A, D = 0, 1, 2


@dataclass
class HMMParams:
    tau: float = 0.1  # emission temperature
    switch: float = 1e-4  # P(H -> A) per frame while alive (and A -> H)
    to_dead: float = 1e-3  # P(alive -> D) per frame
    from_dead: float = 1e-3  # P(D -> alive) per frame, split evenly between H and A

    def log_transitions(self) -> np.ndarray:
        t = np.array([
            [1 - self.switch - self.to_dead, self.switch, self.to_dead],
            [self.switch, 1 - self.switch - self.to_dead, self.to_dead],
            [self.from_dead / 2, self.from_dead / 2, 1 - self.from_dead],
        ])
        return np.log(t)


def viterbi(log_emit: np.ndarray, log_trans: np.ndarray) -> np.ndarray:
    """Most likely state path. ``log_emit`` is (N, 3)."""
    n, k = log_emit.shape
    back = np.empty((n, k), dtype=np.int8)
    score = log_emit[0] - np.log(k)
    lt = log_trans
    for i in range(1, n):
        cand = score[:, None] + lt  # (from, to)
        best = cand.argmax(axis=0)
        back[i] = best
        score = cand[best, np.arange(k)] + log_emit[i]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(score.argmax())
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def decode(proba: np.ndarray, prior: np.ndarray, period: np.ndarray, params: HMMParams):
    """Decode each period independently; frames outside periods get X (3)."""
    log_emit = params.tau * (np.log(np.clip(proba, 1e-6, 1)) - np.log(prior))
    log_trans = params.log_transitions()
    out = np.full(len(proba), 3, dtype=np.int8)
    for p in np.unique(period[period > 0]):
        rows = np.flatnonzero(period == p)
        out[rows] = viterbi(log_emit[rows], log_trans)
    return out
