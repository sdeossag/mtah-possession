"""The one in-memory representation every stage of the pipeline works on.

Two levels on purpose:

* ``Match``  -- clean, identity-level tracking as it comes from an open dataset. Only the
  ground-truth builder and the noise generator ever see it.
* ``Observation`` -- what a possession model is allowed to see: team-level track *slots* that
  appear, vanish and swap, and a ball that is often missing. This is exactly the shape of the
  hackathon's automatic tracking (46 slots for 22 players, no identities), so a model written
  against ``Observation`` runs unchanged on either source.

All coordinates are normalised to [0, 1] on both axes (Metrica convention). Multiply by
``pitch`` to get metres. Arrays are indexed by *frame index* ``i = frame_id - 1``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Match:
    name: str
    fps: int
    pitch: tuple[float, float]  # (length, width) in metres
    period: np.ndarray  # (N,) int8, 0 where the frame belongs to no period
    ball: np.ndarray  # (N, 2) float32, NaN where missing
    players: np.ndarray  # (N, P, 2) float32, NaN where the player is not on the pitch
    team: np.ndarray  # (P,) int8, 0 = home, 1 = away
    events: object = None  # pandas.DataFrame, see events.py
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return len(self.period)


@dataclass
class Observation:
    """Team-level, identity-free view of a match -- the only input a model receives."""

    name: str
    fps: int
    pitch: tuple[float, float]
    period: np.ndarray  # (N,) int8
    ball: np.ndarray  # (N, 2) float32, NaN where missing
    slots: np.ndarray  # (N, S, 2) float32, NaN where the slot is empty
    slot_team: np.ndarray  # (S,) int8, 0 = home, 1 = away
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return len(self.period)


def observe_clean(match: Match) -> Observation:
    """The noise-free observation: one slot per player, ball as tracked."""
    return Observation(
        name=match.name,
        fps=match.fps,
        pitch=match.pitch,
        period=match.period.copy(),
        ball=match.ball.copy(),
        slots=match.players.copy(),
        slot_team=match.team.copy(),
        meta={"source": "clean"},
    )
