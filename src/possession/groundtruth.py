"""Frame-level possession ground truth (H / A / D / X) derived from human event data.

This is only ever applied to the *open* Metrica sample games, which the competition rules
explicitly allow ("hand-labelled data from other games is fine, in any amount"). It is what lets
us validate the pipeline honestly without touching the hackathon's 5 public minutes.

The rules mirror CHALLENGE.md as closely as event data allows:

* **Ownership** changes only on an event that implies *control* by the new team: a pass, carry,
  shot, set piece, recovery, or a "ball lost" (you can only lose what you had). Challenges,
  fouls received, cards and "ball out" never change ownership by themselves -- a contested or
  loose ball stays with the last team that controlled it.
* **Dead ball** runs from the moment play stops (ball out, foul, goal, offside, handball, end of
  half) to the restart that resumes it (any set piece, including goal kicks logged as passes).
  If a restart has no logged stopping event, play is assumed to have stopped at the end of the
  last event before it.
* Before the first kick-off of each period the ball is dead; outside periods the label is X.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import ABSTAIN, DEAD, LABELS

CONTROL_TYPES = {"PASS", "CARRY", "SHOT", "SET PIECE", "RECOVERY", "BALL LOST"}
RESTART_SUBTYPES = ("GOAL KICK",)  # goal kicks are sometimes logged as PASS / BALL LOST
STOP_AT_END_TYPES = {"BALL OUT"}
STOP_SUBTYPE_WORDS = ("OFFSIDE", "HAND BALL", "END HALF", "GOAL", "OUT")
FOUL_WORD = "FAULT"


def _is_restart(row) -> bool:
    if row.type == "SET PIECE":
        return True
    return row.type in ("PASS", "BALL LOST") and row.subtype.startswith(RESTART_SUBTYPES)


def _stop_frame(row) -> int | None:
    """Frame at which this event stops play, or None if it does not."""
    if row.type in STOP_AT_END_TYPES:
        return row.end_frame
    if row.type == "FAULT RECEIVED":
        return row.start_frame
    if row.type == "CHALLENGE" and FOUL_WORD in row.subtype and "ADVANTAGE" not in row.subtype:
        return row.start_frame
    if row.type == "SHOT" and any(w in row.subtype for w in ("OUT", "GOAL")):
        return row.end_frame
    if row.type in ("BALL LOST", "PASS") and any(
            w in row.subtype for w in ("OFFSIDE", "HAND BALL", "END HALF")):
        return row.end_frame
    return None


def build_labels(events: pd.DataFrame, period: np.ndarray) -> np.ndarray:
    """Per-frame label codes (0=H, 1=A, 2=D, 3=X) for frames indexed by ``frame_id - 1``."""
    n = len(period)
    owner = np.full(n, ABSTAIN, dtype=np.int8)
    dead = np.zeros(n, dtype=bool)

    for p in np.unique(period[period > 0]):
        in_period = np.flatnonzero(period == p)
        p_start, p_end = in_period[0], in_period[-1] + 1
        ev = events[(events.period == p) & (events.team >= 0)].reset_index(drop=True)
        if ev.empty:
            continue

        # Ownership: a step function that only moves on control events.
        changes = ev[ev.type.isin(CONTROL_TYPES)]
        current, last = ABSTAIN, p_start
        for row in changes.itertuples():
            f = int(np.clip(row.start_frame - 1, p_start, p_end))
            owner[last:f] = current
            current, last = row.team, f
        owner[last:p_end] = current

        # Dead ball: stoppage -> restart.
        prev_restart_frame = p_start
        pending_stop, last_event_end = None, p_start
        first_restart = True
        for row in ev.itertuples():
            f_start = int(np.clip(row.start_frame - 1, p_start, p_end))
            if _is_restart(row):
                if first_restart:
                    stop = p_start  # before the opening kick-off
                elif pending_stop is not None:
                    stop = pending_stop
                else:
                    stop = last_event_end
                stop = int(np.clip(stop, prev_restart_frame, f_start))
                dead[stop:f_start] = True
                prev_restart_frame, pending_stop, first_restart = f_start, None, False
                last_event_end = int(np.clip(row.end_frame - 1, p_start, p_end))
                continue
            sf = _stop_frame(row)
            if sf is not None and pending_stop is None:
                pending_stop = int(np.clip(sf - 1, p_start, p_end))
            last_event_end = int(np.clip(row.end_frame - 1, p_start, p_end))
        if pending_stop is not None:  # play stopped and never restarted: end of half
            dead[pending_stop:p_end] = True

    labels = owner.copy()
    labels[dead] = DEAD
    labels[period == 0] = ABSTAIN
    return labels


def to_strings(codes: np.ndarray) -> np.ndarray:
    return np.asarray(LABELS)[codes]


def summarise(codes: np.ndarray, fps: int = 25) -> dict:
    """Possession count and mean duration -- the two numbers the challenge cares most about."""
    graded = codes != ABSTAIN
    runs = []
    start = None
    for i in range(len(codes) + 1):
        c = codes[i] if i < len(codes) else -1
        if start is not None and c != codes[start]:
            runs.append((start, i, codes[start]))
            start = None
        if start is None and i < len(codes) and c in (0, 1):
            start = i
    durations = np.array([(b - a) / fps for a, b, _ in runs])
    counts = np.bincount(codes[graded], minlength=3)[:3]
    return {
        "graded_frames": int(graded.sum()),
        "H": int(counts[0]), "A": int(counts[1]), "D": int(counts[2]),
        "possessions": len(runs),
        "mean_possession_s": float(durations.mean()) if len(runs) else 0.0,
        "median_possession_s": float(np.median(durations)) if len(runs) else 0.0,
    }
