"""Unit tests on synthetic data -- they run without downloading anything."""

import numpy as np
import pandas as pd

from possession.degrade import _bursty_dropout, _runs, rematerialise_ball
from possession.evaluate import score
from possession.groundtruth import build_labels
from possession.models.baseline import smooth
from possession.models.hmm import HMMParams, decode, viterbi


def _events(rows):
    return pd.DataFrame(rows, columns=["team", "type", "subtype", "period", "start_frame",
                                       "end_frame"])


def test_ownership_survives_challenges_and_changes_on_recovery():
    period = np.ones(100, dtype=np.int8)
    ev = _events([
        (0, "SET PIECE", "KICK OFF", 1, 1, 1),
        (0, "PASS", "", 1, 1, 10),
        (1, "CHALLENGE", "GROUND-LOST", 1, 20, 20),  # a touch without control
        (0, "BALL LOST", "INTERCEPTION", 1, 40, 45),
        (1, "RECOVERY", "INTERCEPTION", 1, 50, 50),
        (1, "PASS", "", 1, 51, 60),
    ])
    y = build_labels(ev, period)
    assert (y[:49] == 0).all()  # home keeps it through the challenge and the loose ball
    assert (y[49:] == 1).all()


def test_dead_ball_runs_from_ball_out_to_restart():
    period = np.ones(100, dtype=np.int8)
    ev = _events([
        (0, "SET PIECE", "KICK OFF", 1, 1, 1),
        (0, "PASS", "", 1, 1, 20),
        (0, "BALL OUT", "", 1, 20, 30),
        (1, "SET PIECE", "THROW IN", 1, 60, 60),
        (1, "PASS", "", 1, 60, 70),
    ])
    y = build_labels(ev, period)
    assert (y[29:59] == 2).all()
    assert y[28] == 0 and y[59] == 1


def test_outside_period_is_abstain():
    period = np.array([0, 0, 1, 1, 1, 0], dtype=np.int8)
    ev = _events([(0, "SET PIECE", "KICK OFF", 1, 3, 3), (0, "PASS", "", 1, 3, 4)])
    y = build_labels(ev, period)
    assert y[0] == 3 and y[-1] == 3


def test_majority_smoothing_removes_flicker():
    codes = np.array([0] * 20 + [1] + [0] * 20, dtype=np.int8)
    assert (smooth(codes, 5) == 0).all()


def test_viterbi_prefers_sticky_path():
    # noisy evidence that is mostly H, with one confident A frame
    emit = np.log(np.tile([0.6, 0.3, 0.1], (50, 1)))
    emit[25] = np.log([0.1, 0.85, 0.05])
    path = viterbi(emit, HMMParams(switch=1e-4).log_transitions())
    assert (path == 0).all()


def test_decode_marks_outside_periods_x():
    proba = np.full((10, 3), 1 / 3)
    period = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 0], dtype=np.int8)
    out = decode(proba, np.full(3, 1 / 3), period, HMMParams())
    assert out[0] == 3 and out[-1] == 3


def test_perfect_prediction_scores_one():
    gt = np.array([3] * 5 + [0] * 50 + [2] * 20 + [1] * 60 + [3] * 5, dtype=np.int8)
    assert score(gt, gt)["spellF1"] == 1.0


def test_dropout_and_runs():
    rng = np.random.default_rng(0)
    mask = _bursty_dropout(np.full(10_000, 0.01), 20, rng)
    assert 0.05 < mask.mean() < 0.4
    assert _runs(np.array([0, 1, 1, 0, 1], dtype=bool)) == [(1, 3), (4, 5)]


def test_rematerialised_ball_ends_at_restart_spot():
    ball = np.array([[0.5, 0.5], [0.52, 0.5]] + [[np.nan, np.nan]] * 50 + [[0.9, 0.1]],
                    dtype=np.float32)
    out = rematerialise_ball(ball, np.ones(len(ball), dtype=np.int8), np.random.default_rng(0))
    assert not np.isnan(out).any()
    assert np.allclose(out[-2], [0.9, 0.1], atol=1e-5)
