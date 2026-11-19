"""Thin wrapper around the official metric so every score here is the competition's score."""

from __future__ import annotations

import numpy as np

from .official import possession_metrics as pm


def score(gt: np.ndarray, pred: np.ndarray) -> dict:
    r = pm.evaluate(np.asarray(gt, dtype=np.int64), np.asarray(pred, dtype=np.int64))
    return {
        "spellF1": r["spell_f1_mean"],
        "macroF1": r["macro_f1"],
        "poss": r["n_pred_spells"],
        "gt_poss": r["n_gt_spells"],
        "dur": r["mean_spell_duration_s"],
        "gt_dur": r["gt_mean_spell_duration_s"],
        "aliveF1": r["alive_dead_f1"],
        "team_acc": r["team_acc_both_alive"],
    }


def spell_f1(gt: np.ndarray, pred: np.ndarray) -> float:
    """Only the ranking metric -- cheaper than ``score`` inside a search loop."""
    gt = np.asarray(gt, dtype=np.int64)
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(np.asarray(pred, dtype=np.int64), runs)
    return pm.spell_scores(gt, filled, runs)["mean_f1"]
