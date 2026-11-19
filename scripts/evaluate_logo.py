"""Nested leave-one-game-out evaluation on the three Metrica open games.

For each held-out game:

  1. **Inner loop (training games only).** Fit the frame classifier on one training game and
     predict the other, both ways, to get out-of-fold probabilities. Tune the HMM's handful of
     parameters on those -- never on the held-out game.
  2. **Outer fit.** Refit the classifier on both training games and decode the held-out game with
     the tuned parameters. The held-out game is degraded with a *different* noise seed than any
     training data, so the model never sees the exact noise it is scored on.

This mirrors the competition's rule that the 5 public minutes may not be used for tuning: every
choice is made without looking at the data it is scored on.

    python scripts/evaluate_logo.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# joblib probes physical cores through `wmic`, which recent Windows no longer ships
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) // 2)))

from possession.dataset import prepare  # noqa: E402
from possession.evaluate import score, spell_f1  # noqa: E402
from possession.models import emission  # noqa: E402
from possession.models.baseline import smooth  # noqa: E402
from possession.models.hmm import HMMParams, decode  # noqa: E402

GAMES = (1, 2, 3)
TRAIN_SEED, TEST_SEED = 0, 100
GRID = {
    "tau": (0.06, 0.12, 0.25, 0.5, 0.75, 1.0),
    "switch": (1e-5, 3e-5, 1e-4, 3e-4, 1e-3),
    "to_dead": (1e-4, 3e-4, 1e-3, 3e-3),
    "from_dead": (1e-3, 3e-3, 1e-2, 3e-2),
}


def baseline_from_features(x: pd.DataFrame, period: np.ndarray, window: int = 50) -> np.ndarray:
    """baseline_smooth50: nearest player's team, D without a ball, majority vote over +/-2 s."""
    d_home = x["d_home"].fillna(np.inf).to_numpy()
    d_away = x["d_away"].fillna(np.inf).to_numpy()
    vis = (x["ball_vis"].to_numpy() > 0) & np.isfinite(np.minimum(d_home, d_away))
    codes = np.where(vis, np.where(d_home <= d_away, 0, 1), -1).astype(np.int8)
    out = smooth(codes, window)
    out[period == 0] = 3
    return out


def training_pairs(games, seed):
    pairs = []
    for g in games:
        p = prepare(g, seed)
        pairs += [(p.x, p.y), (p.x_swapped, p.y_swapped)]
    return pairs


def tune(oof: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> tuple[HMMParams, float]:
    """Coordinate ascent over GRID on out-of-fold probabilities, two sweeps."""
    params = HMMParams()

    def objective(p: HMMParams) -> float:
        return float(np.mean([spell_f1(y, decode(proba, prior, period, p))
                              for proba, prior, period, y in oof]))

    best = objective(params)
    for _ in range(2):
        for name, values in GRID.items():
            for v in values:
                trial = HMMParams(**{**params.__dict__, name: v})
                s = objective(trial)
                if s > best:
                    best, params = s, trial
    return params, best


def main():
    rows, chosen = [], {}
    t0 = time.time()
    for test in GAMES:
        train = [g for g in GAMES if g != test]

        oof = []
        for a, b in (train, train[::-1]):
            model, prior = emission.fit(training_pairs([a], TRAIN_SEED))
            pb = prepare(b, TRAIN_SEED)
            oof.append((emission.predict_proba(model, pb.x), prior, pb.period, pb.y))
        params, inner = tune(oof)
        chosen[f"g{test}"] = {**params.__dict__, "inner_spellF1": inner}

        model, prior = emission.fit(training_pairs(train, TRAIN_SEED))
        pt = prepare(test, TEST_SEED)
        proba = emission.predict_proba(model, pt.x)

        frame_argmax = proba.argmax(1).astype(np.int8)
        frame_argmax[pt.period == 0] = 3
        candidates = {
            "baseline_smooth50": baseline_from_features(pt.x, pt.period),
            "classifier_argmax": frame_argmax,
            "classifier_hmm": decode(proba, prior, pt.period, params),
        }
        for name, pred in candidates.items():
            rows.append({"test_game": f"g{test}", "system": name, **score(pt.y, pred)})
        print(f"[{time.time() - t0:5.0f}s] g{test} done  params={params}", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "logo_results.csv", index=False)
    (out / "logo_params.json").write_text(json.dumps(chosen, indent=2))

    pd.set_option("display.width", 160)
    print("\n" + df.round(4).to_string(index=False))
    summary = df.groupby("system")[["spellF1", "macroF1", "poss", "gt_poss", "dur", "gt_dur"]]
    print("\nmean over held-out games\n" + summary.mean().round(4).sort_values(
        "spellF1", ascending=False).to_string())


if __name__ == "__main__":
    main()
