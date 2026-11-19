"""Per-frame classifier P(H, A, D | features) -- the evidence fed to the HMM.

Gradient-boosted trees (scikit-learn's HistGradientBoosting): NaN-native, fast on ~10^6 rows,
deterministic with a fixed seed, and no GPU or extra dependency. Every match is added twice, once
as observed and once with home and away exchanged, so the model cannot learn "home has the ball
more" from a handful of games and stays symmetric by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

SUBSAMPLE = 3  # train on every 3rd frame: neighbours are near-duplicates


def make_model(seed: int = 0) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=200,
        l2_regularization=1.0, early_stopping=False, random_state=seed)


def stack(frames: list[tuple[pd.DataFrame, np.ndarray]]):
    xs, ys = [], []
    for x, y in frames:
        keep = np.flatnonzero(y < 3)[::SUBSAMPLE]
        xs.append(x.iloc[keep])
        ys.append(y[keep])
    return pd.concat(xs, ignore_index=True), np.concatenate(ys)


def fit(frames: list[tuple[pd.DataFrame, np.ndarray]], seed: int = 0):
    x, y = stack(frames)
    model = make_model(seed).fit(x, y)
    prior = np.bincount(y, minlength=3) / len(y)
    return model, prior


def predict_proba(model, x: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(x)
