"""Load -> ground truth -> degrade -> features, cached on disk per (game, noise seed)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .degrade import NoiseConfig, degrade
from .features import compute_features, swap_teams
from .groundtruth import build_labels
from .io_metrica import load_sample_game
from .match import Observation

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "outputs" / "cache"


@dataclass
class Prepared:
    name: str
    obs: Observation | None
    x: pd.DataFrame
    y: np.ndarray
    x_swapped: pd.DataFrame
    y_swapped: np.ndarray
    period: np.ndarray


def swap_labels(y: np.ndarray) -> np.ndarray:
    out = y.copy()
    out[y == 0], out[y == 1] = 1, 0
    return out


def prepare(game: int, seed: int, keep_obs: bool = False) -> Prepared:
    CACHE.mkdir(parents=True, exist_ok=True)
    stem = CACHE / f"g{game}_s{seed}"
    files = [stem.with_suffix(".x.parquet"), stem.with_suffix(".xs.parquet"),
             stem.with_suffix(".npz")]
    if all(f.exists() for f in files) and not keep_obs:
        z = np.load(files[2])
        return Prepared(f"g{game}_s{seed}", None, pd.read_parquet(files[0]), z["y"],
                        pd.read_parquet(files[1]), swap_labels(z["y"]), z["period"])

    match = load_sample_game(RAW, game)
    y = build_labels(match.events, match.period)
    obs = degrade(match, NoiseConfig(seed=seed))
    x = compute_features(obs)
    xs = compute_features(swap_teams(obs))
    x.to_parquet(files[0])
    xs.to_parquet(files[1])
    np.savez_compressed(files[2], y=y, period=match.period)
    return Prepared(f"g{game}_s{seed}", obs if keep_obs else None, x, y, xs, swap_labels(y),
                    match.period)
