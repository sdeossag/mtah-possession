"""Build a standalone HTML replay of one held-out game: noisy tracking + truth vs model vs baseline.

The model is trained on the *other* two open games with the HMM parameters that
``evaluate_logo.py`` tuned for this fold, so what you watch is a genuine out-of-sample prediction.
The page is written to ``outputs/`` (git-ignored): it embeds tracking data, which is not
redistributed with this repository.

    python scripts/build_viewer.py --game 3 --start-min 10 --minutes 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) // 2)))

from possession import LABELS  # noqa: E402
from possession.dataset import prepare  # noqa: E402
from possession.evaluate import score  # noqa: E402
from possession.models import emission  # noqa: E402
from possession.models.hmm import HMMParams, decode  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_logo import GAMES, TEST_SEED, TRAIN_SEED, baseline_from_features, training_pairs  # noqa: E402,E501

TIMELINE_STEP = 5  # frames per timeline cell (0.2 s)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--game", type=int, default=3, choices=GAMES)
    ap.add_argument("--start-min", type=float, default=10.0,
                    help="replay start, minutes from the first frame of the file")
    ap.add_argument("--minutes", type=float, default=8.0, help="replay length")
    args = ap.parse_args()

    params_file = ROOT / "outputs" / "logo_params.json"
    if not params_file.exists():
        sys.exit("run scripts/evaluate_logo.py first: it tunes the HMM parameters used here")
    tuned = json.loads(params_file.read_text())[f"g{args.game}"]
    params = HMMParams(**{k: tuned[k] for k in ("tau", "switch", "to_dead", "from_dead")})

    train = [g for g in GAMES if g != args.game]
    print(f"training on games {train}, replaying game {args.game} ...", flush=True)
    model, prior = emission.fit(training_pairs(train, TRAIN_SEED))
    pt = prepare(args.game, TEST_SEED, keep_obs=True)
    proba = emission.predict_proba(model, pt.x)
    hmm = decode(proba, prior, pt.period, params)
    base = baseline_from_features(pt.x, pt.period)

    s_hmm, s_base = score(pt.y, hmm), score(pt.y, base)
    scores = [
        {"name": "model (HMM)", "best": True, **{k: s_hmm[k] for k in ("spellF1", "macroF1", "poss")}},
        {"name": "baseline", "best": False, **{k: s_base[k] for k in ("spellF1", "macroF1", "poss")}},
    ]

    labels = np.asarray(LABELS)
    obs, fps = pt.obs, pt.obs.fps
    a = int(args.start_min * 60 * fps)
    b = min(obs.n_frames, a + int(args.minutes * 60 * fps))
    pitch_dm = np.asarray(obs.pitch, dtype=np.float32) * 10

    ball, players = [], []
    for i in range(a, b):
        bx = obs.ball[i]
        ball.append(None if np.isnan(bx[0]) else [int(round(v)) for v in bx * pitch_dm])
        row = []
        for s in np.flatnonzero(~np.isnan(obs.slots[i, :, 0])):
            x, y = obs.slots[i, s] * pitch_dm
            row += [int(round(x)), int(round(y)), int(obs.slot_team[s])]
        players.append(row)

    touch = (pt.x["since_touch_s"].to_numpy()[a:b] == 0).astype(int).tolist()
    data = {
        "meta": {
            "name": f"Metrica open game {args.game}", "fps": fps, "seed": TEST_SEED,
            "pitch": list(obs.pitch), "timeline_step": TIMELINE_STEP, "scores": scores,
            "gt_poss": s_hmm["gt_poss"],
            "note": (f"Trained on games {train[0]} and {train[1]} only; HMM parameters tuned on "
                     "their out-of-fold predictions. Positions shown are what the model sees: "
                     "anonymous slots, jitter, and a ball missing on ~41% of frames."),
        },
        "timeline": {k: "".join(labels[v[::TIMELINE_STEP]])
                     for k, v in (("gt", pt.y), ("hmm", hmm), ("base", base))},
        "clip": {
            "start": a, "ball": ball, "players": players, "touch": touch,
            "gt": "".join(labels[pt.y[a:b]]), "hmm": "".join(labels[hmm[a:b]]),
            "base": "".join(labels[base[a:b]]),
            "proba": np.round(proba[a:b] * 100).astype(int).tolist(),
        },
    }

    template = (ROOT / "src" / "possession" / "viewer_template.html").read_text(encoding="utf-8")
    html = template.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    out = ROOT / "outputs" / f"viewer_g{args.game}.html"
    out.write_text(html, encoding="utf-8")
    print(f"spellF1 model {s_hmm['spellF1']:.3f} vs baseline {s_base['spellF1']:.3f}")
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
