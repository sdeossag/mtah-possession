# Ball possession from noisy football tracking

**For every frame of a football match, who has the ball — home, away, or nobody (dead ball)?**

An independent implementation of the task posed by
[More than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027) (Metrica Sports ×
FC Barcelona), built and validated on Metrica's **open** sample games, and scored with the
competition's **own, unmodified metric**.

| held-out game (nested leave-one-game-out) | spellF1 ↑ | macroF1 ↑ | possessions (truth) |
|---|---|---|---|
| official baseline (`baseline_smooth50`) | 0.192 | 0.418 | 795 (373) |
| frame classifier alone | 0.116 | 0.844 | 2304 (373) |
| **frame classifier + HMM (this repo)** | **0.472** | **0.852** | 258 (373) |

*Mean over the three games, each scored after training and tuning on the other two only.*

The baseline scores 0.19 here against 0.24 on the real competition match, a sign that the
simulated noise (below) is about as hard as the real automatic tracking.

---

## The problem in one paragraph

Possession is scored with **spellF1**: each possession (a maximal run of `H` or of `A`) is one
object, matched one-to-one against the truth by temporal IoU and averaged over IoU 0.3–0.7. That
punishes flicker brutally. The frame classifier above gets 84% macro-F1 and is still the *worst*
system on spellF1, because it invents 2,300 possessions in a match that has 373. The whole game is
**segmentation**, not per-frame accuracy. That is why the core of this repo is a Hidden Markov Model.

## Approach

```
open tracking + events
  ├─► ground truth (events → H/A/D per frame)        groundtruth.py
  └─► hackathon-like degradation                     degrade.py
          ball missing on 41% of frames, 46 anonymous track slots,
          20.5 visible players/frame, identity swaps, jitter
        └─► per-frame features (51)                  features.py
              └─► P(H, A, D | frame)  gradient boosting   models/emission.py
                    └─► Viterbi over {H, A, D}     models/hmm.py
                          sticky possession, explicit dead-ball transitions
```

**1. Ground truth from events** (`groundtruth.py`). Possession changes only on events that imply
*control* by the new team: pass, carry, shot, set piece, recovery. Challenges and deflections never
change it, and a loose ball stays with the last team that controlled it, as the challenge's rules
require. The ball is dead from the stoppage (ball out, foul, goal, offside) to the restart. Six
unit tests pin these rules.

**2. Making clean data look like automatic tracking** (`degrade.py`). The open games' ball track is
curated: it disappears almost exactly when the ball is dead (84–99% of dead frames). A model trained
on that learns "no ball = dead ball", which is false on real automatic tracking. So the generator
first *puts a plausible ball back* during stoppages (rolling out, then carried to the restart spot),
then *removes* it with bursty gaps that are likelier in crowds and at high ball speed (balls in the
air), calibrated by bisection to the 41% published for the real data. Player tracks are cut into
tracklets and reassigned to 46 anonymous team slots, so no identity survives. The model only ever
sees this `Observation`, which has the same shape as the competition's ATD.

**3. Features** (`features.py`). Distances from each team's nearest player to the ball, players
within 5 m, ball speed, time since and until the ball is seen, last and next known state, team
centroids and spread in a home-attacking frame, player speeds (stoppages are slow), and rolling
windows of 1, 3 and 8 s.

**4. Classifier + HMM** (`models/`). A gradient-boosted classifier gives P(H, A, D) per frame. It is
trained with every match mirrored (home ↔ away), so it is symmetric by construction. Its posteriors
divided by the class priors become HMM emissions, tempered by `tau` because neighbouring frames are
not independent. Viterbi returns the single most likely segmentation. Four interpretable parameters
(`tau`, switch rate, alive → dead rate, dead → alive rate) are tuned **only on out-of-fold
predictions from the training games**.

## Validation protocol

Nested leave-one-game-out over the three games. Every held-out game is degraded with a noise seed
that no training data used. Nothing is tuned on the data it is scored on, the same principle as the
competition's ban on tuning against its 5 public minutes.

## Reproduce

```bash
pip install -r requirements.txt
python scripts/download_data.py        # ~180 MB of Metrica open data into data/ (git-ignored)
python scripts/evaluate_logo.py        # ~7 min: builds caches, trains, tunes, scores
python -m pytest                       # unit tests, no data needed
```

Results land in `outputs/logo_results.csv` and `outputs/logo_params.json`.

## Repository layout

```
src/possession/
  io_metrica.py     loaders: legacy CSV (games 1-2) and FIFA/EPTS (game 3, and the hackathon ATD)
  groundtruth.py    events -> frame labels
  degrade.py        clean tracking -> hackathon-like noisy tracking
  features.py       per-frame features from an identity-free Observation
  models/           baseline.py (official floor), emission.py (classifier), hmm.py (Viterbi)
  evaluate.py       wrapper around the official metric
  official/         possession_metrics.py, vendored verbatim from the competition (MIT)
scripts/            download_data.py, evaluate_logo.py
tests/              unit tests on synthetic data
```

## Limitations and next steps

- **Under-segmentation.** The HMM reports 258 possessions against 373, so it merges short ones.
  Next: duration-aware transitions (a semi-Markov model), plus features that detect a touch or a
  kick (ball acceleration, sudden change of direction).
- **Three games is a small validation set.** The spellF1 noise band on a full match is about ±0.04.
- **The simulated noise is a model of reality, not reality.** Its parameters come from published
  statistics of the real data, not from that data's actual error patterns.
- **Planned:** an interactive pitch viewer that replays a match with predicted and true possession.

## Data and licence

Code: MIT. The match data is **not** in this repository. It is Metrica Sports' open sample data,
downloaded from [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data).
`src/possession/official/possession_metrics.py` is © Metrica Sports, MIT.
