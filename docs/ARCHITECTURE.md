# Architecture and ownership

## Data flow for one take

1. **`Corrector.analyze`** (`pipeline.py`) converts the take to mono 24 kHz and normalizes its loudness. It runs the tracker on a 16 kHz copy with a 5 ms hop, estimates the key and the background noise, and calls `shifter.prepare`, which runs WORLD analysis, WORLD copy-synthesis and the per-segment quality check. The result is cached in `Analysis`, so every later correction of the same take, such as an A/B pair or a manual tweak, only pays for planning and synthesis.
2. **`planner.plan`** splits the contour into notes and takes each note's centre as the median after the onset. It picks the target (nearest semitone, or nearest in-key note) and splits each note into a slow part (drift and scoops) and a vibrato part, using a moving average one vibrato period long. It then builds the correction `strength * [(target - centre) - slow - (1 - keep) * vibrato]`, smoothed with a one-pole glide whose time constant is the retune speed. Frames with voicing below 0.2 are left untouched.
3. **`shifter.apply`** renders the correction with WORLD (F0 multiplied by the ratio, envelope kept) or PSOLA (Praat Manipulation), depending on how the quality check routed each voiced segment. It crossfades back into the original with 10 ms raised-cosine masks.
4. **`Corrector.score`** re-tracks the output to get the reward terms: in-key accuracy *a*, naturalness *n* (the MCD rise over the WORLD copy-synthesis, new pitch jumps and the HNR drop) and the vibrato penalty *p*.
5. **`AdaptiveSession`** (`agent/session.py`) turns the user's feedback into *r = 0.6 f + 0.2 a + 0.2 n − p*. It updates the personal posterior and saves it with the feedback history to SQLite.

## The agent

* **Actions:** 9 strengths × 7 retune speeds (5–400 ms, log-spaced) × 2 snap modes × 4 vibrato levels = 504.
* **Context (8):** bias, mean and spread of the note error, range, vibrato rate and extent, voiced share, and noise.
* **Population part:** ridge regression on a second-order context expansion crossed with the 15-term quadratic action basis. It is trained in simulation (`results/agent/population_model.json`, shipped in the package) and frozen on the device. Retrain it with pilot data once that exists.
* **Personal part:** 15 Gaussian weights over the quadratic basis, starting at zero so there is no cold start. Most of the prior mass sits on the linear terms, because singers differ mainly in where their ideal settings are. Thompson sampling uses an exploration scale of 0.5, which was tuned in simulation.
* **Baselines:** LinUCB, ε-greedy, fixed setting and population only. Discounted Thompson sampling is the extended-work variant for drift.

## Who owns what (proposal section 7)

| Stream | Owner | Modules |
|---|---|---|
| Agent, experiments, ethics, integration and testing | Muhammad Shahzeb Khan | `agent/`, `study/`, `tests/`, CI |
| Data pipeline, trackers, metrics, ONNX | Haris Khan | `data/`, `pitch/`, `metrics.py`, `bench/`, `scripts/export_onnx.py` |
| Correction engine, desktop app, human studies | Afzaal Afzal | `correction/`, `app/`, `packaging/` |

## Settings that need calibrating with real data

* `QualityConfig.mcd_threshold`: run `adaptitune calibrate` on clean development vocals. The default of 12 dB comes from synthetic voices.
* `QualityConfig.hnr_db` (10 dB) and `min_voicing` (0.5): calibrate these on the labelled real recordings and report the share of segments that goes to each route.
* The reward weights and the agent priors: retune them in simulation after the pilot, so the simulator's preference ranges match what pilot singers actually do.
* `metrics.MCD_RISE_FULL`, `JUMP_RATE_FULL` and `HNR_DROP_FULL`: these scale the naturalness indicators. Check them against pilot naturalness ratings.
