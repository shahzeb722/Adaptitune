# Architecture and ownership

## Data flow for one take

1. **`Corrector.analyze`** (`pipeline.py`) converts the take to mono 24 kHz and normalizes its loudness. It runs the tracker on a 16 kHz copy with a 5 ms hop, estimates the key and the background noise, and calls `shifter.prepare`, which runs WORLD analysis, WORLD copy-synthesis and the per-segment quality check. WORLD resynthesizes at its own F0 (DIO and StoneMask), so wherever that F0 is missing or half an octave or more away from the tracker's, the tracker's value is used instead, refined by StoneMask. Without this, DIO's octave jumps on low male voices came out as notes an octave up in the corrected audio. The result is cached in `Analysis`, so every later correction of the same take, such as an A/B pair or a manual tweak, only pays for planning and synthesis.
2. **`planner.plan`** splits the contour into notes and takes each note's centre as the median after the onset. It picks the target (nearest semitone, or nearest in-key note) and splits each note into a slow part (drift and scoops) and a vibrato part, using a moving average one vibrato period long. It then builds the correction `strength * [(target - centre) - slow - (1 - keep) * vibrato]`, smoothed with a one-pole glide whose time constant is the retune speed. Frames with voicing below 0.2 are left untouched.
3. **`shifter.apply`** renders the correction with WORLD (F0 multiplied by the ratio, envelope kept) or PSOLA (Praat Manipulation), depending on how the quality check routed each voiced segment. It crossfades back into the original with 10 ms raised-cosine masks.
4. **`Corrector.score`** re-tracks the output to get the reward terms: in-key accuracy *a*, naturalness *n* (the MCD rise over the WORLD copy-synthesis, new pitch jumps and the HNR drop) and the vibrato penalty *p*.
5. **`AdaptiveSession`** (`agent/session.py`) passes the user's feedback *f* and the measured *a*, *n* and *p* to the agent, logs the reward *r = 0.6 f + 0.2 a + 0.2 n − p*, and saves the personal posterior with the feedback history to SQLite.

## The agent

* **Actions:** 9 strengths × 7 retune speeds (5–400 ms, log-spaced) × 2 snap modes × 4 vibrato levels = 504.
* **Context (8):** bias, mean and spread of the note error, range, vibrato rate and extent, voiced share, and noise.
* **Reward model:** only the feedback *f* depends on the singer, so the agent models the two parts of *r* separately. The expected reward of setting *x* is *0.6 P(f | c, x) + O(c, x)*. Here *logit P(f) = h(c, x) + w · q(x)* and *q(x)* is the 15-term quadratic basis of the four settings.
* **Population part (frozen on the device):** *h*, the population's logit of acceptance, and *O*, the predicted *0.2 a + 0.2 n − p*. Both are ridge regressions on a second-order context expansion crossed with *q(x)*. They are trained on 3,000 simulated singers (`population_model.json`, shipped in the package). Retrain them with pilot data once it exists. *O* depends only on the take and the settings, so it can also be refitted on real takes without any listeners.
* **Personal part:** 15 weights *w*, starting at zero, so a new user first gets the population's choice and there is no cold start. Their Gaussian prior is the spread of individual singers around *h* in the training population. Keep/discard is a Bernoulli outcome, a rating is a fractional outcome counted as three keep/discard decisions, and an A/B pick is a Bradley–Terry comparison. The posterior is the Laplace approximation, refitted after every judgment (a few milliseconds).
* **Thompson sampling** uses half the posterior spread (exploration scale 0.5). The A/B rival is the agent's current best guess, the same rule in the app and in the simulation. The exploration scale and the rating weight were chosen on development singers (`scripts/tune_agent.py`, seed 1). The O4 results use a different set of singers (seed 0).
* **Baselines:** UCB, ε-greedy, fixed setting and population only. Discounted Thompson sampling is the extended-work variant for drift. The first design is kept as the ablation `thompson_linear`: a linear model of *r* itself, which reached only 83% of the oracle.

## Who owns what (proposal section 7)

| Stream | Owner | Modules |
|---|---|---|
| Agent, experiments, ethics, integration and testing | Muhammad Shahzeb Khan | `agent/`, `study/`, `tests/`, CI |
| Data pipeline, trackers, metrics, ONNX, key labelling | Haris Khan | `data/`, `pitch/`, `metrics.py`, `bench/`, `scripts/export_onnx.py`, `tools/key_rater.html` |
| Correction engine, desktop app, human studies | Afzaal Afzal | `correction/`, `app/`, `packaging/` |

## Settings that need calibrating with real data

* `QualityConfig.mcd_threshold`: **calibrated** at 20.93 on the clean takes of the VocalSet corpus's 9 training singers (`scripts/calibrate_quality.py --tracker fcpe`, 15 takes per singer). The tracker finds the voiced segments and guides WORLD's F0, as in the app. With YIN in that role it is 21.85, a little higher because YIN's own octave errors then reach WORLD. Before WORLD took the tracker's F0 where its own jumps an octave, the value was 21.58. Recalibrate the same way if the app's tracker, its sample rate or the WORLD analysis changes.
* `QualityConfig.hnr_db` (10 dB) and `min_voicing` (0.5): calibrate these on the labelled real recordings and report the share of segments that goes to each route.
* The reward weights and the agent's population part: after the pilot, check the simulator's preference ranges against what pilot singers actually do, then retrain. The personal prior comes from the population's spread, so it is re-estimated in the same run.
* `metrics.MCD_RISE_FULL`, `JUMP_RATE_FULL` and `HNR_DROP_FULL`: these scale the naturalness indicators. Check them against pilot naturalness ratings.
* `pitch.key.KeyConfig`: the defaults were chosen on the Essen training volumes (`adaptitune key-eval`). The hand-labelled excerpts are the O2 test set, so report the estimator on them as it is. Do not retune it on them.
* The tracker voicing thresholds are the libraries' defaults and are fixed for O2. If the app needs different ones, tune them on the corpus's development singers, never on MIR-1K or MDB-stem-synth.
