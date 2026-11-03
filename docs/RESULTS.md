# Results so far (synthetic data only)

Every number here comes from **synthetic singing**, because the public datasets could not be downloaded in the build environment. The key results use real melodies (the Essen folksong collection), but these are rendered as synthetic singing too. Treat these results as a check that the code works end to end, not as findings for the report. Run `scripts/reproduce_results.sh`, then repeat it with `--source vocalset`, `--dataset mir1k` and `--dataset mdb` once the data is on disk.

Machine: 4-core Intel Xeon @ 2.8 GHz cloud VM, CPU only. This is a different machine from the reference laptop, so re-measure speed there.

## O1: detuned corpus (144 synthetic takes)

Full datasheet: [O1_CORPUS.md](O1_CORPUS.md).

| | Value |
|---|---|
| Takes / singers | 144 takes from 24 synthetic singers, split by singer (17 train, 4 val, 3 test) |
| Labelled notes | 1,230, each detuned the same way with WORLD and with PSOLA (2,460 labels). 54% are detuned by more than 50 cents and kept as a separate set. |
| QC gate (pYIN re-measures every note) | **98.8%** of labels kept (PSOLA 99.9%, WORLD 97.6%), median residual 0.31 cents |
| Independent re-measurement with FCPE | **99.96%** of kept labels recovered within 10 cents, median residual 0.69 cents, all 578 files match their checksums |
| Unique test audio | 0.04 h. **The O1 target is 2 h, so this must come from VocalSet** (the split planner reserves 2 h of test singers automatically). |

## O2: tracker benchmark (40 noisy synthetic takes, SNR 0–20 dB)

Method, fixed before any real-data run: [O2_PROTOCOL.md](O2_PROTOCOL.md). Every tracker that takes a range searches C1–B6, threads are pinned to 4, and each tracker runs in its own process after a warm-up call.

| Tracker | RPA (95% CI) | RPA within 10 cents | RCA | Overall accuracy | Voicing recall | Voicing false alarm | Octave errors | RTF (4 threads) | Peak MB |
|---|---|---|---|---|---|---|---|---|---|
| FCPE | **0.986** (0.968–0.998) | 0.935 | 0.990 | 0.887 | 0.996 | 0.476 | 0.005 | 0.027 | 823 |
| PESTO (PyTorch) | 0.963 (0.937–0.983) | 0.684 | 0.978 | 0.900 | 0.933 | 0.143 | 0.016 | 0.151 | 1067 |
| PESTO (ONNX export, what the app ships) | 0.956 (0.928–0.979) | 0.720 | 0.969 | 0.896 | 0.932 | 0.144 | 0.012 | 0.037 | 725 |
| CREPE (tiny) | 0.913 (0.850–0.967) | 0.525 | 0.923 | 0.897 | 0.894 | 0.098 | 0.011 | 0.265 | 993 |
| YIN | 0.889 (0.857–0.921) | 0.684 | 0.943 | 0.774 | 0.981 | 0.653 | 0.054 | 0.022 | 724 |
| pYIN | 0.855 (0.803–0.903) | 0.667 | 0.861 | 0.836 | 0.841 | 0.189 | 0.006 | 0.136 | 721 |

* **FCPE is the most accurate and among the fastest.** It is ahead of PyTorch PESTO by 2.3 RPA points (paired 95% CI 0.0 to 5.1), and it is much more precise: 93.5% of frames within 10 cents against about 70% for PESTO. Its weak spot is voicing: it calls 48% of unvoiced frames voiced.
* **The ONNX export of PESTO runs 4× faster than PyTorch PESTO** (RTF 0.037 against 0.151) at the same accuracy. The difference is within the confidence intervals; the export uses a 10 ms step and PyTorch a 5 ms one.
* **CREPE tiny is over the 0.25 RTF budget** on this machine, and its released weights were trained on both MIR-1K and MDB-stem-synth, so the selection rule cannot choose it.
* On these synthetic takes the pre-registered rule would choose FCPE. **The real choice is made on MIR-1K and MDB-stem-synth.**
* **Timing check:** shifting each tracker's output by −30 to +30 ms, the pitch accuracy of every tracker peaks at 0 ms, so frame times are aligned.

## O2: key estimation (Essen folksong collection, held-out volumes)

The estimator was rebuilt and tuned on the training volumes of the Essen collection (4,670 songs). It was then tested on 2,486 different songs, rendered as in-tune and amateur singing. Details are in [O2_PROTOCOL.md](O2_PROTOCOL.md#b2-development-and-held-out-test-on-the-essen-folksong-collection).

| Exact key | Before (Krumhansl–Kessler on frames) | Now (Bellman–Budge on notes, tuning removed, final-note cue) |
|---|---|---|
| In tune, whole song | 64.4% | **94.5%** |
| In tune, 8–20 s excerpt | 46.0% | **80.4%** |
| Amateur, whole song | 52.8% | **90.5%** |
| Amateur, 8–20 s excerpt | 35.8% | **73.8%** (95% CI 72.0–75.5) |
| Minor-key items | 39.7% | **67.5%** |

* **The O2 key target (75% on about 100 hand-labelled excerpts) is within reach but not certain.** Excerpts of amateur-like singing sit right at it, so the labelled excerpts should be long (12–30 s). `adaptitune key-excerpts --source mir1k` makes them that way: one excerpt per MIR-1K song.
* **Detuned singing is not a fair key test** (7–11%, because random 10–100 cent offsets wipe out the pitch-class evidence). O3 is therefore scored with the key the user types.
* **Minor melodies often leave natural minor.** 64% of Essen's minor songs use the raised 7th, and only 6% stay inside natural minor. In-key snapping in a minor key therefore now allows the raised 6th and 7th too (`Key.snap_classes`), so leading tones are no longer pulled down.
* On audio, the key came out the same from pYIN (80.0%), PESTO through ONNX (79.0%) and the true contour (80.0%), so tracker errors are not the bottleneck.

## O3: correction engine (reference setting, 332 QC-passed test labels from 18 takes, FCPE)

Rerun on the new corpus (both shifter copies of every test take, QC-passed labels only).

| Metric | Key typed by user | Key detected | Target |
|---|---|---|---|
| Median error, notes < 50 cents off | **2.5 cents** | 4.4 cents | ≤ 25 |
| Share within 25 cents (< 50 set) | 95.8% | 69.7% | |
| Median error, notes ≥ 50 cents off (reported separately) | 4.3 cents | 99.0 cents | |
| Vibrato rate within ±10% | 96% | 96% | |
| Notes keeping at least 70% of vibrato extent | 80% (median ratio 0.93) | 80% (median ratio 0.94) | ≥ 70% |
| WORLD-detuned vs PSOLA-detuned copies (< 50 set) | 2.3 / 2.7 cents | 4.0 / 6.0 cents | |

The key was detected correctly on only 2 of the 18 detuned test takes. This matches the Essen result for detuned singing (7–11%), so O3 is reported with the key typed by the user. The detected-key column shows what happens when the key is wrong.

## O4: personalization agent (simulation only, drift and fatigue on)

300 simulated **report singers**, 100 per preference model (seed 0). The agent's two settings, its exploration scale and how much a rating counts, were chosen beforehand on 300 different **development singers** (seed 1, `scripts/tune_agent.py`). The population part is trained on a third set of 3,000 singers. The oracle is each singer's best fixed setting in hindsight. Reproduce with `scripts/run_agent_sim.py` (add `--horizon 60` for the curves, `--sensitivity` for the robustness table).

**O4 is met.** After 20 interactions, Thompson sampling's best guess earns **91.4%** of the oracle's reward (95% CI 89.8–92.9%, bootstrap over singers). Its cumulative regret, 3.28, is lower than ε-greedy's (3.98) and the fixed setting's (13.2). Each singer meets every policy with the same feedback noise, so the differences are paired: Thompson sampling minus ε-greedy is −0.70 (95% CI −0.94 to −0.46).

| Policy | After 20: reward / oracle | Ideal point / logistic / threshold | Cumulative regret (20) | After 60: reward / oracle | Cumulative regret (60) |
|---|---|---|---|---|---|
| **Thompson sampling** | **0.914** | 0.897 / 0.923 / 0.922 | **3.28 ± 0.15** | **0.949** | 6.1 ± 0.3 |
| Discounted TS (drift, extended work) | 0.917 | 0.893 / 0.927 / 0.933 | 3.30 ± 0.14 | 0.954 | 6.4 ± 0.3 |
| UCB | 0.895 | 0.860 / 0.914 / 0.913 | 2.85 ± 0.15 | 0.928 | 5.9 ± 0.4 |
| ε-greedy (ε = 0.1) | 0.856 | 0.818 / 0.886 / 0.866 | 3.98 ± 0.17 | 0.874 | 10.3 ± 0.4 |
| First design: linear TS (ablation) | 0.812 | 0.764 / 0.855 / 0.818 | 5.59 ± 0.15 | 0.886 | 11.0 ± 0.4 |
| Population only (no learning) | 0.709 | 0.657 / 0.740 / 0.732 | 4.91 ± 0.24 | 0.703 | 15.1 ± 0.7 |
| Fixed (classic auto-tune) | 0.241 | 0.229 / 0.286 / 0.207 | 13.2 ± 0.2 | 0.246 | 39.4 ± 0.5 |

Regret is mean ± standard error over the 300 singers. Thompson sampling passes 90% at about the 15th interaction (`results/agent_h60/learning_curves.png`). For individual singers, the median reaches 97% of their oracle and 73% of singers reach 90%.

**What changed since the first run (83%).** The agent's model, not its tuning. The first design fitted a linear-Gaussian model to the reward itself. A singer keeps takes made near their preferred settings and rejects almost everything else, and a quadratic in *r* cannot follow such a steep surface. As a check on development singers, a linear model fitted to the noise-free reward of all 504 settings still picks a setting worth only 84% of the oracle. So no amount of data or tuning could reach 90%. The new model splits *r* as the proposal defines it. The singer's feedback *f* gets a logistic link, and the measured *a*, *n* and *p* come from the population, because they do not depend on taste. Its ceiling on the same check is 99%. Details are in `agent/policies.py` and [ARCHITECTURE.md](ARCHITECTURE.md#the-agent). Two smaller changes apply to every policy:

* **The A/B rival** is now the agent's current best guess, as the app already did. The simulation used to compare against the population's choice.
* **Singer seeds are now stable.** They were derived from Python's `hash()` of the model name, which changes from run to run, so the old table could not be reproduced exactly. The study schedule had the same problem and is fixed too.

**How sensitive is the pass?** Each row changes one assumption and reruns the comparison on the report singers:

| Change | TS reward / oracle | TS regret | ε-greedy reward / oracle | ε-greedy regret | O4 |
|---|---|---|---|---|---|
| None (as reported) | 0.914 | 3.28 | 0.856 | 3.98 | pass |
| Keep/discard feedback only, no ratings or A/B | 0.904 | 3.48 | 0.825 | 4.20 | pass |
| Personal prior twice too wide | 0.930 | 3.49 | 0.870 | 3.73 | pass |
| Personal prior half as wide | 0.878 | 3.45 | 0.832 | 4.31 | fail (ratio) |
| Population trained on 300 singers, not 3,000 | 0.893 | 3.46 | 0.829 | 4.24 | fail (ratio) |
| Population trained on logistic singers only | 0.925 | 3.26 | 0.845 | 3.94 | pass |
| … on ideal-point singers only | 0.913 | 3.19 | 0.825 | 4.36 | pass |
| … on threshold singers only | 0.909 | 3.36 | 0.833 | 4.22 | pass |

In every row Thompson sampling beats ε-greedy on both measures. The 90% line itself depends on a personal prior that is wide enough and a well-trained population part.

Caveats for the report:

* **This is simulation.** The population and the personal prior are estimated from the same simulator as the test singers, only from different individuals. Before the study, retrain them on pilot data (ARCHITECTURE.md). The user study is the real test.
* **UCB has lower regret than Thompson sampling** (2.85 against 3.28), though its 20-interaction best guess is worse (0.895). The O4 criterion compares Thompson sampling with ε-greedy and the fixed setting only. Report UCB anyway.
* **Ideal-point singers are the hardest** (89.7%), because their acceptance is a plateau that a quadratic logit only approximates.

Choices made on the development singers (300 singers, seed 1; `results/agent_tuning.csv`):

| Policy | Setting | Reward / oracle (20) | Regret (20) |
|---|---|---|---|
| Thompson sampling | exploration 0.5, rating counts as 1 | 0.899 | 3.50 |
| **Thompson sampling (chosen)** | **exploration 0.5, rating counts as 3** | **0.918** | **3.14** |
| Thompson sampling | exploration 0.75, rating counts as 3 | 0.928 | 3.91 |
| Thompson sampling | exploration 1.0, rating counts as 3 | 0.928 | 4.98 |
| UCB | α = 1.0 (chosen) / 0.5 | 0.892 / 0.857 | 2.91 / 3.16 |
| ε-greedy | ε = 0.1 | 0.850 | 3.99 |
| Linear TS (first design) | as before | 0.822 | 5.66 |

An exploration scale of 0.5 was kept because larger scales gain about one point of reward but lose the regret margin over ε-greedy.

## O5: offline application

| | Value |
|---|---|
| Full-pipeline RTF, FCPE | **0.29** (tracker 0.18, plan 0.004, shift 0.10) |
| Full-pipeline RTF, pYIN | 0.38 |
| Full-pipeline RTF, PESTO (PyTorch) | 0.41 |
| Target | ≤ 0.5 on the reference laptop |
| Offline test | `tests/test_session_study.py::test_full_flow_offline` runs FR1–FR7 with all socket calls blocked |

The quality check sends 90% of segments to PSOLA on synthetic voices, because WORLD copy-synthesis distorts them heavily: the calibrated 95th percentile is 26.7 dB against the 12 dB default. **Calibrate on real clean vocals with `adaptitune calibrate`.**

## O6: study design (power simulation)

These numbers detect a 15-point acceptance gain (C3 vs C1, sessions 3–5; 95% CI excludes 0). They assume a 45% base acceptance and a singer random effect with sd 0.8 on the log-odds scale.

| Singers | 10 | 12 | 15 | 20 |
|---|---|---|---|---|
| Power | 0.50 | 0.58 | 0.62 | 0.73 |

This confirms what the proposal expects: with 10–15 singers, O6 should be reported as an estimate with its confidence interval, and the significance tests as exploratory.
