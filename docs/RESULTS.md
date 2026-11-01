# Results so far (synthetic data only)

Every number here comes from **synthetic singing**, because the public datasets could not be downloaded in the build environment. Treat these results as a check that the code works end to end, not as findings for the report. Run `scripts/reproduce_results.sh`, then repeat it with `--source vocalset`, `--dataset mir1k` and `--dataset mdb` once the data is on disk.

Machine: 4-core Intel Xeon @ 2.8 GHz cloud VM, CPU only. This is a different machine from the reference laptop, so re-measure speed there.

## O1: detuned corpus

| | Value |
|---|---|
| Takes / singers | 120 takes from 20 synthetic singers, split by singer into train, val and test |
| Labelled notes | 1,005, of which 564 are detuned by more than 50 cents and kept as a separate set |
| Offsets recovered from the labels | **98.5%** within 15 cents (median error 1.5 cents; WORLD 1.5, PSOLA 1.3) |
| Unique test audio | 0.04 h. **The O1 target is 2 h, so this must come from VocalSet or MedleyDB.** |

## O2: tracker benchmark (40 noisy synthetic takes, SNR 0–20 dB)

| Tracker | RPA | RCA | Voicing recall | Voicing false alarm | RTF | Peak MB |
|---|---|---|---|---|---|---|
| pYIN | 0.716 | 0.716 | 0.718 | 0.182 | 0.071 | 834 |
| YIN | 0.852 | 0.910 | 0.948 | 0.184 | 0.018 | 873 |
| PESTO (PyTorch) | 0.930 | 0.932 | 0.933 | 0.082 | 0.194 | 1153 |
| FCPE | **0.988** | 0.999 | 1.000 | 0.391 | 0.019 | 851 |
| CREPE (tiny) | 0.954 | 0.954 | 0.955 | 0.119 | 0.267 | 1029 |
| PESTO (ONNX export) | 0.998* | | | 0.041* | **0.064*** | |

\* ONNX results are on the clean 4 s unit-test take, not the noisy set. The export matches PyTorch to within 0.001 cents.

FCPE has the best accuracy, but it marks many unvoiced frames as voiced. The planner only leaves frames untouched when their voicing confidence is below 0.2, so that matters. PESTO through ONNX is the deployment choice, because it is fast, accurate and needs no PyTorch in the app. Decide the winner on MIR-1K and MDB-stem-synth.

**Key detection is a real risk.** On short (6–12 note), randomly generated and heavily detuned test takes, the key was right on only 1 of 18. On longer tonal melodies (unit test) it is at least 75% over all 24 keys. When the key is wrong, notes snap to the wrong semitone (see O3), so the manual key option (FR2) matters, and the ~100 hand-labelled excerpts will tell us how bad this is on real songs.

## O3: correction engine (reference setting, 158 test notes, FCPE)

| Metric | Key typed by user | Key detected | Target |
|---|---|---|---|
| Median error, notes < 50 cents off | **3.0 cents** | 10.5 cents | ≤ 25 |
| Share within 25 cents (< 50 set) | 98.5% | 56.9% | |
| Median error, notes ≥ 50 cents off (reported separately) | 3.9 cents | 100.3 cents | |
| Vibrato rate within ±10% | 94% | 94% | |
| Notes keeping at least 70% of vibrato extent | 87% (median ratio 0.95) | same | ≥ 70% |
| WORLD-detuned vs PSOLA-detuned half (< 50 set) | 3.0 / 3.0 cents | 7.1 / 11.3 | |

## O4: personalization agent (simulation only, drift and fatigue on)

100 singers per preference model (300 in total). The oracle is the best fixed setting for each singer in hindsight.

| Policy | After 20 interactions: reward / oracle | Cumulative regret (20) | After 60: reward / oracle | Cumulative regret (60) |
|---|---|---|---|---|
| **Thompson sampling** | 0.827 | 5.48 ± 0.15 | **0.895** | **10.5 ± 0.5** |
| LinUCB | 0.823 | 5.12 ± 0.15 | 0.845 | 11.0 ± 0.6 |
| ε-greedy | 0.809 | 5.49 ± 0.15 | 0.839 | 13.4 ± 0.6 |
| Discounted TS | 0.805 | 5.69 ± 0.14 | 0.883 | 12.8 ± 0.5 |
| Population only (no learning) | 0.682 | 5.44 ± 0.24 | 0.669 | 16.6 ± 1.0 |
| Fixed (classic auto-tune) | 0.243 | 13.2 ± 0.2 | 0.222 | 41.0 ± 0.6 |

**O4 is not met at 20 interactions.** Thompson sampling reaches 83% of the oracle against a 90% target, and its regret ties ε-greedy rather than beating it. By 60 interactions it reaches 89.5% and has the lowest regret of every policy. The ratio is similar across the three preference models (0.79–0.86 at 20 interactions), so this is not one convenient model carrying the result. Options before the study:

1. **Reduce the action space the agent explores**, for example by fixing snapping per singer after the first session.
2. **Get more judgments per session**, e.g. let the agent also learn from C2 and A/B pairs (this needs a protocol change).
3. **Try the fallback**: pairwise preference optimization in the style of Nielsen et al.
4. **Report it as is.** The proposal already treats a negative result as a valid result.

The plots are in `results/agent/learning_curves.png` and `regret_by_model.png`. Run `scripts/run_agent_sim.py` to regenerate them.

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
