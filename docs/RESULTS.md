# Results

Status on 4 November 2026. O1, both parts of O2 and O3 now run on real data. O2's key test uses the Children's Song Dataset in place of the hand labels, because nobody was available to label keys by ear. O4 and O6 are simulations by design. The O5 app is built for Windows in CI and passes its own checks there, but its speed has to be measured on the reference laptop. The earlier synthetic-only results are kept in the appendix as code checks.

| Objective | Status | Main number |
|---|---|---|
| O1 evaluation corpus | **Met** (VocalSet) | 2.20 h of unique test audio from 9 singers not used in development; 99.5% of kept offsets recovered within 10 cents by a second tracker |
| O2 pitch trackers | **Met** (MIR-1K, MDB-stem-synth); the laptop speeds still have to confirm the choice | 4 trackers on both sets; the rule chooses FCPE, with RPA 0.941 on MDB-stem-synth (target 0.90) |
| O2 key estimation | **Met** on the Children's Song Dataset, which replaced the hand labels. One professional singer, so it is easier than amateur singing | 86.9% exact on excerpts that stop before each song's last note (target 75%; 95% CI 80.3–92.9, resampling songs) |
| O3 correction | **Met** (VocalSet test singers) | Median error 7.7 cents on notes detuned by under 50 cents (target 25); the median vibrato note keeps its rate within 2% and 97% of its extent |
| O4 personalization | **Met** in simulation | Thompson sampling reaches 91.4% of the oracle's reward after 20 interactions |
| O5 offline app | Built for Windows and checked there; laptop run pending | Every feature passes in the packaged Windows app with the network blocked; the full correction path runs at 0.22× real time on real singing on the cloud VM (target 0.5 on the laptop) |
| O6 user study | Design ready | Power simulation below |

**Machine.** The real-data runs used a shared 4-core Intel Xeon @ 2.8 GHz cloud VM, CPU only, with several jobs running side by side at 1 thread each. Accuracy does not depend on that, but speed does, so the speed numbers below are not the reference laptop's. `scripts/o2_laptop_speed.py` measures them there.

## O1: evaluation corpus (VocalSet)

**O1 is met.** Full datasheet: [O1_CORPUS.md](O1_CORPUS.md), section 8. Records: `docs/results/o1/`.

| | Value |
|---|---|
| Unique clean audio | 4.81 h in 1,512 takes from all 20 VocalSet singers. Test **2.20 h** (female1, 2, 4, 7, 8 and male2, 3, 4, 11), train 2.13 h, validation 0.48 h. No singer is in two splits. |
| Labelled notes | 8,916 notes, each detuned the same way with WORLD and with PSOLA (17,832 labels). Offsets 10–100 cents (median 54.5), plus drift, scoops and changed vibrato. 55% are detuned by more than 50 cents and kept as a separate set. |
| QC gate (pYIN) | **89.8%** of labels kept (PSOLA 91.2%, WORLD 88.4%). The test split keeps 7,624 labels. |
| Second measurement (FCPE) | **99.48%** of the 16,011 kept labels recovered within 10 cents, median residual 0.75 cents. All 6,050 files match their checksums. |
| Targets | Each note's target is the semitone nearest the singer's own note centre, as FCPE measures it, and only notes within 20 cents of a semitone are labelled. pYIN puts 20.6% of the kept notes more than 20 cents from their target: the two trackers disagree by a median of 3.8 cents on a note centre, most on vibrato. O3 reports this as its clean floor. |

The corpus was built twice. The first build used PESTO's ONNX export as the analysis tracker, and the O2 benchmark then showed that PESTO reports steady pitch in thirds of a semitone, so the rebuild uses FCPE, the tracker O2 chose. Section 8 of the datasheet has the comparison.

## O2: pitch-tracker benchmark (MIR-1K and MDB-stem-synth)

Method, fixed before any real-data run: [O2_PROTOCOL.md](O2_PROTOCOL.md). MIR-1K: the vocal channel of all 1,000 clips (19 singers). MDB-stem-synth: all 230 stems (85 tracks). Confidence intervals are cluster bootstraps over singers or tracks. Records: `docs/results/o2/`.

| Tracker | Score under the rule | MIR-1K RPA (95% CI) | MDB-stem-synth RPA (95% CI) | Within 10 cents, MIR-1K / MDB | Median error in cents, MIR-1K / MDB | RTF (cloud, 1 thread), MIR-1K / MDB |
|---|---|---|---|---|---|---|
| **FCPE (chosen)** | **0.965** | **0.988** (0.985–0.991) | **0.941** (0.928–0.953) | 0.695 / 0.857 | 6.6 / 1.6 | 0.050 / 0.086 |
| pYIN | 0.929, not eligible (RTF over 0.25) | 0.966 (0.959–0.972) | 0.893 (0.865–0.917) | 0.629 / 0.708 | 7.3 / 4.0 | 0.41 / 0.31 |
| YIN | 0.926 | 0.960 (0.952–0.968) | 0.891 (0.866–0.912) | 0.627 / 0.705 | 7.2 / 3.9 | 0.023 / 0.064 |
| PESTO, ONNX export | 0.882 (MDB only) | 0.983 (0.978–0.987), trained on this set | 0.882 (0.854–0.907) | 0.462 / 0.540 | 10.2 / 7.8 | 0.17 / 0.16 |
| PESTO, PyTorch | 0.886 (MDB only), not eligible (RTF over 0.25) | 0.983 (0.978–0.987), trained on this set | 0.886 (0.858–0.910) | 0.462 / 0.535 | 10.2 / 8.1 | 0.42 / 0.81 |
| CREPE (tiny) | None: trained on both sets | 0.971 (0.966–0.976), trained on this set | Not finished (60 of 230 stems) | 0.598 / – | 7.8 / – | 0.75 / – |

* **O2's tracker criteria are met.** Four trackers are reported on both datasets, and FCPE, the tracker the pre-registered rule chooses, reaches an RPA of 0.941 on MDB-stem-synth against the 0.90 target. The rule would choose FCPE even if training overlap were ignored.
* **FCPE wins clearly.** Against the runner-up, YIN, the paired difference is +2.8 RPA points on MIR-1K (95% CI 2.3 to 3.4) and +5.0 on MDB-stem-synth (3.5 to 6.8). It is also the most precise: 86% of MDB-stem-synth frames within 10 cents, where the next best gets 71%.
* **PESTO reports steady pitch in thirds of a semitone,** in PyTorch and in the ONNX export alike, so its accuracy within 10 cents is the lowest in the table. Its 50-cent RPA on MIR-1K is high, but PESTO was trained on MIR-1K audio, so the rule scores it on MDB-stem-synth only. There it also calls 60% of voiced frames unvoiced (voicing recall 0.405), which costs overall accuracy, not RPA.
* **The reference timing checks out.** With each tracker's output shifted from −30 to +30 ms, every tracker's RPA peaks at 0 ms on both datasets. PESTO's within-10-cent accuracy peaks at +5 ms on MIR-1K, half of its 10 ms step.
* **PyTorch PESTO finished on both sets, and CREPE on MIR-1K only.** The cloud machine restarted after CREPE had done 60 of the 230 MDB-stem-synth stems, and the rest was not rerun: CREPE was trained on both sets, so the rule reports it but cannot choose it. PyTorch PESTO matches its ONNX export (0.886 against 0.882 on MDB-stem-synth, and the same 0.983 on MIR-1K). It would have needed 0.955 to come within the 1-point tie margin of FCPE's score, so the choice does not change. Records: `docs/results/o2/pytorch_pesto_crepe/`.
* **These speeds are not the official ones.** Here every job ran at 1 thread with up to four jobs sharing the VM. FCPE's slowest RTF, 0.086, is a third of the 0.25 limit. The laptop run gives the numbers for the report, and `adaptitune choose-tracker --speed` applies the rule again with them.
* **The app runs FCPE through ONNX Runtime,** with no PyTorch. `scripts/export_onnx.py --model fcpe` makes the export, and on a test clip it matches the PyTorch model within 0.5 cents on every voiced frame, with the same voicing decisions (`tests/test_bench_o2.py`).

## O2: key estimation

**Met on the Children's Song Dataset (CSD), which replaced the hand labels.** O2's key criterion was at least 75% exact on about 100 excerpts labelled by two people. Nobody with the ear to label keys was available, so the test moved to CSD: 50 Korean and 50 English children's songs, each sung in two keys by one professional singer, with MIDI notes aligned by hand. Each recording's reference key comes from those notes: the last note gives the tonic, and the third above it gives the mode. The method was committed before the data arrived ([O2_PROTOCOL.md, B4](O2_PROTOCOL.md#b4-replacement-for-the-hand-labels-the-childrens-song-dataset-4-november-2026)). The app's own path, FCPE through ONNX Runtime and then the key estimator, named the exact key on **86.9%** of the excerpts. Records are in `docs/results/o2/keys_csd/`. The run used commit `df3f977` with one fix, committed next: the check on song pairs had crashed on the song without a reference.

| | Excerpts (the O2 figure) | Whole recordings |
|---|---|---|
| Exact key, all 198 scored recordings | **86.9%** (172) | 92.4% (183) |
| 95% CI, Wilson | 81.5–90.9 | 87.9–95.4 |
| 95% CI, resampling songs | 80.3–92.9 | 87.4–97.0 |
| Same scale (relative keys count as right) | 88.9% | 92.4% |
| MIREX score | 0.893 | 0.930 |
| Exact key, English / Korean | 85.7% / 88.0% | 88.8% / 96.0% |
| Exact key, major references (190) / minor (8) | 90.5% / 0% | 96.3% / 0% |

Each excerpt starts at the first note and stops before the song's last note: 14–25 s, median 24.6 s. "Hokey pokey" never sings the third above its last note, so its two recordings have no reference and are not scored.

* **The references pass the planned checks.** Both versions of all 99 scored songs differ by exactly the transposition between their note lists. No recording's MIDI sits more than half a semitone from the pitch FCPE hears (the largest offset is 0.30 semitone). Krumhansl–Kessler key finding on the same notes agrees with 73.7% of the references. Where it differs, it is mostly in its usual ways: the dominant (22), or a minor key on the third (14) or sixth degree (8). On the 146 excerpts where it agrees, the app scores 90.4%.
* **All 8 minor references are major-key songs that end on their third, so the 0% says nothing about minor keys.** They are four songs, each in both keys: "Butterfly" and "나비야" (the same tune), "The first noel" and "마법의 성". In each, 6–15% of the sung time is on a note that the reference's minor key lacks even with its raised 6th and 7th (B♭ against A minor, for example), while the major key a major third below holds 97–100% of it. That major key is the app's answer on every one of these excerpts and whole recordings (`scripts/csd_misses.py`, `docs/results/o2/keys_csd/misses.json`). As fixed in advance, the references were not changed, so these count as misses. The protocol's by-mode figure leaves them out: 90.5% of excerpts on major references (Wilson 85.5–93.9). The evidence on minor keys is still Essen's 67.5% below.
* **The other 18 misses are close keys, and the song decides.** Twelve are the dominant (5), the relative minor (4), the subdominant (2) or the parallel minor (1), and the whole recording gets 11 of those 12 right. The other 6 are both versions of the only three songs in the set that use 11 or 12 of the 12 pitch classes ("If all the little raindrops", "I'm the music man" and "One for sorrow"), which are missed on the whole recording too. In 16 of the 18 misses the other version of the same song is missed as well, so the interval that resamples songs is the one to quote.
* **This test is easier than the planned one.** It has one professional singer and simple, mostly diatonic songs, and the reference uses a cue the app also uses, the last note. The excerpts leave that note out but keep earlier phrase endings. Amateur singing may still fall short of 75%. Amateur-like Essen excerpts of 8–20 s give 73.8% (below), and on the 31 MIR-1K excerpts where voice and karaoke backing line up, the app matched the backing's key on 18 (58%).
* **MIR-1K's karaoke backing could not replace the labels.** In 60 of the 100 MIR-1K key excerpts, the singer's pitch classes fit the stored backing best at another transposition, 39 of them clearly, so the singer is often not in the backing's key (`scripts/mir1k_backing_check.py`, `docs/results/o2/mir1k_backing_check.json`). That is why the 58% above comes from a small subset and is a rough sign, not a result.

### Development on the Essen folksong collection

The estimator was rebuilt and tuned on the training volumes of the Essen collection (4,670 songs). It was then tested on 2,486 different songs, rendered as in-tune and amateur singing. Details are in [O2_PROTOCOL.md](O2_PROTOCOL.md#b2-development-and-held-out-test-on-the-essen-folksong-collection).

| Exact key | Before (Krumhansl–Kessler on frames) | Now (Bellman–Budge on notes, tuning removed, final-note cue) |
|---|---|---|
| In tune, whole song | 64.4% | **94.5%** |
| In tune, 8–20 s excerpt | 46.0% | **80.4%** |
| Amateur, whole song | 52.8% | **90.5%** |
| Amateur, 8–20 s excerpt | 35.8% | **73.8%** (95% CI 72.0–75.5) |
| Minor-key items | 39.7% | **67.5%** |

* **Amateur excerpts sit right at the target.** Amateur-like excerpts of 8–20 s give 73.8%, which is why the planned hand-labelled excerpts were longer (12–30 s, one per MIR-1K song).
* **Detuned singing is not a fair key test** (7–11%, because random 10–100 cent offsets wipe out the pitch-class evidence). O3 is therefore scored with the key the user types.
* **Minor melodies often leave natural minor.** 64% of Essen's minor songs use the raised 7th, and only 6% stay inside natural minor. In-key snapping in a minor key therefore allows the raised 6th and 7th too (`Key.snap_classes`), so leading tones are not pulled down.
* On audio, the key came out the same from pYIN (80.0%), PESTO through ONNX (79.0%) and the true contour (80.0%), so tracker errors are not the bottleneck.

## O3: correction engine (VocalSet test singers)

**O3 is met.** At the reference setting the median error on notes detuned by under 50 cents is **7.7 cents**, against the 25-cent target. On notes with vibrato, the median note keeps its rate within 2% and 97% of its extent (targets ±10% and 70%).

The reference setting is full strength, a 10 ms retune, in-key snapping and full vibrato preservation. It corrected both detuned copies of every test take that has QC-passed labels: 625 takes and 7,624 labels from the 9 test singers. The corrector runs FCPE with the WORLD quality check calibrated on the training singers. pYIN measures the output, so no note is scored by the tracker that corrected it. The key is the one stored with the take, as a user would type it, because key detection on detuned singing is near chance (see O2 keys). Records are in `docs/results/o3/`, from commit `d227c9d`.

| | Detuned by under 50 cents | 50–100 cents (reported separately) |
|---|---|---|
| Labels | 3,397 | 4,227 |
| **Median error after correction** | **7.7 cents** | 26.1 cents |
| Within 25 cents | 74.9% | 49.5% |
| Clean floor: the same notes in the clean take, measured by pYIN | 11.5 cents (87.2% within 25) | 11.6 cents |
| PSOLA-detuned / WORLD-detuned copies | 7.5 / 8.1 cents | 24.7 / 28.3 cents |
| Targets inside the key (3,281) / outside it (116) | 7.2 / 102 cents | |

| Under 50 cents, by technique | Straight | Slow piano | Messa di voce | Slow forte | Vibrato |
|---|---|---|---|---|---|
| Median error after correction | 2.9 | 7.1 | 9.6 | 12.8 | 13.9 |
| Clean floor | 10.4 | 11.1 | 7.8 | 11.9 | 14.2 |

| Vibrato (5,039 notes with at least 20 cents of vibrato in the clean take) | Result | Target |
|---|---|---|
| Median change in vibrato rate | 2.0% | within ±10% |
| Notes with the rate within ±10% | 85.4% | |
| Median share of vibrato extent kept | 97% | at least 70% |
| Notes keeping at least 70% of their extent | 84.5% | |

* **The corrector ends up closer to the targets than the clean takes are.** The clean floor is what a corrector that exactly undid the injected error would score. pYIN hears the clean notes a median of 11.5 cents from their targets, partly from the singers' own intonation and partly because pYIN and FCPE disagree on note centres (O1). The corrector moves each note to the target FCPE hears, so it lands at 7.7. Straight singing, where the two trackers agree best, comes out at 2.9 cents; vibrato and loud singing, where they disagree most, at about 13.
* **15.7% of the under-50 notes still end more than 50 cents off.** `scripts/o3_miss_causes.py` re-plans each one and sorts it by cause:

  | Cause | Notes | Share of the set |
  |---|---|---|
  | The target is outside the key, so in-key snapping cannot reach it | 109 | 3.2% |
  | The detuned note sat nearer the neighbouring semitone: offset, drift and the singer's own deviation together put it more than 50 cents out | 143 | 4.2% |
  | The corrector merged the note with a neighbouring one | 70 | 2.1% |
  | The corrector's note centre was past halfway for another reason | 58 | 1.7% |
  | Right note, but pYIN hears the output an octave or more off (FCPE hears all 57 within 50 cents of the target) | 57 | 1.7% |
  | Right note, and pYIN misreads the clean note as well | 35 | 1.0% |
  | Right note, output measured 50–600 cents off | 61 | 1.8% |

  The first two causes are beyond any corrector that snaps to the nearest note in the key, and the next two are the corrector's own note segmentation. The proposal's reason for the 50-cent line, that anything further is closer to the next semitone, applies to the second row too: the line is drawn on the constant offset alone, and drift can push a note past it. On the 2,765 notes with an in-key target that were nearer their target than the neighbouring semitone, the median error is 6.6 cents and 80.5% land within 25.
* **A bug this run found and fixed.** WORLD resynthesizes at its own pitch estimate, and on low male voices that estimate jumps an octave on about 5% of voiced frames (7% for male3 and male4). In the first run, 74 under-50 notes with the right note planned measured an octave or more off, and FCPE heard 34 of them an octave up as well: real octave jumps in the audio, which the quality check had not caught. WORLD now takes the tracker's pitch wherever its own is missing or half an octave or more away, and the threshold was recalibrated (20.93, from 21.58). The numbers above are from the rerun. Before the fix the median was also 7.7 cents, with 16.1% misses. After it, 57 notes with the right note planned still measure an octave or more off, and FCPE hears every one of them within 50 cents of its target, so those are pYIN's errors on the processed audio.
* **Mean RTF during this evaluation was 0.31,** at 1 thread with four jobs sharing the VM and PyTorch FCPE. That is not the O5 number.

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

The app is a PySide6 window over the same pipeline and agent as the experiments. `packaging/adaptitune.spec` makes it a one-folder Windows program that needs no Python, PyTorch or internet connection, and the `windows-app` workflow builds it on every push. Two checks are built into the app under Help. Both also run from the command line and write JSON:

* **Check this installation** (`AdaptiTune.exe --self-test`) drives every feature through the window, the way the buttons do, on a 7 s detuned take. Every socket call is blocked and recorded, and the profile is a throwaway folder (`adaptitune/app/checks.py`).
* **Speed test** (`AdaptiTune.exe --speed-test MIR1K_FOLDER`) times the full correction path at 4 threads, best of 3, on the first 64 s of MIR-1K vocals, the same audio as the cloud run below. The report names the CPU, memory and build.

The workflow removes PyTorch from the machine after the build, runs both checks on the packaged exe, and keeps the build as the `AdaptiTune-windows` artifact. The tests run the same self-test on every push (`tests/test_app.py`), along with FR1–FR7 on the session layer with sockets blocked (`tests/test_session_study.py`). [USER_GUIDE.md](USER_GUIDE.md) says how to install it and run the checks on the laptop.

| Requirement | How it is checked | Result (packaged app) |
|---|---|---|
| FR1 record or import a vocal | The self-test imports a WAV through the same path as **Import vocal…**. Recording needs a microphone, so the self-test only reports whether there is one, and `tests/test_app.py` covers Record without one | Pass. Neither machine had a microphone |
| FR2 key by hand or detected | Detects the take's key, then sets G major by hand | Pass: C major detected |
| FR3 correct, play back, compare | Corrects with the profile's settings. Playback (Qt Multimedia) is checked by ear | Pass |
| FR4 keep, rate or pick A/B | A keep, a rating and an A/B pick | Pass: 3 judgments stored |
| FR5 tune the four settings, save | Strength 50%, apply, save, then the profile is read back from disk | Pass |
| FR6 per-note pitch error | One table row per note; median error before and after | Pass: 9 notes, 30 cents before, 13 after (cloud) and 16 after (Windows). The agent samples its settings, so the after-error varies a little |
| FR7 reset or export the profile | Export to JSON, then reset | Pass |
| NFR1 no internet | Every socket call blocked and recorded for the whole self-test | Pass: no connection attempted |
| NFR2 RTF ≤ 0.5 on the laptop | The speed test on the MIR-1K folder | 0.17 on the cloud VM. Laptop pending |
| NFR3 data stays on the device | Lists every file the self-test wrote | Pass: the profile is one SQLite file in the profile folder, and with no connection (NFR1) nothing leaves the device |
| NFR4 Windows 10 or later | The workflow builds the exe on Windows and runs both checks on it | Pass on Windows Server 2025 (build 26100) |

"Pass" means the packaged app passed on the cloud VM (Linux) and on GitHub's Windows runner, both from commit 12e292c. The reference laptop still has to give three results: the speed test on the MIR-1K folder (NFR2), every feature with Wi-Fi off (NFR1), and a recording through its microphone (FR1). Records and the machines they came from: `docs/results/o5/app_checks/`.

| Speed | Value |
|---|---|
| Full-pipeline RTF on real singing | **0.22** on a quiet 4-core 2.8 GHz cloud VM at 4 threads, 3 November (`docs/results/o5/cloud_pipeline_rtf.json`): analysis 0.16 (tracker, WORLD analysis and the quality check), planning 0.001, shifting 0.06. WORLD took 99% of the corrected audio. |
| The same input through the app's speed test | **0.17** packaged and 0.18 run from source (0.168 and 0.176), on a 4-core 2.1 GHz cloud VM, 4 November, so packaging costs nothing. That VM is a different one from 3 November's, so the two days' numbers are not a trend. |
| Windows runner, synthetic singing | 0.16 (AMD EPYC, 4 vCPUs). Synthetic voices send half the audio through PSOLA (A4), so this only shows that the Windows build runs at a normal speed. |
| First analysis after installing | The self-test's 7 s take took 17 s to analyse at the first launch on Windows and 19 s on the cloud VM, against 2 s at the second launch on the cloud VM: librosa's numba code is compiled for the processor once and kept. The speed test reports the best of 3 runs, so this is not in it. |
| Target | RTF ≤ 0.5 on the reference laptop: Help → Speed test on the MIR-1K folder, or `scripts/o2_laptop_speed.py --pipeline-only` in a checkout |
| Pitch tracker | FCPE through ONNX Runtime, so the app needs no PyTorch. The Windows build exports it in CI and checks the export against PyTorch: 0.004 cents apart on a 220 Hz tone |
| Quality check | Calibrated on real singing. The WORLD threshold is the 95th percentile of WORLD's copy-synthesis distortion over 135 clean takes from the 9 training singers: 20.93 (21.58 before WORLD took the tracker's pitch, see O3), in the code's own mel-cepstral distance, which is not comparable with SPTK's. Records in `docs/results/o3/quality_calibration_*.json`. In the O3 run it sent 91.6% of the corrected test audio through WORLD and the rest through PSOLA. |

**What driving the app found.** The unit tests passed while the app itself did not work. Every button that runs in the background (import, correct, A/B, apply) stayed on "Analysing…": PySide discards a job's finished signal when nothing references the job, so the window now holds each running job. Drawing a long take's pitch curve crashed the app with "deallocating None", because PySide6 6.12.0 under Python 3.11 drops a reference to `None` on every call that returns nothing; PySide6 is pinned below 6.12. The packaged app looked for its FCPE model next to the exe, where PyInstaller 6 no longer puts data files, so it fell back to pYIN, which is too slow for NFR2; the self-test now fails a packaged build that does not run the bundled FCPE. Smaller fixes: Record with no microphone left the button stuck, and the import dialog offered m4a, which the bundled libsndfile cannot read. Each fix has a test in `tests/test_app.py`.

## O6: study design (power simulation)

These numbers detect a 15-point acceptance gain (C3 vs C1, sessions 3–5; 95% CI excludes 0). They assume a 45% base acceptance and a singer random effect with sd 0.8 on the log-odds scale.

| Singers | 10 | 12 | 15 | 20 |
|---|---|---|---|---|
| Power | 0.50 | 0.58 | 0.62 | 0.73 |

This confirms what the proposal expects: with 10–15 singers, O6 should be reported as an estimate with its confidence interval, and the significance tests as exploratory.

## Appendix: synthetic checks

These ran before the datasets were on disk. They show that the code works end to end; they are not results for the report. `scripts/reproduce_results.sh` reruns them with no downloads, on the same cloud VM at 4 threads.

### A1. O1 corpus from 144 synthetic takes

| | Value |
|---|---|
| Takes / singers | 144 takes from 24 synthetic singers, split by singer (17 train, 4 val, 3 test) |
| Labelled notes | 1,230, each detuned the same way with WORLD and with PSOLA (2,460 labels). 54% are detuned by more than 50 cents and kept as a separate set. |
| QC gate (pYIN re-measures every note) | **98.8%** of labels kept (PSOLA 99.9%, WORLD 97.6%), median residual 0.31 cents |
| Second measurement with FCPE | **99.96%** of kept labels recovered within 10 cents, median residual 0.69 cents, all 578 files match their checksums |
| Unique test audio | 0.04 h, so `O1_test_hours_ok` is false here. The VocalSet build meets it. |

### A2. O2 trackers on 40 noisy synthetic takes (SNR 0–20 dB, 4 threads)

| Tracker | RPA (95% CI) | RPA within 10 cents | RCA | Overall accuracy | Voicing recall | Voicing false alarm | Octave errors | RTF (4 threads) | Peak MB |
|---|---|---|---|---|---|---|---|---|---|
| FCPE | **0.986** (0.968–0.998) | 0.935 | 0.990 | 0.887 | 0.996 | 0.476 | 0.005 | 0.027 | 823 |
| PESTO (PyTorch) | 0.963 (0.937–0.983) | 0.684 | 0.978 | 0.900 | 0.933 | 0.143 | 0.016 | 0.151 | 1067 |
| PESTO (ONNX export) | 0.956 (0.928–0.979) | 0.720 | 0.969 | 0.896 | 0.932 | 0.144 | 0.012 | 0.037 | 725 |
| CREPE (tiny) | 0.913 (0.850–0.967) | 0.525 | 0.923 | 0.897 | 0.894 | 0.098 | 0.011 | 0.265 | 993 |
| YIN | 0.889 (0.857–0.921) | 0.684 | 0.943 | 0.774 | 0.981 | 0.653 | 0.054 | 0.022 | 724 |
| pYIN | 0.855 (0.803–0.903) | 0.667 | 0.861 | 0.836 | 0.841 | 0.189 | 0.006 | 0.136 | 721 |

The ranking at the top matches the real benchmark: FCPE first and most precise, PESTO's ONNX export as accurate as PyTorch PESTO and four times faster. FCPE's weak spot on these takes was voicing (48% false alarms in the noise). Every tracker's accuracy peaked at 0 ms shift.

### A3. O3 on the synthetic corpus (332 QC-passed test labels from 18 takes, FCPE)

| Metric | Key typed by user | Key detected | Target |
|---|---|---|---|
| Median error, notes < 50 cents off | **2.5 cents** | 4.4 cents | ≤ 25 |
| Share within 25 cents (< 50 set) | 95.8% | 69.7% | |
| Median error, notes ≥ 50 cents off (reported separately) | 4.3 cents | 99.0 cents | |
| Vibrato rate within ±10% | 96% | 96% | |
| Notes keeping at least 70% of vibrato extent | 80% (median ratio 0.93) | 80% (median ratio 0.94) | ≥ 70% |
| WORLD-detuned vs PSOLA-detuned copies (< 50 set) | 2.3 / 2.7 cents | 4.0 / 6.0 cents | |

Here the targets come from the synthetic singers' exact contours, which is one reason these errors are far smaller than the VocalSet ones. The key was detected correctly on only 2 of the 18 detuned takes, in line with the Essen result for detuned singing.

### A4. O5 full-pipeline speed on 20 s of synthetic singing (4 threads)

| | Value |
|---|---|
| Full-pipeline RTF, FCPE (PyTorch) | **0.29** (tracker 0.18, plan 0.004, shift 0.10) |
| Full-pipeline RTF, pYIN | 0.38 |
| Full-pipeline RTF, PESTO (PyTorch) | 0.41 |

On synthetic voices WORLD's copy-synthesis distorts heavily, so the quality check sent 90% of segments to PSOLA. Real singing goes the other way (O5 above).
