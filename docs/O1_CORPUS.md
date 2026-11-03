# O1 evaluation corpus: datasheet

**Objective O1.** Build an evaluation corpus of in-tune singing and detuned copies of it, with the exact injected offset stored for every note.

**Success criterion.** At least 2 hours of unique test audio from singers who are not used in development, where the detuned copies do not count towards the 2 hours. Every injected offset must be recoverable from the stored labels.

Code: `src/adaptitune/data/detune.py` (building, QC, verification, statistics) and `src/adaptitune/data/datasets.py` (loaders, split planner). Tests: `tests/test_corpus_o1.py`, `tests/test_datasets.py`.

## 1. How to build it

```bash
# VocalSet (CC BY 4.0) is enough on its own; add MedleyDB vocal stems if access comes through
adaptitune build-corpus data/corpus --source vocalset --root data/raw/VocalSet --workers 4
adaptitune corpus-stats  data/corpus                       # O1_test_hours_ok and O1_singers_disjoint must both be true
adaptitune verify-corpus data/corpus --tracker fcpe        # independent re-measurement of every kept label
```

A build can be stopped and continued with `--resume`. `--limit 5` builds five takes as a quick check. `--min-test-hours` (default 2.0) sets the test-split target. If the default techniques do not give enough test audio, add more with `--techniques`, for example `straight vibrato slow_forte slow_piano messa forte pp`.

## 2. Sources

| Source | Singers | What is used | Licence |
|---|---|---|---|
| VocalSet (Wilkins et al., 2018) | 20 professionals: 9 female, 11 male | The sustained, pitched techniques in `FULL/<singer>/<context>/<technique>/`: by default `straight`, `vibrato`, `slow_forte`, `slow_piano` and `messa`, over arpeggios, scales, long tones and the song excerpts. Breathy and vocal fry are left out because they are exactly what the shifter's quality check exists for. Spoken, inhaled, lip trill, trill and trillo are left out because they have no stable note centre. `forte`, `pp`, `belt`, `fast_forte` and `fast_piano` are clean too, and can be added with `--techniques`. | CC BY 4.0 |
| MedleyDB vocal stems (optional) | One singer ID per artist | Stems whose instrument is a singer or a vocalist | Non-commercial; needs an access request |
| Synthetic singers | 20–24 | Only for testing the code without downloads | Generated |

The VocalSet loader takes the dataset folder at any depth. If the same file name appears twice (some copies of VocalSet hold the same recordings in more than one folder layout), it counts the file once.

## 3. What happens to each take

1. **Pitch of the clean take.** Synthetic takes use their exact contour. Real takes use the analysis tracker (`--tracker`, default `auto`: an exported ONNX model if one is found, otherwise PESTO or FCPE when PyTorch is installed, otherwise pYIN), with a 5 ms hop.
2. **Notes and targets.** The contour is split into notes. A note gets a label only if it lasts at least 120 ms and its own robust centre is within **20 cents** of a semitone. That semitone becomes the note's intended target, so a professional's small errors are never mistaken for the target. Other notes are left as they are and get no label.
3. **Injected errors**, per labelled note:

   | Error | How often | Range |
   |---|---|---|
   | Constant offset | Every note | 10–100 cents, random sign |
   | Slow drift across the note | 35% | ±15–40 cents |
   | Onset scoop from below | 35% | 40–150 cents, decaying over 50–150 ms |
   | Vibrato changed | 30% | Exaggerated ×1.5–2.2 or suppressed ×0–0.5 |

   10 ms fades at note boundaries stop clicks, and a scoop starts at full depth by definition. The exact per-frame curve is stored.
4. **Two shifters.** The same curve is rendered with **WORLD** and with **PSOLA** (Praat), and both copies are stored. The proposal asks for half the copies from each shifter, reported separately. Rendering both gives the same takes on each side, so the comparison between them is paired, and neither shifter can look good just by undoing its own artifacts. For a strict half/half design, take the WORLD copy of even-numbered takes and the PSOLA copy of odd ones. `eval-corrector` already reports each route separately.
5. **Levels.** The clean take and both copies are scaled by one shared gain, only if a peak would clip, and saved as 16-bit WAV (24 kHz mono).
6. **Highly detuned set.** Notes with an offset above 50 cents are flagged `highly_detuned`, because even a perfect corrector may pick the neighbouring semitone for them. They are reported separately.

Each take has its own random generator, seeded from the corpus seed and the take ID (CRC-32). A take's detuning therefore does not depend on which other takes are built, in what order, or in how many processes. A test checks this.

## 4. QC gate: "every injected offset is recoverable"

After rendering, the QC tracker (`--qc-tracker`, default pYIN) re-tracks the clean take and both copies. For every labelled note, it takes the median over the note's core (20 ms trimmed from each end) of

```
measured (detuned − clean) pitch, in cents  −  stored injection curve, in cents
```

A label passes when this residual is within **10 cents** (`--qc-tol`) and both contours cover at least 60% of the core. A failed label stays in the files with its reason (`low_coverage`, `offset_mismatch` or `octave_error`), but it is not used for evaluation. The pass rate is reported for each route.

10 cents is well under the 25-cent O3 threshold and close to the pitch discrimination threshold, so a kept label is accurate enough to score the corrector against. The QC fields in each label are `qc_pass`, `qc_reason`, `qc_coverage`, `qc_residual_cents` and `recovered_offset_cents`.

`verify-corpus` repeats the measurement with **a different tracker** from the one the QC gate used. It reports the share of kept labels recovered within the tolerance, the median and 95th-percentile residual (overall and per route), and every failure. It also checks every file against `checksums.sha256`. The tests show it catches a label curve shifted by 40 cents and a changed file.

## 5. Splits

Takes are split **by singer**, so no voice is in more than one split. `plan_split` searches for the smallest set of test singers whose clean audio reaches the target plus a 10% margin. It breaks ties by overshoot, then by gender balance. Validation gets two singers (one of each gender when possible), and everyone else goes to training. The plan is deterministic for a given seed and is written to `split.json` with the hours per split. If the target cannot be met, `feasible` is false and the build says so.

`corpus-stats` reads the O1 checks straight off the files:

* `O1_test_hours_ok`: unique clean test audio is at least 2 h. Detuned copies are not counted.
* `O1_singers_disjoint`: no test singer appears in train or val.

## 6. Files

```
clean/<take>.wav                  in-tune take (the reference)
detuned/world/<take>.wav          detuned copy rendered with WORLD
detuned/psola/<take>.wav          the same detuning rendered with PSOLA
labels/<take>.npz                 hop_s, clean_midi, voicing, injection_cents (per frame),
                                  note_index (-1 = unlabelled frame), qc_clean_midi, qc_world_midi, qc_psola_midi
labels/<take>.json                per-take record (used by --resume)
labels.jsonl                      one line per (note, route): the label schema below
manifest.csv                      one row per (take, route): singer, gender, split, key, paths, duration,
                                  notes, labelled notes, QC passes, tuning, gain, context, technique
split.json                        singer -> split, hours per split, whether the target was met
build_info.json                   every parameter, package versions, git commit, failed takes
stats.json                        the corpus-stats report
checksums.sha256                  SHA-256 of every file above
```

**Label schema (version 2).** `take`, `index`, `start` and `end` (in seconds), `target_midi`, `source_center_midi`, `source_dev_cents` (the clean note's distance from its target), `offset_cents`, `drift_cents`, `scoop_cents`, `scoop_s`, `vibrato_factor`, `injected_mean_cents`, `highly_detuned`, `route`, `singer`, `gender`, `split`, `source` and the QC fields. JSON files never contain `NaN`; missing values are `null`.

The per-frame curve in the `.npz` file is the ground truth. `offset_cents` and the other parameters describe how it was made.

## 7. Intended use and limits

* **Use:** scoring pitch correctors (O3). Measure the error of each corrected note against `target_midi` on QC-passed labels. Keep the highly detuned set separate, and report WORLD and PSOLA copies separately.
* **Not for:** benchmarking pitch trackers. The corpus has note-level targets and tracker-derived contours, not hand-annotated frame pitch, so O2 uses MIR-1K and MDB-stem-synth instead.
* **Synthetic errors are not real errors.** Real amateurs miss notes in ways that correlate with the melody, such as interval size, register and fatigue. The proposal's 30–60 minutes of genuinely out-of-tune volunteer takes are the check for this.
* **The shifters leave traces.** A corrector could learn to spot WORLD or PSOLA artifacts. Rendering both copies and reporting them separately makes that visible.
* **Professional source singing.** VocalSet singers are trained, so the clean takes are cleaner than an amateur's best take. Notes more than 20 cents from a semitone are never labelled.
* **Pitch-class targets come from the clean note,** not from a score. A professional who sings a different note than the composer wrote is still "in tune" here.

## 8. Synthetic build (code check only)

The numbers below are from synthetic singers. They show that the code works, not that O1 is met. O1 needs the VocalSet build.

Built with `adaptitune build-corpus --source synthetic --singers 24 --takes 6 --workers 2` in 5 minutes on a 4-core cloud VM:

| Check | Result |
|---|---|
| Takes, singers | 144 takes from 24 synthetic singers. The split is by singer: 17 train, 4 val, 3 test, with no singer in two splits. |
| Labelled notes | 1,230 notes, each with a WORLD and a PSOLA copy, so 2,460 labels |
| Injected errors | Offsets 10.2–99.9 cents (median 54.5), 49% sharp; drift on 37% of notes, scoops on 34%, vibrato changed on 30%; 54% are in the highly detuned set |
| QC gate (pYIN) | **98.8% of labels kept** (PSOLA 99.9%, WORLD 97.6%). The residual between the measured difference and the stored curve has a median of 0.31 cents and a 95th percentile of 2.21 cents. The 30 rejected labels are 21 `low_coverage`, 8 `octave_error` and 1 `offset_mismatch`. |
| Independent re-measurement (FCPE) | **99.96% of the 2,430 kept labels recovered within 10 cents** (WORLD 100%, PSOLA 99.92%). The median residual is 0.69 cents and the 95th percentile 2.63 cents. The one miss is a PSOLA note at −10.4 cents. |
| Integrity | All 578 files match `checksums.sha256` |
| Unique test audio | 0.04 h. Synthetic takes are short, so `O1_test_hours_ok` is false here. The VocalSet build is what has to meet it. |
