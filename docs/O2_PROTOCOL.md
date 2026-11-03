# O2 protocol: pitch-tracker benchmark and key estimation

**Objective O2.** Benchmark at least four pitch trackers (PESTO, FCPE, CREPE, pYIN) for accuracy and CPU speed on MIR-1K and MDB-stem-synth, and choose the analysis unit.

**Success criteria.**

1. At least four trackers are reported on both datasets.
2. The chosen tracker reaches at least 90% raw pitch accuracy on MDB-stem-synth.
3. Key estimation is correct on at least 75% of about 100 hand-labelled excerpts, each labelled separately by two people.

This file fixes the method **before** any real-data result exists. The git history of this file and of the constants in `src/adaptitune/bench/trackers.py` is the record of that.

---

## Part A. Tracker benchmark

### A1. Data

| Dataset | What it is | How it is used |
|---|---|---|
| **MIR-1K** | 1,000 clips of Chinese pop sung by 19 amateurs. Vocal and accompaniment are on separate channels. Reference pitch is given every 20 ms as semitones (`PitchLabel/*.pv`, first frame at 20 ms, 0 = unvoiced). | The vocal (right) channel. Optional extra conditions: the accompaniment mixed back in (`--mix-snr 0`) and pink noise (`--noise-snr 20`, `10`). |
| **MDB-stem-synth** | 230 monophonic MedleyDB stems of 25 instruments, voices included, resynthesized so the reference pitch is exact by construction. Annotations come every 128 samples at 44.1 kHz (about 2.9 ms). | Every stem, as in the CREPE, PESTO and FCPE papers. A header line in the CSV files is tolerated. |

The loaders read both datasets at 16 kHz, which is every tracker's analysis rate. MIR-1K is recorded at 16 kHz, and MDB-stem-synth is resampled from 44.1 kHz. The loaders are tested against fake copies of both folder layouts (`tests/test_datasets.py`).

### A2. Trackers

| Name | Implementation | Voicing threshold | Already trained on | Notes |
|---|---|---|---|---|
| `pyin` | librosa pYIN, 10 ms internal hop, 0.25-semitone states, refined with YIN's parabolic estimate | HMM decision | nothing (no training) | Baseline from the proposal |
| `yin` | librosa YIN, with a periodicity-based voicing estimate | 0.45 | nothing | Extra fast baseline |
| `pesto` | PESTO (PyTorch), the `mir-1k_g7` checkpoint | 0.5 | **MIR-1K audio** (self-supervised, no labels): it is the only checkpoint PESTO ships | |
| `pesto-onnx` | The same PESTO model exported to ONNX and run with ONNX Runtime | 0.5 | **MIR-1K audio** | What the app ships. It must match `pesto` to under 1 cent (a test checks this). |
| `fcpe` | torchfcpe, decoded without its voicing mask so every frame has a raw pitch | 0.006 | M4Singer and VCTK only; MIR-1K is a test set in the FCPE paper | |
| `crepe` | torchcrepe, `tiny` model, decoded as in the CREPE paper: Viterbi path, then the weighted mean of the path bin and its 4 neighbours on each side (see A9) | 0.4 periodicity | **MIR-1K and MDB-stem-synth** (plus Bach10, RWC-synth, MedleyDB and NSynth, per the CREPE README) | `full` is too slow for the CPU budget |

**Training overlap.** A model that has already heard a benchmark set gets an inflated score there. That accuracy is still reported, but flagged, and it is left out of that tracker's selection score (A6). The overlap is recorded in `TRAINED_ON` in `src/adaptitune/bench/trackers.py`, with the source for each entry.

The voicing thresholds are each library's documented default. They were fixed before the benchmark and are never tuned on test data. They affect only overall accuracy, voicing recall and false alarms, not raw pitch accuracy.

**Search range.** Every tracker that takes a range gets the same one: **C1–B6 (32.7–1975.5 Hz)**, the native range of CREPE and FCPE. MDB-stem-synth has bass and flute stems outside the singing range, and a narrower range would score those frames as errors for some trackers and not others. PESTO has no range setting. The app itself searches 65–1100 Hz, so the benchmark also reports accuracy on the reference frames inside that range (`rpa_sing`).

### A3. Accuracy metrics

All metrics come from `mir_eval` (`adaptitune.metrics.pitch_metrics`). Per item:

* **RPA / RCA at 50 cents (the O2 metric), voicing-independent.** Each tracker returns its pitch guess on every frame, including frames it calls unvoiced. Those frames are passed to mir_eval as negative frequencies, which is mir_eval's convention for "unvoiced, but this is my guess". RPA then measures pitch only, which is how the CREPE, PESTO and FCPE papers report it. Without this, a tracker with a cautious voicing threshold would lose RPA for a voicing decision.
* **RPA at 25 and 10 cents.** A pitch corrector needs better than 50-cent accuracy, so these finer tolerances are reported too.
* **Overall accuracy, voicing recall and voicing false alarm,** using the tracker's own voicing decision.
* **Median error** of the frames counted as correct, and the **octave-error rate**.
* **`rpa_sing`:** RPA over the reference frames between 65 and 1100 Hz, plus the share of voiced reference frames inside that range.

### A4. Speed and memory

* **RTF** = tracking time ÷ audio duration, summed over all items. Every numeric library is pinned to **4 threads** (`OMP`, `MKL`, `OpenBLAS`, `Numba`, `torch.set_num_threads` and the ONNX Runtime intra-op threads). Change this with `--threads`.
* Each tracker runs in **its own fresh Python process**. One tracker's memory or crash cannot affect another.
* A **warm-up call** on the first second of audio comes before timing starts, so model loading and JIT compilation are not counted.
* **Peak memory** is the process's peak resident set. **Model memory** is the resident set after loading minus before.
* `<label>_run.json` records the CPU name, cores, RAM, OS, Python, every relevant package version, the git commit and the time. **The O2 speed numbers that go in the report must be measured on the reference laptop.**

### A5. Statistics

* Accuracy is averaged **per item**. Pooled-frame RPA is reported as well.
* **95% confidence intervals** use a cluster bootstrap with 2,000 resamples. It resamples singers for MIR-1K and tracks for MDB-stem-synth, never single clips, because clips from one voice are not independent.
* Two trackers are compared with a **paired** bootstrap of the per-item difference on the items both ran (`paired_diff_ci`).

### A6. Selection rule (fixed in advance)

The rule is implemented in `choose_tracker`, with these constants: `MAX_TRACKER_RTF = 0.25`, `TIE_PP = 1.0`, `O2_MIN_TRACKERS = 4`, `O2_MIN_MDB_RPA = 0.90` and `TRAINED_ON`.

1. **Eligible:** RTF ≤ 0.25 on every dataset. This is half of the 0.5 full-pipeline budget from O5.
2. **Score:** mean of the 50-cent RPA over MIR-1K and MDB-stem-synth, **leaving out any dataset the tracker was trained on**. PESTO is therefore scored on MDB-stem-synth only. CREPE has seen both sets, so it is reported but cannot be chosen.
3. **Choice:** the eligible tracker with the best score. If other eligible trackers are within **1 RPA point** of it, the fastest of those wins, because a smaller difference is within what changes between runs and datasets.
4. **O2 passes** when at least four trackers are reported on both datasets and the chosen tracker's RPA on MDB-stem-synth is at least 0.90.

The result goes to `chosen_tracker.json`. It holds every tracker's eligibility, score and training overlap, the paired difference between the winner and the runner-up with its confidence interval, and whether the run was official (both datasets present). It also records the tracker the same rule would choose if training overlap were ignored. If that differs, the report says so.

### A7. Checking the references themselves

`--audit` re-scores every tracker with its output shifted from −30 to +30 ms in 5 ms steps. If every tracker prefers the same non-zero shift, the reference times are off, for example because of a wrong frame-time convention, and that would be found before any result is reported. `alignment_report` summarizes this. Read the pitch-only column (`best_shift_ms_rpa10`): overall accuracy also moves with each tracker's voicing lag at note onsets. On the synthetic takes, every tracker's pitch accuracy peaks at a shift of 0 ms.

### A8. Commands

```bash
adaptitune bench-trackers --dataset mir1k --root data/raw/MIR-1K          --audit --out results/trackers
adaptitune bench-trackers --dataset mdb   --root data/raw/MDB-stem-synth  --audit --out results/trackers
adaptitune choose-tracker --results results/trackers                      # applies A6 and writes chosen_tracker.json
# optional robustness conditions (reported, not used for the choice)
adaptitune bench-trackers --dataset mir1k --root data/raw/MIR-1K --mix-snr 0    --out results/trackers
adaptitune bench-trackers --dataset mir1k --root data/raw/MIR-1K --noise-snr 10 --out results/trackers
```

A long run can be split into interleaved parts that run side by side, then joined. The joined files are the same as one run's, because every metric is a per-item mean or sum:

```bash
for k in 0 1 2 3; do adaptitune bench-trackers --dataset mdb --root data/raw --shard $k/4 --threads 1 --audit --out results/mdb_$k & done; wait
adaptitune bench-merge results/mdb_0 results/mdb_1 results/mdb_2 results/mdb_3 --label mdb --out results/trackers
```

Outputs per condition: `<label>_per_item.csv`, `<label>_summary.csv` and `.md`, `<label>_run.json`, and with `--audit` also `<label>_alignment.csv` and `<label>_alignment_per_item.csv`. `bench-trackers` prints the table and runs `choose-tracker` on its own once both official summaries exist. `scripts/export_onnx.py --out models/pesto.onnx` must run first, or `pesto-onnx` is skipped.

### A9. Changes made on first contact with the real data

Items 1–3 and 5 were made on 3 November 2026, after a smoke test on two MDB-stem-synth stems (`AClassicEducation_NightOwl` stems 1 and 8) and before any full run. Items 4, 6 and 7 record how the full run was done, one finding from it and what followed from the choice. None of them touches the selection rule, its constants, the voicing thresholds, the search range or the data.

1. **Long recordings are tracked in 30 s segments**, each with 1.5 s of extra audio on both sides that is tracked and then dropped (`SEGMENT_S`, `SEGMENT_CONTEXT_S`). In the smoke test PESTO needed 5.8 GB of memory for one 171 s stem, and MDB-stem-synth has stems up to 1,061 s long. The context is longer than PESTO's longest CQT kernel (about ±1.02 s) and the other trackers' frames. On a 70 s test glide, pYIN, YIN and PESTO give identical frames with and without segments, the ONNX export differs by at most 0.001 cents, and FCPE and CREPE by at most 1 cent. The app uses the same code, so the benchmark measures what ships.
2. **CREPE is decoded the way the CREPE paper does it.** torchcrepe's default Viterbi decoder returns the centre of the chosen 20-cent bin plus random triangular dither of up to ±20 cents. Two runs on the same audio differed by up to 37 cents, which would make CREPE's numbers irreproducible and penalize it on the 10- and 25-cent tolerances. The Viterbi path is kept, and the pitch is the salience-weighted mean of the path bin and its four neighbours on each side, as in `crepe --viterbi`.
3. **Loaders match the released archives.** MDB-stem-synth's folders are `audio_stems/` and `annotation_stems/` (plural), and both archives carry macOS `._` files with audio file names that are not audio. Both are now handled, and each dataset is found at any depth below `--root` (`tests/test_datasets.py`).
4. **Where it ran.** The accuracy numbers were computed on a 4-core cloud machine, with each tracker's run split into interleaved parts (8 for MIR-1K, 24 for MDB-stem-synth), each part with 1 thread and several parts at a time, then joined with `bench-merge`. The parts of the trackers the rule can choose ran first and PyTorch PESTO and CREPE last; the order changes no number. The speed columns of that run come from a loaded shared machine at 1 thread. **They are not the O2 speed result**, which must come from the reference laptop at 4 threads (A4): `scripts/o2_laptop_speed.py` times every tracker there on the first part of each dataset, and `choose-tracker --speed` then takes every RTF from that run. PyTorch PESTO and CREPE ran last, partly while the O3 evaluation had priority on the same machine and some of their parts were briefly paused, so their cloud speeds are even less representative than the others'.
5. **MIR-1K's source.** The original archive at mirlab.org now returns 404, so the team used a mirror copy (the README links the Figshare upload of `MIR-1K.rar`, article 5802891). The benchmark reads its 1,000 clips in `Wavfile/` and 1,000 files in `PitchLabel/`; `data/manifests/mir1k.sha256` lists the SHA-256 of every file used, so the copy can be matched later.
6. **PESTO's output moves in steps of a third of a semitone.** With the library's default decoding (`reduction="alwa"`), both PyTorch PESTO and the ONNX export return +0 or +33.3 cents for steady tones at +10, +20 and +30 cents, where FCPE, pYIN and YIN are within 3.5 cents. This is PESTO as released, so the benchmark keeps it. It shows in PESTO's accuracy within 10 cents and its median error, not in the 50-cent RPA the rule uses. It did matter elsewhere: the first O1 corpus build used PESTO's ONNX export as its analysis tracker, so it was rebuilt with FCPE (`docs/O1_CORPUS.md`, section 8).
7. **The app now runs the tracker the rule chose.** A2 lists `pesto-onnx` as what the app ships, which was true when this protocol was written. The rule chose FCPE, so the app now runs FCPE's own ONNX export with ONNX Runtime (`fcpe-onnx`, made by `scripts/export_onnx.py --model fcpe`). On voiced frames it matches PyTorch FCPE within 0.5 cents and makes the same voicing decisions (`tests/test_bench_o2.py`). It is not part of the choice; the laptop run times it for O5.

---

## Part B. Key estimation

### B1. The estimator (`adaptitune.pitch.key`)

1. Split the contour into notes, and take each note's robust centre.
2. **Remove the global tuning.** This is the circular mean of the note centres' distance from the nearest semitone, weighted by duration. A singer who is 40 cents flat throughout would otherwise have many notes nearer the wrong semitone.
3. Build a **pitch-class histogram of notes**, weighted by duration^0.5. Each note's vote is shared softly with the neighbouring class when its centre is near the boundary between them.
4. Correlate the histogram with each of the 24 rotated **Bellman–Budge** key profiles (Krumhansl–Schmuckler method). Add 0.1 to keys whose tonic is the last note's pitch class.
5. **Confidence** is the margin over the best key with a *different scale*, so a C major vs. A minor tie (same notes) does not count as uncertain for in-key snapping.

### B2. Development and held-out test on the Essen folksong collection

Before the hand labels exist, the estimator was developed on the Essen collection that ships with music21 (`adaptitune key-eval`, `src/adaptitune/bench/keys.py`):

* **Ground truth.** The ABC `K:` field gives the tonic. The mode comes from the score: major if the major third above the tonic is sung for longer than the minor third, and minor if the reverse. The estimator never sees either. 83% of the European songs end on their `K:` tonic, and 13% are minor.
* **Singing.** Each song is moved to a random key and rendered as a pitch contour with vibrato, scoops, drift and jitter. There are three conditions: `in_tune`; `amateur` (a global tuning error plus 25-cent note scatter); and `corpus` (the O1 detuning, 10–100 cent offsets). Each is tested as a whole song and as an 8–20 s excerpt that stops before the last note, so the final tonic is not given away.
* **Selection on training volumes only.** The 240 configurations are 5 profiles (Krumhansl–Kessler, Temperley–Kostka–Payne, Bellman–Budge, Aarden–Essen, simple) × 2 scoring rules (correlation, log-likelihood) × 3 histograms (frames; notes weighted by duration^0.5 or by duration) × tuning removal on/off × 4 final-note bonuses (0, 0.05, 0.1, 0.2). The one with the best **mode-balanced** accuracy (the mean of major and minor accuracy, so the 87% of major songs cannot outvote the minor ones) on 4,670 songs from the training volumes became the default. Aarden's profiles are derived from Essen itself, so they are reported but were never eligible. The Han Chinese volumes give the pentatonic "gong" note in `K:` rather than the final, so they are left out of selection and scored only on whether the key's scale contains the melody.
* **Held-out test:** 2,486 European songs and 622 Han songs from other volumes.

**Results on the held-out test volumes** (exact key, 95% Wilson interval):

| Condition | Items | Chosen estimator (95% CI) | Same scale | MIREX score | Before this work (KK, frames) | Aarden (Essen-derived, reference) |
|---|---|---|---|---|---|---|
| In tune, whole song | 2,486 | **94.5%** (93.5–95.3) | 95.3% | 0.961 | 64.4% | 91.5% |
| In tune, 8–20 s excerpt | 2,470 | **80.4%** (78.8–81.9) | 84.3% | 0.860 | 46.0% | 72.3% |
| Amateur, whole song | 2,486 | **90.5%** (89.2–91.6) | 91.5% | 0.926 | 52.8% | 86.4% |
| Amateur, 8–20 s excerpt | 2,470 | **73.8%** (72.0–75.5) | 78.1% | 0.801 | 35.8% | 67.1% |
| O1-style detuned, whole song | 2,486 | **11.1%** (9.9–12.4) | 12.3% | 0.153 | 12.6% | 16.2% |
| O1-style detuned, excerpt | 2,470 | **7.6%** (6.6–8.7) | 9.3% | 0.119 | 7.4% | 11.8% |
| Major-key items (in tune and amateur, songs and excerpts) | 8,478 | **87.7%** (87.0–88.4) | 90.0% | 0.911 | 51.5% | 81.2% |
| Minor-key items (same) | 1,434 | **67.5%** (65.0–69.9) | 71.1% | 0.746 | 39.7% | 68.5% |
| The 70% of those items with the highest confidence | 6,938 | **92.9%** (92.3–93.5) | 95.1% | 0.946 | 52.1% | 88.9% |

Han Chinese test songs, scored only on whether the estimated key's scale contains the whole melody: 82.9% (before: 75.8%).

**Audio check.** 200 European test excerpts (in tune and amateur) were synthesized to audio and tracked. The key was right for 80.0% from the true contour, 80.0% from pYIN and 79.0% from PESTO through ONNX, so tracker errors do not change the picture.

What this says:

* **Excerpts land at 74–80%, right at the 75% target.** Whole songs are much easier (90–95%). The hand-labelled excerpts should therefore be long enough to hear the key (12–30 s).
* **Detuned singing is not a fair key test.** Notes moved by 10–100 cents in random directions destroy the pitch-class evidence (7–11%). This is why O3 is scored with the key the user types (`--key-source user`). The estimated key is reported as a secondary condition.
* **Minor keys are the weak side** (67.5% against 87.7% for major). Minor melodies also leave their scale far more often. Only 6% of Essen's minor songs stay inside natural minor, against 84% of major songs inside the major scale. 64% use the raised 7th (the leading tone), 41% the raised 6th, and 71% touch the major third briefly (about 3% of their duration). **In-key snapping to natural minor would pull leading tones and raised 6ths down a semitone**, so O3's in-key mode should allow the raised 6th and 7th in minor keys (the harmonic and melodic forms).
* **Confidence is useful.** On the 70% of items with the highest confidence the key is right 92.9% of the time, so the app can ask the user to confirm the key when confidence is low.

### B3. The hand-labelled test (the O2 criterion)

**Excerpts.** Run `adaptitune key-excerpts` to write 100 excerpts, `excerpts.json` and a blank label template:

* **Recommended: `--source mir1k`.** This gives one excerpt per song (MIR-1K has about 110 songs by 19 amateur singers). It joins one singer's consecutive clips of a song into 12–30 s, takes the vocal channel only, and spreads the excerpts round-robin over singers. The excerpt numbers are shuffled so they do not reveal the singer. Amateur singers performing real songs are AdaptiTune's actual users. The key estimator was never tuned on MIR-1K, so it is a held-out test.
* **`--source corpus`** cuts excerpts from the O1 test split. It takes songs first, then arpeggios and scales, and never single long tones. VocalSet only has three songs ("Row, row, row your boat", "Caro mio ben" and "Dona nobis pacem"), so most excerpts repeat the same three tunes. Its interval is therefore computed by resampling songs, and it is best used as a second, smaller check.

The estimator's output is not written anywhere a rater can see it.

**Instructions for the raters.** Two team members label **separately**, without talking about the excerpts:

1. Open `tools/key_rater.html` in any browser, with no internet needed, and load the excerpt folder. Each rater gets their own shuffled order.
2. For each excerpt, choose the key from the 24-key grid. Use the tonic drone and the scale player to check: the tonic is the note the melody feels at rest on, and the mode is the scale that fits the melody best.
3. If the excerpt changes key, label the key it spends the most time in and add a note. If you cannot hear a key (it is too short, too chromatic, or pentatonic with no clear centre), choose **No clear key** instead of guessing.
4. Set a confidence from 1 to 3, and export the CSV when done. Work is saved in the browser as you go.

**Agreement and adjudication.** `adaptitune key-agreement rater1.csv rater2.csv --out key_labels.csv` reports raw agreement, same-scale agreement (for example C major against A minor) and **Cohen's kappa**. It writes the merged file with an `adjudicated` column that is already filled in where both raters agree. The two raters then listen to each disagreement together and fill in the rest.

**Scoring.** `adaptitune eval-keys key_labels.csv --audio-root key_excerpts --out per_excerpt.csv`:

* The reference is the adjudicated key. Before adjudication, it is the key both raters agreed on.
* Excerpts whose reference is "no clear key" are counted and reported, but not scored.
* **O2 key criterion: exact accuracy ≥ 75%** on the scored excerpts. It is reported with a Wilson 95% interval, and with a song-resampling interval when songs repeat.
* Also reported: same-scale accuracy, the MIREX weighted score, accuracy on the excerpts the raters agreed on from the start, rater agreement and kappa, and the tracker used (the chosen O2 tracker, `auto` by default).

---

## What still needs the real data

| Step | Command | Needs |
|---|---|---|
| Tracker benchmark | `bench-trackers --dataset mir1k` and `--dataset mdb` (A8), then `choose-tracker` | MIR-1K, MDB-stem-synth, the reference laptop |
| Key excerpts | `key-excerpts data/raw/MIR-1K --source mir1k` | MIR-1K |
| Labelling | `tools/key_rater.html`, then `key-agreement`, then `eval-keys` | Two raters, about 1.5 hours each |
