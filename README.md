# AdaptiTune

On-device vocal pitch correction that learns each singer's preferences.

Final Year Project, BS Computer Science, COMSATS University Islamabad (Lahore Campus), 2024–2028.
Team: Muhammad Shahzeb Khan (lead), Haris Khan, Afzaal Afzal. Supervisor: Ms. Humera Faisal.

AdaptiTune corrects a recorded vocal the way classic auto-tune does: it detects the pitch, works out the intended note and shifts the voice towards it while keeping the formants. What's new is that it doesn't use one setting for everybody. A small contextual bandit (Thompson sampling) learns each singer's preferred correction strength, retune speed, scale snapping and vibrato from their keep, rating and A/B feedback. Everything runs offline on a normal laptop, and the profile never leaves the device.

```
 audio ─► preprocess ─► pitch tracker ─► key ─► planner ─► WORLD / PSOLA ─► corrected audio
                         (FCPE via ONNX;        (notes,     (quality check
                          pYIN, PESTO, CREPE     targets,    routes each
                          benchmarked)           4 settings) segment)
                                     ▲                                │
                                     │ settings                       ▼
                        personalization agent ◄── reward ◄── feedback + objective scores
                        (population part + 15-weight personal part, SQLite profile)
```

## Quick start

```bash
python3.11 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[app,dev,study]"
pytest                                                     # about 115 tests, 2-4 minutes
adaptitune-app                                             # the desktop app
adaptitune correct my_take.wav fixed.wav --key "A minor"   # one file from the command line
```

The app's pitch tracker is **FCPE**, which the O2 benchmark chose. It runs FCPE's ONNX export with ONNX Runtime, so it needs no PyTorch: make the export once with `python scripts/export_onnx.py --model fcpe` on a machine with the neural extra (`pip install -e ".[neural]"` adds FCPE, PESTO and CREPE and pulls in PyTorch). With the neural extra installed it can also run FCPE directly. Without either it falls back to PESTO's export, then pYIN. `pip install -e ".[eval]"` adds music21 for the Essen key validation (`adaptitune key-eval`).

## What is where

| Proposal | Code | Check |
|---|---|---|
| 5.1 Audio input and preprocessing | `adaptitune/audio.py` | |
| 5.3 Pitch analysis unit (PESTO, FCPE, CREPE, pYIN) | `adaptitune/pitch/trackers.py`, `bench/trackers.py` | O2 `adaptitune bench-trackers`, `choose-tracker` |
| 5.3 Key estimation, user override | `adaptitune/pitch/key.py`, `bench/keys.py`, `bench/key_labels.py`, `tools/key_rater.html` | O2 `adaptitune key-eval`, `key-excerpts`, `key-agreement`, `eval-keys` |
| 5.3 Correction planner and four settings | `adaptitune/correction/planner.py`, `settings.py` | O3 |
| 5.3 WORLD / PSOLA with quality check | `adaptitune/correction/shifter.py` | O3, `adaptitune calibrate` |
| 5.2 Synthetic detuning corpus | `adaptitune/data/detune.py`, `datasets.py` | O1 `adaptitune build-corpus`, `verify-corpus`, `corpus-stats` |
| 5.4 Contextual bandit and reward | `adaptitune/agent/policies.py`, `reward.py`, `features.py` | O4 |
| 5.4 Simulated singers (3 models, drift, fatigue) | `adaptitune/agent/simulator.py`, `experiment.py` | O4 `adaptitune simulate` |
| 5.5 Offline deployment, ONNX, SQLite profile | `adaptitune/app/`, `agent/profile.py`, `scripts/export_onnx.py` | O5 `adaptitune rtf` |
| 5.6 FR1–FR7, NFR1–NFR4 | `adaptitune/app/main.py`, `agent/session.py` | `tests/test_session_study.py`, `tests/test_app.py` |
| 5.7 Study protocol and analysis | `adaptitune/study/` | O6 `adaptitune study-plan`, `study-analyze` |
| Metrics (RPA/RCA, cents error, vibrato, naturalness) | `adaptitune/metrics.py` | |

Results so far, and what still needs real data, are in [docs/RESULTS.md](docs/RESULTS.md). The O1 corpus datasheet is [docs/O1_CORPUS.md](docs/O1_CORPUS.md), and the O2 method, fixed before any real-data run, is [docs/O2_PROTOCOL.md](docs/O2_PROTOCOL.md). The architecture notes and a guide to who owns what are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). A code knowledge graph made with graphify is in `graphify-out/` (open `graph.html`).

## Using the real datasets

None of the datasets are redistributed. Download them, then:

| Dataset | Where | Used for |
|---|---|---|
| VocalSet | [zenodo.org/records/1442513](https://zenodo.org/records/1442513) (CC BY 4.0) | O1 corpus (clean in-tune singing) |
| MIR-1K | [Figshare copy of MIR-1K.rar](https://figshare.com/articles/dataset/MIR-1K_rar/5802891) (797 MB, MD5 `1810d01457ccbb84a0b41c4da53eee74`). The original mirlab.org link now returns 404. Only `Wavfile/` and `PitchLabel/` are needed. | O2 tracker benchmark, O2 key excerpts |
| MDB-stem-synth | [zenodo.org/records/1481172](https://zenodo.org/records/1481172) | O2 tracker benchmark |
| MedleyDB (optional) | access request at medleydb.weebly.com | More O1 singers |

```bash
python scripts/export_onnx.py --model pesto                                 # PESTO for ONNX Runtime, benchmarked as pesto-onnx
python scripts/export_onnx.py --model fcpe                                  # FCPE for ONNX Runtime, what the app runs
adaptitune bench-trackers --dataset mir1k --root data/raw/MIR-1K --audit    # O2, see docs/O2_PROTOCOL.md
adaptitune bench-trackers --dataset mdb   --root data/raw/MDB-stem-synth --audit
adaptitune choose-tracker --results results/trackers                        # the pre-registered rule
adaptitune build-corpus data/corpus --source vocalset --root data/raw/VocalSet --tracker fcpe --workers 4   # O1, see docs/O1_CORPUS.md
adaptitune corpus-stats data/corpus                                         # O1_test_hours_ok, O1_singers_disjoint
adaptitune verify-corpus data/corpus --tracker fcpe                         # every kept offset re-measured
adaptitune key-excerpts data/raw/MIR-1K --source mir1k --out key_excerpts   # then label with tools/key_rater.html
python scripts/calibrate_quality.py data/corpus --tracker fcpe             # WORLD quality-check threshold, training singers only
adaptitune eval-corrector data/corpus --tracker fcpe --measure-tracker pyin # O3 on the test singers (add --shard k/n to split it)
```

`scripts/reproduce_results.sh` runs the whole set on synthetic singing, so it works with no downloads.

## Desktop app

* **Personalized:** the agent picks the settings and you keep, discard or rate the result (FR3, FR4).
* **Compare A / B:** you get two versions and pick the one you prefer (FR4).
* **Manual:** you set the four sliders yourself and save the result as your setting (FR5). The time you spend and the number of adjustments are logged for the study.
* The pitch view and the table show how far each note was off before and after (FR6).
* The profile can be reset or exported (FR7). It lives in `~/.adaptitune/profile.sqlite`; set `ADAPTITUNE_HOME` to move it.
* Recording needs `sounddevice` and a microphone, while importing works without them.

To build the Windows app, run `packaging\build_windows.bat` (PyInstaller, one-folder build).

## License

Code: MIT. Corpus outputs inherit their source licenses (VocalSet CC BY 4.0, MedleyDB non-commercial).
