# AdaptiTune

On-device vocal pitch correction that learns each singer's preferences.

Final Year Project, BS Computer Science, COMSATS University Islamabad (Lahore Campus), 2024–2028.
Team: Muhammad Shahzeb Khan (lead), Haris Khan, Afzaal Afzal. Supervisor: Ms. Humera Faisal.

AdaptiTune corrects a recorded vocal the way classic auto-tune does: it detects the pitch, works out the intended note and shifts the voice towards it while keeping the formants. What's new is that it doesn't use one setting for everybody. A small contextual bandit (Thompson sampling) learns each singer's preferred correction strength, retune speed, scale snapping and vibrato from their keep, rating and A/B feedback. Everything runs offline on a normal laptop, and the profile never leaves the device.

```
 audio ─► preprocess ─► pitch tracker ─► key ─► planner ─► WORLD / PSOLA ─► corrected audio
                         (pYIN, PESTO,          (notes,     (quality check
                          FCPE, CREPE, ONNX)     targets,    routes each
                                                 4 settings) segment)
                                     ▲                                │
                                     │ settings                       ▼
                        personalization agent ◄── reward ◄── feedback + objective scores
                        (population part + 15-weight personal part, SQLite profile)
```

## Quick start

```bash
python3.11 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[app,dev,study]"
pytest                                                     # 51 tests, about 30 s
adaptitune-app                                             # the desktop app
adaptitune correct my_take.wav fixed.wav --key "A minor"   # one file from the command line
```

The neural trackers are optional: `pip install -e ".[neural]"` adds PESTO, FCPE and CREPE (this pulls in PyTorch). Without them the app falls back to pYIN, and with an exported ONNX model it uses ONNX Runtime and needs no PyTorch at all.

## What is where

| Proposal | Code | Check |
|---|---|---|
| 5.1 Audio input and preprocessing | `adaptitune/audio.py` | |
| 5.3 Pitch analysis unit (PESTO, FCPE, CREPE, pYIN) | `adaptitune/pitch/trackers.py` | O2 `adaptitune bench-trackers` |
| 5.3 Key estimation, user override | `adaptitune/pitch/key.py` | O2 `adaptitune eval-keys` |
| 5.3 Correction planner and four settings | `adaptitune/correction/planner.py`, `settings.py` | O3 |
| 5.3 WORLD / PSOLA with quality check | `adaptitune/correction/shifter.py` | O3, `adaptitune calibrate` |
| 5.2 Synthetic detuning corpus | `adaptitune/data/detune.py`, `datasets.py` | O1 `adaptitune verify-corpus` |
| 5.4 Contextual bandit and reward | `adaptitune/agent/policies.py`, `reward.py`, `features.py` | O4 |
| 5.4 Simulated singers (3 models, drift, fatigue) | `adaptitune/agent/simulator.py`, `experiment.py` | O4 `adaptitune simulate` |
| 5.5 Offline deployment, ONNX, SQLite profile | `adaptitune/app/`, `agent/profile.py`, `scripts/export_onnx.py` | O5 `adaptitune rtf` |
| 5.6 FR1–FR7, NFR1–NFR4 | `adaptitune/app/main.py`, `agent/session.py` | `tests/test_session_study.py`, `tests/test_app.py` |
| 5.7 Study protocol and analysis | `adaptitune/study/` | O6 `adaptitune study-plan`, `study-analyze` |
| Metrics (RPA/RCA, cents error, vibrato, naturalness) | `adaptitune/metrics.py` | |

Results so far, and what still needs real data, are in [docs/RESULTS.md](docs/RESULTS.md). The architecture notes and a guide to who owns what are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). A code knowledge graph made with graphify is in `graphify-out/` (open `graph.html`).

## Using the real datasets

None of the datasets are redistributed. Download them, then:

```bash
adaptitune bench-trackers --dataset mir1k --root /data/MIR-1K
adaptitune bench-trackers --dataset mdb   --root /data/MDB-stem-synth
adaptitune build-corpus results/corpus --source vocalset --root /data/VocalSet
adaptitune calibrate results/corpus/clean/*.wav        # WORLD quality-check threshold
adaptitune eval-corrector results/corpus --key-source user
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
