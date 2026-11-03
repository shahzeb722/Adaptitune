# Graph Report - src  (2026-11-03)

## Corpus Check
- 23 files · ~33,106 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 725 nodes · 1851 edges · 42 communities (23 shown, 19 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 69 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Key Estimation and Labelling
- WORLD/PSOLA Shifter
- Agent Features and Simulator
- Pitch Trackers
- Metrics and Synthetic Singing
- Audio Preprocessing and Datasets
- Adaptive Session
- Bandit Baselines
- Personal Posterior Profile
- CLI and Excerpt Tools
- Desktop Main Window
- Benchmark Environment and Items
- Corpus Build and Splits
- Correction Planner
- Corrector Evaluation
- Tracker Selection Rule
- Detuning Design and QC
- Desktop App Jobs
- Population Model
- Population-only Policy
- RTF and Synthetic Takes
- Study Analysis
- Agent Simulation Experiments
- Study Protocol
- Recorder and Users
- Pitch View Widget
- Corpus Verification
- Thompson Sampling
- Benchmark Noise
- Fixed Policy
- Audio Player
- Pitch View Painting

## God Nodes (most connected - your core abstractions)
1. `MainWindow` - 41 edges
2. `Audio` - 37 edges
3. `Key` - 36 edges
4. `main()` - 25 edges
5. `AdaptiveSession` - 23 edges
6. `load()` - 22 edges
7. `_process_take()` - 21 edges
8. `evaluate_key_labels()` - 20 edges
9. `get_tracker()` - 20 edges
10. `PitchTrack` - 19 edges

## Surprising Connections (you probably didn't know these)
- `Analysis` --uses--> `PitchTrack`  [INFERRED]
  adaptitune/pipeline.py → adaptitune/pitch/trackers.py
- `Corrector` --uses--> `CorrectionSettings`  [INFERRED]
  adaptitune/pipeline.py → adaptitune/correction/settings.py
- `Corrector` --uses--> `Tracker`  [INFERRED]
  adaptitune/pipeline.py → adaptitune/pitch/trackers.py
- `TakeResult` --uses--> `CorrectionSettings`  [INFERRED]
  adaptitune/pipeline.py → adaptitune/correction/settings.py
- `TakeResult` --uses--> `PitchTrack`  [INFERRED]
  adaptitune/pipeline.py → adaptitune/pitch/trackers.py

## Import Cycles
- None detected.

## Communities (42 total, 19 thin omitted)

### Community 0 - "Key Estimation and Labelling"
Cohesion: 0.05
Nodes (44): cohens_kappa(), evaluate_key_labels(), _excerpt_songs(), merge_raters(), parse_label(), _read_rater(), _song_of(), _audio_check() (+36 more)

### Community 1 - "WORLD/PSOLA Shifter"
Cohesion: 0.08
Nodes (24): Audio, save(), Plan, apply(), _fit_len(), _mask_from_frames(), prepare(), Prepared (+16 more)

### Community 2 - "Agent Features and Simulator"
Cohesion: 0.06
Nodes (17): action_quad(), context_from_analysis(), context_from_track(), joint(), quad(), settings_quad(), Feedback, reward() (+9 more)

### Community 3 - "Pitch Trackers"
Cohesion: 0.06
Nodes (14): available_trackers(), CrepeTracker, default_threads(), FcpeTracker, find_onnx_model(), get_tracker(), _interp_log(), OnnxTracker (+6 more)

### Community 4 - "Metrics and Synthetic Singing"
Cohesion: 0.08
Nodes (25): contour(), _envelope(), note_contour(), random_melody(), sing(), SungNote, SynthResult, contour_midi() (+17 more)

### Community 5 - "Audio Preprocessing and Datasets"
Cohesion: 0.12
Nodes (16): estimate_noise_db(), frame_times(), normalize_loudness(), preprocess(), resample(), rms_db(), to_mono(), Item (+8 more)

### Community 7 - "Bandit Baselines"
Cohesion: 0.15
Nodes (3): EpsilonGreedy, LinUCB, Policy

### Community 8 - "Personal Posterior Profile"
Cohesion: 0.15
Nodes (3): PersonalPosterior, default_path(), ProfileStore

### Community 9 - "CLI and Excerpt Tools"
Cohesion: 0.18
Nodes (10): main(), calibrate_mcd_threshold(), select_excerpts(), select_mir1k_excerpts(), dataset_label(), json_safe(), save_choice(), main() (+2 more)

### Community 11 - "Benchmark Environment and Items"
Cohesion: 0.17
Nodes (11): bench_tracker(), _cpu_name(), _git_commit(), _items(), machine_info(), _peak_rss_mb(), _rss_mb(), _run_isolated() (+3 more)

### Community 12 - "Corpus Build and Splits"
Cohesion: 0.16
Nodes (12): plan_split(), split_by_singer(), build_corpus(), detune_take(), DetuneSpec, _git_commit(), _label(), NoteLabel (+4 more)

### Community 13 - "Correction Planner"
Cohesion: 0.17
Nodes (7): Note, plan(), _robust_center(), _runs(), segment_notes(), _smooth(), _vibrato_window()

### Community 14 - "Corrector Evaluation"
Cohesion: 0.15
Nodes (7): load(), evaluate(), evaluate_keys(), summarize(), corpus_stats(), load_labels(), load_manifest()

### Community 15 - "Tracker Selection Rule"
Cohesion: 0.16
Nodes (9): alignment_report(), _apply_rule(), choose_tracker(), _load_condition(), paired_diff_ci(), _rtf(), run(), summarize() (+1 more)

### Community 16 - "Detuning Design and QC"
Cohesion: 0.14
Nodes (9): design_take(), _fit(), injection_curve(), _nan_to_none(), note_qc(), _process_take(), render_routes(), take_rng() (+1 more)

### Community 18 - "Population Model"
Cohesion: 0.17
Nodes (3): expand_context(), PopulationModel, load_population()

### Community 20 - "RTF and Synthetic Takes"
Cohesion: 0.26
Nodes (6): _cpu_name(), measure(), _synthetic_items(), make_singers(), synth_clean_take(), SynthSinger

### Community 21 - "Study Analysis"
Cohesion: 0.23
Nodes (6): analyze(), cluster_bootstrap_diff(), holm(), mixed_logit(), naturalness_noninferiority(), simulate_study()

### Community 22 - "Agent Simulation Experiments"
Cohesion: 0.30
Nodes (7): _markdown(), plot(), run(), run_singer(), RunLog, summarize(), train_population()

### Community 23 - "Study Protocol"
Cohesion: 0.24
Nodes (6): consistency(), flag_inconsistent(), save_trials(), schedule(), session_conditions(), Trial

### Community 26 - "Corpus Verification"
Cohesion: 0.39
Nodes (6): _corpus_files(), _sha256(), _tracker(), verify_checksums(), verify_labels(), write_checksums()

### Community 29 - "Benchmark Noise"
Cohesion: 0.47
Nodes (3): active_power(), add_noise(), pink_noise()

### Community 32 - "Pitch View Painting"
Cohesion: 0.83
Nodes (3): curve(), X(), Y()

## Knowledge Gaps
- **19 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Key` connect `WORLD/PSOLA Shifter` to `Key Estimation and Labelling`, `Metrics and Synthetic Singing`, `Adaptive Session`, `Desktop Main Window`, `Corpus Build and Splits`, `Correction Planner`, `Detuning Design and QC`, `Desktop App Jobs`, `Population-only Policy`, `Pitch View Widget`, `App Result Handlers`?**
  _High betweenness centrality (0.085) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `MainWindow` (e.g. with `AdaptiveSession` and `PitchView`) actually correct?**
  _`MainWindow` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Should `Key Estimation and Labelling` be split into smaller, more focused modules?**
  _Cohesion score 0.052659716653301256 - nodes in this community are weakly interconnected._
- **Why does `MainWindow` connect `Desktop Main Window` to `WORLD/PSOLA Shifter`, `Adaptive Session`, `CLI and Excerpt Tools`, `Desktop App Jobs`, `Recorder and Users`, `Pitch View Widget`, `App Result Handlers`?**
  _High betweenness centrality (0.084) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `Audio` (e.g. with `AdaptiveSession` and `Player`) actually correct?**
  _`Audio` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Should `WORLD/PSOLA Shifter` be split into smaller, more focused modules?**
  _Cohesion score 0.08311688311688312 - nodes in this community are weakly interconnected._
- **Why does `AdaptiveSession` connect `Adaptive Session` to `WORLD/PSOLA Shifter`, `Personal Posterior Profile`, `Desktop Main Window`, `Desktop App Jobs`, `Population Model`, `Population-only Policy`?**
  _High betweenness centrality (0.067) - this node is a cross-community bridge._