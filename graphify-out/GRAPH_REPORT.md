# Graph Report - src  (2026-11-01)

## Corpus Check
- Corpus is ~19,752 words - fits in a single context window. You may not need a graph.

## Summary
- 543 nodes · 1448 edges · 20 communities (11 shown, 9 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 84 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Bandit Policies
- Desktop App
- Desktop App
- Desktop App
- WORLD/PSOLA Shifter
- Bandit Policies
- Desktop App
- Agent Simulation Experiments
- Bandit Policies
- Pitch Trackers
- Pitch Trackers

## God Nodes (most connected - your core abstractions)
1. `MainWindow` - 42 edges
2. `CorrectionSettings` - 38 edges
3. `Key` - 30 edges
4. `Audio` - 28 edges
5. `AdaptiveSession` - 27 edges
6. `load()` - 20 edges
7. `TakeResult` - 20 edges
8. `main()` - 19 edges
9. `Corrector` - 19 edges
10. `PitchTrack` - 19 edges

## Surprising Connections (you probably didn't know these)
- `train_population()` --uses--> `PopulationModel`  [INFERRED]
  adaptitune/agent/experiment.py → adaptitune/agent/policies.py
- `run()` --uses--> `PopulationModel`  [INFERRED]
  adaptitune/agent/experiment.py → adaptitune/agent/policies.py
- `settings_quad()` --uses--> `CorrectionSettings`  [INFERRED]
  adaptitune/agent/features.py → adaptitune/correction/settings.py
- `Policy` --uses--> `CorrectionSettings`  [INFERRED]
  adaptitune/agent/policies.py → adaptitune/correction/settings.py
- `AdaptiveSession` --uses--> `ProfileStore`  [INFERRED]
  adaptitune/agent/session.py → adaptitune/agent/profile.py

## Import Cycles
- None detected.

## Communities (20 total, 9 thin omitted)

### Community 0 - "Bandit Policies"
Cohesion: 0.06
Nodes (24): action_quad(), context_from_analysis(), context_from_track(), joint(), quad(), settings_quad(), DiscountedThompson, expand_context() (+16 more)

### Community 1 - "Desktop App"
Cohesion: 0.06
Nodes (14): Job, _Signals, evaluate(), evaluate_keys(), summarize(), cents(), hz_to_midi(), Key (+6 more)

### Community 2 - "Desktop App"
Cohesion: 0.06
Nodes (23): FixedPolicy, main(), save_choice(), main(), _settings(), action_index(), CorrectionSettings, load_labels() (+15 more)

### Community 3 - "Desktop App"
Cohesion: 0.07
Nodes (7): MainWindow, Player, Recorder, PitchView, curve(), X(), Y()

### Community 4 - "WORLD/PSOLA Shifter"
Cohesion: 0.09
Nodes (29): calibrate_mcd_threshold(), _runs(), apply(), _fit_len(), _mask_from_frames(), prepare(), Prepared, _psola_apply() (+21 more)

### Community 5 - "Bandit Policies"
Cohesion: 0.07
Nodes (7): EpsilonGreedy, LinUCB, PersonalPosterior, Policy, PopulationOnly, default_path(), ProfileStore

### Community 6 - "Desktop App"
Cohesion: 0.09
Nodes (20): Audio, estimate_noise_db(), frame_times(), load(), normalize_loudness(), preprocess(), resample(), rms_db() (+12 more)

### Community 7 - "Agent Simulation Experiments"
Cohesion: 0.10
Nodes (13): _markdown(), plot(), run(), run_singer(), RunLog, summarize(), train_population(), assert_shapes() (+5 more)

### Community 8 - "Bandit Policies"
Cohesion: 0.10
Nodes (8): PopulationModel, Feedback, reward(), AdaptiveSession, load_population(), Take, action_grid(), TakeResult

### Community 9 - "Pitch Trackers"
Cohesion: 0.08
Nodes (9): CrepeTracker, FcpeTracker, get_tracker(), OnnxTracker, PestoTracker, PitchTrack, PyinTracker, Tracker (+1 more)

### Community 10 - "Pitch Trackers"
Cohesion: 0.10
Nodes (18): _cpu_name(), measure(), _items(), pick_tracker(), run(), _synthetic_items(), _worker(), make_singers() (+10 more)

## Knowledge Gaps
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CorrectionSettings` connect `Desktop App` to `Bandit Policies`, `Desktop App`, `Desktop App`, `Bandit Policies`, `Desktop App`, `Agent Simulation Experiments`, `Bandit Policies`?**
  _High betweenness centrality (0.117) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `MainWindow` (e.g. with `AdaptiveSession` and `PitchView`) actually correct?**
  _`MainWindow` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Should `Bandit Policies` be split into smaller, more focused modules?**
  _Cohesion score 0.06440677966101695 - nodes in this community are weakly interconnected._
- **Why does `MainWindow` connect `Desktop App` to `Bandit Policies`, `Desktop App`, `Desktop App`?**
  _High betweenness centrality (0.106) - this node is a cross-community bridge._
- **Are the 10 inferred relationships involving `CorrectionSettings` (e.g. with `settings_quad()` and `FixedPolicy`) actually correct?**
  _`CorrectionSettings` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Should `Desktop App` be split into smaller, more focused modules?**
  _Cohesion score 0.05928614640048397 - nodes in this community are weakly interconnected._
- **Why does `AdaptiveSession` connect `Bandit Policies` to `Desktop App`, `Desktop App`, `Desktop App`, `Bandit Policies`, `Desktop App`?**
  _High betweenness centrality (0.059) - this node is a cross-community bridge._