# O6 dry run records

The whole five-session study, run through the study's own runner and analysis by 15 simulated participants who judge real MIR-1K singing. The write-up is in [RESULTS.md](../../../RESULTS.md#o6-user-study-dry-run-and-power), and the plan the run follows is [O6_PROTOCOL.md](../../../O6_PROTOCOL.md).

**Simulated people, real audio.** The keep or discard answers, the ratings, the hand-tuned settings and all the times are simulated, so the acceptance numbers follow the simulator's assumptions. The pitch error, in-key accuracy, naturalness score and vibrato penalty of every version are measured on the real corrected audio.

| File | Contents |
|---|---|
| `summary.json` | The analysis (`adaptitune.study.analysis.analyze`) plus the tables below, the machine and the commit |
| `trials.csv`, `manual.csv`, `sessions.csv` | What **Study → Export study data…** writes, for all 75 sessions (900 judged versions) |
| `acceptance_by_session.csv` | Keep rate of unflagged judgments by session and condition |
| `objective_by_condition.csv` | Measured metrics of the original (not repeated) versions, by condition |
| `agent_distance_by_session.csv` | Distance from the agent's C3 settings to the setting saved by hand that session, and to the simulated singer's true ideal |
| `acceptance_by_model_s3_5.csv` | Keep rate in sessions 3–5 by the simulated singer's preference model |
| `summary_manual_noise_0.15.json`, `summary_manual_noise_0.25.json` | The same run with less accurate hand tuning (`--manual-noise`) |

**Reproduce.** With MIR-1K in `data/raw/MIR-1K` and FCPE's ONNX export made (`python scripts/export_onnx.py --model fcpe`):

```
python scripts/o6_dry_run.py data/raw/MIR-1K --participants 15 --workers 4 --out results/o6_dry_run
python scripts/o6_dry_run.py data/raw/MIR-1K --manual-noise 0.15 --out results/o6_dry_run_noise015
python scripts/o6_dry_run.py data/raw/MIR-1K --manual-noise 0.25 --out results/o6_dry_run_noise025
```

**Provenance.** Commit ac6466d, 4 November 2026, on a 4-core Intel Xeon @ 2.1 GHz cloud VM with 15.7 GB of RAM, Python 3.11.18, four workers at one thread each. Pitch tracker: FCPE through ONNX Runtime (`fcpe-onnx`). An earlier run of the same simulation, made before the analysis gained the time measure, gave identical acceptance numbers. Each run took 4–5 minutes. The seeds are fixed: participant *i*, counted from 0, uses `default_rng(1000 + i)` for the simulated person and seed `100 i + s` for the agent in session *s*. Participants are numbered in the order of MIR-1K's singers with the most clips (amy, leon, geniusturtle, yifen, davidson, jmzen, Kenshin, bobon, fdps, stool, titon, khair, abjones, annar, ariel), and their preference models cycle through logistic, ideal point and threshold.
