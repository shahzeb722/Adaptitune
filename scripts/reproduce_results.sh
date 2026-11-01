#!/usr/bin/env bash
# Regenerates every number in docs/RESULTS.md on synthetic data (no downloads needed).
# With the real datasets, swap --dataset/--source and add --root (see README).
set -euo pipefail
OUT=${OUT:-results}
adaptitune bench-trackers --dataset synthetic --limit 40 --out "$OUT/trackers"
adaptitune build-corpus "$OUT/corpus" --source synthetic --singers 20 --takes 6
adaptitune verify-corpus "$OUT/corpus" --tracker fcpe > "$OUT/corpus_verify.json"
adaptitune eval-corrector "$OUT/corpus" --tracker fcpe --key-source user --out "$OUT/corrector_userkey"
adaptitune eval-corrector "$OUT/corpus" --tracker fcpe --key-source estimated --out "$OUT/corrector_estkey"
for t in pyin fcpe pesto; do adaptitune rtf --tracker $t > "$OUT/rtf_$t.json"; done
python scripts/run_agent_sim.py --singers 100 --horizon 20 --out "$OUT/agent"
python scripts/run_agent_sim.py --singers 50 --horizon 60 --out "$OUT/agent_h60"
python -c "from adaptitune.study.analysis import simulate_study as s; import json; print(json.dumps([s(n_singers=n) for n in (10,12,15,20)], indent=1))" > "$OUT/study_power.json"
