"""Calibrate the quality check's WORLD threshold on real clean vocals (proposal 5.3).

The threshold is the 95th percentile of WORLD copy-synthesis MCD over voiced segments of clean
takes. It must come from TRAINING singers only, never from the test singers that O3 is scored on.

    python scripts/calibrate_quality.py data/corpus --takes-per-singer 15 --out results/quality_calibration.json

Then set ``QualityConfig.mcd_threshold`` in src/adaptitune/correction/shifter.py to the value.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from adaptitune.bench.corrector import calibrate_mcd_threshold
from adaptitune.data.detune import load_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--split", default="train")
    ap.add_argument("--takes-per-singer", type=int, default=15)
    ap.add_argument("--percentile", type=float, default=95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/quality_calibration.json")
    a = ap.parse_args()
    corpus = Path(a.corpus)
    man = load_manifest(corpus).drop_duplicates("take")
    man = man[man["split"] == a.split]
    if man.empty:
        raise SystemExit(f"no {a.split} takes in {corpus}")
    rng = np.random.default_rng(a.seed)
    picked = []
    for singer, g in sorted(man.groupby("singer"), key=lambda x: x[0]):
        idx = rng.permutation(len(g))[: a.takes_per_singer]
        picked += [corpus / c for c in g.iloc[idx]["clean"]]
    t0 = time.time()
    thr = calibrate_mcd_threshold(picked, a.percentile)
    res = dict(mcd_threshold_db=round(thr, 2), percentile=a.percentile, split=a.split, singers=int(man.singer.nunique()),
               takes=len(picked), files=[str(p.relative_to(corpus)) for p in picked], seconds=round(time.time() - t0, 1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"mcd_threshold = {thr:.2f} dB from {len(picked)} {a.split} takes of {res['singers']} singers -> {a.out}")


if __name__ == "__main__":
    main()
