"""Why do some O3 notes still land more than 50 cents from their target?

For every note of the "under 50 cents" set in an `eval-corrector` run, this re-plans the detuned
copy the way the corrector did (same tracker, the stored key, the reference setting; planning
only, no shifting) and sorts each miss by its first matching cause:

1. the target is outside the key, so in-key snapping cannot reach it;
2. the detuned note sat nearer the neighbouring semitone: its offset plus drift plus the singer's
   own deviation put its centre more than 50 cents from the target (from the stored curve and
   the analysis contour, not from the corrector);
3. the planner merged the note with a neighbouring one (its note is over 1.5 times longer);
4. the planner's own centre for the note was past halfway for another reason;
5. the planner chose another target although its centre was within 50 cents;
6. the planner chose the right target, but the measuring tracker hears the output an octave or
   more away;
7. the planner chose the right target, and the measuring tracker also misreads the clean note;
8. the planner chose the right target and the output measures 50-600 cents off.

    python scripts/o3_miss_causes.py results/corrector_vocalset_fcpe data/corpus --out docs/results/o3/miss_causes.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adaptitune import metrics as M
from adaptitune.audio import load
from adaptitune.correction.planner import plan
from adaptitune.correction.settings import REFERENCE
from adaptitune.music import Key
from adaptitune.pipeline import Corrector

CAUSES = {
    1: "target outside the key",
    2: "detuned note nearer the neighbouring semitone",
    3: "planner merged it with a neighbouring note",
    4: "planner's note centre past halfway",
    5: "planner chose another note, centre within 50 cents",
    6: "right note planned; output measured an octave or more off",
    7: "right note planned; the clean note is misread too",
    8: "right note planned; output measured 50-600 cents off",
}


def cause(r) -> int:
    if r.plan_target != r.target:
        if not r.target_in_key:
            return 1
        if abs(r.detuned_centre) > 50:
            return 2
        if r.plan_len > 1.5 * r.label_len:
            return 3
        return 4 if abs(r.plan_centre) > 50 else 5
    if abs(r.error) > 600:
        return 6
    return 7 if abs(r.clean_error) > 50 else 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="an eval-corrector output folder (notes.csv, run.json)")
    ap.add_argument("corpus")
    ap.add_argument("--tracker", help="default: the run's corrector tracker")
    ap.add_argument("--out", default="results/o3_miss_causes.json")
    a = ap.parse_args()
    run, corpus = Path(a.run), Path(a.corpus)
    info = json.loads((run / "run.json").read_text())
    notes = pd.read_csv(run / "notes.csv")
    low = notes[~notes.highly_detuned]
    labels = pd.read_json(corpus / "labels.jsonl", lines=True)[["take", "index", "route", "start", "end"]]
    low = low.merge(labels, on=["take", "index", "route"], how="left")
    keys = pd.read_csv(corpus / "manifest.csv").drop_duplicates("take").set_index("take")["key"]
    c = Corrector(a.tracker or info["tracker"])
    rows = []
    for (take, route), g in low.groupby(["take", "route"]):
        det = load(corpus / "detuned" / route / f"{take}.wav", normalize=False)
        if det.sr != c.work_sr:
            det = det.resampled(c.work_sr)
        p = plan(c.tracker.track(det.samples, det.sr, c.hop_s), REFERENCE, Key.parse(keys[take]))
        z = np.load(corpus / "labels" / f"{take}.npz")
        hop = float(z["hop_s"])
        det_midi = z["clean_midi"].astype(float) + z["injection_cents"].astype(float) / 100
        det_midi[~(z["clean_midi"] > 0)] = np.nan
        for r in g.itertuples(index=False):
            a_, b_ = int(round(r.start / hop)), int(round(r.end / hop))
            core = a_ + int(0.15 * (b_ - a_))
            over = [n for n in p.notes if n.start < b_ and n.end > core]
            main_ = max(over, key=lambda n: min(b_, n.end) - max(core, n.start)) if over else None
            rows.append(dict(take=take, index=r.index, route=route, technique=r.technique, target=r.target,
                             target_in_key=bool(r.target_in_key), error=r.error_cents, clean_error=r.clean_error_cents,
                             detuned_centre=100 * (M.note_center(det_midi, a_, b_) - r.target),
                             plan_target=main_.target if main_ else -1,
                             plan_centre=100 * (main_.center - r.target) if main_ else np.nan,
                             plan_len=(main_.end - main_.start) if main_ else 0, label_len=b_ - a_))
    d = pd.DataFrame(rows)
    miss = d[d.error.abs() > 50].copy()
    miss["cause"] = miss.apply(cause, axis=1)
    counts = miss.cause.value_counts().sort_index()
    fair = d[d.target_in_key & (d.detuned_centre.abs() <= 50)]
    res = dict(
        run=str(run), tracker=c.tracker.name, notes_under_50=len(d), misses=len(miss), miss_share=round(len(miss) / len(d), 4),
        causes={f"{k}. {CAUSES[k]}": dict(notes=int(counts.get(k, 0)), share_of_notes=round(int(counts.get(k, 0)) / len(d), 4)) for k in CAUSES},
        in_key_and_nearer_target=dict(notes=len(fair), median_abs_error=round(float(fair.error.abs().median()), 2),
                                      share_within_25=round(float((fair.error.abs() <= 25).mean()), 4),
                                      miss_share=round(float((fair.error.abs() > 50).mean()), 4)),
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    miss.to_csv(Path(a.out).with_suffix(".csv"), index=False)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
