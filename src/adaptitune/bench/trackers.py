"""Tracker benchmark (objective O2, proposal 5.3).

Raw pitch accuracy (RPA), raw chroma accuracy (RCA), voicing recall / false alarm,
real-time factor and peak memory for every installed tracker, on MIR-1K and
MDB-stem-synth (or a synthetic stand-in). Each tracker runs in its own subprocess so
peak resident memory is measured per tracker and one crash cannot sink the whole run.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import metrics as M


def _items(dataset: str, root: str | None, limit: int | None, seed: int = 0):
    from ..data import datasets as D

    if dataset == "mir1k":
        it = D.mir1k(root)
    elif dataset == "mdb":
        it = D.mdb_stem_synth(root)
    elif dataset == "synthetic":
        it = _synthetic_items(limit or 40, seed)
    else:
        raise ValueError(dataset)
    for i, item in enumerate(it):
        if limit and i >= limit:
            break
        yield item


def _synthetic_items(n: int, seed: int):
    from ..data.datasets import Item
    from ..data.detune import make_singers, synth_clean_take

    rng = np.random.default_rng(seed)
    singers = make_singers(max(4, n // 4), seed + 1)
    for i in range(n):
        a, tr, _ = synth_clean_take(singers[i % len(singers)], rng, 16_000)
        yield Item(f"synth/{i}", singers[i % len(singers)].name, a, tr.times, tr.f0)


def _worker(name: str, dataset: str, root: str | None, limit: int | None, q):
    import warnings

    warnings.filterwarnings("ignore")
    try:
        import torch

        torch.set_num_threads(4)
    except Exception:
        pass
    from ..pitch.trackers import get_tracker

    rows = []
    try:
        trk = get_tracker(name)
        first = True
        for item in _items(dataset, root, limit):
            if first:  # warm-up (JIT, model load) is not part of the steady-state RTF
                trk.track(item.audio.samples[: item.audio.sr], item.audio.sr)
                first = False
            t0 = time.perf_counter()
            tr = trk.track(item.audio.samples, item.audio.sr)
            el = time.perf_counter() - t0
            acc = M.raw_pitch_accuracy(item.ref_times, item.ref_f0, tr.times, tr.f0)
            rows.append(dict(tracker=name, item=item.id, duration=item.audio.duration, seconds=el, **acc))
        peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        q.put(dict(ok=True, rows=rows, peak_mb=peak_mb))
    except Exception as e:  # pragma: no cover
        q.put(dict(ok=False, error=f"{type(e).__name__}: {e}", rows=rows))


def run(
    dataset: str = "synthetic",
    root: str | None = None,
    trackers: list[str] | None = None,
    limit: int | None = None,
    out_dir: str | Path = "results/trackers",
) -> pd.DataFrame:
    from ..pitch.trackers import available_trackers

    trackers = trackers or available_trackers()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    per_item, summary = [], []
    for name in trackers:
        q = ctx.Queue()
        p = ctx.Process(target=_worker, args=(name, dataset, root, limit, q))
        p.start()
        res = q.get()
        p.join()
        if not res["ok"]:
            print(f"[{name}] failed: {res['error']}")
            continue
        df = pd.DataFrame(res["rows"])
        per_item.append(df)
        summary.append(
            dict(
                tracker=name,
                dataset=dataset,
                items=len(df),
                rpa=df.rpa.mean(),
                rca=df.rca.mean(),
                voicing_recall=df.vr.mean(),
                voicing_false_alarm=df.vfa.mean(),
                rtf=df.seconds.sum() / df.duration.sum(),
                peak_mb=res["peak_mb"],
            )
        )
    s = pd.DataFrame(summary)
    if per_item:
        pd.concat(per_item).to_csv(out / f"{dataset}_per_item.csv", index=False)
    s.to_csv(out / f"{dataset}_summary.csv", index=False)
    (out / f"{dataset}_summary.md").write_text(s.round(4).to_markdown(index=False) if len(s) else "no results\n")
    return s


def pick_tracker(summaries: list[pd.DataFrame], max_rtf: float = 0.25, weight_speed: float = 0.1) -> str:
    """Best accuracy/speed trade-off: highest mean RPA among trackers fast enough to leave room
    for WORLD/PSOLA inside the 0.5 full-pipeline budget, with a small speed bonus."""
    s = pd.concat(summaries).groupby("tracker").agg(rpa=("rpa", "mean"), rtf=("rtf", "max"))
    ok = s[s.rtf <= max_rtf]
    if ok.empty:
        ok = s
    score = ok.rpa - weight_speed * ok.rtf
    return str(score.idxmax())


def save_choice(name: str, path: str | Path = "results/chosen_tracker.json") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"tracker": name}))
