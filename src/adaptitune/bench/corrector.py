"""Corrector evaluation on the detuned corpus (objective O3) and key-estimation accuracy
(the key part of O2).

O3 is measured at the fixed reference setting (full strength, in-key snapping, vibrato
preservation on), on test-split notes only. The corrected output and the clean original are
measured with ``measure_tracker``; use a tracker other than the corrector's own so the corrector
is not graded by the tracker it was built on.

* median |error from intended semitone| <= 25 cents on notes detuned by < 50 cents,
* vibrato rate within +-10 % and extent >= 70 % of the clean original (notes with vibrato),
* the 50-100 cent set reported separately, and every number split by detuning shifter.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .. import metrics as M
from ..audio import load
from ..correction.settings import REFERENCE, CorrectionSettings
from ..data.detune import load_labels, load_manifest
from ..pipeline import Corrector


def evaluate(
    corpus_dir: str | Path,
    tracker: str = "auto",
    settings: CorrectionSettings = REFERENCE,
    split: str = "test",
    out_dir: str | Path = "results/corrector",
    max_takes: int | None = None,
    key_source: str = "user",
    measure_tracker: str | None = None,
    shard: tuple[int, int] | None = None,
) -> dict:
    """``key_source="user"`` passes the take's key from the manifest, as a user who types the
    key in would (this isolates the corrector, which is what O3 measures); ``"estimated"``
    detects it from the detuned audio, which shows the cost of key-detection errors."""
    corpus = Path(corpus_dir)
    man = load_manifest(corpus)
    keys = man.drop_duplicates("take").set_index("take")["key"].to_dict()
    tech = man.drop_duplicates("take").set_index("take")["technique"].to_dict() if "technique" in man else {}
    det_path = {(r["take"], r["route"]): r["detuned"] for _, r in man.iterrows()}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # only labels that passed the corpus QC gate (all of them for schema-1 corpora)
    labels = pd.DataFrame([asdict(lb) for lb in load_labels(corpus, qc_only=True)])
    if split:
        labels = labels[labels.split == split]
    if shard:  # every n-th take (both routes of a take stay together); join with merge_shards
        k, n = shard
        takes = sorted(labels["take"].unique())
        labels = labels[labels["take"].isin(set(takes[k::n]))]
    c = Corrector(tracker)
    hop = c.hop_s
    if measure_tracker and measure_tracker != tracker:
        from ..pitch.trackers import get_tracker

        meas = get_tracker(measure_tracker)
    else:
        meas = c.tracker
    rows, clean_cache = [], {}
    for k, ((take, route), g) in enumerate(labels.groupby(["take", "route"])):
        if max_takes and k >= max_takes:
            break
        det = load(corpus / det_path.get((take, route), f"detuned/{take}.wav"), normalize=False)
        if take not in clean_cache:
            clean = load(corpus / "clean" / f"{take}.wav", normalize=False)
            clean_cache = {take: M.contour_midi(meas.track(clean.samples, clean.sr, hop).f0)}
        mc = clean_cache[take]
        an = c.analyze(det, key=keys[take] if key_source == "user" else None)
        res = c.correct(an, settings, score=False)
        snap_pcs = set(an.key.snap_classes) if an.key else set(range(12))
        out_tr = meas.track(res.audio.samples, res.audio.sr, hop)
        mo = M.contour_midi(out_tr.f0)
        for _, r in g.iterrows():
            a, b = int(round(r.start / hop)), int(round(r.end / hop))
            ctr = M.note_center(mo, a, b)
            r0, e0 = M.vibrato(mc[a:b], hop)
            r1, e1 = M.vibrato(mo[a:b], hop)
            rows.append(
                dict(
                    take=take, index=r["index"], route=route, highly_detuned=r.highly_detuned,
                    offset_cents=r.offset_cents, target=r.target_midi,
                    error_cents=100 * (ctr - r.target_midi) if not np.isnan(ctr) else np.nan,
                    vib_rate_clean=r0, vib_rate_out=r1, vib_extent_clean=e0, vib_extent_out=e1,
                    key_used=an.key.name if an.key else "", key_true=keys[take], rtf=res.rtf,
                    technique=tech.get(take, ""), target_in_key=int(r.target_midi) % 12 in snap_pcs,
                )
            )
    df = pd.DataFrame(rows)
    df.to_csv(out / "notes.csv", index=False)
    summ = summarize(df)
    pd.DataFrame([summ]).T.to_csv(out / "summary.csv", header=["value"])
    (out / "run.json").write_text(json.dumps(dict(
        corpus=str(corpus), tracker=c.tracker.name, measure_tracker=meas.name, settings=asdict(settings), split=split,
        key_source=key_source, shard=list(shard) if shard else None, takes=int(df["take"].nunique()) if len(df) else 0,
    ), indent=2))
    return summ


def merge_shards(shard_dirs: list[str | Path], out_dir: str | Path) -> dict:
    """Join sharded ``evaluate`` runs: checks that the shards share their settings and cover
    0..n-1 once, then summarizes all notes together."""
    runs = [json.loads((Path(d) / "run.json").read_text()) for d in shard_dirs]
    same = {k: {json.dumps(r[k], sort_keys=True) for r in runs} for k in ("corpus", "tracker", "measure_tracker", "settings", "split", "key_source")}
    bad = [k for k, v in same.items() if len(v) > 1]
    if bad:
        raise ValueError(f"shards differ in {bad}")
    n = {r["shard"][1] for r in runs if r["shard"]}
    ks = sorted(r["shard"][0] for r in runs if r["shard"])
    if len(n) != 1 or ks != list(range(next(iter(n)))):
        raise ValueError(f"shards do not cover 0..n-1 exactly once: {ks} of {n}")
    df = pd.concat([pd.read_csv(Path(d) / "notes.csv") for d in shard_dirs], ignore_index=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "notes.csv", index=False)
    summ = summarize(df)
    pd.DataFrame([summ]).T.to_csv(out / "summary.csv", header=["value"])
    meta = {k: runs[0][k] for k in same}
    (out / "run.json").write_text(json.dumps(dict(meta, merged_from=[str(d) for d in shard_dirs], takes=int(df["take"].nunique())), indent=2))
    return summ


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    df = df.copy()
    df["abs_err"] = df.error_cents.abs()
    low = df[~df.highly_detuned]
    high = df[df.highly_detuned]
    has_vib = df[(df.vib_extent_clean >= 20) & df.vib_rate_clean.notna()]
    rate_ok = (has_vib.vib_rate_out - has_vib.vib_rate_clean).abs() <= 0.1 * has_vib.vib_rate_clean
    ext_ok = has_vib.vib_extent_out >= 0.7 * has_vib.vib_extent_clean
    s = {
        "notes": len(df),
        "key_correct_share": float((df.groupby("take").key_used.first() == df.groupby("take").key_true.first()).mean()),
        "takes": int(df["take"].nunique()),
        "median_abs_err_lt50": float(low.abs_err.median()),
        "share_within_25_lt50": float((low.abs_err <= 25).mean()),
        "median_abs_err_ge50": float(high.abs_err.median()) if len(high) else float("nan"),
        "share_within_25_ge50": float((high.abs_err <= 25).mean()) if len(high) else float("nan"),
        "vibrato_notes": len(has_vib),
        "vibrato_rate_within_10pct": float(rate_ok.mean()) if len(has_vib) else float("nan"),
        "vibrato_extent_ge_70pct": float(ext_ok.mean()) if len(has_vib) else float("nan"),
        "median_vibrato_extent_ratio": float((has_vib.vib_extent_out / has_vib.vib_extent_clean).median()) if len(has_vib) else float("nan"),
        "mean_rtf": float(df.groupby(["take", "route"]).rtf.first().mean()),
    }
    for route, g in low.groupby("route"):
        s[f"median_abs_err_lt50_{route}"] = float(g.abs_err.median())
    if "target_in_key" in df:
        # a take that changes key has notes outside any one key; in-key snapping cannot reach them
        ink = low[low.target_in_key.astype(bool)]
        s["target_in_key_share"] = float(df.target_in_key.astype(bool).mean())
        s["median_abs_err_lt50_target_in_key"] = float(ink.abs_err.median()) if len(ink) else float("nan")
        s["share_within_25_lt50_target_in_key"] = float((ink.abs_err <= 25).mean()) if len(ink) else float("nan")
    s["O3_pass"] = bool(
        s["median_abs_err_lt50"] <= 25
        and (np.isnan(s["vibrato_extent_ge_70pct"]) or s["median_vibrato_extent_ratio"] >= 0.7)
    )
    return s


def evaluate_keys(label_csv: str | Path, tracker: str = "auto", audio_root: str | Path = ".") -> dict:
    """Key accuracy on the hand-labelled excerpts (see :mod:`adaptitune.bench.key_labels`)."""
    from .key_labels import evaluate_key_labels

    return evaluate_key_labels(label_csv, tracker, audio_root)


def calibrate_mcd_threshold(clean_files: list[str | Path], percentile: float = 95, hop_s: float = 0.005) -> float:
    """95th percentile of WORLD copy-synthesis MCD over voiced segments of clean vocals
    (proposal 5.3). Use the result as ``QualityConfig.mcd_threshold``."""
    from ..correction.planner import _runs
    from ..correction.shifter import _world_analyze, world_copy
    from ..pitch.trackers import get_tracker

    trk = get_tracker("auto")
    vals = []
    for f in clean_files:
        a = load(f)
        tr = trk.track(a.samples, a.sr, hop_s)
        wa = _world_analyze(a.samples, a.sr, hop_s * 1000)
        copy = world_copy(a.samples, a.sr, hop_s, wa)
        for s, e in _runs(tr.f0 > 0):
            lo, hi = int(s * hop_s * a.sr), int(e * hop_s * a.sr)
            if hi - lo > 0.1 * a.sr:
                vals.append(M.mel_cepstral_distortion(a.samples[lo:hi], copy[lo:hi], a.sr))
    vals = np.array([v for v in vals if not np.isnan(v)])
    return float(np.percentile(vals, percentile)) if len(vals) else float("nan")
