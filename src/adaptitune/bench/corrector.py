"""Corrector evaluation on the detuned corpus (objective O3) and key-estimation accuracy
(the key part of O2).

O3 is measured at the fixed reference setting (full strength, in-key snapping, vibrato
preservation on), on test-split notes only:

* median |error from intended semitone| <= 25 cents on notes detuned by < 50 cents,
* vibrato rate within +-10 % and extent >= 70 % of the clean original (notes with vibrato),
* the 50-100 cent set reported separately, and every number split by detuning shifter.
"""

from __future__ import annotations

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
) -> dict:
    """``key_source="user"`` passes the take's key from the manifest, as a user who types the
    key in would (this isolates the corrector, which is what O3 measures); ``"estimated"``
    detects it from the detuned audio, which shows the cost of key-detection errors."""
    corpus = Path(corpus_dir)
    man = load_manifest(corpus)
    keys = man.drop_duplicates("take").set_index("take")["key"].to_dict()
    det_path = {(r["take"], r["route"]): r["detuned"] for _, r in man.iterrows()}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # only labels that passed the corpus QC gate (all of them for schema-1 corpora)
    labels = pd.DataFrame([asdict(lb) for lb in load_labels(corpus, qc_only=True)])
    if split:
        labels = labels[labels.split == split]
    c = Corrector(tracker)
    hop = c.hop_s
    rows, clean_cache = [], {}
    for k, ((take, route), g) in enumerate(labels.groupby(["take", "route"])):
        if max_takes and k >= max_takes:
            break
        det = load(corpus / det_path.get((take, route), f"detuned/{take}.wav"), normalize=False)
        if take not in clean_cache:
            clean = load(corpus / "clean" / f"{take}.wav", normalize=False)
            clean_cache = {take: M.contour_midi(c.tracker.track(clean.samples, clean.sr, hop).f0)}
        mc = clean_cache[take]
        an = c.analyze(det, key=keys[take] if key_source == "user" else None)
        res = c.correct(an, settings, score=False)
        out_tr = c.tracker.track(res.audio.samples, res.audio.sr, hop)
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
                )
            )
    df = pd.DataFrame(rows)
    df.to_csv(out / "notes.csv", index=False)
    summ = summarize(df)
    pd.DataFrame([summ]).T.to_csv(out / "summary.csv", header=["value"])
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
