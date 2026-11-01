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
from ..data.detune import load_labels
from ..music import Key
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
    keys = pd.read_csv(corpus / "manifest.csv").set_index("take")["key"].to_dict()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels = pd.DataFrame([asdict(lb) for lb in load_labels(corpus)])
    if split:
        labels = labels[labels.split == split]
    c = Corrector(tracker)
    rows = []
    for k, (take, g) in enumerate(labels.groupby("take")):
        if max_takes and k >= max_takes:
            break
        det = load(corpus / "detuned" / f"{take}.wav", normalize=False)
        clean = load(corpus / "clean" / f"{take}.wav", normalize=False)
        an = c.analyze(det, key=keys[take] if key_source == "user" else None)
        res = c.correct(an, settings, score=False)
        out_tr = c.tracker.track(res.audio.samples, res.audio.sr, c.hop_s)
        clean_tr = c.tracker.track(clean.samples, clean.sr, c.hop_s)
        mo, mc = M.contour_midi(out_tr.f0), M.contour_midi(clean_tr.f0)
        hop = c.hop_s
        for _, r in g.iterrows():
            a, b = int(r.start / hop), int(r.end / hop)
            ctr = M.note_center(mo, a, b)
            r0, e0 = M.vibrato(mc[a:b], hop)
            r1, e1 = M.vibrato(mo[a:b], hop)
            rows.append(
                dict(
                    take=take, index=r["index"], route=r.route, highly_detuned=r.highly_detuned,
                    offset_cents=r.offset_cents, target=r.target_midi,
                    error_cents=100 * (ctr - r.target_midi) if not np.isnan(ctr) else np.nan,
                    vib_rate_clean=r0, vib_rate_out=r1, vib_extent_clean=e0, vib_extent_out=e1,
                    key_used=an.key.name, key_true=keys[take], rtf=res.rtf,
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
        "median_abs_err_lt50": float(low.abs_err.median()),
        "share_within_25_lt50": float((low.abs_err <= 25).mean()),
        "median_abs_err_ge50": float(high.abs_err.median()) if len(high) else float("nan"),
        "share_within_25_ge50": float((high.abs_err <= 25).mean()) if len(high) else float("nan"),
        "vibrato_notes": len(has_vib),
        "vibrato_rate_within_10pct": float(rate_ok.mean()) if len(has_vib) else float("nan"),
        "vibrato_extent_ge_70pct": float(ext_ok.mean()) if len(has_vib) else float("nan"),
        "median_vibrato_extent_ratio": float((has_vib.vib_extent_out / has_vib.vib_extent_clean).median()) if len(has_vib) else float("nan"),
        "mean_rtf": float(df.groupby("take").rtf.first().mean()),
    }
    for route, g in low.groupby("route"):
        s[f"median_abs_err_lt50_{route}"] = float(g.abs_err.median())
    s["O3_pass"] = bool(
        s["median_abs_err_lt50"] <= 25
        and (np.isnan(s["vibrato_extent_ge_70pct"]) or s["median_vibrato_extent_ratio"] >= 0.7)
    )
    return s


def evaluate_keys(label_csv: str | Path, tracker: str = "auto", audio_root: str | Path = ".") -> dict:
    """Key accuracy on hand-labelled excerpts plus inter-rater agreement (O2: >= 75 %).

    CSV columns: ``path,key_rater1,key_rater2`` (e.g. ``A minor``). The reference is rater 1
    where both agree; disagreements are reported and excluded from the accuracy.
    """
    from ..data.datasets import read_key_labels
    from ..pitch.key import estimate_key, same_key_or_relative

    rows = read_key_labels(label_csv)
    c = Corrector(tracker)
    out = []
    for r in rows:
        k1, k2 = Key.parse(r["key_rater1"]), Key.parse(r.get("key_rater2") or r["key_rater1"])
        audio = load(Path(audio_root) / r["path"])
        tr = c.tracker.track(audio.samples, audio.sr)
        est, conf = estimate_key(tr.f0, tr.voicing)
        out.append(dict(path=r["path"], agree=k1 == k2, ref=k1.name, est=est.name, exact=est == k1, scale_equiv=same_key_or_relative(est, k1), conf=conf))
    df = pd.DataFrame(out)
    agreed = df[df.agree]
    return {
        "excerpts": len(df),
        "rater_agreement": float(df.agree.mean()),
        "accuracy_exact": float(agreed.exact.mean()),
        "accuracy_scale_equivalent": float(agreed.scale_equiv.mean()),
        "O2_key_pass": bool(agreed.exact.mean() >= 0.75),
    }


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
