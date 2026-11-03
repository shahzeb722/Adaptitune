"""Formant-preserving pitch shifter (proposal 5.1 and 5.3).

WORLD (pyworld) is the main path: it separates F0, spectral envelope and aperiodicity, so we
can move F0 and keep the envelope (timbre). WORLD assumes clearly periodic input, so every
voiced segment first goes through a quality check and is routed to PSOLA (Praat via
Parselmouth) when any of these holds:

* median harmonics-to-noise ratio below ``hnr_db`` (10 dB),
* mean tracker voicing probability below ``min_voicing`` (0.5),
* WORLD copy-synthesis mel-cepstral distortion above ``mcd_threshold`` (95th percentile on
  clean vocals of the O1 corpus's training singers, calibrated by
  ``scripts/calibrate_quality.py``; see ``QualityConfig``).

Segments with voicing below 0.2 are left untouched by the planner already. All thresholds are
starting values and are exposed so they can be calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..audio import SYNTH_SR, resample
from .planner import Plan, _runs

WORLD, PSOLA, UNTOUCHED = "world", "psola", "untouched"
FADE_S = 0.01


@dataclass
class QualityConfig:
    hnr_db: float = 10.0
    min_voicing: float = 0.5
    # 95th percentile of WORLD copy-synthesis MCD over the voiced segments of 135 clean takes
    # (15 per singer) from the 9 training singers of the VocalSet corpus, with FCPE finding the
    # segments (docs/results/o3/quality_calibration_fcpe.json; YIN gives 21.46). The units are
    # those of metrics.mel_cepstral_distortion, which are not comparable with SPTK-based MCD.
    mcd_threshold: float = 21.58
    force: str | None = None  # "world" | "psola" to bypass routing (benchmarks, detuning)


@dataclass
class Segment:
    start: int  # frame
    end: int
    route: str
    hnr_db: float = float("nan")
    voicing: float = float("nan")
    mcd_db: float = float("nan")


@dataclass
class ShiftResult:
    samples: np.ndarray
    sr: int
    segments: list[Segment] = field(default_factory=list)

    def route_shares(self) -> dict[str, float]:
        tot = sum(s.end - s.start for s in self.segments) or 1
        out = {WORLD: 0.0, PSOLA: 0.0, UNTOUCHED: 0.0}
        for s in self.segments:
            out[s.route] += (s.end - s.start) / tot
        return out


# --------------------------------------------------------------------------- WORLD


def _world_analyze(x: np.ndarray, sr: int, frame_ms: float):
    import pyworld as pw

    x = np.ascontiguousarray(x, dtype=np.float64)
    f0, t = pw.dio(x, sr, f0_floor=60.0, f0_ceil=1100.0, frame_period=frame_ms)
    f0 = pw.stonemask(x, f0, t, sr)
    sp = pw.cheaptrick(x, f0, t, sr)
    ap = pw.d4c(x, f0, t, sr)
    return f0, t, sp, ap


def world_shift(x: np.ndarray, sr: int, ratio: np.ndarray, hop_s: float, analysis=None) -> np.ndarray:
    """Multiply F0 by ``ratio`` (per planner frame) keeping the spectral envelope."""
    import pyworld as pw

    f0, t, sp, ap = analysis if analysis is not None else _world_analyze(x, sr, hop_s * 1000)
    r = np.interp(t, np.arange(len(ratio)) * hop_s, ratio, left=1.0, right=1.0)
    y = pw.synthesize(f0 * r, sp, ap, sr, frame_period=hop_s * 1000)
    return _fit_len(y, len(x))


def world_copy(x: np.ndarray, sr: int, hop_s: float, analysis=None) -> np.ndarray:
    return world_shift(x, sr, np.ones(int(len(x) / sr / hop_s) + 1), hop_s, analysis)


# --------------------------------------------------------------------------- PSOLA


def psola_shift(x: np.ndarray, sr: int, ratio: np.ndarray, hop_s: float) -> np.ndarray:
    """TD-PSOLA through Praat's Manipulation object."""
    import parselmouth
    from parselmouth.praat import call

    snd = parselmouth.Sound(np.ascontiguousarray(x, dtype=np.float64), sampling_frequency=sr)
    manip = call(snd, "To Manipulation", 0.01, 60, 1100)
    tier = call(manip, "Extract pitch tier")
    n = int(call(tier, "Get number of points"))
    new = call("Create PitchTier", "shifted", snd.xmin, snd.xmax)
    rt = np.arange(len(ratio)) * hop_s
    for i in range(1, n + 1):
        tt = call(tier, "Get time from index", i)
        f = call(tier, "Get value at index", i)
        call(new, "Add point", tt, f * float(np.interp(tt, rt, ratio, left=1, right=1)))
    call([manip, new], "Replace pitch tier")
    out = call(manip, "Get resynthesis (overlap-add)")
    return _fit_len(out.values[0].astype(np.float64), len(x))


# --------------------------------------------------------------------------- helpers


def _fit_len(y: np.ndarray, n: int) -> np.ndarray:
    if len(y) >= n:
        return y[:n]
    return np.pad(y, (0, n - len(y)))


def _segment_hnr(x: np.ndarray, sr: int) -> float:
    import parselmouth
    from parselmouth.praat import call

    if len(x) < 0.05 * sr:
        return float("nan")
    snd = parselmouth.Sound(x, sampling_frequency=sr)
    harm = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    vals = harm.values[0]
    vals = vals[vals > -200]
    return float(np.median(vals)) if len(vals) else float("nan")


def _mask_from_frames(frames: np.ndarray, hop_s: float, n: int, sr: int) -> np.ndarray:
    """Upsample a 0/1 frame mask to samples with short raised-cosine fades."""
    t = np.arange(n) / sr
    m = np.interp(t, np.arange(len(frames)) * hop_s, frames.astype(float), left=0, right=0)
    k = max(1, int(FADE_S * sr))
    win = np.hanning(2 * k + 1)
    win /= win.sum()
    return np.clip(np.convolve(m, win, mode="same"), 0, 1)


@dataclass
class Prepared:
    """Settings-independent analysis of one take, cached so A/B versions and repeated
    corrections only pay for planning and synthesis."""

    y: np.ndarray
    sr: int
    hop_s: float
    segments: list[Segment]  # one per voiced run, route decided by the quality check
    world: tuple | None = None
    copy: np.ndarray | None = None  # WORLD copy-synthesis (also used by the naturalness score)
    _psola: tuple | None = None

    def psola_objects(self):
        if self._psola is None:
            import parselmouth
            from parselmouth.praat import call

            snd = parselmouth.Sound(np.ascontiguousarray(self.y), sampling_frequency=self.sr)
            manip = call(snd, "To Manipulation", 0.01, 60, 1100)
            tier = call(manip, "Extract pitch tier")
            n = int(call(tier, "Get number of points"))
            pts = np.array(
                [(call(tier, "Get time from index", i), call(tier, "Get value at index", i)) for i in range(1, n + 1)]
            ).reshape(-1, 2)
            self._psola = (snd, manip, pts)
        return self._psola


def prepare(
    x: np.ndarray,
    sr: int,
    midi: np.ndarray,
    voicing: np.ndarray,
    hop_s: float,
    quality: QualityConfig | None = None,
    work_sr: int = SYNTH_SR,
) -> Prepared:
    """Run WORLD analysis and the per-segment quality check once per take."""
    from ..metrics import mel_cepstral_distortion

    q = quality or QualityConfig()
    y = resample(x, sr, work_sr)
    voiced = ~np.isnan(midi)
    segs = [Segment(a, b, WORLD) for a, b in _runs(voiced)]
    world = _world_analyze(y, work_sr, hop_s * 1000)
    copy = world_copy(y, work_sr, hop_s, world)
    pad = int(0.02 * work_sr)
    for s in segs:
        if q.force:
            s.route = q.force
            continue
        a, b = int(s.start * hop_s * work_sr), min(len(y), int(s.end * hop_s * work_sr))
        lo, hi = max(0, a - pad), b + pad
        s.voicing = float(np.mean(voicing[s.start : s.end])) if len(voicing) else 1.0
        s.hnr_db = _segment_hnr(y[lo:hi], work_sr)
        s.mcd_db = mel_cepstral_distortion(y[lo:hi], copy[lo:hi], work_sr)
        if (
            (not np.isnan(s.hnr_db) and s.hnr_db < q.hnr_db)
            or s.voicing < q.min_voicing
            or (not np.isnan(s.mcd_db) and s.mcd_db > q.mcd_threshold)
        ):
            s.route = PSOLA
    return Prepared(y, work_sr, hop_s, segs, world, copy)


def _psola_apply(prep: Prepared, ratio: np.ndarray) -> np.ndarray:
    from parselmouth.praat import call

    snd, manip, pts = prep.psola_objects()
    new = call("Create PitchTier", "shifted", snd.xmin, snd.xmax)
    rt = np.arange(len(ratio)) * prep.hop_s
    r = np.interp(pts[:, 0], rt, ratio, left=1, right=1) if len(pts) else []
    for (tt, f), k in zip(pts, r):
        call(new, "Add point", float(tt), float(f * k))
    call([manip, new], "Replace pitch tier")
    out = call(manip, "Get resynthesis (overlap-add)")
    return _fit_len(out.values[0].astype(np.float64), len(prep.y))


def apply(prep: Prepared, p: Plan) -> ShiftResult:
    """Apply a plan with the cached routing, crossfading each path into the original."""
    hop = prep.hop_s
    ratio = p.ratio()
    nfr = len(ratio)
    active = np.abs(p.correction) > 1e-4
    segs = []
    world_frames = np.zeros(nfr, bool)
    psola_frames = np.zeros(nfr, bool)
    for s in prep.segments:
        route = s.route if active[s.start : s.end].any() else UNTOUCHED
        segs.append(Segment(s.start, s.end, route, s.hnr_db, s.voicing, s.mcd_db))
        if route == WORLD:
            world_frames[s.start : s.end] = True
        elif route == PSOLA:
            psola_frames[s.start : s.end] = True

    y = prep.y
    out = y.copy()
    if world_frames.any():
        yw = world_shift(y, prep.sr, np.where(world_frames, ratio, 1.0), hop, prep.world)
        mw = _mask_from_frames(world_frames, hop, len(y), prep.sr)
        out = out * (1 - mw) + yw * mw
    if psola_frames.any():
        yp = _psola_apply(prep, np.where(psola_frames, ratio, 1.0))
        mp = _mask_from_frames(psola_frames, hop, len(y), prep.sr)
        out = out * (1 - mp) + yp * mp
    return ShiftResult(out, prep.sr, segs)


def shift(
    x: np.ndarray,
    sr: int,
    p: Plan,
    voicing: np.ndarray,
    quality: QualityConfig | None = None,
    work_sr: int = SYNTH_SR,
) -> ShiftResult:
    """One-shot convenience: prepare + apply."""
    return apply(prepare(x, sr, p.midi, voicing, p.hop_s, quality, work_sr), p)
