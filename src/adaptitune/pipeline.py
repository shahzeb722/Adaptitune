"""End-to-end correction pipeline: preprocess -> track -> key -> plan -> shift -> score.

A take is analysed once (:meth:`Corrector.analyze`) and can then be corrected with any
number of settings (:meth:`Corrector.correct`), which is what A/B comparison, the
personalization agent and the benchmarks need.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import metrics as M
from .audio import SYNTH_SR, Audio, estimate_noise_db, preprocess
from .correction.planner import Plan, plan
from .correction.settings import CorrectionSettings
from .correction.shifter import Prepared, QualityConfig, ShiftResult, apply, prepare
from .music import Key
from .pitch.key import estimate_key
from .pitch.trackers import DEFAULT_HOP_S, PitchTrack, Tracker, get_tracker


@dataclass
class Analysis:
    audio: Audio
    track: PitchTrack
    key: Key
    key_confidence: float
    noise_db: float
    prepared: Prepared
    seconds: float = 0.0


@dataclass
class TakeResult:
    audio: Audio
    plan: Plan
    shift: ShiftResult
    settings: CorrectionSettings
    key: Key | None
    seconds: dict = field(default_factory=dict)
    out_track: PitchTrack | None = None
    accuracy: float = float("nan")  # a
    natural: M.Naturalness | None = None  # n
    vib_penalty: float = 0.0  # p

    @property
    def rtf(self) -> float:
        return sum(self.seconds.values()) / max(self.audio.duration, 1e-9)

    def note_report(self) -> list[dict]:
        """FR6: per-note pitch error before and after."""
        rows = []
        out_m = M.contour_midi(self.out_track.f0) if self.out_track is not None else self.plan.corrected_midi
        for i, n in enumerate(self.plan.notes):
            after = M.note_center(out_m, n.start, n.end)
            rows.append(
                dict(
                    index=i,
                    start=round(n.t_start, 3),
                    end=round(n.t_end, 3),
                    target=n.target,
                    before_cents=round(100 * (n.center - n.target), 1),
                    after_cents=round(100 * (after - n.target), 1) if not np.isnan(after) else None,
                )
            )
        return rows


class Corrector:
    def __init__(
        self,
        tracker: Tracker | str = "auto",
        quality: QualityConfig | None = None,
        hop_s: float = DEFAULT_HOP_S,
        work_sr: int = SYNTH_SR,
    ):
        self.tracker = get_tracker(tracker) if isinstance(tracker, str) else tracker
        self.quality = quality or QualityConfig()
        self.hop_s = hop_s
        self.work_sr = work_sr

    # ------------------------------------------------------------------ analysis
    def analyze(self, x: np.ndarray | Audio, sr: int | None = None, key: Key | str | None = None) -> Analysis:
        t0 = time.perf_counter()
        audio = x if isinstance(x, Audio) else preprocess(x, sr, self.work_sr)
        if audio.sr != self.work_sr:
            audio = audio.resampled(self.work_sr)
        track = self.tracker.track(audio.samples, audio.sr, self.hop_s)
        if key is None:
            k, conf = estimate_key(track.f0, track.voicing)
        else:
            k, conf = (Key.parse(key) if isinstance(key, str) else key), 1.0
        noise = estimate_noise_db(audio.samples, audio.sr)
        prep = prepare(audio.samples, audio.sr, M.contour_midi(track.f0), track.voicing, self.hop_s, self.quality, self.work_sr)
        return Analysis(audio, track, k, conf, noise, prep, time.perf_counter() - t0)

    # ------------------------------------------------------------------ correction
    def correct(
        self,
        an: Analysis,
        settings: CorrectionSettings,
        key: Key | None = None,
        score: bool = True,
    ) -> TakeResult:
        secs = {"analysis": an.seconds}
        t0 = time.perf_counter()
        k = key or an.key
        p = plan(an.track, settings, k)
        secs["plan"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        sh = apply(an.prepared, p)
        secs["shift"] = time.perf_counter() - t0
        res = TakeResult(Audio(sh.samples, sh.sr), p, sh, settings, p.key, secs)
        if score:
            self.score(an, res)
        return res

    def score(self, an: Analysis, res: TakeResult) -> TakeResult:
        """Objective reward terms. Re-tracks the output, so it is not counted in the RTF."""
        out_track = self.tracker.track(res.audio.samples, res.audio.sr, self.hop_s)
        n = min(len(out_track.f0), len(an.track.f0))
        out_m = M.contour_midi(out_track.f0[:n])
        in_m = M.contour_midi(an.track.f0[:n])
        res.out_track = out_track
        centers = [M.note_center(out_m, nt.start, nt.end) for nt in res.plan.notes]
        targets = [nt.target for nt in res.plan.notes]
        res.accuracy = M.in_key_accuracy(np.array(centers), np.array(targets))
        copy = an.prepared.copy
        res.natural = M.naturalness(an.audio.samples, res.audio.samples, copy, an.audio.sr, in_m, out_m, self.hop_s)
        # vibrato penalty on the longest note with real vibrato
        pens = []
        for nt in res.plan.notes:
            if (nt.end - nt.start) * self.hop_s < 0.5:
                continue
            _, e0 = M.vibrato(in_m[nt.start : nt.end], self.hop_s)
            _, e1 = M.vibrato(out_m[nt.start : nt.end], self.hop_s)
            pens.append(M.vibrato_penalty(e0, e1))
        res.vib_penalty = float(np.mean(pens)) if pens else 0.0
        return res


def correct_file(path_in: str, path_out: str, settings: CorrectionSettings | None = None, key: str | None = None, tracker: str = "auto") -> TakeResult:
    from .audio import load, save

    c = Corrector(tracker)
    an = c.analyze(load(path_in), key=key)
    res = c.correct(an, settings or CorrectionSettings())
    save(path_out, res.audio)
    return res
