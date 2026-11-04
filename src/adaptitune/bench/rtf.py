"""Full-pipeline real-time factor (NFR2, O5): tracker + key + planner + WORLD/PSOLA, measured
on the machine it runs on. The target is <= 0.5 on the reference laptop (4-core 8th-gen i5,
8 GB RAM, CPU only). Re-tracking the output for scoring is not part of the user-facing path
and is excluded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..audio import Audio, load, save
from ..correction.settings import DEFAULT
from ..machine import cpu_name as _cpu_name  # noqa: F401 (kept for callers)
from ..machine import machine_info
from ..pipeline import Corrector
from ..pitch.trackers import default_threads

TARGET_RTF = 0.5


def mir1k_minute(mir1k_root: str | Path, path: str | Path, seconds: float = 60.0) -> Path:
    """The O5 test input: the vocal channel of the first MIR-1K clips in file-name order, with
    0.3 s gaps, until ``seconds`` long, written at 16 kHz. The cloud run and the laptop runs all
    time this same audio (63.8 s)."""
    from ..data.datasets import mir1k

    parts, total, sr = [], 0.0, 16_000
    for item in mir1k(mir1k_root, sr=sr):
        parts += [item.audio.samples, np.zeros(int(0.3 * sr))]
        total += item.audio.duration + 0.3
        if total >= seconds:
            break
    if not parts:
        raise FileNotFoundError(f"no MIR-1K clips with pitch labels under {mir1k_root}")
    path = Path(path)
    save(path, Audio(np.concatenate(parts), sr), "PCM_16")
    return path


def _synthetic(seconds: float = 20.0) -> Audio:
    from ..data.detune import make_singers, synth_clean_take

    rng = np.random.default_rng(0)
    audio, _, _ = synth_clean_take(make_singers(1)[0], rng)
    while audio.duration < seconds:
        more, _, _ = synth_clean_take(make_singers(1)[0], rng)
        audio = Audio(np.concatenate([audio.samples, more.samples]), audio.sr)
    return audio


def measure(src: str | Path | Audio | None = None, tracker: str = "auto", repeats: int = 3) -> dict:
    """Time the user-facing path (analysis, planning and shifting at the default setting) on a
    file, an ``Audio`` or, with nothing given, 20 s of synthetic singing. The RTF is the best of
    ``repeats`` runs after a warm-up; the median is reported too."""
    import time

    audio = src if isinstance(src, Audio) else load(src) if src else _synthetic()
    c = Corrector(tracker)
    c.analyze(Audio(audio.samples[: audio.sr * 2], audio.sr))  # warm-up (JIT, model load)
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        an = c.analyze(audio)
        res = c.correct(an, DEFAULT, score=False)
        runs.append(time.perf_counter() - t0)
        stages = res.seconds
    best = min(runs)
    info = machine_info(default_threads(), ("numpy", "scipy", "librosa", "onnxruntime", "pyworld", "praat-parselmouth", "PySide6"))
    return {
        "tracker": c.tracker.name,
        "audio_seconds": round(audio.duration, 2),
        "rtf": round(best / audio.duration, 3),
        "rtf_median": round(float(np.median(runs)) / audio.duration, 3),
        "stages_rtf": {k: round(v / audio.duration, 3) for k, v in stages.items()},
        "route_shares": {k: round(float(v), 3) for k, v in res.shift.route_shares().items()},
        "cpu": info["cpu"],
        "cores": info["logical_cores"],
        "threads": info["threads"],
        "repeats": repeats,
        "target": TARGET_RTF,
        "pass": best / audio.duration <= TARGET_RTF,
        "machine": info,
    }
