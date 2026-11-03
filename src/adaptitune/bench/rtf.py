"""Full-pipeline real-time factor (NFR2, O5): tracker + key + planner + WORLD/PSOLA, measured
on the machine it runs on. The target is <= 0.5 on the reference laptop (4-core 8th-gen i5,
8 GB RAM, CPU only). Re-tracking the output for scoring is not part of the user-facing path
and is excluded.
"""

from __future__ import annotations

import os
import platform
import time

import numpy as np

from ..audio import Audio, load
from ..correction.settings import DEFAULT
from ..pipeline import Corrector
from ..pitch.trackers import default_threads


def _cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def measure(path: str | None = None, tracker: str = "auto", repeats: int = 3) -> dict:
    if path:
        audio = load(path)
    else:
        from ..data.detune import make_singers, synth_clean_take

        rng = np.random.default_rng(0)
        audio, _, _ = synth_clean_take(make_singers(1)[0], rng)
        while audio.duration < 20:
            more, _, _ = synth_clean_take(make_singers(1)[0], rng)
            audio = Audio(np.concatenate([audio.samples, more.samples]), audio.sr)
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
    return {
        "tracker": c.tracker.name,
        "audio_seconds": round(audio.duration, 2),
        "rtf": round(best / audio.duration, 3),
        "rtf_median": round(float(np.median(runs)) / audio.duration, 3),
        "stages_rtf": {k: round(v / audio.duration, 3) for k, v in stages.items()},
        "route_shares": {k: round(float(v), 3) for k, v in res.shift.route_shares().items()},
        "cpu": _cpu_name(),
        "cores": os.cpu_count(),
        "threads": default_threads(),
        "target": 0.5,
        "pass": best / audio.duration <= 0.5,
    }
