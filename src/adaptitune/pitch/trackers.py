"""Pitch analysis unit (proposal 5.1 and 5.3).

Every tracker returns a :class:`PitchTrack` on a common hop so the rest of the pipeline does
not care which one produced it. Neural trackers are optional and imported lazily; pYIN is
the always-available fallback.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..audio import ANALYSIS_SR, resample

DEFAULT_HOP_S = 0.005
FMIN, FMAX = 65.0, 1100.0  # C2 .. C#6, covers bass to soprano


@dataclass
class PitchTrack:
    times: np.ndarray  # seconds
    f0: np.ndarray  # Hz, 0 where unvoiced
    voicing: np.ndarray  # voicing probability / confidence in [0, 1]
    hop_s: float
    tracker: str = ""
    seconds: float = 0.0  # wall-clock time spent

    @property
    def voiced(self) -> np.ndarray:
        return self.f0 > 0

    def resampled(self, hop_s: float, n_frames: int | None = None) -> "PitchTrack":
        if abs(hop_s - self.hop_s) < 1e-9 and n_frames in (None, len(self.f0)):
            return self
        if n_frames is None:
            n_frames = int(np.floor(self.times[-1] / hop_s)) + 1 if len(self.times) else 0
        t = np.arange(n_frames) * hop_s
        v = np.interp(t, self.times, self.voicing, left=0, right=0)
        # interpolate in log-frequency, only between voiced neighbours
        logf = np.where(self.f0 > 0, np.log2(np.maximum(self.f0, 1e-6)), np.nan)
        idx = np.clip(np.searchsorted(self.times, t), 1, max(len(self.times) - 1, 1))
        lo, hi = idx - 1, idx
        a = logf[lo] if len(logf) else np.array([])
        b = logf[hi] if len(logf) else np.array([])
        w = np.clip((t - self.times[lo]) / np.maximum(self.times[hi] - self.times[lo], 1e-9), 0, 1)
        f = np.where(np.isnan(a) | np.isnan(b), np.nan, a + w * (b - a))
        # nearest-neighbour fallback at voicing edges
        near = np.where(w < 0.5, a, b)
        f = np.where(np.isnan(f), near, f)
        f0 = np.where(np.isnan(f), 0.0, 2**f)
        return PitchTrack(t, f0, v, hop_s, self.tracker, self.seconds)


class Tracker(ABC):
    name = "base"
    analysis_sr = ANALYSIS_SR

    def track(self, x: np.ndarray, sr: int, hop_s: float = DEFAULT_HOP_S) -> PitchTrack:
        y = resample(x, sr, self.analysis_sr)
        t0 = time.perf_counter()
        pt = self._track(y, self.analysis_sr, hop_s)
        pt.seconds = time.perf_counter() - t0
        pt.tracker = self.name
        n = int(np.ceil(len(x) / sr / hop_s))
        return pt.resampled(hop_s, n)

    @abstractmethod
    def _track(self, y: np.ndarray, sr: int, hop_s: float) -> PitchTrack: ...

    @classmethod
    def available(cls) -> bool:
        return True


class PyinTracker(Tracker):
    """Probabilistic YIN with HMM smoothing (Mauch & Dixon 2014), via librosa."""

    name = "pyin"

    def __init__(
        self,
        fmin: float = FMIN,
        fmax: float = FMAX,
        frame_length: int = 1024,
        internal_hop_s: float = 0.01,
        resolution: float = 0.25,
        refine: bool = True,
    ):
        # librosa's pYIN cost grows with frames x pitch states; a 10 ms hop and 0.2-semitone
        # state grid keep it near RTF 0.1 on a laptop CPU, and the contour is then
        # interpolated to the 5 ms planner hop.
        self.fmin, self.fmax, self.frame_length = fmin, fmax, frame_length
        self.internal_hop_s, self.resolution, self.refine = internal_hop_s, resolution, refine

    def _track(self, y, sr, hop_s):
        import librosa

        hop = max(1, int(round(max(hop_s, self.internal_hop_s) * sr)))
        f0, voiced, prob = librosa.pyin(
            y,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=sr,
            frame_length=self.frame_length,
            hop_length=hop,
            center=True,
            resolution=self.resolution,
        )
        f0 = np.where(voiced & ~np.isnan(f0), f0, 0.0)
        # pYIN reports the centre of its pitch state, i.e. it is quantized to `resolution`.
        # Refine with YIN's parabolic-interpolated estimate wherever both agree within a
        # quarter tone, keeping pYIN's voicing and octave decisions.
        if self.refine:
            fy = librosa.yin(y, fmin=self.fmin, fmax=self.fmax, sr=sr, frame_length=self.frame_length, hop_length=hop)
            k = min(len(fy), len(f0))
            agree = (f0[:k] > 0) & (np.abs(12 * np.log2(np.maximum(fy[:k], 1) / np.maximum(f0[:k], 1))) < 0.5)
            f0[:k] = np.where(agree, fy[:k], f0[:k])
        t = np.arange(len(f0)) * hop / sr
        return PitchTrack(t, f0, np.nan_to_num(prob), hop / sr)


class YinTracker(Tracker):
    """Plain YIN (de Cheveigné & Kawahara 2002). Fast, no voicing model; we derive voicing
    from the aperiodicity so it can still be used as a cheap baseline."""

    name = "yin"

    def __init__(self, fmin: float = FMIN, fmax: float = FMAX, frame_length: int = 1024):
        self.fmin, self.fmax, self.frame_length = fmin, fmax, frame_length

    def _track(self, y, sr, hop_s):
        import librosa

        hop = max(1, int(round(hop_s * sr)))
        f0 = librosa.yin(y, fmin=self.fmin, fmax=self.fmax, sr=sr, frame_length=self.frame_length, hop_length=hop)
        rms = librosa.feature.rms(y=y, frame_length=self.frame_length, hop_length=hop)[0][: len(f0)]
        loud = rms > (np.max(rms) * 0.05 if len(rms) else 0)
        # crude periodicity: normalized autocorrelation at the detected lag
        frames = librosa.util.frame(np.pad(y, self.frame_length // 2), frame_length=self.frame_length, hop_length=hop)
        frames = frames[:, : len(f0)]
        lag = np.clip(np.round(sr / f0).astype(int), 1, self.frame_length - 1)
        num = np.array([np.dot(frames[: -l, i], frames[l:, i]) for i, l in enumerate(lag)])
        den = np.sqrt(np.sum(frames**2, axis=0) + 1e-12) ** 2
        per = np.clip(num / den * 2, 0, 1)
        prob = per * loud
        voiced = prob > 0.45
        f0 = np.where(voiced, f0, 0.0)
        t = np.arange(len(f0)) * hop / sr
        return PitchTrack(t, f0, prob, hop / sr)


class PestoTracker(Tracker):
    """PESTO (Riou et al. 2023), ~130k-parameter self-supervised tracker."""

    name = "pesto"

    @classmethod
    def available(cls) -> bool:
        try:
            import pesto  # noqa: F401
            import torch  # noqa: F401

            return True
        except Exception:
            return False

    def _track(self, y, sr, hop_s):
        import pesto
        import torch

        x = torch.tensor(y, dtype=torch.float32)
        t, f0, conf, _ = pesto.predict(x, sr, step_size=hop_s * 1000, convert_to_freq=True)
        t, f0, conf = (a.detach().cpu().numpy().astype(np.float64) for a in (t, f0, conf))
        if t.max() > 10 * len(y) / sr:  # some versions return ms
            t = t / 1000
        conf = np.clip(conf, 0, 1)
        f0 = np.where(conf > 0.5, f0, 0.0)
        return PitchTrack(t, f0, conf, hop_s)


class FcpeTracker(Tracker):
    """FCPE (Luo et al. 2024) via torchfcpe."""

    name = "fcpe"
    _model = None

    @classmethod
    def available(cls) -> bool:
        try:
            import torchfcpe  # noqa: F401

            return True
        except Exception:
            return False

    def _track(self, y, sr, hop_s):
        import torch
        from torchfcpe import spawn_bundled_infer_model

        if FcpeTracker._model is None:
            FcpeTracker._model = spawn_bundled_infer_model(device="cpu")
        n = int(np.ceil(len(y) / sr / hop_s))
        x = torch.tensor(y, dtype=torch.float32)[None, :, None]
        with torch.no_grad():
            f0 = FcpeTracker._model.infer(
                x, sr=sr, decoder_mode="local_argmax", threshold=0.006,
                f0_min=FMIN, f0_max=FMAX, interp_uv=False, output_interp_target_length=n,
            )
        f0 = f0.squeeze().cpu().numpy().astype(np.float64)
        f0 = np.where(f0 > FMIN * 0.9, f0, 0.0)
        conf = (f0 > 0).astype(np.float64)
        return PitchTrack(np.arange(len(f0)) * hop_s, f0, conf, hop_s)


class CrepeTracker(Tracker):
    """CREPE (Kim et al. 2018) via torchcrepe, 'tiny' model by default for CPU speed."""

    name = "crepe"

    def __init__(self, model: str = "tiny"):
        self.model = model

    @classmethod
    def available(cls) -> bool:
        try:
            import torchcrepe  # noqa: F401

            return True
        except Exception:
            return False

    def _track(self, y, sr, hop_s):
        import torch
        import torchcrepe

        hop = int(round(hop_s * sr))
        x = torch.tensor(y, dtype=torch.float32)[None]
        f0, per = torchcrepe.predict(
            x, sr, hop, FMIN, FMAX, self.model, return_periodicity=True, batch_size=512, device="cpu"
        )
        f0 = f0.squeeze().numpy().astype(np.float64)
        per = per.squeeze().numpy().astype(np.float64)
        f0 = np.where(per > 0.4, f0, 0.0)
        return PitchTrack(np.arange(len(f0)) * hop / sr, f0, per, hop / sr)


class OnnxTracker(Tracker):
    """Runs an exported tracker with ONNX Runtime (deployment layer, proposal 5.5).

    The model must take a float32 waveform ``[1, n]`` at ``analysis_sr`` and return
    ``(f0_hz[1, T], confidence[1, T])`` at ``model_hop_s``. ``scripts/export_onnx.py`` produces
    such a model from PESTO.
    """

    name = "onnx"

    def __init__(self, path: str, model_hop_s: float = 0.01, threshold: float = 0.5):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.model_hop_s = model_hop_s
        self.threshold = threshold

    def _track(self, y, sr, hop_s):
        inp = self.sess.get_inputs()[0].name
        f0, conf = self.sess.run(None, {inp: y.astype(np.float32)[None]})
        f0, conf = f0[0].astype(np.float64), np.clip(conf[0].astype(np.float64), 0, 1)
        f0 = np.where(conf > self.threshold, f0, 0.0)
        return PitchTrack(np.arange(len(f0)) * self.model_hop_s, f0, conf, self.model_hop_s)


REGISTRY: dict[str, type[Tracker]] = {
    "pyin": PyinTracker,
    "yin": YinTracker,
    "pesto": PestoTracker,
    "fcpe": FcpeTracker,
    "crepe": CrepeTracker,
}


def get_tracker(name: str = "auto", **kw) -> Tracker:
    """'auto' picks the benchmark winner if installed (PESTO, then FCPE) else pYIN."""
    if name == "auto":
        for cand in ("pesto", "fcpe"):
            if REGISTRY[cand].available():
                return REGISTRY[cand](**kw)
        return PyinTracker(**kw)
    if name.endswith(".onnx"):
        return OnnxTracker(name, **kw)
    cls = REGISTRY[name]
    if not cls.available():
        raise RuntimeError(f"tracker {name!r} is not installed; pip install 'adaptitune[neural]'")
    return cls(**kw)


def available_trackers() -> list[str]:
    return [k for k, v in REGISTRY.items() if v.available()]
