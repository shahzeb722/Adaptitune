"""Pitch analysis unit (proposal 5.1 and 5.3).

Every tracker returns a :class:`PitchTrack` on a common hop so the rest of the pipeline does
not care which one produced it. Neural trackers are optional and imported lazily; pYIN is
the always-available fallback.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..audio import ANALYSIS_SR, resample

DEFAULT_HOP_S = 0.005
FMIN, FMAX = 65.0, 1100.0  # C2 .. C#6, covers bass to soprano (the app's search range)
# The O2 benchmark searches C1 .. B6 for every tracker that takes a range: the native range
# of CREPE and FCPE, and wide enough for the bass and flute stems in MDB-stem-synth.
BENCH_FMIN, BENCH_FMAX = 32.70, 1975.5
# Long inputs are tracked in segments of SEGMENT_S seconds, each with SEGMENT_CONTEXT_S of
# extra audio on both sides that is tracked and then dropped. This bounds memory: PESTO needs
# about 34 MB per second of audio in one piece (5.8 GB for a 3-minute MDB stem), and several
# trackers grow similarly. The context is longer than PESTO's longest CQT kernel (about
# +-1.02 s at 16 kHz) and pYIN/CREPE's frames, so the kept frames see the same audio as an
# unsegmented run would.
SEGMENT_S, SEGMENT_CONTEXT_S = 30.0, 1.5


def default_threads() -> int:
    """CPU threads for model inference: $ADAPTITUNE_THREADS, else up to 4 (the reference
    laptop has 4 cores, proposal 5.3)."""
    try:
        return max(1, int(os.environ["ADAPTITUNE_THREADS"]))
    except (KeyError, ValueError):
        return max(1, min(4, os.cpu_count() or 1))


def _interp_log(src_t: np.ndarray, f: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Resample a frequency series (0 = no value) in the log domain. Values are interpolated
    only between neighbours that both have one; at the edges of a run the nearest value is
    used, and frames with no nearby value stay 0."""
    if len(src_t) == 0:
        return np.zeros(len(t))
    logf = np.where(f > 0, np.log2(np.maximum(f, 1e-6)), np.nan)
    idx = np.clip(np.searchsorted(src_t, t), 1, max(len(src_t) - 1, 1))
    lo, hi = idx - 1, idx
    a, b = logf[lo], logf[hi]
    w = np.clip((t - src_t[lo]) / np.maximum(src_t[hi] - src_t[lo], 1e-9), 0, 1)
    out = np.where(np.isnan(a) | np.isnan(b), np.nan, a + w * (b - a))
    near = np.where(w < 0.5, a, b)  # nearest-neighbour fallback at voicing edges
    out = np.where(np.isnan(out), near, out)
    return np.where(np.isnan(out), 0.0, 2**out)


@dataclass
class PitchTrack:
    times: np.ndarray  # seconds
    f0: np.ndarray  # Hz, 0 where unvoiced
    voicing: np.ndarray  # voicing probability / confidence in [0, 1]
    hop_s: float
    tracker: str = ""
    seconds: float = 0.0  # wall-clock time spent
    # The tracker's pitch estimate on every frame, before its voicing decision (0 where it
    # has none). Used for voicing-independent raw pitch accuracy (mir_eval convention).
    raw_f0: np.ndarray | None = None

    @property
    def voiced(self) -> np.ndarray:
        return self.f0 > 0

    def resampled(self, hop_s: float, n_frames: int | None = None) -> "PitchTrack":
        if abs(hop_s - self.hop_s) < 1e-9 and n_frames in (None, len(self.f0)):
            return self
        if n_frames is None:
            n_frames = int(np.floor(self.times[-1] / hop_s)) + 1 if len(self.times) else 0
        t = np.arange(n_frames) * hop_s
        v = np.interp(t, self.times, self.voicing, left=0, right=0) if len(self.times) else np.zeros(n_frames)
        f0 = _interp_log(self.times, self.f0, t)
        raw = None if self.raw_f0 is None else _interp_log(self.times, self.raw_f0, t)
        if raw is not None:
            # near a voicing edge the two interpolations see different neighbours; a voiced frame
            # must report one pitch, so the raw curve takes the voiced value there
            raw = np.where(f0 > 0, f0, raw)
        return PitchTrack(t, f0, v, hop_s, self.tracker, self.seconds, raw)


class Tracker(ABC):
    name = "base"
    analysis_sr = ANALYSIS_SR

    def track(self, x: np.ndarray, sr: int, hop_s: float = DEFAULT_HOP_S) -> PitchTrack:
        y = resample(x, sr, self.analysis_sr)
        t0 = time.perf_counter()
        pt = self._track_segmented(y, self.analysis_sr, hop_s)
        pt.seconds = time.perf_counter() - t0
        pt.tracker = self.name
        n = int(np.ceil(len(x) / sr / hop_s))
        return pt.resampled(hop_s, n)

    def _track_segmented(self, y: np.ndarray, sr: int, hop_s: float) -> PitchTrack:
        """``_track`` on SEGMENT_S pieces with SEGMENT_CONTEXT_S of context, joined again."""
        seg, ctx = int(round(SEGMENT_S * sr)), int(round(SEGMENT_CONTEXT_S * sr))
        if len(y) <= seg + 2 * ctx:
            return self._track(y, sr, hop_s)
        times, f0, voicing, raw, hops = [], [], [], [], []
        for s0 in range(0, len(y), seg):
            a, b = max(0, s0 - ctx), min(len(y), s0 + seg + ctx)
            p = self._track(y[a:b], sr, hop_s)
            t = p.times + a / sr
            # keep the frames that belong to [s0, s0 + seg); the quarter-hop margin keeps a frame
            # that sits exactly on a boundary in exactly one segment despite rounding
            m = 0.25 * p.hop_s
            keep = (t >= s0 / sr - m) & (t < (s0 + seg) / sr - m)
            times.append(t[keep])
            f0.append(p.f0[keep])
            voicing.append(p.voicing[keep])
            raw.append(None if p.raw_f0 is None else p.raw_f0[keep])
            hops.append(p.hop_s)
        raw_f0 = None if any(r is None for r in raw) else np.concatenate(raw)
        return PitchTrack(np.concatenate(times), np.concatenate(f0), np.concatenate(voicing), hops[0], raw_f0=raw_f0)

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
            fill_na=None,  # best-guess pitch on unvoiced frames too (raw pitch)
        )
        raw = np.nan_to_num(np.asarray(f0, np.float64))
        # pYIN reports the centre of its pitch state, i.e. it is quantized to `resolution`.
        # Refine with YIN's parabolic-interpolated estimate wherever both agree within a
        # quarter tone, keeping pYIN's voicing and octave decisions.
        if self.refine:
            fy = librosa.yin(y, fmin=self.fmin, fmax=self.fmax, sr=sr, frame_length=self.frame_length, hop_length=hop)
            k = min(len(fy), len(raw))
            agree = (raw[:k] > 0) & (np.abs(12 * np.log2(np.maximum(fy[:k], 1) / np.maximum(raw[:k], 1))) < 0.5)
            raw[:k] = np.where(agree, fy[:k], raw[:k])
        f0 = np.where(voiced, raw, 0.0)
        t = np.arange(len(f0)) * hop / sr
        return PitchTrack(t, f0, np.nan_to_num(prob), hop / sr, raw_f0=raw)


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
        raw = np.asarray(f0, np.float64)
        f0 = np.where(voiced, raw, 0.0)
        t = np.arange(len(f0)) * hop / sr
        return PitchTrack(t, f0, prob, hop / sr, raw_f0=raw)


class PestoTracker(Tracker):
    """PESTO (Riou et al. 2023), ~130k-parameter self-supervised tracker."""

    name = "pesto"

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

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
        return PitchTrack(t, np.where(conf > self.threshold, f0, 0.0), conf, hop_s, raw_f0=f0)


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

    def __init__(self, threshold: float = 0.006, fmin: float = FMIN, fmax: float = FMAX):
        self.threshold = threshold  # torchfcpe's default voicing threshold on the peak latent
        self.fmin, self.fmax = fmin, fmax

    @classmethod
    def _load(cls):
        if cls._model is None:
            import contextlib
            import sys

            from torchfcpe import spawn_bundled_infer_model

            with contextlib.redirect_stdout(sys.stderr):  # torchfcpe prints INFO lines to stdout
                cls._model = spawn_bundled_infer_model(device="cpu")
        return cls._model

    def _track(self, y, sr, hop_s):
        import torch

        m = self._load()
        x = torch.tensor(y, dtype=torch.float32)[None, :, None]
        with torch.no_grad():
            # Same steps as torchfcpe's ``infer`` (mel -> network -> local-argmax decoding),
            # but without its voicing mask, so the raw pitch of every frame is kept.
            latent = m.model.forward(m.wav2mel(x, sr))
            cents = m.model.latent2cents_local_decoder(latent, mask=False)
            raw = m.model.cent_to_f0(cents).squeeze().cpu().numpy().astype(np.float64)
            conf = latent.max(dim=-1).values.squeeze().cpu().numpy().astype(np.float64)
        raw = np.atleast_1d(np.clip(np.nan_to_num(raw), 0, self.fmax))
        conf = np.atleast_1d(conf)
        voiced = (conf > self.threshold) & (raw > self.fmin * 0.9)
        f0 = np.where(voiced, raw, 0.0)
        hop = m.get_hop_size() / m.get_model_sr()
        # voicing stays binary as before, so the planner and quality check behave the same
        return PitchTrack(np.arange(len(f0)) * hop, f0, voiced.astype(np.float64), hop, raw_f0=raw)


class CrepeTracker(Tracker):
    """CREPE (Kim et al. 2018) via torchcrepe, 'tiny' model by default for CPU speed."""

    name = "crepe"

    def __init__(self, model: str = "tiny", threshold: float = 0.4, fmin: float = FMIN, fmax: float = FMAX):
        self.model, self.threshold = model, threshold
        self.fmin, self.fmax = fmin, fmax

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
            x, sr, hop, self.fmin, self.fmax, self.model, decoder=_crepe_viterbi_local_average,
            return_periodicity=True, batch_size=512, device="cpu",
        )
        raw = f0.squeeze().numpy().astype(np.float64)
        per = per.squeeze().numpy().astype(np.float64)
        f0 = np.where(per > self.threshold, raw, 0.0)
        return PitchTrack(np.arange(len(f0)) * hop / sr, f0, per, hop / sr, raw_f0=raw)


def _crepe_viterbi_local_average(salience):
    """CREPE's own decoding with Viterbi smoothing (Kim et al. 2018, ``crepe --viterbi``): the
    Viterbi path picks a bin per frame, then the pitch is the salience-weighted mean of the
    path bin and its four neighbours on each side. torchcrepe's default decoder returns the
    bin centre plus random triangular dither of up to +-20 cents instead, which made two runs
    on the same audio differ by up to 37 cents and is not what the paper evaluates."""
    import torch
    import torchcrepe

    bins, _ = torchcrepe.decode.viterbi(salience)  # its frequencies carry the dither; only the path is used
    sal = torch.where(torch.isfinite(salience), salience, torch.zeros_like(salience))
    idx = torch.arange(sal.shape[1], device=sal.device)[None, :, None]
    w = sal * ((idx >= bins[:, None, :] - 4) & (idx <= bins[:, None, :] + 4))
    cents = torchcrepe.CENTS_PER_BIN * idx.to(sal.dtype) + 1997.3794084376191
    cents = (w * cents).sum(1) / w.sum(1).clamp_min(1e-12)
    return bins, torchcrepe.convert.cents_to_frequency(cents)


class OnnxTracker(Tracker):
    """Runs an exported tracker with ONNX Runtime (deployment layer, proposal 5.5).

    The model must take a float32 waveform ``[1, n]`` at ``analysis_sr`` and return
    ``(f0_hz[1, T], confidence[1, T])`` at ``model_hop_s``. ``scripts/export_onnx.py`` produces
    such a model from PESTO.
    """

    name = "onnx"

    def __init__(self, path: str, model_hop_s: float = 0.01, threshold: float = 0.5, threads: int | None = None):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads or default_threads()
        so.inter_op_num_threads = 1
        self.path = path
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.model_hop_s = model_hop_s
        self.threshold = threshold

    def _track(self, y, sr, hop_s):
        inp = self.sess.get_inputs()[0].name
        raw, conf = self.sess.run(None, {inp: y.astype(np.float32)[None]})
        raw, conf = raw[0].astype(np.float64), np.clip(conf[0].astype(np.float64), 0, 1)
        f0 = np.where(conf > self.threshold, raw, 0.0)
        return PitchTrack(np.arange(len(f0)) * self.model_hop_s, f0, conf, self.model_hop_s, raw_f0=raw)


class PestoOnnxTracker(OnnxTracker):
    """PESTO exported with ``scripts/export_onnx.py`` (the deployment form of PESTO). Listed
    in the benchmark so the export is measured exactly as the app runs it."""

    name = "pesto-onnx"
    model = "pesto"

    def __init__(self, path: str | None = None, **kw):
        path = path or find_onnx_model(self.model)
        if not path:
            raise RuntimeError(f"no exported model found; run scripts/export_onnx.py --model {self.model}")
        super().__init__(path, **kw)

    @classmethod
    def available(cls) -> bool:
        try:
            import onnxruntime  # noqa: F401
        except Exception:
            return False
        return find_onnx_model(cls.model) is not None


class FcpeOnnxTracker(PestoOnnxTracker):
    """FCPE exported with ``scripts/export_onnx.py --model fcpe``: the same front end, network
    and decoder as :class:`FcpeTracker`, run with ONNX Runtime, so the app needs no PyTorch.
    Voicing and range handling are FCPE's, as in :class:`FcpeTracker`."""

    name = "fcpe-onnx"
    model = "fcpe"

    def __init__(self, path: str | None = None, threshold: float = 0.006, fmin: float = FMIN, fmax: float = FMAX, **kw):
        super().__init__(path, threshold=threshold, **kw)
        self.fmin, self.fmax = fmin, fmax

    def _track(self, y, sr, hop_s):
        inp = self.sess.get_inputs()[0].name
        raw, conf = self.sess.run(None, {inp: y.astype(np.float32)[None]})
        raw = np.atleast_1d(np.clip(np.nan_to_num(raw[0].astype(np.float64)), 0, self.fmax))
        conf = np.atleast_1d(conf[0].astype(np.float64))
        voiced = (conf > self.threshold) & (raw > self.fmin * 0.9)
        f0 = np.where(voiced, raw, 0.0)
        hop = self.model_hop_s
        return PitchTrack(np.arange(len(f0)) * hop, f0, voiced.astype(np.float64), hop, raw_f0=raw)


REGISTRY: dict[str, type[Tracker]] = {
    "pyin": PyinTracker,
    "yin": YinTracker,
    "pesto": PestoTracker,
    "fcpe": FcpeTracker,
    "crepe": CrepeTracker,
    "pesto-onnx": PestoOnnxTracker,
    "fcpe-onnx": FcpeOnnxTracker,
}


def find_onnx_model(model: str = "pesto") -> str | None:
    """An exported tracker ``<model>.onnx`` (pesto or fcpe), from $ADAPTITUNE_ONNX (PESTO) or
    $ADAPTITUNE_FCPE_ONNX, ~/.adaptitune, ./models or next to a frozen (PyInstaller) executable."""
    import os
    import sys
    from pathlib import Path

    cands = [os.environ.get("ADAPTITUNE_ONNX" if model == "pesto" else f"ADAPTITUNE_{model.upper()}_ONNX")]
    home = Path(os.environ.get("ADAPTITUNE_HOME", Path.home() / ".adaptitune"))
    cands += [str(home / f"{model}.onnx"), f"models/{model}.onnx"]
    if getattr(sys, "frozen", False):
        # PyInstaller 6 puts bundled data in _internal/ (sys._MEIPASS), not next to the exe
        for base in (getattr(sys, "_MEIPASS", None), Path(sys.executable).parent):
            if base:
                cands.append(str(Path(base) / "models" / f"{model}.onnx"))
    for c in cands:
        if c and Path(c).is_file():
            return c
    return None


# The tracker the O2 benchmark chose with the rule fixed in advance (docs/O2_PROTOCOL.md A6):
# FCPE, RPA 0.988 on MIR-1K and 0.941 on MDB-stem-synth (docs/RESULTS.md). "auto" means this one,
# run through its ONNX export when that is there (the deployment path, no PyTorch needed).
APP_TRACKER = "fcpe"


def get_tracker(name: str = "auto", **kw) -> Tracker:
    """'auto' is the tracker O2 chose (``APP_TRACKER``): its ONNX export if one is found, else
    the PyTorch model. If neither is installed, it falls back to the PESTO ONNX export, then
    PESTO or FCPE if PyTorch is installed, then pYIN."""
    if name == "auto":
        for cand in (f"{APP_TRACKER}-onnx", APP_TRACKER):
            if cand in REGISTRY and REGISTRY[cand].available():
                return REGISTRY[cand](**kw)
        onnx = find_onnx_model()
        if onnx:
            try:
                return OnnxTracker(onnx, **kw)
            except Exception:
                pass
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
