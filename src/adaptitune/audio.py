"""Audio input and preprocessing (proposal section 5.1, first component).

Everything downstream works on mono float64 audio in [-1, 1] at a known sample rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ANALYSIS_SR = 16_000  # what pitch trackers see
SYNTH_SR = 24_000  # what WORLD/PSOLA run at (see proposal 5.5: 24 kHz if 44.1 kHz is too slow)
TARGET_LUFS_RMS_DB = -20.0  # simple RMS loudness target, dBFS


@dataclass
class Audio:
    samples: np.ndarray  # mono float64
    sr: int

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sr

    def resampled(self, sr: int) -> "Audio":
        return Audio(resample(self.samples, self.sr, sr), sr)


def to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        # soundfile gives (frames, channels)
        x = x.mean(axis=1)
    return x


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return np.asarray(x, dtype=np.float64)
    g = np.gcd(int(sr_in), int(sr_out))
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float64)


def rms_db(x: np.ndarray) -> float:
    r = np.sqrt(np.mean(np.square(x)) + 1e-12)
    return 20 * np.log10(r)


def normalize_loudness(x: np.ndarray, target_db: float = TARGET_LUFS_RMS_DB) -> np.ndarray:
    """RMS-normalize, then make sure we never clip."""
    if not np.any(x):
        return x
    gain = 10 ** ((target_db - rms_db(x)) / 20)
    y = x * gain
    peak = np.max(np.abs(y))
    if peak > 0.99:
        y = y * (0.99 / peak)
    return y


def load(path: str | Path, sr: int = SYNTH_SR, normalize: bool = True) -> Audio:
    """Load any file soundfile can read. For mp3/m4a etc. we fall back to librosa/FFmpeg."""
    path = Path(path)
    try:
        data, file_sr = sf.read(str(path), always_2d=False)
    except Exception:  # pragma: no cover - depends on system codecs
        import librosa

        data, file_sr = librosa.load(str(path), sr=None, mono=False)
        data = data.T if data.ndim == 2 else data
    x = to_mono(data)
    x = resample(x, file_sr, sr)
    if normalize:
        x = normalize_loudness(x)
    return Audio(x, sr)


def save(path: str | Path, audio: Audio) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(audio.samples, -1, 1).astype(np.float32), audio.sr)


def preprocess(x: np.ndarray, sr: int, out_sr: int = SYNTH_SR) -> Audio:
    """Mono, resample, normalize. Framing is done by each consumer at its own hop."""
    return Audio(normalize_loudness(resample(to_mono(x), sr, out_sr)), out_sr)


def frame_times(n_frames: int, hop_s: float) -> np.ndarray:
    return np.arange(n_frames) * hop_s


def estimate_noise_db(x: np.ndarray, sr: int, frame_s: float = 0.03) -> float:
    """Background-noise estimate: 10th-percentile frame RMS in dBFS."""
    n = max(1, int(frame_s * sr))
    k = len(x) // n
    if k == 0:
        return rms_db(x)
    frames = x[: k * n].reshape(k, n)
    r = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    return float(20 * np.log10(np.percentile(r, 10)))
