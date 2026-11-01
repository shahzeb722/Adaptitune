"""Objective metrics (proposal 5.3, 5.4 and Table 3).

* Tracker accuracy: raw pitch / raw chroma accuracy (mir_eval).
* Corrector accuracy: per-note distance from the intended semitone, at the note centre.
* Expressiveness: vibrato rate and extent of a note.
* Reward terms: in-key accuracy ``a`` and naturalness ``n`` from three artifact indicators.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .music import hz_to_midi

# ----------------------------------------------------------------- tracker accuracy


def raw_pitch_accuracy(ref_t, ref_f0, est_t, est_f0, cent_tolerance: float = 50) -> dict:
    import mir_eval

    ref_v, ref_c, est_v, est_c = mir_eval.melody.to_cent_voicing(
        np.asarray(ref_t, float), np.asarray(ref_f0, float), np.asarray(est_t, float), np.asarray(est_f0, float)
    )
    return {
        "rpa": float(mir_eval.melody.raw_pitch_accuracy(ref_v, ref_c, est_v, est_c, cent_tolerance)),
        "rca": float(mir_eval.melody.raw_chroma_accuracy(ref_v, ref_c, est_v, est_c, cent_tolerance)),
        "vr": float(mir_eval.melody.voicing_recall(ref_v, est_v)),
        "vfa": float(mir_eval.melody.voicing_false_alarm(ref_v, est_v)),
    }


# ----------------------------------------------------------------- notes and vibrato


def note_center(midi: np.ndarray, start: int, end: int) -> float:
    seg = midi[start:end]
    seg = seg[~np.isnan(seg)]
    if len(seg) == 0:
        return float("nan")
    skip = int(0.15 * len(seg))
    core = seg[skip:] if len(seg) - skip >= 3 else seg
    return float(np.median(core))


def vibrato(midi: np.ndarray, hop_s: float, fmin: float = 3.0, fmax: float = 9.0) -> tuple[float, float]:
    """(rate Hz, extent cents) of a single note's contour. Extent = half peak-to-peak,
    estimated as sqrt(2) * RMS of the band-passed contour so noise does not dominate."""
    from scipy.signal import butter, sosfiltfilt

    m = midi[~np.isnan(midi)]
    n = len(m)
    if n * hop_s < 0.35:
        return float("nan"), float("nan")
    fs = 1 / hop_s
    sos = butter(2, [fmin, min(fmax, fs / 2 * 0.9)], btype="band", fs=fs, output="sos")
    v = sosfiltfilt(sos, m - np.mean(m))
    trim = int(0.1 * n)
    v = v[trim : n - trim] if n - 2 * trim > 8 else v
    extent = float(np.sqrt(2) * np.std(v) * 100)
    spec = np.abs(np.fft.rfft(v * np.hanning(len(v)), n=8 * len(v)))
    freqs = np.fft.rfftfreq(8 * len(v), hop_s)
    band = (freqs >= fmin) & (freqs <= fmax)
    rate = float(freqs[band][np.argmax(spec[band])]) if band.any() else float("nan")
    return rate, extent


@dataclass
class NoteScore:
    intended: float
    center: float
    error_cents: float
    vib_rate: float = float("nan")
    vib_extent: float = float("nan")


def score_notes(midi: np.ndarray, hop_s: float, truth: list[dict]) -> list[NoteScore]:
    """Score a contour against ground-truth notes (dicts with start, end, intended_midi)."""
    out = []
    for nt in truth:
        a, b = int(round(nt["start"] / hop_s)), int(round(nt["end"] / hop_s))
        c = note_center(midi, a, b)
        r, e = vibrato(midi[a:b], hop_s)
        out.append(NoteScore(nt["intended_midi"], c, 100 * (c - nt["intended_midi"]), r, e))
    return out


# ----------------------------------------------------------------- reward terms


def in_key_accuracy(centers: np.ndarray, targets: np.ndarray, tol_cents: float = 25) -> float:
    """``a`` in the reward: share of notes whose centre is within ``tol`` of the target."""
    centers, targets = np.asarray(centers, float), np.asarray(targets, float)
    ok = ~np.isnan(centers)
    if not ok.any():
        return 0.0
    return float(np.mean(np.abs(100 * (centers[ok] - targets[ok])) <= tol_cents))


def _log_mel_cepstrum(x: np.ndarray, sr: int, n_mels: int = 40, n_ceps: int = 24) -> np.ndarray:
    import librosa
    from scipy.fft import dct

    n_fft = 1024 if sr > 16000 else 512
    S = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=n_fft, hop_length=n_fft // 4, n_mels=n_mels, power=1.0)
    return dct(np.log(S + 1e-10), axis=0, type=2, norm="ortho")[1 : n_ceps + 1]


def mel_cepstral_distortion(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    """Frame-averaged MCD in dB between two time-aligned signals (energy term excluded)."""
    n = min(len(a), len(b))
    if n < 0.03 * sr:
        return float("nan")
    ca, cb = _log_mel_cepstrum(a[:n], sr), _log_mel_cepstrum(b[:n], sr)
    # ignore near-silent frames
    import librosa

    rms = librosa.feature.rms(y=a[:n], frame_length=1024 if sr > 16000 else 512, hop_length=256 if sr > 16000 else 128)[0]
    k = min(ca.shape[1], len(rms))
    keep = rms[:k] > 0.05 * (rms.max() + 1e-12)
    if not keep.any():
        return float("nan")
    d = (10 / np.log(10)) * np.sqrt(2 * np.sum((ca[:, :k] - cb[:, :k]) ** 2, axis=0))
    return float(np.mean(d[keep]))


def new_pitch_jumps(orig_midi: np.ndarray, out_midi: np.ndarray, hop_s: float, cents_thr: float = 50, within_s: float = 0.01) -> float:
    """Share of voiced frames with a > 50 cent jump within 10 ms that the original did not have."""
    k = max(1, int(round(within_s / hop_s)))

    def jumps(m):
        d = np.abs(m[k:] - m[:-k]) * 100
        return np.nan_to_num(d, nan=0) > cents_thr

    if len(orig_midi) <= k:
        return 0.0
    jo, jc = jumps(orig_midi), jumps(out_midi)
    voiced = (~np.isnan(out_midi[k:])).sum() or 1
    return float(np.sum(jc & ~jo) / voiced)


def hnr_db(x: np.ndarray, sr: int) -> float:
    import parselmouth
    from parselmouth.praat import call

    snd = parselmouth.Sound(np.ascontiguousarray(x, dtype=np.float64), sampling_frequency=sr)
    h = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0).values[0]
    h = h[h > -200]
    return float(np.mean(h)) if len(h) else 0.0


@dataclass
class Naturalness:
    mcd_rise_db: float
    jump_rate: float
    hnr_drop_db: float
    score: float  # n in [0, 1]


# scales: an indicator reaching these values counts as fully bad (1.0)
MCD_RISE_FULL = 4.0  # dB above the WORLD copy-synthesis of the same audio
JUMP_RATE_FULL = 0.02  # 2% of voiced frames with new jumps
HNR_DROP_FULL = 6.0  # dB


def naturalness(
    orig: np.ndarray,
    corrected: np.ndarray,
    copy_synth: np.ndarray,
    sr: int,
    orig_midi: np.ndarray,
    out_midi: np.ndarray,
    hop_s: float,
) -> Naturalness:
    """n = 1 - mean of the three indicators, each scaled to [0, 1] (proposal 5.4)."""
    mcd_c = mel_cepstral_distortion(orig, corrected, sr)
    mcd_ref = mel_cepstral_distortion(orig, copy_synth, sr)
    rise = max(0.0, np.nan_to_num(mcd_c - mcd_ref))
    jr = new_pitch_jumps(orig_midi, out_midi, hop_s)
    drop = max(0.0, hnr_db(orig, sr) - hnr_db(corrected, sr))
    parts = [min(1, rise / MCD_RISE_FULL), min(1, jr / JUMP_RATE_FULL), min(1, drop / HNR_DROP_FULL)]
    return Naturalness(rise, jr, drop, float(1 - np.mean(parts)))


def vibrato_penalty(orig_extent: float, new_extent: float, amount: float = 0.1) -> float:
    """p in the reward: small penalty when vibrato extent falls below half the original."""
    if np.isnan(orig_extent) or np.isnan(new_extent) or orig_extent < 15:  # no real vibrato
        return 0.0
    return amount if new_extent < 0.5 * orig_extent else 0.0


def contour_midi(f0: np.ndarray) -> np.ndarray:
    return hz_to_midi(np.where(f0 > 0, f0, np.nan))
