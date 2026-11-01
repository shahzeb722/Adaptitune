"""Synthetic singing voice generator.

Used for unit tests, for the simulator's audio mode and as a stand-in corpus when the
public datasets are not available. Every note's intended pitch, injected detuning, vibrato
and scoop are known exactly, so it gives perfect ground truth.

The voice is additive harmonic synthesis shaped by a vowel formant envelope, plus a little
breath noise, jitter and shimmer. It is not meant to sound like a real singer, only to have
the properties that matter to trackers and shifters: a clear F0, formants that should not
move when the pitch moves, vibrato, scoops and onsets/offsets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..music import midi_to_hz

VOWELS = {
    # F1, F2, F3, F4 (Hz), rough averages for an adult voice
    "a": (800, 1150, 2900, 3900),
    "e": (400, 2100, 2800, 3800),
    "i": (300, 2300, 3000, 3900),
    "o": (450, 800, 2830, 3800),
    "u": (325, 700, 2530, 3500),
}
BANDWIDTHS = (80, 90, 120, 150)


@dataclass
class SungNote:
    midi: float  # intended pitch (integer for in-tune material)
    dur: float  # seconds
    offset_cents: float = 0.0  # constant detuning
    drift_cents: float = 0.0  # linear drift reached at the end of the note
    scoop_cents: float = 0.0  # starts this far away and glides in
    scoop_s: float = 0.08
    vibrato_rate: float = 5.5  # Hz
    vibrato_cents: float = 0.0  # extent = half peak-to-peak
    vowel: str = "a"
    gap_after: float = 0.12  # silence after the note


@dataclass
class SynthResult:
    samples: np.ndarray
    sr: int
    hop_s: float
    f0: np.ndarray  # actual sung F0 per hop frame (Hz, 0 = unvoiced)
    notes: list[dict] = field(default_factory=list)  # start, end, intended midi, offset, ...


def _envelope(freqs: np.ndarray, formants, tilt_db_per_oct: float = -9.0) -> np.ndarray:
    amp = np.zeros_like(freqs)
    gains = (1.0, 0.7, 0.35, 0.2)
    for fk, bk, gk in zip(formants, BANDWIDTHS, gains):
        amp += gk / (1 + ((freqs - fk) / bk) ** 2)
    tilt = 10 ** (tilt_db_per_oct * np.log2(np.maximum(freqs, 50) / 100) / 20)
    return (amp + 0.02) * tilt


def note_contour(n: SungNote, t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """MIDI contour (fractional) for one note at times t (relative to note start)."""
    c = np.full_like(t, n.midi + n.offset_cents / 100)
    if n.dur > 0:
        c += (n.drift_cents / 100) * (t / n.dur)
    if n.scoop_cents:
        c += (n.scoop_cents / 100) * np.exp(-t / max(n.scoop_s / 3, 1e-3))
    if n.vibrato_cents:
        # vibrato fades in over ~250 ms like a real singer
        ramp = np.clip(t / 0.25, 0, 1)
        phase = rng.uniform(0, 2 * np.pi)
        c += ramp * (n.vibrato_cents / 100) * np.sin(2 * np.pi * n.vibrato_rate * t + phase)
    return c


def sing(
    notes: list[SungNote],
    sr: int = 24_000,
    hop_s: float = 0.005,
    lead_in: float = 0.15,
    jitter_cents: float = 3.0,
    breath: float = 0.05,
    seed: int = 0,
) -> SynthResult:
    rng = np.random.default_rng(seed)
    total = lead_in + sum(n.dur + n.gap_after for n in notes)
    N = int(total * sr)
    t_all = np.arange(N) / sr
    midi = np.full(N, np.nan)
    amp = np.zeros(N)
    vowel_idx = np.zeros(N, dtype=int)
    vowel_keys = list(VOWELS)
    meta = []
    t0 = lead_in
    for n in notes:
        a, b = int(t0 * sr), int((t0 + n.dur) * sr)
        tt = t_all[a:b] - t0
        midi[a:b] = note_contour(n, tt, rng)
        att, rel = min(0.03, n.dur / 4), min(0.05, n.dur / 4)
        env = np.minimum(1, np.minimum(tt / att, (n.dur - tt) / rel))
        amp[a:b] = np.clip(env, 0, 1)
        vowel_idx[a:b] = vowel_keys.index(n.vowel)
        meta.append(
            dict(
                start=t0,
                end=t0 + n.dur,
                intended_midi=n.midi,
                offset_cents=n.offset_cents,
                drift_cents=n.drift_cents,
                scoop_cents=n.scoop_cents,
                vibrato_rate=n.vibrato_rate,
                vibrato_cents=n.vibrato_cents,
            )
        )
        t0 += n.dur + n.gap_after

    voiced = ~np.isnan(midi)
    # slow random jitter (smoothed noise), a few cents
    jit = rng.standard_normal(N // 200 + 2)
    jit = np.interp(np.arange(N) / 200, np.arange(len(jit)), jit) * (jitter_cents / 100)
    m = np.where(voiced, midi + jit, 0)
    f0 = np.where(voiced, midi_to_hz(m), 0.0)

    phase = 2 * np.pi * np.cumsum(f0) / sr
    out = np.zeros(N)
    fmax = sr / 2 * 0.95
    f0_safe = np.where(voiced, f0, 100.0)
    max_h = int(fmax / max(np.min(f0_safe), 50))
    shimmer = 1 + 0.03 * np.interp(np.arange(N) / 300, np.arange(N // 300 + 2), rng.standard_normal(N // 300 + 2))
    for vi, vk in enumerate(vowel_keys):
        mask = voiced & (vowel_idx == vi)
        if not mask.any():
            continue
        for h in range(1, max_h + 1):
            fh = h * f0_safe
            ok = mask & (fh < fmax)
            if not ok.any():
                break
            out[ok] += _envelope(fh[ok], VOWELS[vk]) * np.sin(h * phase[ok])
    out *= amp * shimmer
    h_rms = np.sqrt(np.mean(out[voiced] ** 2)) if voiced.any() else 1.0
    out /= h_rms or 1.0
    # breath noise relative to the harmonic RMS (breath=0.1 is about 20 dB HNR)
    noise = rng.standard_normal(N) * breath
    out += noise * (0.3 + amp) * voiced
    out += rng.standard_normal(N) * 1e-3  # room noise floor, about -60 dB
    peak = np.max(np.abs(out)) or 1.0
    out = 0.5 * out / peak

    hop = int(round(hop_s * sr))
    f0_frames = f0[::hop].copy()
    f0_frames[amp[::hop] < 0.3] = 0.0
    return SynthResult(out, sr, hop_s, f0_frames, meta)


def random_melody(
    rng: np.random.Generator,
    n_notes: int = 8,
    key_tonic: int = 0,
    mode: str = "major",
    low: int = 55,
    high: int = 72,
) -> list[int]:
    from ..music import Key

    pcs = Key(key_tonic, mode).pitch_classes
    pool = [m for m in range(low, high + 1) if m % 12 in pcs]
    cur = rng.choice(pool)
    out = [int(cur)]
    for _ in range(n_notes - 1):
        i = pool.index(cur) + int(rng.integers(-2, 3))
        cur = pool[int(np.clip(i, 0, len(pool) - 1))]
        out.append(int(cur))
    return out
