"""Small music-theory helpers: Hz/MIDI/cents conversion, keys and scales."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR_STEPS = (0, 2, 4, 5, 7, 9, 11)
MINOR_STEPS = (0, 2, 3, 5, 7, 8, 10)  # natural minor


def hz_to_midi(f: np.ndarray | float) -> np.ndarray:
    f = np.asarray(f, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = 69 + 12 * np.log2(f / 440.0)
    return np.where(f > 0, m, np.nan)


def midi_to_hz(m: np.ndarray | float) -> np.ndarray:
    return 440.0 * 2 ** ((np.asarray(m, dtype=np.float64) - 69) / 12)


def cents(a_midi, b_midi):
    """Signed distance a - b in cents."""
    return 100.0 * (np.asarray(a_midi) - np.asarray(b_midi))


@dataclass(frozen=True)
class Key:
    tonic: int  # pitch class 0..11
    mode: str  # "major" | "minor"

    @property
    def pitch_classes(self) -> tuple[int, ...]:
        steps = MAJOR_STEPS if self.mode == "major" else MINOR_STEPS
        return tuple((self.tonic + s) % 12 for s in steps)

    @property
    def name(self) -> str:
        return f"{NOTE_NAMES[self.tonic]} {self.mode}"

    @staticmethod
    def parse(text: str) -> "Key":
        """'A minor', 'C# major', 'Bb major', 'Am', 'F#'."""
        t = text.strip().replace("♯", "#").replace("♭", "b")
        mode = "major"
        low = t.lower()
        if low.endswith(("minor", " min")):
            mode = "minor"
            t = t[: t.lower().rfind("min")].strip()
        elif low.endswith(("major", " maj")):
            t = t[: t.lower().rfind("maj")].strip()
        elif len(t) >= 2 and t.endswith("m"):
            mode, t = "minor", t[:-1]
        flats = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B", "Fb": "E"}
        t = t[0].upper() + t[1:]
        t = flats.get(t, t)
        if t == "E#":
            t = "F"
        if t == "B#":
            t = "C"
        return Key(NOTE_NAMES.index(t), mode)

    @staticmethod
    def all() -> list["Key"]:
        return [Key(t, m) for m in ("major", "minor") for t in range(12)]


def nearest_in_scale(midi: float, key: Key | None) -> int:
    """Nearest semitone (chromatic if key is None) to a fractional MIDI value."""
    if key is None:
        return int(np.round(midi))
    base = int(np.floor(midi))
    best, best_d = base, 1e9
    for cand in range(base - 2, base + 3):
        if cand % 12 in key.pitch_classes:
            d = abs(cand - midi)
            if d < best_d - 1e-9:
                best, best_d = cand, d
    return best


def tuning_offset_cents(midi: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Global tuning of a set of pitches relative to A440 equal temperament, in cents
    (-50, 50]: the weighted circular mean of their distance from the nearest semitone. A choir
    or singer that is 30 cents flat throughout gives about -30."""
    m = np.asarray(midi, dtype=np.float64)
    w = np.ones_like(m) if weights is None else np.asarray(weights, dtype=np.float64)
    ok = ~np.isnan(m) & (w > 0)
    if not ok.any():
        return 0.0
    z = np.sum(w[ok] * np.exp(2j * np.pi * m[ok]))
    if abs(z) < 1e-12:
        return 0.0
    c = float(np.angle(z) / (2 * np.pi) * 100)
    return 50.0 if c <= -50 else c
