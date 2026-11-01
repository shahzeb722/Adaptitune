"""Correction planner (proposal 5.1 and 5.3).

Turns an F0 contour into notes, picks an intended target for each note, and builds a
per-frame correction curve (in semitones) from the four settings.

Model of a sung note::

    m(t) = center + slow(t) + vib(t)

``slow`` is everything below vibrato rate (drift, scoops, onset glides) and ``vib`` is the
oscillation around it. The ideal output is ``target + keep * vib(t)``, so the ideal
correction is::

    c*(t) = (target - center) - slow(t) - (1 - keep) * vib(t)

Strength scales it, and retune speed is a one-pole glide from the current correction to
``c*`` with time constant ``retune_ms`` (that is why slow retune also keeps more vibrato and
scoops, like the classic effect).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import uniform_filter1d

from ..music import Key, hz_to_midi, nearest_in_scale
from ..pitch.trackers import PitchTrack
from .settings import CorrectionSettings

VOICING_UNTOUCHED = 0.2  # below this we cannot estimate a reliable target (proposal 5.3)
MIN_NOTE_S = 0.06
SPLIT_SEMITONES = 0.6
SPLIT_HOLD_S = 0.04
VIBRATO_WINDOW_S = 0.2  # ~one vibrato cycle at 5 Hz, averages it out


@dataclass
class Note:
    start: int  # first frame (inclusive)
    end: int  # last frame (exclusive)
    center: float  # fractional MIDI
    target: int  # intended MIDI note
    hop_s: float = 0.005

    @property
    def error_cents(self) -> float:
        return 100 * (self.center - self.target)

    @property
    def t_start(self) -> float:
        return self.start * self.hop_s

    @property
    def t_end(self) -> float:
        return self.end * self.hop_s


@dataclass
class Plan:
    hop_s: float
    midi: np.ndarray  # input contour (nan = unvoiced)
    notes: list[Note]
    correction: np.ndarray  # semitones per frame, 0 where untouched
    slow: np.ndarray = field(repr=False, default=None)
    vib: np.ndarray = field(repr=False, default=None)
    key: Key | None = None
    settings: CorrectionSettings | None = None

    @property
    def corrected_midi(self) -> np.ndarray:
        return self.midi + self.correction

    def ratio(self) -> np.ndarray:
        """Per-frame frequency ratio to apply (1 where untouched)."""
        return 2 ** (self.correction / 12)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) < 3 or n < 2:
        return x.copy()
    n = min(n, len(x) if len(x) % 2 else len(x) - 1)
    return uniform_filter1d(x, size=max(n, 1), mode="nearest")


def _robust_center(m: np.ndarray) -> float:
    k = len(m)
    skip = int(0.15 * k)  # ignore the onset where scoops live
    core = m[skip:] if k - skip >= 3 else m
    return float(np.median(core))


def _vibrato_window(seg: np.ndarray, hop: float, default: int) -> int:
    """Moving-average length of exactly one vibrato period removes the oscillation best."""
    if len(seg) * hop < 0.35:
        return default
    from ..metrics import vibrato

    rate, extent = vibrato(seg, hop)
    if np.isnan(rate) or extent < 10:
        return default
    return max(3, int(round(1 / (rate * hop))))


def segment_notes(midi: np.ndarray, hop_s: float) -> list[tuple[int, int]]:
    """Split voiced runs into notes where the smoothed pitch moves away and stays away."""
    voiced = ~np.isnan(midi)
    win = max(1, int(VIBRATO_WINDOW_S / hop_s))
    hold = max(1, int(SPLIT_HOLD_S / hop_s))
    min_len = max(2, int(MIN_NOTE_S / hop_s))
    out: list[tuple[int, int]] = []
    for a, b in _runs(voiced):
        if b - a < min_len:
            continue
        s = _smooth(midi[a:b], win)
        segs, start, away = [], 0, 0
        for i in range(len(s)):
            ref = np.median(s[start : max(i, start + 1)])
            if abs(s[i] - ref) > SPLIT_SEMITONES and i - start > min_len:
                away += 1
                if away >= hold:
                    cut = i - hold + 1
                    segs.append((start, cut))
                    start, away = cut, 0
            else:
                away = 0
        segs.append((start, len(s)))
        # merge tiny pieces into the closer neighbour
        merged: list[list[int]] = []
        for sa, sb in segs:
            if merged and sb - sa < min_len:
                merged[-1][1] = sb
            else:
                merged.append([sa, sb])
        if len(merged) > 1 and merged[0][1] - merged[0][0] < min_len:
            merged[1][0] = merged[0][0]
            merged.pop(0)
        out.extend((a + sa, a + sb) for sa, sb in merged)
    return out


def plan(
    track: PitchTrack,
    settings: CorrectionSettings,
    key: Key | None = None,
) -> Plan:
    hop = track.hop_s
    midi = hz_to_midi(track.f0)
    untouchable = (track.voicing < VOICING_UNTOUCHED) & ~np.isnan(midi)
    midi_for_notes = np.where(untouchable, np.nan, midi)

    snap_key = key if settings.snap == "in_key" else None
    if settings.snap == "in_key" and key is None:
        from ..pitch.key import estimate_key

        snap_key, _ = estimate_key(track.f0, track.voicing)

    notes: list[Note] = []
    n = len(midi)
    slow = np.zeros(n)
    vib = np.zeros(n)
    ideal = np.zeros(n)
    win = max(1, int(VIBRATO_WINDOW_S / hop))
    for a, b in segment_notes(midi_for_notes, hop):
        seg = midi[a:b]
        center = _robust_center(seg)
        target = nearest_in_scale(center, snap_key)
        smooth = _smooth(seg, _vibrato_window(seg, hop, win))
        slow[a:b] = smooth - center
        vib[a:b] = seg - smooth
        ideal[a:b] = (target - center) - slow[a:b] - (1 - settings.vibrato) * vib[a:b]
        notes.append(Note(a, b, center, target, hop))

    in_note = np.zeros(n, bool)
    for nt in notes:
        in_note[nt.start : nt.end] = True

    # retune glide: one-pole smoothing; reset in unvoiced gaps
    alpha = 1 - np.exp(-hop / (settings.retune_ms / 1000))
    c = np.zeros(n)
    prev = 0.0
    for i in range(n):
        if not in_note[i]:
            prev = 0.0
            continue
        prev = prev + alpha * (settings.strength * ideal[i] - prev)
        c[i] = prev
    c[np.isnan(midi)] = 0.0
    return Plan(hop, midi, notes, c, slow, vib, snap_key, settings)
