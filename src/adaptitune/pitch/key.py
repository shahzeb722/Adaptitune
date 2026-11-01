"""Key estimation from a monophonic F0 contour (proposal 5.3).

A duration-weighted pitch-class histogram of the voiced frames is correlated with the
Krumhansl-Kessler key profiles for all 24 keys. Each frame is soft-assigned to its two
neighbouring pitch classes so a singer who is a little flat still votes mostly for the right
class. The user can always override the result (FR2).
"""

from __future__ import annotations

import numpy as np

from ..music import Key, hz_to_midi

KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def pitch_class_histogram(f0: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    m = hz_to_midi(f0)
    ok = ~np.isnan(m)
    m = m[ok]
    w = np.ones_like(m) if weights is None else np.asarray(weights)[ok]
    hist = np.zeros(12)
    lo = np.floor(m).astype(int)
    frac = m - lo
    np.add.at(hist, lo % 12, w * (1 - frac))
    np.add.at(hist, (lo + 1) % 12, w * frac)
    return hist


def key_scores(hist: np.ndarray) -> dict[Key, float]:
    out = {}
    if hist.sum() <= 0:
        return {k: 0.0 for k in Key.all()}
    for key in Key.all():
        prof = KK_MAJOR if key.mode == "major" else KK_MINOR
        out[key] = float(np.corrcoef(hist, np.roll(prof, key.tonic))[0, 1])
    return out


def estimate_key(f0: np.ndarray, voicing: np.ndarray | None = None) -> tuple[Key, float]:
    """Return the best key and a confidence in [0, 1] (margin between the top two scores)."""
    scores = key_scores(pitch_class_histogram(f0, voicing))
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (best, s1), (_, s2) = ranked[0], ranked[1]
    conf = float(np.clip((s1 - s2) * 5, 0, 1))
    return best, conf


def same_key_or_relative(a: Key, b: Key) -> bool:
    """Relative major/minor share all pitch classes, so in-key snapping is identical."""
    return set(a.pitch_classes) == set(b.pitch_classes)
