"""Context and action features for the personalization agent (proposal 5.4).

Context (per take): mean and spread of note pitch error, vocal range, vibrato rate and
extent, share of voiced frames and background noise, each scaled to about [0, 1], plus a
bias term. Action: the four settings mapped to [0, 1]^4 and expanded to a quadratic basis
(15 terms), which is what the personal part of the model learns weights for.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .. import metrics as M
from ..correction.planner import segment_notes, _robust_center
from ..correction.settings import CorrectionSettings, action_matrix

CONTEXT_NAMES = ("bias", "err_mean", "err_spread", "range", "vib_rate", "vib_extent", "voiced", "noise")
D_CONTEXT = len(CONTEXT_NAMES)
D_ACTION = 15


def context_from_track(f0: np.ndarray, hop_s: float, noise_db: float) -> np.ndarray:
    midi = M.contour_midi(f0)
    notes = segment_notes(midi, hop_s)
    errs, centers, rates, exts = [], [], [], []
    for a, b in notes:
        c = _robust_center(midi[a:b])
        centers.append(c)
        errs.append(100 * (c - np.round(c)))
        r, e = M.vibrato(midi[a:b], hop_s)
        if not np.isnan(r) and e > 10:
            rates.append(r)
            exts.append(e)
    errs = np.abs(np.array(errs)) if errs else np.zeros(1)
    rng = (max(centers) - min(centers)) if len(centers) > 1 else 0.0
    voiced = float(np.mean(f0 > 0)) if len(f0) else 0.0
    return np.array(
        [
            1.0,
            np.clip(np.mean(errs) / 50, 0, 1),
            np.clip(np.std(errs) / 30, 0, 1),
            np.clip(rng / 24, 0, 1),
            np.clip((np.mean(rates) if rates else 0.0) / 8, 0, 1),
            np.clip((np.mean(exts) if exts else 0.0) / 80, 0, 1),
            voiced,
            np.clip((noise_db + 80) / 60, 0, 1),
        ]
    )


def context_from_analysis(an) -> np.ndarray:
    return context_from_track(an.track.f0, an.track.hop_s, an.noise_db)


def quad(x: np.ndarray) -> np.ndarray:
    """Quadratic basis over normalized settings. x: (..., 4) -> (..., 15)."""
    x = np.asarray(x, float)
    parts = [np.ones(x.shape[:-1] + (1,)), x, x**2]
    parts.append(np.stack([x[..., i] * x[..., j] for i, j in combinations(range(4), 2)], axis=-1))
    return np.concatenate(parts, axis=-1)


_Q = None


def action_quad() -> np.ndarray:
    """(504, 15) quadratic features of the action grid (cached)."""
    global _Q
    if _Q is None:
        _Q = quad(action_matrix())
    return _Q


def joint(context: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Population features: outer product context x quadratic action basis."""
    return np.einsum("...c,...q->...cq", np.broadcast_to(context, q.shape[:-1] + context.shape[-1:]), q).reshape(
        q.shape[:-1] + (context.shape[-1] * q.shape[-1],)
    )


def settings_quad(s: CorrectionSettings) -> np.ndarray:
    return quad(s.normalized())
