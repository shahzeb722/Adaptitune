"""Key estimation from a monophonic F0 contour (proposal 5.3, part of O2).

The default method was chosen on the training volumes of the Essen folksong melodies and checked
on separate test volumes (``adaptitune.bench.keys``, ``docs/O2_PROTOCOL.md``):

1. **Tuning.** The weighted circular mean of the notes' distance from the nearest semitone is
   removed first, so a singer who is consistently flat or sharp still votes for the right
   pitch classes.
2. **Notes, not frames.** The contour is split into notes with the planner's segmentation.
   Each note's centre (median after the onset, so scoops and vibrato do not move it) gives
   its pitch class. A centre that sits between two semitones is shared between them.
3. **Weighting.** Each note votes with ``duration ** alpha``: long notes count more, but one
   held note cannot outvote a phrase.
4. **Scoring.** The 12-bin histogram is correlated with each of the 24 rotated key profiles
   (Krumhansl-Schmuckler scoring; Bellman-Budge profiles by default).
5. **Tonic cue.** Keys whose tonic is the last note get a small bonus (0.1 on the correlation
   scale), because phrases tend to end on the tonic.

``estimate_key`` returns the key and a *scale* confidence: the margin over the best key with a
different set of pitch classes. Relative major and minor keys share every pitch class, so
in-key snapping cannot tell them apart and they should not lower the confidence. The user can
always override the key (FR2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..music import Key, hz_to_midi, tuning_offset_cents

# Key profiles, index 0 = tonic. Values as in music21.analysis.discrete (checked by a test).
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])  # Krumhansl & Kessler 1982
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
TKP_MAJOR = np.array([0.748, 0.060, 0.488, 0.082, 0.670, 0.460, 0.096, 0.715, 0.104, 0.366, 0.057, 0.400])  # Temperley 2007 (Kostka-Payne)
TKP_MINOR = np.array([0.712, 0.084, 0.474, 0.618, 0.049, 0.460, 0.105, 0.747, 0.404, 0.067, 0.133, 0.330])
BB_MAJOR = np.array([16.80, 0.86, 12.95, 1.41, 13.49, 11.93, 1.25, 20.28, 1.80, 8.04, 0.62, 10.57])  # Bellman & Budge 2005
BB_MINOR = np.array([18.16, 0.69, 12.99, 13.34, 1.07, 11.15, 1.38, 21.07, 7.49, 1.53, 0.92, 10.21])
# Aarden 2003, derived from the Essen collection itself: reported, never used as the default.
AE_MAJOR = np.array([17.7661, 0.145624, 14.9265, 0.160186, 19.8049, 11.3587, 0.291248, 22.062, 0.145624, 8.15494, 0.232998, 4.95122])
AE_MINOR = np.array([18.2648, 0.737619, 14.0499, 16.8599, 0.702494, 14.4362, 0.702494, 18.6161, 4.56621, 1.93186, 7.37619, 1.75623])
SIMPLE_MAJOR = np.array([2, 0, 1, 0, 1, 1, 0, 2, 0, 1, 0, 1], float)  # Sapp's simple weights
SIMPLE_MINOR = np.array([2, 0, 1, 1, 0, 1, 0, 2, 1, 0, 0.5, 0.5])

PROFILES: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "krumhansl": (KK_MAJOR, KK_MINOR),
    "temperley": (TKP_MAJOR, TKP_MINOR),
    "bellman": (BB_MAJOR, BB_MINOR),
    "aarden": (AE_MAJOR, AE_MINOR),
    "simple": (SIMPLE_MAJOR, SIMPLE_MINOR),
}
# profiles not derived from the Essen collection, so they can be judged fairly on it
FAIR_PROFILES = ("krumhansl", "temperley", "bellman", "simple")
KEYS = Key.all()


@dataclass(frozen=True)
class KeyConfig:
    """Defaults = the configuration chosen on the Essen training volumes (``adaptitune
    key-eval``; mode-balanced training accuracy 82.5 % against 48.8 % for the Krumhansl-Kessler
    frame histogram used before). See docs/O2_PROTOCOL.md."""

    profile: str = "bellman"
    scoring: str = "corr"  # "corr" (Krumhansl-Schmuckler) or "loglik"
    unit: str = "notes"  # "notes" or "frames"
    alpha: float = 0.5  # note weight = duration ** alpha
    tuning: bool = True  # remove the global tuning offset first
    final_bonus: float = 0.1  # added to the score of keys whose tonic is the last note
    soft_width: float = 0.15  # semitones; how sharply a centre is assigned to its nearest class


DEFAULT_CONFIG = KeyConfig()


@dataclass
class KeyResult:
    key: Key
    confidence: float  # scale confidence in [0, 1], see module docstring
    key_confidence: float  # margin over the runner-up key, including the relative key
    scores: dict[Key, float] = field(repr=False, default_factory=dict)
    tuning_cents: float = 0.0
    n_notes: int = 0

    def ranked(self) -> list[tuple[Key, float]]:
        return sorted(self.scores.items(), key=lambda kv: kv[1], reverse=True)


# ----------------------------------------------------------------------------- histograms


def pitch_class_histogram(f0: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Frame-level histogram: every voiced frame split linearly between its two neighbouring
    pitch classes (the original method, kept as the ``unit="frames"`` option)."""
    m = hz_to_midi(f0)
    ok = ~np.isnan(m)
    m = m[ok]
    w = np.ones_like(m) if weights is None else np.asarray(weights, float)[ok]
    hist = np.zeros(12)
    lo = np.floor(m).astype(int)
    frac = m - lo
    np.add.at(hist, lo % 12, w * (1 - frac))
    np.add.at(hist, (lo + 1) % 12, w * frac)
    return hist


def note_events(f0: np.ndarray, hop_s: float = 0.005) -> tuple[np.ndarray, np.ndarray]:
    """(centre MIDI, duration s) of each note, using the planner's note segmentation."""
    from ..correction.planner import _robust_center, segment_notes

    midi = hz_to_midi(np.where(np.asarray(f0) > 0, f0, np.nan))
    notes = segment_notes(midi, hop_s)
    if not notes:
        return np.zeros(0), np.zeros(0)
    centres = np.array([_robust_center(midi[a:b]) for a, b in notes])
    durs = np.array([(b - a) * hop_s for a, b in notes])
    ok = ~np.isnan(centres)
    return centres[ok], durs[ok]


def soft_histogram(centres: np.ndarray, weights: np.ndarray, soft_width: float = 0.15) -> np.ndarray:
    """Each centre votes for its nearest pitch class, sharing its vote with the next class in
    proportion to how close it sits to the boundary between them (50/50 exactly halfway)."""
    hist = np.zeros(12)
    if len(centres) == 0:
        return hist
    base = np.round(centres)
    d = centres - base  # in [-0.5, 0.5]
    p = 1 / (1 + np.exp(-(0.5 - np.abs(d)) / max(soft_width, 1e-3)))
    np.add.at(hist, base.astype(int) % 12, weights * p)
    np.add.at(hist, (base + np.sign(d)).astype(int) % 12, weights * (1 - p))
    return hist


# ----------------------------------------------------------------------------- scoring


def _profile_matrix(profile: str, scoring: str) -> np.ndarray:
    """24 x 12 matrix: one rotated profile per key, in ``Key.all()`` order."""
    maj, mnr = PROFILES[profile]
    rows = []
    for k in KEYS:
        prof = (maj if k.mode == "major" else mnr).astype(float)
        if scoring == "loglik":
            prof = prof + 0.02 * prof.max()  # floor so unseen classes are unlikely, not impossible
            prof = np.log(prof / prof.sum())
        rows.append(np.roll(prof, k.tonic))
    return np.array(rows)


def score_histogram(hist: np.ndarray, cfg: KeyConfig = DEFAULT_CONFIG, final_pc: int | None = None) -> np.ndarray:
    """Scores for the 24 keys (``Key.all()`` order); higher is better."""
    if hist.sum() <= 0:
        return np.zeros(24)
    mat = _profile_matrix(cfg.profile, cfg.scoring)
    if cfg.scoring == "corr":
        h = hist - hist.mean()
        p = mat - mat.mean(axis=1, keepdims=True)
        denom = np.sqrt((h**2).sum() * (p**2).sum(axis=1))
        s = (p @ h) / np.where(denom > 0, denom, 1)
    elif cfg.scoring == "loglik":
        s = mat @ (hist / hist.sum())
    else:
        raise ValueError(cfg.scoring)
    if final_pc is not None and cfg.final_bonus:
        s = s + cfg.final_bonus * np.array([k.tonic == final_pc for k in KEYS], float)
    return s


def histogram_and_cues(f0: np.ndarray, voicing: np.ndarray | None = None, hop_s: float = 0.005, cfg: KeyConfig = DEFAULT_CONFIG):
    """(histogram, last note's pitch class or None, tuning in cents, number of notes)."""
    f0 = np.asarray(f0, float)
    centres, durs = note_events(f0, hop_s)
    tuning = tuning_offset_cents(centres, durs) if (cfg.tuning and len(centres) >= 3) else 0.0
    if cfg.unit == "frames":
        m = hz_to_midi(np.where(f0 > 0, f0, np.nan)) - tuning / 100
        f = np.where(np.isnan(m), 0.0, 440 * 2 ** ((m - 69) / 12))
        hist = pitch_class_histogram(f, voicing)
    else:
        c = centres - tuning / 100
        hist = soft_histogram(c, durs**cfg.alpha, cfg.soft_width)
    final_pc = int(np.round(centres[-1] - tuning / 100)) % 12 if len(centres) else None
    return hist, final_pc, tuning, len(centres)


def analyze_key(f0: np.ndarray, voicing: np.ndarray | None = None, hop_s: float = 0.005, cfg: KeyConfig = DEFAULT_CONFIG) -> KeyResult:
    hist, final_pc, tuning, n = histogram_and_cues(f0, voicing, hop_s, cfg)
    s = score_histogram(hist, cfg, final_pc)
    scores = dict(zip(KEYS, map(float, s)))
    order = np.argsort(-s)
    best = KEYS[order[0]]
    spread = float(s[order[0]] - np.median(s)) or 1.0
    key_conf = float(np.clip((s[order[0]] - s[order[1]]) / spread, 0, 1)) if hist.sum() > 0 else 0.0
    other = next((i for i in order[1:] if not same_key_or_relative(KEYS[i], best)), order[1])
    scale_conf = float(np.clip((s[order[0]] - s[other]) / spread, 0, 1)) if hist.sum() > 0 else 0.0
    return KeyResult(best, scale_conf, key_conf, scores, float(tuning), int(n))


def estimate_key(f0: np.ndarray, voicing: np.ndarray | None = None, hop_s: float = 0.005, cfg: KeyConfig = DEFAULT_CONFIG) -> tuple[Key, float]:
    """The best key and its scale confidence in [0, 1]."""
    r = analyze_key(f0, voicing, hop_s, cfg)
    return r.key, r.confidence


def key_scores(hist: np.ndarray, profile: str = "krumhansl") -> dict[Key, float]:
    """Krumhansl-Schmuckler correlation of a histogram with every key (kept for callers that
    build their own histogram)."""
    s = score_histogram(hist, KeyConfig(profile=profile, scoring="corr"))
    return dict(zip(KEYS, map(float, s)))


def fit_key(ranked: list[tuple[Key, float]], required_pcs: set[int] | list[int]) -> Key:
    """The best-ranked key whose scale contains every pitch class in ``required_pcs`` (the key a
    singer would type in for a take whose notes we know); if none does, the one that contains
    the most of them, ties broken by rank."""
    req = {int(p) % 12 for p in required_pcs}
    if not ranked:
        return Key(0, "major")
    best = max(ranked, key=lambda ks: (len(req & set(ks[0].pitch_classes)), ks[1]))
    return best[0]


# ----------------------------------------------------------------------------- comparisons


def same_key_or_relative(a: Key, b: Key) -> bool:
    """Relative major/minor share all pitch classes, so in-key snapping is identical."""
    return set(a.pitch_classes) == set(b.pitch_classes)


def mirex_key_score(est: Key, ref: Key) -> float:
    """MIREX audio key detection score: 1 exact, 0.5 perfect fifth (same mode, tonic a fifth
    up or down), 0.3 relative, 0.2 parallel, 0 otherwise."""
    if est == ref:
        return 1.0
    if est.mode == ref.mode and (est.tonic - ref.tonic) % 12 in (5, 7):
        return 0.5
    if est.mode != ref.mode and same_key_or_relative(est, ref):
        return 0.3
    if est.mode != ref.mode and est.tonic == ref.tonic:
        return 0.2
    return 0.0
