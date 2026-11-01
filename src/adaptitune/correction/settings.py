"""The four correction settings and the 504-action grid (proposal 5.3 and 5.4)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np

STRENGTH_LEVELS = tuple(np.round(np.linspace(0, 1, 9), 4))  # 9 levels
RETUNE_MS_LEVELS = tuple(np.round(np.geomspace(5, 400, 7), 1))  # 7 log-spaced levels
SNAP_MODES = ("chromatic", "in_key")  # 2 modes
VIBRATO_LEVELS = (0.0, 1 / 3, 2 / 3, 1.0)  # 4 levels of vibrato preservation


@dataclass(frozen=True)
class CorrectionSettings:
    strength: float = 1.0  # 0 = untouched, 1 = remove all of the error
    retune_ms: float = 20.0  # time constant of the glide to the target
    snap: str = "in_key"  # "chromatic" | "in_key"
    vibrato: float = 1 / 3  # share of natural vibrato kept

    def __post_init__(self):
        if self.snap not in SNAP_MODES:
            raise ValueError(f"snap must be one of {SNAP_MODES}")
        if not 0 <= self.strength <= 1 or not 0 <= self.vibrato <= 1 or self.retune_ms <= 0:
            raise ValueError("strength/vibrato in [0,1], retune_ms > 0")

    def as_dict(self) -> dict:
        return asdict(self)

    def normalized(self) -> np.ndarray:
        """Map to [0,1]^4 for the personalization agent."""
        lo, hi = np.log(RETUNE_MS_LEVELS[0]), np.log(RETUNE_MS_LEVELS[-1])
        r = (np.log(self.retune_ms) - lo) / (hi - lo)
        return np.array([self.strength, np.clip(r, 0, 1), float(self.snap == "in_key"), self.vibrato])

    @staticmethod
    def from_normalized(v) -> "CorrectionSettings":
        lo, hi = np.log(RETUNE_MS_LEVELS[0]), np.log(RETUNE_MS_LEVELS[-1])
        return CorrectionSettings(
            strength=float(np.clip(v[0], 0, 1)),
            retune_ms=float(np.exp(lo + np.clip(v[1], 0, 1) * (hi - lo))),
            snap="in_key" if v[2] >= 0.5 else "chromatic",
            vibrato=float(np.clip(v[3], 0, 1)),
        )

    def snapped(self) -> "CorrectionSettings":
        """Nearest point on the action grid."""
        return CorrectionSettings(
            strength=float(min(STRENGTH_LEVELS, key=lambda s: abs(s - self.strength))),
            retune_ms=float(min(RETUNE_MS_LEVELS, key=lambda r: abs(np.log(r / self.retune_ms)))),
            snap=self.snap,
            vibrato=float(min(VIBRATO_LEVELS, key=lambda v: abs(v - self.vibrato))),
        )


# C1 "classic auto-tune": hard, fast, in-key, flattens vibrato.
FIXED_CLASSIC = CorrectionSettings(strength=1.0, retune_ms=RETUNE_MS_LEVELS[0], snap="in_key", vibrato=0.0)
# C2 our default: firm but natural.
DEFAULT = CorrectionSettings(strength=0.75, retune_ms=RETUNE_MS_LEVELS[3], snap="in_key", vibrato=2 / 3)
# O3 reference setting: full strength, in-key snapping (vibrato preservation on).
REFERENCE = CorrectionSettings(strength=1.0, retune_ms=RETUNE_MS_LEVELS[1], snap="in_key", vibrato=1.0)


@lru_cache(maxsize=1)
def action_grid() -> tuple[CorrectionSettings, ...]:
    return tuple(
        CorrectionSettings(float(s), float(r), m, float(v))
        for s in STRENGTH_LEVELS
        for r in RETUNE_MS_LEVELS
        for m in SNAP_MODES
        for v in VIBRATO_LEVELS
    )


def action_matrix() -> np.ndarray:
    """(504, 4) normalized settings."""
    return np.stack([a.normalized() for a in action_grid()])


def action_index(s: CorrectionSettings) -> int:
    s = s.snapped()
    for i, a in enumerate(action_grid()):
        if (
            abs(a.strength - s.strength) < 1e-6
            and abs(a.retune_ms - s.retune_ms) < 1e-6
            and a.snap == s.snap
            and abs(a.vibrato - s.vibrato) < 1e-6
        ):
            return i
    raise KeyError(s)
