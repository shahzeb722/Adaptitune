"""Reward (proposal 5.4): r = 0.6 f + 0.2 a + 0.2 n - p."""

from __future__ import annotations

from dataclasses import dataclass

W_FEEDBACK, W_ACCURACY, W_NATURAL = 0.6, 0.2, 0.2


@dataclass
class Feedback:
    kind: str  # "keep" | "rating" | "ab"
    value: float  # keep: 1/0, rating: 1..5, ab: 1 if this version was picked else 0

    def scaled(self) -> float:
        if self.kind == "rating":
            return (min(max(self.value, 1), 5) - 1) / 4
        return float(self.value > 0.5)


def reward(f: float, a: float, n: float, p: float = 0.0, weights=(W_FEEDBACK, W_ACCURACY, W_NATURAL)) -> float:
    wf, wa, wn = weights
    return wf * f + wa * a + wn * n - p
