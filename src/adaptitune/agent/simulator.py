"""Simulated singers for developing and sanity-checking the agent (proposal 5.4, O4).

The simulator follows our own assumptions, so it only checks that the learning code works;
the real test is the user study. To avoid tuning the agent to one convenient model there are
three preference models:

* ``logistic``    - keep/choose probability is a logistic function of a weighted squared
                    distance from the singer's ideal settings,
* ``ideal_point`` - satisfaction falls off non-linearly (Gaussian) with distance, then is
                    thresholded with noise,
* ``threshold``   - any audible artifact above the singer's tolerance gets the take rejected,
                    otherwise behaves like ``logistic``.

Every singer also has two realistic flaws: the ideal point drifts slowly between sessions,
and choices get noisier with each judgment in a session (listening fatigue).

The objective reward terms (in-key accuracy ``a``, naturalness ``n``, vibrato penalty ``p``)
come from a cheap analytic surrogate of the real pipeline so thousands of singers can be
simulated in seconds. ``scripts/check_surrogate.py`` compares the surrogate with the real
pipeline on synthetic audio.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..correction.settings import action_matrix
from .features import D_CONTEXT
from .reward import reward

MODELS = ("logistic", "ideal_point", "threshold")


def sample_context(rng: np.random.Generator) -> np.ndarray:
    beginner = rng.random() < 0.5
    err = rng.beta(4, 3) if beginner else rng.beta(2, 5)
    vib_ext = rng.beta(1.5, 4) if beginner else rng.beta(3, 2.5)
    return np.array(
        [
            1.0,
            err,
            np.clip(err * rng.uniform(0.5, 1.2), 0, 1),
            rng.uniform(0.3, 0.8),
            rng.uniform(0.55, 0.8) if vib_ext > 0.15 else 0.0,
            vib_ext,
            rng.uniform(0.5, 0.85),
            rng.uniform(0.1, 0.5),
        ]
    )


def population_ideal(c: np.ndarray) -> np.ndarray:
    """Where the typical singer with context c wants the four settings (normalized)."""
    err, vib_ext = c[1], c[5]
    return np.array(
        [
            np.clip(0.35 + 0.6 * err, 0, 1),  # sloppier singers want firmer correction
            np.clip(0.25 + 0.5 * vib_ext, 0, 1),  # vibrato singers want slower retune
            1.0,  # most people want in-key snapping
            np.clip(0.15 + 0.85 * vib_ext, 0, 1),  # ... and want their vibrato kept
        ]
    )


@dataclass
class SimSinger:
    context: np.ndarray
    ideal: np.ndarray
    weights: np.ndarray  # importance of each setting to this singer
    model: str
    temp: float = 0.08  # choice noise at the start of a session
    fatigue: float = 0.12  # temp grows by this share per judgment in a session
    drift_sd: float = 0.04  # ideal-point random walk per session
    artifact_tol: float = 0.35  # threshold model: max acceptable (1 - n)
    width: float = 0.35  # ideal-point model: tolerance radius
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    judgments_in_session: int = 0

    @staticmethod
    def sample(rng: np.random.Generator, model: str | None = None) -> "SimSinger":
        c = sample_context(rng)
        ideal = population_ideal(c) + rng.normal(0, [0.2, 0.2, 0.0, 0.2])
        ideal[2] = 1.0 if rng.random() < 0.8 else 0.0
        ideal = np.clip(ideal, 0, 1)
        w = rng.dirichlet([2, 2, 1, 2]) * 4
        return SimSinger(
            c, ideal, w, model or str(rng.choice(MODELS)),
            temp=float(rng.uniform(0.05, 0.12)), artifact_tol=float(rng.uniform(0.25, 0.5)),
            width=float(rng.uniform(0.25, 0.45)), rng=rng,
        )

    # ------------------------------------------------------------------ objective surrogate
    def objective(self, x: np.ndarray) -> tuple[float, float, float]:
        x = np.atleast_2d(x)
        s, r, snap, v = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        err, vib_ext = self.context[1], self.context[5]
        # in-key accuracy: residual error shrinks with strength; chromatic snapping sometimes
        # picks an out-of-key neighbour on sloppy notes; very slow retune leaves onsets off
        resid = err * (1 - s) + 0.15 * err * r
        a = np.clip(1 - 1.6 * resid - 0.15 * err * (1 - snap), 0, 1)
        # naturalness: hard + fast correction sounds robotic; flattening real vibrato hurts
        n = np.clip(1 - 0.55 * s * (1 - r) ** 2 - 0.45 * vib_ext * (1 - v) * s - 0.1 * s * err, 0, 1)
        p = np.where((vib_ext > 0.25) & (v < 0.5) & (s > 0.5), 0.1, 0.0)
        return a, n, p

    # ------------------------------------------------------------------ preferences
    def utility(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        d2 = np.sum(self.weights * (x - self.ideal) ** 2, axis=1)
        return -d2

    def p_accept(self, x: np.ndarray, temp: float | None = None) -> np.ndarray:
        """Probability the singer keeps a take made with settings x."""
        t = self.temp if temp is None else temp
        x = np.atleast_2d(x)
        u = self.utility(x)
        if self.model == "logistic":
            return 1 / (1 + np.exp(-(u + 0.25) / t))
        if self.model == "ideal_point":
            d = np.sqrt(-u)
            sat = np.exp(-(d**2) / (2 * self.width**2))
            return 1 / (1 + np.exp(-(sat - 0.5) / t))
        # threshold
        _, n, _ = self.objective(x)
        ok = 1 / (1 + np.exp(-((self.artifact_tol - (1 - n)) / 0.03)))
        return ok / (1 + np.exp(-(u + 0.25) / t))

    def current_temp(self) -> float:
        return self.temp * (1 + self.fatigue * self.judgments_in_session)

    def expected_reward(self, x: np.ndarray) -> np.ndarray:
        """E[r] for settings x (rows), with the keep/discard probability as f."""
        a, n, p = self.objective(x)
        return reward(self.p_accept(x), a, n, p)

    # ------------------------------------------------------------------ interaction
    def judge(self, x: np.ndarray, kind: str, other: np.ndarray | None = None) -> float:
        """Return f in [0, 1] for settings x. ``kind`` in keep | rating | ab."""
        t = self.current_temp()
        self.judgments_in_session += 1
        if kind == "keep":
            return float(self.rng.random() < self.p_accept(x, t)[0])
        if kind == "rating":
            pa = self.p_accept(x, t)[0]
            stars = np.clip(np.round(1 + 4 * pa + self.rng.normal(0, 0.6 * t / self.temp)), 1, 5)
            return float((stars - 1) / 4)
        # A/B: Bradley-Terry on the acceptance logits
        pa, pb = self.p_accept(x, t)[0], self.p_accept(other, t)[0]
        la, lb = np.log(pa / (1 - pa + 1e-9) + 1e-9), np.log(pb / (1 - pb + 1e-9) + 1e-9)
        return float(self.rng.random() < 1 / (1 + np.exp(-(la - lb))))

    def new_session(self) -> None:
        self.judgments_in_session = 0
        self.ideal = np.clip(self.ideal + self.rng.normal(0, self.drift_sd, 4) * np.array([1, 1, 0, 1]), 0, 1)


def oracle_best(singer: SimSinger) -> tuple[int, float]:
    """Best fixed action for this singer right now (index, expected reward)."""
    er = singer.expected_reward(action_matrix())
    i = int(np.argmax(er))
    return i, float(er[i])


def assert_shapes():  # pragma: no cover - quick self check used in tests
    rng = np.random.default_rng(0)
    s = SimSinger.sample(rng)
    assert s.context.shape == (D_CONTEXT,)
    assert s.expected_reward(action_matrix()).shape == (504,)
