"""Contextual bandit policies (proposal 5.4).

Model of expected reward for context c and action a::

    r(c, a) = pop(c, a) + w . q(a) + noise

* ``pop`` is the population part: ridge regression on joint(c, q(a)), trained in simulation
  and on pilot data, then frozen on the device.
* ``w`` (15 weights) is the personal part, a quadratic over the four settings. It starts at
  zero, so a new user first gets the population's choice (no cold start), and has a Gaussian
  prior whose scale is set so a handful of interactions can move it.

Because the personal part is a smooth function of the settings, feedback on one setting
also updates its neighbours, and all 504 candidates are scored in a few milliseconds.

Policies: linear Thompson sampling (main), LinUCB, epsilon-greedy, fixed setting, and a
discounted Thompson sampling variant for drift (extended work).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from ..correction.settings import FIXED_CLASSIC, CorrectionSettings, action_grid, action_index
from .features import D_ACTION, D_CONTEXT, action_quad, joint


# ----------------------------------------------------------------------------- population


def expand_context(c: np.ndarray) -> np.ndarray:
    """Second-order context expansion: [c, c_i * c_j for i <= j, i, j >= 1]."""
    c = np.asarray(c, float)
    rest = c[..., 1:]
    iu = np.triu_indices(rest.shape[-1])
    pairs = (rest[..., :, None] * rest[..., None, :])[..., iu[0], iu[1]]
    return np.concatenate([c, pairs], axis=-1)


@dataclass
class PopulationModel:
    coef: np.ndarray  # (D_CONTEXT_EXPANDED * D_ACTION,) or (D_CONTEXT * D_ACTION,)
    expanded: bool = True

    def _ctx(self, context):
        return expand_context(context) if self.expanded else np.asarray(context, float)

    def predict(self, context: np.ndarray) -> np.ndarray:
        """Predicted reward of all 504 actions for this context."""
        return joint(self._ctx(context), action_quad()) @ self.coef

    @staticmethod
    def fit(contexts: np.ndarray, action_idx: np.ndarray, rewards: np.ndarray, ridge: float = 1e-2, expanded: bool = True) -> "PopulationModel":
        ctx = expand_context(contexts) if expanded else contexts
        X = joint(ctx, action_quad()[action_idx])
        A = X.T @ X + ridge * np.eye(X.shape[1])
        return PopulationModel(np.linalg.solve(A, X.T @ rewards), expanded)

    @staticmethod
    def neutral() -> "PopulationModel":
        """Untrained fallback: prefers moderate settings (peak near the default)."""
        from ..correction.settings import DEFAULT
        from .features import quad

        x0 = DEFAULT.normalized()
        # -(x - x0)^2 summed = sum(-x^2 + 2 x0 x - x0^2) in the quad basis
        w = np.zeros(D_ACTION)
        w[0] = 0.8 - np.sum(x0**2) * 0.3
        w[1:5] = 0.6 * x0
        w[5:9] = -0.3
        coef = np.zeros((D_CONTEXT, D_ACTION))
        coef[0] = w  # bias row only
        assert np.isclose(quad(x0) @ w, 0.8, atol=1e-6)
        return PopulationModel(coef.ravel(), expanded=False)

    def to_json(self) -> str:
        return json.dumps({"coef": self.coef.tolist(), "expanded": self.expanded})

    @staticmethod
    def from_json(s: str) -> "PopulationModel":
        d = json.loads(s)
        return PopulationModel(np.array(d["coef"]), d.get("expanded", False))


# ----------------------------------------------------------------------------- personal part


# Prior variance shape over the quadratic basis [1, x(4), x^2(4), x_i x_j(6)]. A singer
# mostly differs from the population in *where* their ideal settings are, which moves the
# linear terms of the reward bowl; its curvature (square and cross terms) is shared more.
# Putting most prior mass on the linear terms leaves ~5 effective personal parameters, so
# about 20 noisy judgments are enough to move them.
PRIOR_SHAPE = np.array([0.5] + [1.0] * 4 + [0.15] * 4 + [0.05] * 6)


@dataclass
class PersonalPosterior:
    # tuned in simulation (scripts/run_agent_sim.py --sweep); see docs/RESULTS.md
    prior_var: float = 0.1
    noise_var: float = 0.01
    precision: np.ndarray = field(default=None)
    b: np.ndarray = field(default=None)
    n: int = 0
    prior_mean: np.ndarray = field(default=None)
    curvature: float = 0.0  # > 0 enables the bowl prior (see anchor); off by default

    def __post_init__(self):
        if self.prior_mean is None:
            self.prior_mean = np.zeros(D_ACTION)
        if self.precision is None:
            self.precision = self.prior_precision()
        if self.b is None:
            self.b = self.prior_precision() @ self.prior_mean

    def anchor(self, center: np.ndarray) -> None:
        """Centre the personal prior on a bowl -c * ||x - center||^2.

        The population part alone is too flat to keep a linear personal correction from
        running to the edges of the grid. With a known-curvature bowl as the prior mean,
        feedback moves the *centre* of the bowl (the singer's ideal settings) instead.
        Only valid before any update."""
        c = self.curvature
        m0 = np.zeros(D_ACTION)
        m0[0] = -c * float(np.sum(center**2))
        m0[1:5] = 2 * c * center
        m0[5:9] = -c
        self.prior_mean = m0
        self.precision = self.prior_precision()
        self.b = self.precision @ m0

    def prior_precision(self) -> np.ndarray:
        return np.diag(1 / (self.prior_var * PRIOR_SHAPE))

    @property
    def cov(self) -> np.ndarray:
        return np.linalg.inv(self.precision)

    @property
    def mean(self) -> np.ndarray:
        return np.linalg.solve(self.precision, self.b)

    def update(self, q: np.ndarray, residual: float, discount: float = 1.0) -> None:
        if discount < 1.0:
            prior = self.prior_precision()
            self.precision = discount * self.precision + (1 - discount) * prior
            self.b = discount * self.b + (1 - discount) * prior @ self.prior_mean
        self.precision = self.precision + np.outer(q, q) / self.noise_var
        self.b = self.b + q * residual / self.noise_var
        self.n += 1

    def to_dict(self) -> dict:
        return dict(
            prior_var=self.prior_var, noise_var=self.noise_var, precision=self.precision.tolist(),
            b=self.b.tolist(), n=self.n, prior_mean=self.prior_mean.tolist(), curvature=self.curvature,
        )

    @staticmethod
    def from_dict(d: dict) -> "PersonalPosterior":
        return PersonalPosterior(
            d["prior_var"], d["noise_var"], np.array(d["precision"]), np.array(d["b"]), d["n"],
            np.array(d.get("prior_mean", np.zeros(D_ACTION))), d.get("curvature", 0.6),
        )


# ----------------------------------------------------------------------------- policies


class Policy:
    name = "policy"

    def __init__(self, population: PopulationModel | None = None, rng: np.random.Generator | None = None, **post_kw):
        self.pop = population or PopulationModel.neutral()
        self.post = PersonalPosterior(**post_kw)
        self.rng = rng or np.random.default_rng()
        self.anchored = False

    def _ensure_anchor(self, context: np.ndarray) -> None:
        if not self.anchored and self.post.n == 0 and self.post.curvature > 0:
            from ..correction.settings import action_matrix

            self.post.anchor(action_matrix()[int(np.argmax(self.pop.predict(context)))])
        self.anchored = True

    # scores for all actions
    def mean_scores(self, context: np.ndarray) -> np.ndarray:
        return self.pop.predict(context) + action_quad() @ self.post.mean

    def select(self, context: np.ndarray) -> int:
        raise NotImplementedError

    def __getattribute__(self, name):
        attr = object.__getattribute__(self, name)
        if name in ("select", "greedy", "update", "mean_scores"):
            def wrapped(context, *a, _f=attr, **k):
                object.__getattribute__(self, "_ensure_anchor")(context)
                return _f(context, *a, **k)
            return wrapped
        return attr

    def greedy(self, context: np.ndarray) -> int:
        return int(np.argmax(self.mean_scores(context)))

    def update(self, context: np.ndarray, action: int, r: float) -> None:
        residual = r - float(self.pop.predict(context)[action])
        self.post.update(action_quad()[action], residual)

    def choose(self, context: np.ndarray) -> CorrectionSettings:
        return action_grid()[self.select(context)]

    def learn(self, context: np.ndarray, settings: CorrectionSettings, r: float) -> None:
        self.update(context, action_index(settings), r)


class ThompsonSampling(Policy):
    """Linear Thompson sampling (Agrawal & Goyal 2013). ``explore`` scales the posterior
    spread used for sampling (1 = textbook; < 1 trades a little learning speed for less
    regret, a standard practical tweak with ~20 interactions per user)."""

    name = "thompson"

    def __init__(self, *a, explore: float = 0.5, **kw):
        super().__init__(*a, **kw)
        self.explore = explore

    def select(self, context):
        w = self.rng.multivariate_normal(self.post.mean, self.explore**2 * self.post.cov)
        return int(np.argmax(self.pop.predict(context) + action_quad() @ w))


class DiscountedThompson(ThompsonSampling):
    name = "thompson_discounted"

    def __init__(self, *a, discount: float = 0.95, **kw):
        super().__init__(*a, **kw)
        self.discount = discount

    def update(self, context, action, r):
        residual = r - float(self.pop.predict(context)[action])
        self.post.update(action_quad()[action], residual, self.discount)


class LinUCB(Policy):
    name = "linucb"

    def __init__(self, *a, alpha: float = 0.5, **kw):
        super().__init__(*a, **kw)
        self.alpha = alpha

    def select(self, context):
        Q = action_quad()
        bonus = np.sqrt(np.einsum("ij,jk,ik->i", Q, self.post.cov, Q))
        return int(np.argmax(self.mean_scores(context) + self.alpha * bonus))


class EpsilonGreedy(Policy):
    name = "epsilon_greedy"

    def __init__(self, *a, epsilon: float = 0.1, **kw):
        super().__init__(*a, **kw)
        self.epsilon = epsilon

    def select(self, context):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(action_grid())))
        return self.greedy(context)


class FixedPolicy(Policy):
    """C1: always the same setting (classic auto-tune by default). Never learns."""

    name = "fixed"

    def __init__(self, *a, settings: CorrectionSettings = FIXED_CLASSIC, **kw):
        super().__init__(*a, **kw)
        self.idx = action_index(settings)

    def select(self, context):
        return self.idx

    def greedy(self, context):
        return self.idx

    def update(self, context, action, r):
        pass


class PopulationOnly(Policy):
    """C2-like: population model's best guess, no personal learning."""

    name = "population"

    def select(self, context):
        return int(np.argmax(self.pop.predict(context)))

    greedy = select

    def update(self, context, action, r):
        pass


POLICIES = {
    p.name: p for p in (ThompsonSampling, DiscountedThompson, LinUCB, EpsilonGreedy, FixedPolicy, PopulationOnly)
}
