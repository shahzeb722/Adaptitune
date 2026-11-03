"""Contextual bandit policies (proposal 5.4).

The reward is the proposal's r = 0.6 f + 0.2 a + 0.2 n - p. Here f is the singer's feedback on a
take, and a, n and p are measured on the corrected take by the pipeline. Only f depends on who
is singing, so the agent models the two parts separately:

    E[r | c, x] = 0.6 P(f | c, x) + O(c, x)
    logit P(f | c, x) = h(c, x) + w . q(x)

c is the take's context (features.py), x is one of the 504 settings and q(x) is its 15-term
quadratic basis.

* h (the population's logit) and O (the objective part) are fitted in simulation, and on pilot
  data when it exists, then frozen on the device (``PopulationModel``).
* w is the singer's own deviation from the population. It starts at 0, so a new user first gets
  the population's choice (no cold start). Its prior N(0, S) comes from the population too:
  S is how far individual singers' preferences sit from h.
* Feedback enters through a logistic likelihood. A keep/discard decision is a Bernoulli outcome.
  A 1-5 rating, scaled to [0, 1], is a fractional outcome. It counts as ``rating_weight``
  keep/discard decisions, because a rating carries more information. An A/B pick is a
  Bradley-Terry comparison of the two versions' logits. The posterior over w is the Laplace
  approximation, refitted with a few Newton steps after each judgment (Chapelle & Li 2011).

A linear model of r does not work here. A singer's acceptance rises steeply near their
preferred settings and is close to zero elsewhere, and a quadratic in r cannot follow that.
Fitted to the noise-free rewards of all 504 settings, its best setting reaches 84% of the
oracle's reward on simulated singers. The logistic model reaches 99% (docs/RESULTS.md, O4). The
linear agent is kept as the ablation ``thompson_linear``.

Policies: Thompson sampling (main), UCB, epsilon-greedy, discounted Thompson sampling for drift
(extended work), the population's choice without personal learning, a fixed setting, and the
linear ablation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

from ..correction.settings import FIXED_CLASSIC, CorrectionSettings, action_grid, action_index
from .features import D_ACTION, D_CONTEXT, action_quad
from .reward import W_FEEDBACK, reward

# Acceptance probabilities are clipped to [LAPSE, 1 - LAPSE] when the population is fitted: like
# a lapse rate in psychophysics, no setting is ever certainly kept or certainly discarded.
LAPSE = 0.02
D_CONTEXT_EXPANDED = D_CONTEXT + (D_CONTEXT - 1) * D_CONTEXT // 2


def expand_context(c: np.ndarray) -> np.ndarray:
    """Second-order context expansion: [c, c_i * c_j for i <= j, i, j >= 1]."""
    c = np.asarray(c, float)
    rest = c[..., 1:]
    iu = np.triu_indices(rest.shape[-1])
    pairs = (rest[..., :, None] * rest[..., None, :])[..., iu[0], iu[1]]
    return np.concatenate([c, pairs], axis=-1)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, float)))


def fit_logistic(Z, y, offset=None, weights=None, prior_precision=None, w0=None, iters: int = 100):
    """MAP fit of y ~ sigmoid(offset + Z w) with a N(0, prior_precision^-1) prior.

    y may be fractional (0..1). Returns (mode, precision at the mode), the Laplace posterior.
    The objective is strictly convex, so damped Newton steps converge from any start."""
    Z = np.atleast_2d(np.asarray(Z, float))
    y = np.asarray(y, float)
    d = Z.shape[1]
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, float)
    wt = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    P = 1e-6 * np.eye(d) if prior_precision is None else np.asarray(prior_precision, float)
    w = np.zeros(d) if w0 is None else np.array(w0, float)

    def loss(v):
        eta = off + Z @ v
        # -[y log s + (1 - y) log(1 - s)] = log(1 + e^eta) - y eta
        return float(np.sum(wt * (np.logaddexp(0.0, eta) - y * eta)) + 0.5 * v @ P @ v)

    cur = loss(w)
    H = P
    for _ in range(iters):
        mu = sigmoid(off + Z @ w)
        g = Z.T @ (wt * (mu - y)) + P @ w
        H = (Z * (wt * mu * (1 - mu))[:, None]).T @ Z + P
        step = np.linalg.solve(H, g)
        t = 1.0
        while t > 1e-4 and loss(w - t * step) > cur + 1e-12:
            t *= 0.5
        if t <= 1e-4:  # no step lowers the loss: we are at the minimum (to rounding)
            break
        w = w - t * step
        cur = loss(w)
        if np.max(np.abs(t * step)) < 1e-7:
            break
    mu = sigmoid(off + Z @ w)
    H = (Z * (wt * mu * (1 - mu))[:, None]).T @ Z + P
    return w, H


# ----------------------------------------------------------------------------- population


@dataclass
class PopulationModel:
    """The frozen population part.

    ``logit_coef`` and ``objective_coef`` (36 x 15) map the expanded context to the quadratic
    weights of h and O. ``prior_cov`` (15 x 15) is the spread of individual singers' logit
    weights around h. ``reward_coef`` is a linear model of r itself, used only by the
    ``thompson_linear`` ablation."""

    logit_coef: np.ndarray
    objective_coef: np.ndarray
    prior_cov: np.ndarray
    reward_coef: np.ndarray | None = None

    def logit_weights(self, context: np.ndarray) -> np.ndarray:
        return expand_context(context) @ self.logit_coef

    def logit(self, context: np.ndarray) -> np.ndarray:
        """Population logit of acceptance for all 504 actions."""
        return action_quad() @ self.logit_weights(context)

    def objective(self, context: np.ndarray) -> np.ndarray:
        """Predicted 0.2 a + 0.2 n - p for all 504 actions."""
        return action_quad() @ (expand_context(context) @ self.objective_coef)

    def predict(self, context: np.ndarray) -> np.ndarray:
        """Expected reward of all 504 actions for a singer the agent knows nothing about."""
        return W_FEEDBACK * sigmoid(self.logit(context)) + self.objective(context)

    def predict_linear(self, context: np.ndarray) -> np.ndarray:
        if self.reward_coef is None:
            raise ValueError("this population model has no linear reward part")
        return action_quad() @ (expand_context(context) @ self.reward_coef)

    @staticmethod
    def neutral() -> "PopulationModel":
        """Untrained fallback: acceptance peaks at the default settings, objective part flat."""
        from ..correction.settings import DEFAULT

        x0 = DEFAULT.normalized()
        # logit = 2 - 8 ||x - x0||^2 in the quadratic basis [1, x, x^2, x_i x_j]
        w = np.zeros(D_ACTION)
        w[0] = 2 - 8 * np.sum(x0**2)
        w[1:5] = 16 * x0
        w[5:9] = -8
        logit = np.zeros((D_CONTEXT_EXPANDED, D_ACTION))
        logit[0] = w  # the first expanded feature is the bias
        return PopulationModel(logit, np.zeros_like(logit), 4.0 * np.eye(D_ACTION), logit * 0.1)

    def to_json(self) -> str:
        d = {
            "version": 2,
            "lapse": LAPSE,
            "logit_coef": self.logit_coef.tolist(),
            "objective_coef": self.objective_coef.tolist(),
            "prior_cov": self.prior_cov.tolist(),
            "reward_coef": None if self.reward_coef is None else self.reward_coef.tolist(),
        }
        return json.dumps(d)

    @staticmethod
    def from_json(s: str) -> "PopulationModel":
        d = json.loads(s)
        if d.get("version") != 2:
            raise ValueError("old population model format; retrain with scripts/run_agent_sim.py")
        rc = d.get("reward_coef")
        return PopulationModel(
            np.array(d["logit_coef"]), np.array(d["objective_coef"]), np.array(d["prior_cov"]),
            None if rc is None else np.array(rc),
        )


# ----------------------------------------------------------------------------- personal part


@dataclass
class Judgment:
    """One judgment of a take. ``y`` is 1/0 for keep/discard, the rating scaled to [0, 1], or
    for A/B 1 if the version made with ``action`` was picked over the one made with ``rival``."""

    kind: str
    context: list
    action: int
    y: float
    rival: int | None = None


class LogisticPosterior:
    """Laplace posterior over the singer's own logit weights w (see the module docstring)."""

    kind = "logistic"

    def __init__(self, prior_cov, rating_weight: float = 3.0, discount: float = 1.0, judgments=None):
        self.prior_cov = np.asarray(prior_cov, float)
        self.prior_precision = np.linalg.inv(self.prior_cov)
        self.rating_weight = rating_weight
        self.discount = discount
        self.judgments: list[Judgment] = list(judgments or [])
        self.mean = np.zeros(D_ACTION)
        self.precision = self.prior_precision.copy()

    @property
    def n(self) -> int:
        return len(self.judgments)

    @property
    def cov(self) -> np.ndarray:
        return np.linalg.inv(self.precision)

    def design(self, pop: PopulationModel):
        """Rows z, offsets, outcomes and weights of the logistic likelihood."""
        Q = action_quad()
        Z, off, y, wt = [], [], [], []
        for j in self.judgments:
            z = Q[j.action] - Q[j.rival] if j.kind == "ab" else Q[j.action]
            Z.append(z)
            off.append(float(z @ pop.logit_weights(np.asarray(j.context, float))))
            y.append(j.y)
            wt.append(self.rating_weight if j.kind == "rating" else 1.0)
        wt = np.array(wt)
        if self.discount < 1.0:  # older judgments count less (drift)
            wt = wt * self.discount ** np.arange(len(wt))[::-1]
        return np.array(Z), np.array(off), np.array(y), wt

    def refit(self, pop: PopulationModel) -> None:
        if not self.judgments:
            self.mean, self.precision = np.zeros(D_ACTION), self.prior_precision.copy()
            return
        Z, off, y, wt = self.design(pop)
        self.mean, self.precision = fit_logistic(Z, y, off, wt, self.prior_precision, w0=self.mean)

    def add(self, judgment: Judgment, pop: PopulationModel) -> None:
        self.judgments.append(judgment)
        self.refit(pop)

    def to_dict(self) -> dict:
        return dict(
            kind=self.kind, prior_cov=self.prior_cov.tolist(), rating_weight=self.rating_weight,
            discount=self.discount, judgments=[asdict(j) for j in self.judgments],
            mean=self.mean.tolist(), precision=self.precision.tolist(),
        )

    @staticmethod
    def from_dict(d: dict) -> "LogisticPosterior":
        p = LogisticPosterior(d["prior_cov"], d["rating_weight"], d["discount"], [Judgment(**j) for j in d["judgments"]])
        p.mean, p.precision = np.array(d["mean"]), np.array(d["precision"])
        return p


# Prior variance shape of the linear ablation over [1, x(4), x^2(4), x_i x_j(6)]: most mass on
# the linear terms, which move the singer's ideal settings.
PRIOR_SHAPE = np.array([0.5] + [1.0] * 4 + [0.15] * 4 + [0.05] * 6)


@dataclass
class PersonalPosterior:
    """Gaussian posterior of the linear ablation (r = pop(c, x) + w . q(x) + noise)."""

    prior_var: float = 0.1
    noise_var: float = 0.01
    precision: np.ndarray = field(default=None)
    b: np.ndarray = field(default=None)
    n: int = 0
    kind = "linear"

    def __post_init__(self):
        if self.precision is None:
            self.precision = np.diag(1 / (self.prior_var * PRIOR_SHAPE))
        if self.b is None:
            self.b = np.zeros(D_ACTION)

    @property
    def cov(self) -> np.ndarray:
        return np.linalg.inv(self.precision)

    @property
    def mean(self) -> np.ndarray:
        return np.linalg.solve(self.precision, self.b)

    def update(self, q: np.ndarray, residual: float) -> None:
        self.precision = self.precision + np.outer(q, q) / self.noise_var
        self.b = self.b + q * residual / self.noise_var
        self.n += 1

    def to_dict(self) -> dict:
        return dict(kind=self.kind, prior_var=self.prior_var, noise_var=self.noise_var,
                    precision=self.precision.tolist(), b=self.b.tolist(), n=self.n)

    @staticmethod
    def from_dict(d: dict) -> "PersonalPosterior":
        return PersonalPosterior(d["prior_var"], d["noise_var"], np.array(d["precision"]), np.array(d["b"]), d["n"])


def posterior_from_dict(d: dict):
    return LogisticPosterior.from_dict(d) if d.get("kind") == "logistic" else PersonalPosterior.from_dict(d)


# ----------------------------------------------------------------------------- policies


def _measured_reward(f: float, measured) -> float:
    return reward(f, *measured)


class Policy:
    """Base class: the logistic model of feedback plus the population's objective part.

    The app and the simulation use the same calls: ``select`` picks the settings for a take,
    ``rival`` the other version of an A/B pair, ``observe``/``observe_pair`` learn from the
    judgment, and ``greedy`` is the agent's current best guess."""

    name = "policy"

    def __init__(self, population: PopulationModel | None = None, rng: np.random.Generator | None = None,
                 rating_weight: float = 3.0, discount: float = 1.0):
        self.pop = population or PopulationModel.neutral()
        self.rng = rng or np.random.default_rng()
        self.rating_weight, self.discount = rating_weight, discount
        self.post = self.new_posterior()

    def new_posterior(self):
        return LogisticPosterior(self.pop.prior_cov, self.rating_weight, self.discount)

    def set_posterior(self, post) -> None:
        """Use a stored posterior. One of another kind (an older profile) starts afresh."""
        if isinstance(post, LogisticPosterior):
            post.refit(self.pop)  # the shipped population may have been retrained
            self.post = post
        else:
            self.post = self.new_posterior()

    # ------------------------------------------------------------------ scores
    def _expected(self, context: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Expected reward of every action when w ~ N(mean, cov). E[sigmoid] uses the probit
        approximation sigmoid(mu / sqrt(1 + pi s^2 / 8)) (MacKay 1992)."""
        Q = action_quad()
        mu = self.pop.logit(context) + Q @ mean
        s2 = np.einsum("ij,jk,ik->i", Q, cov, Q)
        return W_FEEDBACK * sigmoid(mu / np.sqrt(1 + np.pi * s2 / 8)) + self.pop.objective(context)

    def mean_scores(self, context: np.ndarray) -> np.ndarray:
        """Posterior expected reward of every action."""
        return self._expected(context, self.post.mean, self.post.cov)

    def population_choice(self, context: np.ndarray) -> int:
        """Best action for a singer the agent knows nothing about (the prior). This is also
        the greedy choice of every learning policy before its first judgment."""
        return int(np.argmax(self._expected(context, np.zeros(D_ACTION), self.pop.prior_cov)))

    def sample_scores(self, context: np.ndarray, scale: float = 1.0) -> np.ndarray:
        w = self.rng.multivariate_normal(self.post.mean, scale**2 * self.post.cov)
        return W_FEEDBACK * sigmoid(self.pop.logit(context) + action_quad() @ w) + self.pop.objective(context)

    def select(self, context: np.ndarray) -> int:
        raise NotImplementedError

    def greedy(self, context: np.ndarray) -> int:
        return int(np.argmax(self.mean_scores(context)))

    def rival(self, context: np.ndarray, action: int) -> int:
        """The other version of an A/B pair: the agent's best guess, or the population's choice
        if that is the same setting, or else the next best setting."""
        b = self.greedy(context)
        if b == action:
            b = self.population_choice(context)
        if b == action:
            s = self.mean_scores(context)
            s[action] = -np.inf
            b = int(np.argmax(s))
        return b

    # ------------------------------------------------------------------ learning
    def observe(self, context: np.ndarray, action: int, kind: str, f: float, measured) -> float:
        """Learn from a keep/discard or rating judgment of the take made with ``action``.

        f is the scaled feedback (1/0, or the rating in [0, 1]); ``measured`` is the pipeline's
        (a, n, p) for that take. Returns the proposal's reward r for the log."""
        self.post.add(Judgment(kind, [float(v) for v in context], int(action), float(f)), self.pop)
        return _measured_reward(f, measured)

    def observe_pair(self, context: np.ndarray, action: int, rival: int, picked_action: bool, measured_a, measured_b) -> tuple[float, float]:
        """Learn from an A/B judgment between the versions made with ``action`` and ``rival``."""
        y = float(bool(picked_action))
        self.post.add(Judgment("ab", [float(v) for v in context], int(action), y, int(rival)), self.pop)
        return _measured_reward(y, measured_a), _measured_reward(1 - y, measured_b)

    def choose(self, context: np.ndarray) -> CorrectionSettings:
        return action_grid()[self.select(context)]


class ThompsonSampling(Policy):
    """Thompson sampling with the Laplace posterior (Chapelle & Li 2011). ``explore`` scales the
    posterior spread used for sampling (1 = textbook). Chapelle & Li found that a smaller scale
    helps with few interactions; 0.5 was chosen on development singers (scripts/tune_agent.py)."""

    name = "thompson"

    def __init__(self, *a, explore: float = 0.5, **kw):
        super().__init__(*a, **kw)
        self.explore = explore

    def select(self, context):
        return int(np.argmax(self.sample_scores(context, self.explore)))


class DiscountedThompson(ThompsonSampling):
    """Thompson sampling that discounts older judgments, for singers whose taste drifts."""

    name = "thompson_discounted"

    def __init__(self, *a, discount: float = 0.95, **kw):
        super().__init__(*a, discount=discount, **kw)


class UCB(Policy):
    """Optimism instead of sampling: the logit is raised by ``alpha`` posterior standard deviations."""

    name = "ucb"

    def __init__(self, *a, alpha: float = 1.0, **kw):
        super().__init__(*a, **kw)
        self.alpha = alpha

    def select(self, context):
        Q = action_quad()
        mu = self.pop.logit(context) + Q @ self.post.mean
        sd = np.sqrt(np.einsum("ij,jk,ik->i", Q, self.post.cov, Q))
        return int(np.argmax(W_FEEDBACK * sigmoid(mu + self.alpha * sd) + self.pop.objective(context)))


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

    def observe(self, context, action, kind, f, measured):
        return _measured_reward(f, measured)

    def observe_pair(self, context, action, rival, picked_action, measured_a, measured_b):
        y = float(bool(picked_action))
        return _measured_reward(y, measured_a), _measured_reward(1 - y, measured_b)


class PopulationOnly(FixedPolicy):
    """C2-like: the population's best guess, no personal learning."""

    name = "population"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

    def select(self, context):
        return self.population_choice(context)

    greedy = select


class LinearThompson(Policy):
    """Ablation: the first design, linear-Gaussian Thompson sampling on r itself.

    r = pop_linear(c, x) + w . q(x) + noise, with a Gaussian prior on w and a fixed noise
    variance."""

    name = "thompson_linear"

    def __init__(self, population=None, rng=None, explore: float = 0.5, prior_var: float = 0.1, noise_var: float = 0.01, **kw):
        self.prior_var, self.noise_var = prior_var, noise_var
        super().__init__(population, rng, **kw)
        self.explore = explore

    def new_posterior(self):
        return PersonalPosterior(self.prior_var, self.noise_var)

    def set_posterior(self, post) -> None:
        self.post = post if isinstance(post, PersonalPosterior) else self.new_posterior()

    def population_choice(self, context):
        return int(np.argmax(self.pop.predict_linear(context)))

    def mean_scores(self, context):
        return self.pop.predict_linear(context) + action_quad() @ self.post.mean

    def select(self, context):
        w = self.rng.multivariate_normal(self.post.mean, self.explore**2 * self.post.cov)
        return int(np.argmax(self.pop.predict_linear(context) + action_quad() @ w))

    def _update(self, context, action, r):
        self.post.update(action_quad()[action], r - float(self.pop.predict_linear(context)[action]))

    def observe(self, context, action, kind, f, measured):
        r = _measured_reward(f, measured)
        self._update(context, action, r)
        return r

    def observe_pair(self, context, action, rival, picked_action, measured_a, measured_b):
        y = float(bool(picked_action))
        ra, rb = _measured_reward(y, measured_a), _measured_reward(1 - y, measured_b)
        self._update(context, action, ra)
        self._update(context, rival, rb)
        return ra, rb


POLICIES = {
    p.name: p
    for p in (ThompsonSampling, DiscountedThompson, UCB, EpsilonGreedy, FixedPolicy, PopulationOnly, LinearThompson)
}
