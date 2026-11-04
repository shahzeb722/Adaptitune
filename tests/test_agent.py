import numpy as np
import pytest

from adaptitune.agent.experiment import run, run_singer, singer_seed, train_population
from adaptitune.agent.features import D_ACTION, action_quad, context_from_track, quad
from adaptitune.agent.policies import (
    POLICIES, Judgment, LogisticPosterior, PersonalPosterior, PopulationModel, fit_logistic, posterior_from_dict, sigmoid,
)
from adaptitune.agent.reward import Feedback, reward
from adaptitune.agent.simulator import MODELS, SimSinger, oracle_best
from adaptitune.correction.settings import FIXED_CLASSIC, action_index, action_matrix


@pytest.fixture(scope="module")
def pop():
    return train_population(400, seed=7)


def test_reward_formula():
    assert reward(1, 1, 1, 0) == pytest.approx(1.0)
    assert reward(0, 0.5, 0.5, 0.1) == pytest.approx(0.1)
    assert Feedback("rating", 5).scaled() == 1.0
    assert Feedback("rating", 1).scaled() == 0.0
    assert Feedback("ab", 1).scaled() == 1.0


def test_quad_features():
    assert action_quad().shape == (504, D_ACTION) == (504, 15)
    q = quad(np.array([0.5, 0.2, 1.0, 0.0]))
    assert q[0] == 1 and q[1] == 0.5 and q[5] == 0.25


def test_context_features(detuned):
    c = context_from_track(detuned.f0, detuned.hop_s, -60)
    assert c.shape == (8,) and c[0] == 1
    assert np.all((c >= 0) & (c <= 1))
    assert c[1] > 0.3  # notes are 20-40 cents off
    assert c[5] > 0.2  # there is vibrato


def test_population_beats_random(pop):
    rng = np.random.default_rng(0)
    X = action_matrix()
    gain = []
    for _ in range(100):
        s = SimSinger.sample(rng)
        er = s.expected_reward(X)
        gain.append(er[np.argmax(pop.predict(s.context))] - er.mean())
    assert np.mean(gain) > 0.2


def test_linear_posterior_learns_linear_signal():
    post = PersonalPosterior()
    w_true = np.zeros(15)
    w_true[1] = 0.3
    Q = action_quad()
    rng = np.random.default_rng(0)
    for _ in range(200):
        i = rng.integers(504)
        post.update(Q[i], Q[i] @ w_true + rng.normal(0, 0.05))
    assert post.mean[1] == pytest.approx(0.3, abs=0.1)
    assert posterior_from_dict(post.to_dict()).n == 200


def test_fit_logistic_recovers_weights():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(4000, 3))
    w = np.array([1.5, -2.0, 0.5])
    y = (rng.random(4000) < sigmoid(0.3 + Z @ w)).astype(float)
    est, H = fit_logistic(Z, y, offset=np.full(4000, 0.3))
    assert est == pytest.approx(w, abs=0.15)
    assert np.all(np.linalg.eigvalsh(H) > 0)
    # fractional outcomes give the same answer as their expectation
    est2, _ = fit_logistic(Z, sigmoid(0.3 + Z @ w), offset=np.full(4000, 0.3))
    assert est2 == pytest.approx(w, abs=1e-3)


def test_logistic_posterior_learns_a_singers_taste(pop):
    """Keep/discard, rating and A/B judgments from a singer whose logit is h + w_true . q all
    move the posterior towards w_true, and the stored form gives back the same posterior."""
    rng = np.random.default_rng(1)
    c = SimSinger.sample(rng).context
    Q, h = action_quad(), pop.logit(c)
    w_true = rng.multivariate_normal(np.zeros(15), pop.prior_cov)
    p_true = sigmoid(h + Q @ w_true)
    post = LogisticPosterior(pop.prior_cov)
    err0 = np.linalg.norm(post.mean - w_true)
    for t in range(300):
        a, b = rng.integers(504, size=2)
        kind = ("keep", "rating", "ab")[t % 3]
        if kind == "keep":
            post.judgments.append(Judgment(kind, list(c), int(a), float(rng.random() < p_true[a])))
        elif kind == "rating":
            post.judgments.append(Judgment(kind, list(c), int(a), float(np.clip(p_true[a] + rng.normal(0, 0.1), 0, 1))))
        else:
            win = rng.random() < sigmoid((h[a] + Q[a] @ w_true) - (h[b] + Q[b] @ w_true))
            post.judgments.append(Judgment(kind, list(c), int(a), float(win), int(b)))
    post.refit(pop)
    assert np.linalg.norm(post.mean - w_true) < 0.5 * err0
    back = posterior_from_dict(post.to_dict())
    assert isinstance(back, LogisticPosterior) and back.n == 300
    assert np.allclose(back.mean, post.mean) and np.allclose(back.precision, post.precision)


def test_cold_start_uses_population(pop):
    s = SimSinger.sample(np.random.default_rng(3))
    ts = POLICIES["thompson"](pop, rng=np.random.default_rng(0))
    assert ts.greedy(s.context) == ts.population_choice(s.context) == POLICIES["population"](pop).select(s.context)


def test_rival_is_always_a_different_setting(pop):
    s = SimSinger.sample(np.random.default_rng(4))
    for name in POLICIES:
        p = POLICIES[name](pop, rng=np.random.default_rng(0))
        for a in (p.greedy(s.context), p.population_choice(s.context), 0, 503):
            assert p.rival(s.context, a) != a


def test_fixed_and_population_policies_never_move(pop):
    s = SimSinger.sample(np.random.default_rng(1))
    for name, start in (("fixed", action_index(FIXED_CLASSIC)), ("population", None)):
        f = POLICIES[name](pop)
        start = f.select(s.context) if start is None else start
        assert f.select(s.context) == start
        f.observe(s.context, 0, "keep", 1.0, (1.0, 1.0, 0.0))
        f.observe_pair(s.context, 0, 1, True, (1.0, 1.0, 0.0), (0.0, 0.0, 0.0))
        assert f.select(s.context) == f.greedy(s.context) == start


def test_observe_returns_the_proposal_reward(pop):
    s = SimSinger.sample(np.random.default_rng(2))
    for name in ("thompson", "thompson_linear"):
        p = POLICIES[name](pop, rng=np.random.default_rng(0))
        assert p.observe(s.context, 7, "rating", 0.75, (0.9, 0.8, 0.1)) == pytest.approx(reward(0.75, 0.9, 0.8, 0.1))
        ra, rb = p.observe_pair(s.context, 7, 9, False, (0.9, 0.8, 0.0), (0.5, 0.5, 0.0))
        assert (ra, rb) == (pytest.approx(reward(0, 0.9, 0.8, 0)), pytest.approx(reward(1, 0.5, 0.5, 0)))
        assert p.post.n == (2 if name == "thompson" else 3)  # an A/B pick is one judgment for the logistic model


def test_singer_seeds_are_stable_and_disjoint():
    assert singer_seed(0, "logistic", 5) == 5
    dev = {singer_seed(1, m, i) for m in MODELS for i in range(1000)}
    rep = {singer_seed(0, m, i) for m in MODELS for i in range(1000)}
    assert not dev & rep and len(rep) == 3000


@pytest.mark.parametrize("model", MODELS)
def test_simulator_models(model):
    rng = np.random.default_rng(5)
    s = SimSinger.sample(rng, model)
    X = action_matrix()
    p = s.p_accept(X)
    assert p.shape == (504,) and np.all((p >= 0) & (p <= 1))
    i, best = oracle_best(s)
    assert best == pytest.approx(s.expected_reward(X).max())
    assert s.judge(X[i][None], "keep") in (0.0, 1.0)
    assert 0 <= s.judge(X[i][None], "rating") <= 1


def test_fatigue_and_drift():
    s = SimSinger.sample(np.random.default_rng(2))
    t0 = s.current_temp()
    for _ in range(5):
        s.judge(action_matrix()[0][None], "keep")
    assert s.current_temp() > t0
    ideal = s.ideal.copy()
    s.new_session()
    assert s.judgments_in_session == 0 and not np.allclose(ideal, s.ideal)


@pytest.mark.parametrize("model", MODELS)
def test_thompson_improves_over_population(pop, model):
    """Learning sanity check: after 20 interactions TS is better than the cold start."""
    gains = []
    for i in range(20):
        rng = np.random.default_rng(100 + i)
        s = SimSinger.sample(rng, model)
        p = POLICIES["thompson"](pop, rng=np.random.default_rng(i))
        start = p.population_choice(s.context)
        log = run_singer(p, s, 20, 4, rng)
        gains.append(log.greedy_er[-1] - log.er_table[-1][start])
    assert np.mean(gains) > 0.03


def test_experiment_runs(pop, tmp_path):
    df, s = run(("thompson", "fixed"), singers_per_model=2, horizon=6, population=pop, out_dir=tmp_path)
    assert set(df.policy) == {"thompson", "fixed"}
    assert (tmp_path / "learning_curves.png").exists()
    assert s["table"]


def test_population_json_roundtrip(pop):
    back = PopulationModel.from_json(pop.to_json())
    c = SimSinger.sample(np.random.default_rng(0)).context
    assert np.allclose(back.predict(c), pop.predict(c))
    assert np.allclose(back.predict_linear(c), pop.predict_linear(c))
    assert np.allclose(back.prior_cov, pop.prior_cov)


def test_population_prior_is_a_covariance(pop):
    ev = np.linalg.eigvalsh(pop.prior_cov)
    assert np.allclose(pop.prior_cov, pop.prior_cov.T) and ev.min() > 0


def test_shipped_population_model_loads():
    from adaptitune.agent.session import load_population

    m = load_population()
    assert m.predict(np.ones(8)).shape == (504,)
    assert m.prior_cov.shape == (15, 15)


def test_neutral_population_peaks_at_the_default():
    from adaptitune.correction.settings import DEFAULT, action_grid

    m = PopulationModel.neutral()
    c = np.r_[1.0, np.full(7, 0.5)]
    assert action_grid()[int(np.argmax(m.logit(c)))] == DEFAULT
