import numpy as np
import pytest

from adaptitune.agent.experiment import run, run_singer, train_population
from adaptitune.agent.features import D_ACTION, action_quad, context_from_track, quad
from adaptitune.agent.policies import POLICIES, PersonalPosterior, PopulationModel
from adaptitune.agent.reward import Feedback, reward
from adaptitune.agent.simulator import MODELS, SimSinger, oracle_best
from adaptitune.correction.settings import FIXED_CLASSIC, action_grid, action_index, action_matrix


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


def test_posterior_learns_linear_signal():
    post = PersonalPosterior()
    w_true = np.zeros(15)
    w_true[1] = 0.3
    Q = action_quad()
    rng = np.random.default_rng(0)
    for _ in range(200):
        i = rng.integers(504)
        post.update(Q[i], Q[i] @ w_true + rng.normal(0, 0.05))
    assert post.mean[1] == pytest.approx(0.3, abs=0.1)
    assert PersonalPosterior.from_dict(post.to_dict()).n == 200


def test_cold_start_uses_population(pop):
    s = SimSinger.sample(np.random.default_rng(3))
    ts = POLICIES["thompson"](pop, rng=np.random.default_rng(0))
    assert ts.greedy(s.context) == int(np.argmax(pop.predict(s.context)))


def test_fixed_policy_never_moves(pop):
    f = POLICIES["fixed"](pop)
    s = SimSinger.sample(np.random.default_rng(1))
    assert f.select(s.context) == action_index(FIXED_CLASSIC)
    f.update(s.context, 0, 1.0)
    assert f.select(s.context) == action_index(FIXED_CLASSIC)


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


def test_thompson_improves_over_population(pop):
    """Learning sanity check: after 30 interactions TS is better than the cold start."""
    gains = []
    for i in range(30):
        rng = np.random.default_rng(100 + i)
        s = SimSinger.sample(rng, "logistic")
        p = POLICIES["thompson"](pop, rng=np.random.default_rng(i))
        log = run_singer(p, s, 30, 6, rng)
        gains.append(log.greedy_er[-1] - log.er_table[-1][int(np.argmax(pop.predict(s.context)))])
    assert np.mean(gains) > 0


def test_experiment_runs(pop, tmp_path):
    df, s = run(("thompson", "fixed"), singers_per_model=2, horizon=6, population=pop, out_dir=tmp_path)
    assert set(df.policy) == {"thompson", "fixed"}
    assert (tmp_path / "learning_curves.png").exists()
    assert s["table"]


def test_population_json_roundtrip(pop):
    back = PopulationModel.from_json(pop.to_json())
    c = SimSinger.sample(np.random.default_rng(0)).context
    assert np.allclose(back.predict(c), pop.predict(c))


def test_shipped_population_model_loads():
    from adaptitune.agent.session import load_population

    m = load_population()
    assert m.predict(np.ones(8)).shape == (504,)
