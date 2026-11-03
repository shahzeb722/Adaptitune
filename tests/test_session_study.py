import socket

import numpy as np
import pandas as pd
import pytest

from adaptitune.agent.profile import ProfileStore
from adaptitune.agent.session import AdaptiveSession
from adaptitune.audio import Audio
from adaptitune.correction.settings import CorrectionSettings
from adaptitune.pipeline import Corrector
from adaptitune.study import analysis as SA
from adaptitune.study import protocol as SP


@pytest.fixture()
def session(tmp_path):
    return AdaptiveSession("ali", ProfileStore(tmp_path / "p.sqlite"), Corrector("yin"), seed=0)


@pytest.fixture()
def offline(monkeypatch):
    """NFR1: any attempt to open a network connection fails the test."""

    def boom(*a, **k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(socket, "getaddrinfo", boom)


def test_full_flow_offline(session, detuned, offline, tmp_path):
    take = session.load_take(Audio(detuned.samples, detuned.sr), key="C major")  # FR1, FR2
    res = session.propose(take)  # FR3
    assert len(res.note_report()) == 5  # FR6
    session.feedback(take, res, "keep", 1)  # FR4 keep
    session.feedback(take, res, "rating", 4)  # FR4 rating
    a, b = session.propose_ab(take)
    assert a.settings != b.settings
    session.feedback_ab(take, a, b, "a")  # FR4 A/B
    man = session.manual(take, CorrectionSettings(0.5, 44.7, "in_key", 1.0))  # FR5
    session.save_manual(take, man.settings, seconds=30, adjustments=4)
    assert session.last_manual().strength == 0.5
    assert session.interactions == 3  # keep, rating and one A/B pick
    assert len(session.store.history("ali")) == 4  # the A/B pick logs both versions
    out = session.export(tmp_path / "me.json")  # FR7 export
    assert out.exists() and "posterior" in out.read_text()
    session.reset()  # FR7 reset
    assert session.interactions == 0


def test_profile_persists(tmp_path, detuned):
    store = ProfileStore(tmp_path / "p.sqlite")
    s1 = AdaptiveSession("sara", store, Corrector("yin"), seed=0)
    take = s1.load_take(Audio(detuned.samples, detuned.sr), key="C major")
    s1.feedback(take, s1.propose(take), "keep", 1)
    a, b = s1.propose_ab(take)
    s1.feedback_ab(take, a, b, "b")
    s2 = AdaptiveSession("sara", ProfileStore(tmp_path / "p.sqlite"), Corrector("yin"))
    assert s2.interactions == 2
    assert len(s2.store.history("sara")) == 3
    assert np.allclose(s2.policy.post.mean, s1.policy.post.mean)
    assert s2.current_best(take) == s1.current_best(take)


def test_old_linear_profile_starts_afresh(tmp_path):
    """A profile saved by the first (linear) agent is replaced by a fresh logistic posterior."""
    from adaptitune.agent.policies import LogisticPosterior, PersonalPosterior

    store = ProfileStore(tmp_path / "p.sqlite")
    old = PersonalPosterior()
    old.update(np.ones(15), 0.5)
    store.save("old", old)
    s = AdaptiveSession("old", store, Corrector("yin"), seed=0)
    assert isinstance(s.policy.post, LogisticPosterior) and s.interactions == 0


def test_schedule_design():
    stim = [f"rec{i}" for i in range(8)]
    for session in range(1, 6):
        tr = SP.schedule("p01", session, stim, 0)
        assert len(tr) == SP.TAKES_PER_SESSION
        assert sum(t.repeat_of is not None for t in tr) == SP.REPEATS_PER_SESSION
        conds = [t.condition for t in tr if t.repeat_of is None]
        assert conds.count("C3") == 4
        assert ("C4" in conds) == (session >= 3)
        for t in tr:
            if t.repeat_of:
                orig = tr[t.repeat_of - 1]
                assert orig.condition == t.condition and orig.stimulus == t.stimulus and orig.position < t.position


def test_consistency_flag():
    tr = SP.schedule("p01", 1, ["a", "b", "c"], 0)
    for t in tr:
        t.keep = 1
    reps = [t for t in tr if t.repeat_of]
    for r in reps:
        r.keep = 0  # disagree on every repeat
    assert SP.consistency(tr) == 0
    SP.flag_inconsistent(tr)
    assert any(t.flagged for t in tr)


def _fake_study(n=12, effect=0.25, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        u = rng.normal(0, 0.5)
        for s in range(1, 6):
            for t in SP.schedule(f"p{i}", s, ["a", "b", "c", "d"], i):
                base = {"C1": 0.4, "C2": 0.5, "C3": 0.4 + effect, "C4": 0.7}[t.condition]
                t.keep = int(rng.random() < 1 / (1 + np.exp(-(np.log(base / (1 - base)) + u))))
                t.naturalness = int(np.clip(round(rng.normal(3.5 if t.condition != "C1" else 3.2, 0.7)), 1, 5))
                rows.append(t.__dict__)
    return pd.DataFrame(rows)


def test_analysis_end_to_end(tmp_path):
    df = _fake_study()
    p = tmp_path / "trials.csv"
    df.to_csv(p, index=False)
    out = SA.analyze(str(p))
    est, lo, hi = out["C3_minus_C1"]
    assert lo < est < hi and est > 0.1
    assert out["naturalness_noninferiority"]["pass"]
    assert out["mixed_logit"] is not None


def test_holm():
    adj = SA.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)


def test_power_simulation_monotone():
    small = SA.simulate_study(n_singers=6, n_sims=150)
    big = SA.simulate_study(n_singers=30, n_sims=150)
    assert big["power"] >= small["power"]
