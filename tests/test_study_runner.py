"""The O6 study runner: one session as a participant goes through it, the rules it enforces,
and the files the analysis reads."""

import csv
import json

import numpy as np
import pytest

from adaptitune.agent.profile import ProfileStore
from adaptitune.agent.session import AdaptiveSession
from adaptitune.audio import Audio
from adaptitune.correction.settings import DEFAULT, FIXED_CLASSIC, CorrectionSettings
from adaptitune.data.synth import SungNote, sing
from adaptitune.pipeline import Corrector
from adaptitune.study import runner as R
from adaptitune.study.analysis import analyze


def _takes(n=3):
    out = []
    for k in range(n):
        notes = [SungNote(60 + 2 * ((k + j) % 4), 0.45, offset_cents=(-1) ** j * 30, vibrato_cents=20 * (j % 2)) for j in range(4)]
        s = sing(notes, seed=k)
        out.append(Audio(s.samples, s.sr))
    return out


@pytest.fixture
def study(tmp_path):
    store = ProfileStore(tmp_path / "profiles.sqlite")
    corrector = Corrector("yin")

    def run_session(code, session, keep=lambda t: 1, manual=CorrectionSettings(0.6, 40.0, "in_key", 0.5)):
        code = R.check_code(code)  # the window does the same before it makes the agent session
        api = AdaptiveSession(code, store, corrector, seed=session)
        run = R.StudySession(code, session, api, root=tmp_path / "study", build="test")
        for a in _takes():
            run.add_take(a, key="C major")
        run.start()
        for i, t in enumerate(run.trials):
            run.render(i)
            run.judge(i, keep(t), 4 if t.condition != "C1" else 3, seconds=5.0)
        run.manual(manual)
        run.finish_manual(manual, seconds=90, adjustments=7, rating=4)
        return run, run.finish()

    return run_session, store, tmp_path / "study"


def test_session_follows_the_protocol(study):
    run_session, store, root = study
    run, summary = run_session("p01", 1)
    assert run.participant == "P01" and len(run.trials) == 12
    conds = [t.condition for t in run.trials if not t.repeat_of]
    assert conds.count("C3") == 4 and "C4" not in conds
    # every condition hears different takes before any take repeats
    for c in ("C1", "C2"):
        stims = [t.stimulus for t in run.trials if t.condition == c and not t.repeat_of]
        assert len(set(stims)) == len(stims)
    # C1 and C2 are the fixed settings; only C3 originals teach the agent
    for i, t in enumerate(run.trials):
        s = run.results[i].settings
        if t.condition == "C1":
            assert s == FIXED_CLASSIC
        if t.condition == "C2":
            assert s == DEFAULT
        if t.repeat_of:
            assert run.results[i] is run.results[t.repeat_of - 1]  # the same audio, not a new version
    learned = sum(1 for t in run.trials if t.condition == "C3" and not t.repeat_of)
    assert summary["agent_judgments_after"] - summary["agent_judgments_before"] == learned == 4
    assert summary["consistency"] == 1.0 and summary["flagged_trials"] == 0
    assert R.next_session("P01", root) == 2


def test_c4_is_the_setting_saved_last_session(study):
    run_session, store, root = study
    saved = CorrectionSettings(0.5, 80.0, "chromatic", 0.25)
    run_session("P02", 1, manual=saved)
    run_session("P02", 2, manual=saved)
    run3, _ = run_session("P02", 3)
    c4 = [i for i, t in enumerate(run3.trials) if t.condition == "C4"]
    assert len(c4) >= 2 and all(run3.results[i].settings == saved for i in c4)
    with pytest.raises(ValueError, match="already done"):
        R.StudySession("P02", 3, AdaptiveSession("P02", store, Corrector("yin")), root=root)


def test_runner_refuses_steps_out_of_order(study, tmp_path):
    _, store, root = study
    api = AdaptiveSession("P03", store, Corrector("yin"), seed=0)
    run = R.StudySession("P03", 1, api, root=root, build="test")
    with pytest.raises(RuntimeError):
        run.start()
    for a in _takes():
        run.add_take(a)
    run.start()
    first_c3 = next(i for i, t in enumerate(run.trials) if t.condition == "C3")
    if first_c3 > 0:
        with pytest.raises(RuntimeError, match="learn"):
            run.render(first_c3)  # the agent has not heard the earlier judgments yet
    with pytest.raises(RuntimeError, match="next"):
        run.judge(1, 1, 3)
    with pytest.raises(ValueError):
        run.judge(0, 1, 7)
    with pytest.raises(RuntimeError):
        run.finish()
    for bad in ("Ali Khan", "", "12", "P-1"):
        with pytest.raises(ValueError):
            R.StudySession(bad, 1, AdaptiveSession(bad, store, Corrector("yin")), root=root)


def test_inconsistent_repeats_are_flagged(study):
    run_session, _, root = study
    # keep every original, discard every repeat: agreement on repeats is 0
    run, summary = run_session("P04", 1, keep=lambda t: 0 if t.repeat_of else 1)
    assert summary["consistency"] == 0.0 and summary["flagged_trials"] > 0
    with open(root / "sessions" / "P04_s1_trials.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert any(r["flagged"] == "True" for r in rows)


def test_export_feeds_the_analysis_and_withdrawal_deletes(study, tmp_path):
    run_session, store, root = study
    rng = np.random.default_rng(0)
    for k in range(4):
        for s in (1, 2, 3):
            run_session(f"P1{k}", s, keep=lambda t: int(rng.random() < (0.8 if t.condition == "C3" else 0.4)))
    paths = R.export(root, tmp_path / "export")
    with open(paths["trials"], newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4 * 3 * 12 and set(R.TRIAL_FIELDS) <= set(rows[0])
    assert all(r["participant"].startswith("P1") for r in rows)  # codes only
    out = analyze(str(paths["trials"]), str(paths["manual"]))
    assert out["participants"] == 4 and out["manual_seconds_mean"] == 90
    assert set(out["acceptance_by_condition_s3_5"]) == {"C1", "C2", "C3", "C4"}
    t = out["time_to_kept_take_s3_5"]  # 5 s per judgment against 90 s by hand
    assert t["n"] == 4 and t["manual_seconds"] == 90 and t["agent_seconds"] < 90 and t["pass"]
    assert out["O6_parts"]["less_time"] is True
    assert json.loads((root / "sessions" / "P10_s3.json").read_text())["c4_setting"] is not None
    n = R.withdraw("P10", store, root)
    assert n > 0 and not list((root / "sessions").glob("P10_*")) and not list((root / "takes").glob("P10_*"))
    assert store.last_manual("P10") is None
    assert b"P10" not in store.path.read_bytes()  # not even in the database file's free pages


def test_time_to_a_kept_take_counts_a_singer_who_kept_nothing_as_tuning_by_hand():
    import pandas as pd

    from adaptitune.study.analysis import time_to_kept_take

    trials = pd.DataFrame({"participant": ["A"] * 4 + ["B"] * 4, "condition": ["C3"] * 8,
                           "keep": [1, 0, 1, 0, 0, 0, 0, 0], "seconds": [10.0] * 8})
    manual = pd.DataFrame({"participant": ["A", "B"], "seconds": [100.0, 60.0]})
    t = time_to_kept_take(trials, manual)
    # A: 40 s of judging for 2 kept versions = 20 s. B kept none: 40 s judging, then 60 s by hand
    assert t["agent_seconds"] == (20 + 100) / 2 and t["manual_seconds"] == 80
    assert t["saved"][0] == ((100 - 20) + (60 - 100)) / 2 and t["pass"]
