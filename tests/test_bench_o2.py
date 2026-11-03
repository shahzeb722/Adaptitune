"""O2 tracker benchmark: metrics, statistics, the pre-registered selection rule and the harness."""

import json

import numpy as np
import pandas as pd
import pytest

from adaptitune.bench import trackers as B
from adaptitune.metrics import pitch_metrics, raw_pitch_accuracy


def _ref():
    t = np.arange(200) * 0.01
    return t, np.where((t > 0.2) & (t < 1.8), 220.0, 0.0)


def test_pitch_metrics_perfect():
    t, ref = _ref()
    m = pitch_metrics(t, ref, t, ref, ref)
    assert m["rpa"] == m["rca"] == m["oa"] == m["vr"] == 1.0
    assert m["vfa"] == 0.0 and m["median_err_cents"] == 0.0 and m["octave_err"] == 0.0
    assert m["n_ref_voiced"] == int((ref > 0).sum())


def test_rpa_is_voicing_independent_with_raw_pitch():
    """A tracker that knows the pitch but calls half the voiced frames unvoiced keeps its raw
    pitch accuracy, while voicing recall and overall accuracy still show the voicing miss."""
    t, ref = _ref()
    est = np.where((t > 0.2) & (t < 1.0), 220.0 * 2 ** (5 / 1200), 0.0)
    raw = np.full_like(t, 220.0 * 2 ** (5 / 1200))
    raw[t > 1.5] = 440.0  # an octave error near the end
    with_raw = pitch_metrics(t, ref, t, est, raw)
    without = pitch_metrics(t, ref, t, est)
    assert without["rpa"] == pytest.approx(without["vr"])
    assert with_raw["rpa"] > without["rpa"] + 0.25
    assert with_raw["rca"] == pytest.approx(1.0)  # octave errors are right in chroma
    assert with_raw["octave_err"] == pytest.approx(1 - with_raw["rpa"], abs=1e-9)
    assert with_raw["vr"] == without["vr"] and with_raw["oa"] == without["oa"]
    assert with_raw["median_err_cents"] == pytest.approx(5.0, abs=0.01)
    assert set(raw_pitch_accuracy(t, ref, t, est)) == {"rpa", "rca", "vr", "vfa"}


def test_finer_tolerances_are_stricter():
    t, ref = _ref()
    est = np.where(ref > 0, ref * 2 ** (18 / 1200), 0.0)  # 18 cents sharp everywhere
    m = pitch_metrics(t, ref, t, est, est)
    assert m["rpa"] == 1.0 and m["rpa_25"] == 1.0 and m["rpa_10"] == 0.0


def test_trackers_return_raw_pitch():
    from adaptitune.data.synth import SungNote, sing
    from adaptitune.pitch.trackers import get_tracker

    s = sing([SungNote(57, 0.8), SungNote(60, 0.8)], sr=16_000, seed=0)
    for name in ("pyin", "yin"):
        tr = get_tracker(name).track(s.samples, s.sr)
        assert tr.raw_f0 is not None and len(tr.raw_f0) == len(tr.f0)
        v = tr.f0 > 0
        assert np.allclose(tr.raw_f0[v], tr.f0[v])  # voiced frames keep the same pitch
        assert (tr.raw_f0 > 0).sum() >= v.sum()


def test_bootstrap_ci_resamples_clusters():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"cluster": np.repeat(list("abcdefgh"), 5), "x": rng.normal(0.8, 0.05, 40)})
    lo, hi = B.bootstrap_ci(df, lambda d: d.x.mean(), n_boot=300)
    assert lo < df.x.mean() < hi
    single = pd.DataFrame({"cluster": ["a"] * 5, "x": [1.0] * 5})
    assert B.bootstrap_ci(single, lambda d: d.x.mean(), n_boot=50) == (1.0, 1.0)


def test_add_noise_hits_requested_snr():
    sr = 16_000
    x = np.concatenate([np.zeros(sr), 0.3 * np.sin(2 * np.pi * 200 * np.arange(2 * sr) / sr), np.zeros(sr)])
    y = B.add_noise(x, sr, 10.0, seed=3)
    noise = y - x
    snr = 10 * np.log10(B.active_power(x, sr) / np.mean(noise**2))
    assert snr == pytest.approx(10.0, abs=0.3)
    assert np.array_equal(y, B.add_noise(x, sr, 10.0, seed=3))  # deterministic


def _write_condition(out, label, rows):
    pd.DataFrame(rows).to_csv(out / f"{label}_summary.csv", index=False)


def test_speed_can_come_from_another_machine(tmp_path):
    """Accuracy from the big run, RTF from a timing run (the reference laptop)."""
    acc, lap = tmp_path / "acc", tmp_path / "laptop"
    acc.mkdir()
    lap.mkdir()
    for ds in ("mir1k", "mdb"):
        # on the loaded cloud run "a" looks too slow; on the laptop it is fast enough
        _write_condition(acc, ds, [dict(tracker="a", rpa=0.97, rtf=0.40), dict(tracker="b", rpa=0.93, rtf=0.10)])
        _write_condition(lap, ds, [dict(tracker="a", rpa=0.97, rtf=0.12), dict(tracker="b", rpa=0.93, rtf=0.03)])
        (lap / f"{ds}_run.json").write_text(json.dumps({"machine": {"cpu": "laptop"}}))
    assert B.choose_tracker(acc)["tracker"] == "b"
    c = B.choose_tracker(acc, speed_dir=lap)
    assert c["tracker"] == "a" and c["speed_from"]["machine"]["mdb"] == {"cpu": "laptop"}
    _write_condition(lap, "mdb", [dict(tracker="b", rpa=0.93, rtf=0.03)])
    with pytest.raises(ValueError):  # never mix machines for one tracker
        B.choose_tracker(acc, speed_dir=lap)


def test_selection_rule(tmp_path):
    # five trackers: "slow" is the most accurate but too slow; "a" and "b" tie within 1 point
    def rows(ds_shift):
        return [
            dict(tracker="slow", rpa=0.99 - ds_shift, rtf=0.60),
            dict(tracker="a", rpa=0.955 - ds_shift, rtf=0.20),
            dict(tracker="b", rpa=0.950 - ds_shift, rtf=0.05),
            dict(tracker="c", rpa=0.900 - ds_shift, rtf=0.01),
            dict(tracker="d", rpa=0.700 - ds_shift, rtf=0.01),
        ]

    _write_condition(tmp_path, "mir1k", rows(0.02))
    _write_condition(tmp_path, "mdb", rows(0.0))
    c = B.choose_tracker(tmp_path)
    assert c["official"] and c["best_by_accuracy"] == "a"
    assert c["tracker"] == "b"  # tied within 1 RPA point and faster
    assert c["O2_tracker_pass"] and c["trackers_reported_on_both"] == 5
    B.save_choice(c, tmp_path / "choice.json")
    assert json.loads((tmp_path / "choice.json").read_text())["tracker"] == "b"  # strict JSON

    # the chosen tracker must reach 90 % on MDB-stem-synth
    _write_condition(tmp_path, "mdb", [dict(r, rpa=r["rpa"] - 0.1) for r in rows(0.0)])
    assert not B.choose_tracker(tmp_path)["O2_tracker_pass"]


def test_selection_leaves_out_data_a_model_was_trained_on(tmp_path):
    """CREPE's released weights saw both benchmark sets, PESTO's checkpoint saw MIR-1K audio."""
    _write_condition(tmp_path, "mir1k", [dict(tracker="crepe", rpa=0.99, rtf=0.1), dict(tracker="pesto", rpa=0.99, rtf=0.05),
                                         dict(tracker="fcpe", rpa=0.95, rtf=0.05), dict(tracker="pyin", rpa=0.90, rtf=0.05)])
    _write_condition(tmp_path, "mdb", [dict(tracker="crepe", rpa=0.99, rtf=0.1), dict(tracker="pesto", rpa=0.93, rtf=0.05),
                                       dict(tracker="fcpe", rpa=0.94, rtf=0.05), dict(tracker="pyin", rpa=0.85, rtf=0.05)])
    c = B.choose_tracker(tmp_path)
    rows = {r["tracker"]: r for r in c["table"]}
    assert not rows["crepe"]["eligible"] and rows["crepe"]["score"] is None and rows["crepe"]["trained_on"] == "mir1k,mdb"
    assert rows["pesto"]["score"] == pytest.approx(0.93) and rows["pesto"]["score_datasets"] == "mdb"
    assert rows["fcpe"]["score"] == pytest.approx(0.945)
    assert c["tracker"] == "fcpe"  # 1.5 points ahead of PESTO's unseen-data score
    assert c["chosen_if_training_overlap_ignored"] == "crepe"
    assert c["O2_tracker_pass"]


def test_selection_without_official_data_is_marked(tmp_path):
    _write_condition(tmp_path, "synthetic", [dict(tracker="a", rpa=0.95, rtf=0.1), dict(tracker="b", rpa=0.9, rtf=0.1)])
    c = B.choose_tracker(tmp_path)
    assert c["tracker"] == "a" and not c["official"] and not c["O2_tracker_pass"]


def test_harness_end_to_end(tmp_path):
    s = B.run("synthetic", None, ["yin"], limit=2, out_dir=tmp_path, n_boot=50, audit=True)
    assert len(s) == 1
    row = s.iloc[0]
    for col in ("rpa", "rpa_lo", "rpa_hi", "rca", "rpa_25", "rpa_10", "oa", "vr", "vfa", "median_err_cents", "octave_err",
                "rpa_sing", "ref_sing_share", "rtf", "peak_mb"):
        assert col in s.columns
    assert 0 <= row.rpa <= 1 and row.rtf > 0
    assert row.ref_sing_share == 1.0 and row.rpa_sing == pytest.approx(row.rpa)  # synthetic voices are all in range
    run_info = json.loads((tmp_path / "synthetic_run.json").read_text())
    assert run_info["machine"]["threads"] == 4 and run_info["machine"]["cpu"]
    assert run_info["search_range_hz"] == [32.7, 1975.5]
    assert (tmp_path / "synthetic_summary.md").read_text().startswith("| Tracker")
    rep = B.alignment_report(tmp_path / "synthetic_alignment.csv")
    assert list(rep.tracker) == ["yin"]


def test_pesto_onnx_matches_pytorch():
    from adaptitune.pitch.trackers import PestoOnnxTracker, PestoTracker

    if not (PestoTracker.available() and PestoOnnxTracker.available()):
        pytest.skip("needs pesto, torch, onnxruntime and an exported model (scripts/export_onnx.py)")
    from adaptitune.data.synth import SungNote, sing

    s = sing([SungNote(57, 0.7, vibrato_cents=40), SungNote(62, 0.9), SungNote(64, 0.6, scoop_cents=-80)], sr=16_000, seed=2)
    a = PestoTracker().track(s.samples, s.sr)
    b = PestoOnnxTracker().track(s.samples, s.sr)
    v = (a.raw_f0 > 0) & (b.raw_f0 > 0) & (s.f0[: len(a.f0)] > 0)
    diff = np.abs(1200 * np.log2(a.raw_f0[v] / b.raw_f0[v]))
    assert np.median(diff) < 1.0


def test_every_ranged_tracker_gets_the_same_search_range():
    from adaptitune.pitch.trackers import BENCH_FMAX, BENCH_FMIN, REGISTRY

    for name in ("pyin", "yin", "fcpe", "crepe"):
        if not REGISTRY[name].available():
            continue
        trk = B.bench_tracker(name)
        assert (trk.fmin, trk.fmax) == (BENCH_FMIN, BENCH_FMAX), name


def test_wide_range_finds_a_low_bass_note():
    """A 50 Hz tone (a bass stem in MDB-stem-synth) is below the app's 65 Hz floor but inside
    the benchmark range."""
    sr = 16_000
    x = 0.4 * np.sin(2 * np.pi * 50.0 * np.arange(2 * sr) / sr)
    tr = B.bench_tracker("pyin").track(x, sr)
    mid = tr.raw_f0[len(tr.raw_f0) // 4: 3 * len(tr.raw_f0) // 4]
    assert np.median(np.abs(1200 * np.log2(mid / 50.0))) < 20


def test_long_input_is_tracked_in_segments_with_the_same_result():
    from adaptitune.pitch import trackers as T

    sr = 16_000
    t = np.arange(int(70 * sr)) / sr
    f = 220 * 2 ** (np.sin(2 * np.pi * 0.1 * t) / 2)  # a slow +-6 semitone glide over 70 s
    x = 0.3 * np.sin(2 * np.pi * np.cumsum(f) / sr)
    trk = T.YinTracker()
    whole, seg = trk._track(x, sr, 0.01), trk._track_segmented(x, sr, 0.01)
    assert np.allclose(seg.times, whole.times) and np.allclose(seg.f0, whole.f0)


def test_segments_bound_the_input_and_cover_every_frame_once():
    from adaptitune.pitch import trackers as T

    seen = []

    class Probe(T.Tracker):
        name = "probe"

        def _track(self, y, sr, hop_s):
            seen.append(len(y))
            n = len(y) // int(round(sr * hop_s))
            return T.PitchTrack(np.arange(n) * hop_s, np.full(n, 100.0), np.ones(n), hop_s, raw_f0=np.full(n, 100.0))

    sr = 16_000
    pt = Probe()._track_segmented(np.zeros(int(95.3 * sr)), sr, 0.01)
    assert len(seen) == 4 and max(seen) <= (T.SEGMENT_S + 2 * T.SEGMENT_CONTEXT_S) * sr
    assert len(pt.times) == 9530 and np.allclose(np.diff(pt.times), 0.01) and pt.times[0] == 0


def test_shards_merge_to_the_unsharded_result(tmp_path):
    from adaptitune.bench import trackers as B

    full = B.run("synthetic", None, ["yin"], limit=8, out_dir=tmp_path / "full", n_boot=50)
    for k in range(2):
        B.run("synthetic", None, ["yin"], limit=8, out_dir=tmp_path / f"s{k}", n_boot=50, shard=f"{k}/2")
    merged = B.merge_shards([tmp_path / "s0", tmp_path / "s1"], tmp_path / "m", "synthetic", n_boot=50)
    for col in ("items", "clusters", "rpa", "rca", "oa", "rpa_pooled", "octave_err"):
        assert merged[col].iloc[0] == pytest.approx(full[col].iloc[0])
    with pytest.raises(ValueError):  # a missing shard is an error, not a smaller benchmark
        B.merge_shards([tmp_path / "s0"], tmp_path / "m2", "synthetic")


def test_auto_is_the_tracker_o2_chose():
    """The app's default tracker is the one the O2 selection rule chose."""
    from adaptitune.pitch.trackers import APP_TRACKER, REGISTRY, get_tracker

    if not REGISTRY[APP_TRACKER].available():
        pytest.skip(f"{APP_TRACKER} is not installed")
    assert get_tracker("auto").name == APP_TRACKER == "fcpe"
