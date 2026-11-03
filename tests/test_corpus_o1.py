"""O1 detuned corpus: paired WORLD/PSOLA copies, per-frame labels, the QC gate, independent
verification, integrity checks and reproducibility."""

import json

import numpy as np
import pytest

from adaptitune.data import detune as T


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    out = tmp_path_factory.mktemp("corpus")
    T.build_corpus(out, "synthetic", n_singers=3, takes_per_singer=1, seed=5)
    return out


def test_layout_and_schema(corpus):
    for name in ("labels.jsonl", "manifest.csv", "split.json", "build_info.json", "checksums.sha256", "stats.json"):
        assert (corpus / name).exists(), name
    man = T.load_manifest(corpus)
    assert set(man["route"]) == {"world", "psola"}
    assert len(man) == 2 * man["take"].nunique()  # one copy per shifter for every take
    for _, r in man.iterrows():
        assert (corpus / r["clean"]).exists() and (corpus / r["detuned"]).exists() and (corpus / r["labels"]).exists()
    labels = T.load_labels(corpus)
    assert labels and all(lb.schema == 2 for lb in labels)
    # every labelled note appears once per route, with identical injected parameters
    by_note = {}
    for lb in labels:
        by_note.setdefault((lb.take, lb.index), []).append(lb)
    for pair in by_note.values():
        assert sorted(lb.route for lb in pair) == ["psola", "world"]
        assert len({(lb.offset_cents, lb.drift_cents, lb.scoop_cents, lb.vibrato_factor) for lb in pair}) == 1
    for lb in labels:
        assert 10 <= abs(lb.offset_cents) <= 100
        assert abs(lb.source_dev_cents) <= T.KEEP_WITHIN_CENTS + 1e-6
        assert lb.highly_detuned == (abs(lb.offset_cents) > T.HIGH_DETUNE_CENTS)
        assert lb.end - lb.start >= T.MIN_LABEL_NOTE_S - 1e-6
    # strict JSON (no NaN tokens)
    for line in (corpus / "labels.jsonl").read_text().splitlines():
        json.loads(line, parse_constant=lambda c: pytest.fail(f"non-standard JSON constant {c}"))


def test_sidecar_curves_match_labels(corpus):
    labels = [lb for lb in T.load_labels(corpus) if lb.route == "world"]
    take = labels[0].take
    z = np.load(corpus / "labels" / f"{take}.npz")
    inj, idx = z["injection_cents"], z["note_index"]
    assert float(z["hop_s"]) == T.HOP_S
    for lb in (lb for lb in labels if lb.take == take):
        a, b = int(round(lb.start / T.HOP_S)), int(round(lb.end / T.HOP_S))
        assert (idx[a:b] == lb.index).all()
        assert np.mean(inj[a:b]) == pytest.approx(lb.injected_mean_cents, abs=0.05)
    assert np.all(inj[idx < 0] == 0)  # unlabelled frames are untouched


def test_qc_gate_and_stats(corpus):
    s = T.corpus_stats(corpus, write=False)
    assert s["qc_pass_share"] >= 0.9
    assert s["qc_abs_residual_cents"]["median"] < 3
    assert s["O1_singers_disjoint"]
    assert set(s["qc_pass_share_by_route"]) == {"world", "psola"}
    for lb in T.load_labels(corpus):
        if lb.qc_pass:
            assert abs(lb.qc_residual_cents) <= T.QC_TOL_CENTS and lb.qc_coverage >= T.QC_MIN_COVERAGE
            assert lb.recovered_offset_cents == pytest.approx(lb.offset_cents + lb.qc_residual_cents, abs=0.02)
        else:
            assert lb.qc_reason in ("low_coverage", "offset_mismatch", "octave_error")


def test_independent_verification_recovers_every_kept_label(corpus):
    res = T.verify_labels(corpus, tracker="yin")
    assert res["independent_of_build_qc"]
    assert res["notes_checked"] == sum(lb.qc_pass for lb in T.load_labels(corpus))
    assert res["recovered_share"] == 1.0
    assert res["files_bad"] == [] and res["files_checked"] > 0


def test_verification_catches_a_wrong_label_and_a_changed_file(corpus, tmp_path):
    import shutil

    bad = tmp_path / "bad"
    shutil.copytree(corpus, bad)
    lb = next(lb for lb in T.load_labels(bad) if lb.qc_pass)
    z = dict(np.load(bad / "labels" / f"{lb.take}.npz"))
    a, b = int(round(lb.start / T.HOP_S)), int(round(lb.end / T.HOP_S))
    z["injection_cents"] = z["injection_cents"].copy()
    z["injection_cents"][a:b] += 40.0  # the stored curve no longer matches the audio
    np.savez_compressed(bad / "labels" / f"{lb.take}.npz", **z)
    res = T.verify_labels(bad, tracker="pyin")
    assert res["recovered_share"] < 1.0
    assert any(f["take"] == lb.take and f["index"] == lb.index for f in res["failures"])
    assert f"labels/{lb.take}.npz" in res["files_bad"]


def test_note_qc_reasons():
    n = 100
    mc = np.full(n, 60.0)
    inj = np.full(n, 30.0)
    assert T.note_qc(mc, mc + 0.30, inj, 0, n, 0.005)["qc_pass"]
    assert T.note_qc(mc, mc + 0.50, inj, 0, n, 0.005)["qc_reason"] == "offset_mismatch"
    assert T.note_qc(mc, mc + 12.3, inj, 0, n, 0.005)["qc_reason"] == "octave_error"
    holes = mc + 0.3
    holes[::2] = np.nan
    holes[1::4] = np.nan
    assert T.note_qc(mc, holes, inj, 0, n, 0.005)["qc_reason"] == "low_coverage"


def test_takes_are_reproducible_and_order_independent(corpus, tmp_path):
    """A take's detuning depends only on the seed and its id, not on which takes are built."""
    one = tmp_path / "one"
    T.build_corpus(one, "synthetic", n_singers=3, takes_per_singer=1, seed=5, limit=1)
    a = [lb for lb in T.load_labels(one)]
    b = [lb for lb in T.load_labels(corpus) if lb.take == a[0].take]
    assert [(x.index, x.route, x.offset_cents, x.qc_residual_cents) for x in a] == [(x.index, x.route, x.offset_cents, x.qc_residual_cents) for x in b]


def test_corrector_eval_shards_merge_to_the_unsharded_result(corpus, tmp_path):
    """O3 on a corpus: two shards joined give the same notes and summary as one run, and the
    output can be measured with a tracker other than the corrector's own."""
    from adaptitune.bench import corrector as BC

    full = BC.evaluate(corpus, "yin", split=None, out_dir=tmp_path / "full", measure_tracker="pyin")
    parts = [BC.evaluate(corpus, "yin", split=None, out_dir=tmp_path / f"s{k}", measure_tracker="pyin", shard=(k, 2)) for k in range(2)]
    assert all(p["takes"] >= 1 for p in parts)
    merged = BC.merge_shards([tmp_path / "s1", tmp_path / "s0"], tmp_path / "merged")
    assert merged["notes"] == full["notes"] and merged["takes"] == full["takes"]
    assert merged["median_abs_err_lt50"] == pytest.approx(full["median_abs_err_lt50"])
    # the corrector's own WORLD/PSOLA split is recorded per take and summarized
    assert 0 <= full["corrector_world_share"] <= 1
    assert merged["corrector_world_share"] == pytest.approx(full["corrector_world_share"])
    run = json.loads((tmp_path / "merged" / "run.json").read_text())
    assert run["tracker"] == "yin" and run["measure_tracker"] == "pyin"
    with pytest.raises(ValueError):
        BC.merge_shards([tmp_path / "s0"], tmp_path / "bad")
