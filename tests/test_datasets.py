"""Dataset loaders against small fake copies of the real folder layouts, and the O1 split planner."""

import numpy as np
import pytest
import soundfile as sf

from adaptitune.data import datasets as D

SR = 16_000


def _tone(f, seconds=1.0, sr=SR, amp=0.3):
    t = np.arange(int(seconds * sr)) / sr
    return amp * np.sin(2 * np.pi * f * t)


def test_mir1k_layout(tmp_path):
    (tmp_path / "Wavfile").mkdir()
    (tmp_path / "PitchLabel").mkdir()
    voc = _tone(220.0)
    acc = 0.3 * np.random.default_rng(0).standard_normal(SR)
    sf.write(tmp_path / "Wavfile" / "abjones_1_01.wav", np.stack([acc, voc], axis=1), SR)
    semis = np.full(49, 57.0)
    semis[:3] = 0
    np.savetxt(tmp_path / "PitchLabel" / "abjones_1_01.pv", semis, fmt="%.4f")
    (item,) = list(D.mir1k(tmp_path))
    assert item.singer == "abjones" and item.id == "mir1k/abjones_1_01"
    assert item.ref_times[0] == pytest.approx(0.02) and item.ref_times[1] - item.ref_times[0] == pytest.approx(0.02)
    assert item.ref_f0[0] == 0 and item.ref_f0[5] == pytest.approx(220.0, rel=1e-6)
    assert np.corrcoef(item.audio.samples, voc)[0, 1] > 0.999  # right channel = vocal
    (mixed,) = list(D.mir1k(tmp_path, mix_snr=0.0))
    resid = mixed.audio.samples - voc
    assert np.mean(resid**2) == pytest.approx(np.mean(voc**2), rel=1e-3)  # 0 dB vocal-to-accompaniment (16-bit file)


def test_mdb_layout_with_and_without_header(tmp_path):
    (tmp_path / "audio_stem").mkdir()
    (tmp_path / "annotation_stem").mkdir()
    for k, header in ((1, False), (2, True)):
        stem = f"Artist_Track_STEM_0{k}.RESYN"
        sf.write(tmp_path / "audio_stem" / f"{stem}.wav", _tone(330.0, sr=44_100), 44_100)
        t = np.arange(100) * 128 / 44_100
        rows = "\n".join(f"{a},{330.0}" for a in t)
        (tmp_path / "annotation_stem" / f"{stem}.csv").write_text(("time,f0\n" if header else "") + rows + "\n")
    items = list(D.mdb_stem_synth(tmp_path))
    assert len(items) == 2 and {it.singer for it in items} == {"Artist_Track"}
    assert items[0].audio.sr == 16_000 and np.allclose(items[1].ref_f0, 330.0)


def test_vocalset_layout_any_root_and_deduplicated(tmp_path):
    full = tmp_path / "VocalSet11" / "FULL"
    files = {
        full / "female1" / "arpeggios" / "straight" / "f1_arpeggios_straight_a.wav": True,
        full / "male2" / "excerpts" / "vibrato" / "m2_caro_vibrato.wav": True,
        full / "female1" / "long_tones" / "breathy" / "f1_long_breathy_a.wav": False,  # not a clean technique
        # the same recording again in another of VocalSet's layouts must not count twice
        tmp_path / "VocalSet11" / "data_by_technique" / "female1" / "arpeggios" / "straight" / "f1_arpeggios_straight_a.wav": False,
    }
    for p in files:
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(p, _tone(200.0, 0.5), SR)
    found = D.vocalset_files(tmp_path)
    assert sorted(f["singer"] for f in found) == ["female1", "male2"]
    by = {f["singer"]: f for f in found}
    assert by["female1"]["gender"] == "F" and by["male2"]["gender"] == "M"
    assert by["female1"]["context"] == "arpeggios" and by["female1"]["technique"] == "straight"
    assert by["male2"]["context"] == "excerpts" and by["male2"]["technique"] == "vibrato"
    items = list(D.vocalset(tmp_path, sr=24_000))
    assert len(items) == 2 and items[0].audio.sr == 24_000


def test_plan_split_guarantees_test_hours():
    rng = np.random.default_rng(1)
    hours = {f"female{i}": float(rng.uniform(0.15, 0.3)) for i in range(1, 10)}
    hours |= {f"male{i}": float(rng.uniform(0.15, 0.3)) for i in range(1, 12)}
    groups = {k: "F" if k.startswith("female") else "M" for k in hours}
    r = D.plan_split(hours, groups, min_test_hours=2.0)
    a = r["assignment"]
    assert r["feasible"] and r["hours"]["test"] >= 2.0
    assert set(a) == set(hours) and set(a.values()) == {"train", "val", "test"}
    test = [s for s in a if a[s] == "test"]
    dev = [s for s in a if a[s] != "test"]
    assert not set(test) & set(dev)
    f = sum(groups[s] == "F" for s in test)
    assert abs(f - (len(test) - f)) <= 1  # gender-balanced test set
    assert {groups[s] for s in a if a[s] == "val"} == {"F", "M"}
    assert r == D.plan_split(hours, groups, min_test_hours=2.0)  # deterministic


def test_plan_split_reports_infeasible():
    hours = {f"s{i}": 0.1 for i in range(6)}
    r = D.plan_split(hours, None, min_test_hours=2.0)
    assert not r["feasible"]
