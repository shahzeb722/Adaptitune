"""O2 key estimation: profiles, tuning robustness, scoring helpers, the labelling workflow and
the Essen ground-truth helpers."""

import csv

import numpy as np
import pytest

from adaptitune.bench import key_labels as KL
from adaptitune.bench import keys as KB
from adaptitune.data.synth import SungNote, sing
from adaptitune.music import Key, midi_to_hz, tuning_offset_cents
from adaptitune.pitch import key as K


def test_profiles_match_music21():
    d = pytest.importorskip("music21.analysis.discrete")
    pairs = {"krumhansl": d.KrumhanslKessler, "temperley": d.TemperleyKostkaPayne, "bellman": d.BellmanBudge,
             "aarden": d.AardenEssen, "simple": d.SimpleWeights}
    for name, cls in pairs.items():
        maj, mnr = K.PROFILES[name]
        assert np.allclose(maj, cls().getWeights("major")), name
        assert np.allclose(mnr, cls().getWeights("minor")), name


def _melody_f0(midis, durs, cents=0.0, hop=0.005):
    f0 = []
    for m, d in zip(midis, durs):
        f0 += [midi_to_hz(m + cents / 100)] * int(d / hop) + [0.0] * int(0.06 / hop)
    return np.array(f0, float)


TUNE = [67, 64, 64, 65, 62, 62, 60, 62, 64, 65, 67, 67, 67, 67, 64, 64, 65, 62, 62, 60, 64, 67, 67, 60]  # C major
DURS = [0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.5, 1.5]


def test_tuning_offset_estimate():
    m = np.array([60.3, 62.31, 64.28, 65.3, 67.29])
    assert tuning_offset_cents(m) == pytest.approx(29.6, abs=0.5)
    assert tuning_offset_cents(m - 0.6) == pytest.approx(-30.4, abs=0.5)


@pytest.mark.parametrize("cents", [0, -45, 45])
def test_global_tuning_is_estimated(cents):
    res = K.analyze_key(_melody_f0(TUNE, DURS, cents))
    assert res.key == Key(0, "major")
    assert res.tuning_cents == pytest.approx(cents, abs=3)


def _scattered_f0(offset, sd, seed, hop=0.005):
    rng = np.random.default_rng(seed)
    f0 = []
    for m, d in zip(TUNE, DURS):
        f0 += [midi_to_hz(m + (offset + rng.normal(0, sd)) / 100)] * int(d / hop) + [0.0] * int(0.06 / hop)
    return np.array(f0, float)


def test_tuning_removal_rescues_a_singer_near_the_half_semitone():
    """A singer about 45 cents off, with ordinary note-to-note scatter, puts many notes nearer
    the wrong semitone; removing the global tuning first keeps the key right."""
    on = off = n = 0
    for offset in (-45, 45):
        for seed in range(30):
            f0 = _scattered_f0(offset, 12, seed)
            on += K.analyze_key(f0).key == Key(0, "major")
            off += K.analyze_key(f0, cfg=K.KeyConfig(tuning=False)).key == Key(0, "major")
            n += 1
    assert on / n >= 0.95
    assert off / n <= on / n - 0.2


def test_confidence_ignores_the_relative_key():
    res = K.analyze_key(_melody_f0(TUNE, DURS))
    assert 0 < res.key_confidence <= res.confidence <= 1


def test_estimate_key_on_synthetic_audio():
    from adaptitune.pitch.trackers import get_tracker

    s = sing([SungNote(m, d * 0.7, vibrato_cents=30) for m, d in zip(TUNE, DURS)], sr=16_000, seed=0)
    tr = get_tracker("pyin").track(s.samples, s.sr)
    key, conf = K.estimate_key(tr.f0, tr.voicing, tr.hop_s)
    assert key == Key(0, "major") and conf > 0


def test_mirex_score_and_fit_key():
    C, G, F, Am, Cm = Key(0, "major"), Key(7, "major"), Key(5, "major"), Key(9, "minor"), Key(0, "minor")
    assert K.mirex_key_score(C, C) == 1.0
    assert K.mirex_key_score(G, C) == 0.5 and K.mirex_key_score(F, C) == 0.5
    assert K.mirex_key_score(Am, C) == 0.3
    assert K.mirex_key_score(Cm, C) == 0.2
    assert K.mirex_key_score(Key(1, "major"), C) == 0.0
    ranked = [(G, 0.9), (C, 0.8), (F, 0.7)]
    assert K.fit_key(ranked, {0, 4, 5}) == C  # G major lacks F natural
    assert K.fit_key(ranked, {7}) == G


def test_cohens_kappa_textbook_example():
    # 50 items: both yes 20, A yes/B no 5, A no/B yes 10, both no 15 -> kappa 0.4
    a = ["y"] * 20 + ["y"] * 5 + ["n"] * 10 + ["n"] * 15
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert KL.cohens_kappa(a, b) == pytest.approx(0.4)
    assert KL.cohens_kappa(a, a) == 1.0


def test_labelling_workflow_end_to_end(tmp_path):
    from adaptitune.audio import Audio, save

    (tmp_path / "audio").mkdir()
    melodies = {"ex001": (0, TUNE), "ex002": (7, TUNE), "ex003": (2, TUNE)}
    for ex, (shift, tune) in melodies.items():
        s = sing([SungNote(m + shift, d * 0.7) for m, d in zip(tune, DURS)], sr=24_000, seed=1)
        save(tmp_path / "audio" / f"{ex}.wav", Audio(s.samples, s.sr))
    r1 = [("ex001", "C major"), ("ex002", "G major"), ("ex003", "D major")]
    r2 = [("ex001", "C major"), ("ex002", "E minor"), ("ex003", "none")]
    for name, rows in (("r1.csv", r1), ("r2.csv", r2)):
        with open(tmp_path / name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["excerpt", "path", "key", "confidence", "notes", "rater", "labelled_at"])
            for ex, k in rows:
                w.writerow([ex, f"audio/{ex}.wav", k, 3, "", name, ""])
    agree = KL.merge_raters(tmp_path / "r1.csv", tmp_path / "r2.csv", tmp_path / "merged.csv")
    assert agree["excerpts_in_both"] == 3 and agree["raw_agreement"] == pytest.approx(1 / 3)
    assert agree["scale_agreement"] == pytest.approx(2 / 3)  # G major vs E minor share a scale
    assert agree["to_adjudicate"] == ["ex002", "ex003"]
    # before adjudication only the agreed excerpt is scored; then the raters settle the rest
    res = KL.evaluate_key_labels(tmp_path / "merged.csv", "pyin", tmp_path)
    assert res["scored"] == 1 and res["unresolved_disagreements"] == 2
    with open(tmp_path / "merged.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["adjudicated"] = {"ex001": "C major", "ex002": "G major", "ex003": "D major"}[r["excerpt"]]
    with open(tmp_path / "merged.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    res = KL.evaluate_key_labels(tmp_path / "merged.csv", "pyin", tmp_path, tmp_path / "per_excerpt.csv")
    assert res["scored"] == 3 and res["accuracy_exact"] == 1.0 and res["O2_key_pass"]
    assert res["rater_cohens_kappa"] < 1


def test_essen_ground_truth_helpers():
    assert KB.parse_key_field("G") == (7, None)
    assert KB.parse_key_field("Bb") == (10, None)
    assert KB.parse_key_field("Dm") == (2, "minor")
    assert KB.parse_key_field("Ador") == (9, "minor")
    assert KB.parse_key_field("Gmix") == (7, "major")
    notes = [(60, 1.0), (64, 2.0), (63, 0.5), (None, 1.0)]
    assert KB.mode_by_third(notes, 0) == "major"
    assert KB.mode_by_third([(60, 1.0), (63, 1.0)], 0) == "minor"
    assert KB.mode_by_third([(60, 1.0), (67, 1.0)], 0) is None
    assert KB.wilson(75, 100)[0] < 0.75 < KB.wilson(75, 100)[1]


def test_essen_rendering_keeps_the_key():
    """Rendering transposes the song and its key together, and an excerpt stops before the
    final note."""
    song = dict(id="x/1", file="x", region="x", tonic=7, mode="major", notes=[[67, 1], [71, 1], [74, 1], [None, 1], [72, 1], [71, 1], [69, 1], [67, 2]] * 3)
    items = KB.make_items([song], ("in_tune",), ("full", "excerpt"), seed=0)
    full = next(it for it in items if it["kind"] == "full")
    est = K.analyze_key(np.zeros(1))  # empty contour must not crash
    assert est.confidence == 0
    shift = (full["ref"].tonic - 7) % 12
    assert full["ref"].mode == "major"
    assert full["pc_dur"][(7 + shift) % 12] > 0  # transposed melody contains the transposed tonic
    assert all(it["feat"]["centres"].size > 3 for it in items)


def test_mir1k_excerpts_one_per_song(tmp_path):
    """Consecutive clips of a song are joined (vocal channel only) up to the length cap, every
    excerpt is a different song, and singers are spread round-robin."""
    import json

    import soundfile as sf

    sr = 16_000
    wav = tmp_path / "MIR-1K" / "Wavfile"
    wav.mkdir(parents=True)
    t = np.arange(int(7 * sr)) / sr
    for singer in ("abjones", "amy", "bobon"):
        for song in (1, 2, 3):
            for clip in (1, 2, 3, 4, 5, 6):
                voc = 0.2 * np.sin(2 * np.pi * (200 + 10 * song) * t)
                acc = 0.2 * np.sin(2 * np.pi * 90 * t)
                sf.write(wav / f"{singer}_{song}_{clip:02d}.wav", np.stack([acc, voc], 1), sr)
    sf.write(wav / "short_1_01.wav", np.zeros((sr, 2)), sr)  # too short to use
    df = KL.select_mir1k_excerpts(tmp_path / "MIR-1K", tmp_path / "ex", n=7, seed=0)
    assert len(df) == 7 and df.song.is_unique
    assert set(df.singer) == {"abjones", "amy", "bobon"} and df.singer.value_counts().max() == 3
    assert (df.duration <= 30.0).all() and (df.duration >= 12.0).all()
    meta = json.loads((tmp_path / "ex" / "excerpts.json").read_text())
    first = meta["excerpts"][0]
    assert first["clips"][0].endswith("_01") and len(first["clips"]) == 4  # 4 x 7 s + 3 gaps fits under 30 s
    y, ysr = sf.read(tmp_path / "ex" / first["path"])
    assert ysr == sr and y.ndim == 1
    spec = np.abs(np.fft.rfft(y[: 7 * sr]))
    assert np.argmax(spec) / 7 > 150  # the vocal tone, not the 90 Hz accompaniment
    tpl = (tmp_path / "ex" / "key_labels_template.csv").read_text().splitlines()
    assert tpl[0].startswith("excerpt,path") and len(tpl) == 8


def test_vocalset_song_id_for_clustering():
    from collections import namedtuple

    R = namedtuple("R", "take context source_path clean")
    a = R("vocalset__female3__excerpts__vibrato__f3_caro_vibrato", "excerpts", "", "clean/x.wav")
    b = R("vocalset__male2__excerpts__straight__m2_caro_straight", "excerpts", "/data/VocalSet/FULL/male2/excerpts/straight/m2_caro_straight.wav", "")
    c = R("vocalset__male2__scales__straight__m2_scales_straight_a", "scales", "", "")
    assert KL._song_of(a) == KL._song_of(b) == "vocalset/caro"  # the same tune, different singers
    assert KL._song_of(c) == c.take  # an exercise is its own item
