import numpy as np
import pytest

from adaptitune import metrics as M
from adaptitune.correction.planner import plan
from adaptitune.correction.settings import FIXED_CLASSIC, REFERENCE, CorrectionSettings
from adaptitune.correction.shifter import PSOLA, WORLD, QualityConfig, shift
from adaptitune.music import Key
from adaptitune.pipeline import Corrector
from adaptitune.pitch.trackers import get_tracker

C = Key(0, "major")


@pytest.fixture(scope="module")
def tracker():
    return get_tracker("yin")


@pytest.mark.parametrize("route", [WORLD, PSOLA])
def test_shifters_reach_target(detuned, truth_track, tracker, route):
    p = plan(truth_track, CorrectionSettings(1.0, 5.0, "in_key", 1.0), C)
    out = shift(detuned.samples, detuned.sr, p, truth_track.voicing, QualityConfig(force=route))
    tr = tracker.track(out.samples, out.sr)
    scores = M.score_notes(M.contour_midi(tr.f0), tr.hop_s, detuned.notes)
    assert np.median([abs(s.error_cents) for s in scores]) <= 10
    assert all(s.route in (route, "untouched") for s in out.segments)


def _lpc_envelope_db(x, sr):
    import librosa

    y = librosa.resample(x, orig_sr=sr, target_sr=10_000)
    a = librosa.lpc(y * np.hanning(len(y)), order=12)
    w = np.linspace(200, 4500, 200)
    z = np.exp(-2j * np.pi * w / 10_000)
    resp = 1 / np.abs(np.polyval(a[::-1], z))
    return 20 * np.log10(resp / resp.max())


@pytest.mark.parametrize("route", [WORLD, PSOLA])
def test_formants_preserved(route):
    """Shift a sustained vowel up 4 semitones: the LPC envelope must stay much closer to the
    original than a naive resample ("chipmunk") shift does."""
    from adaptitune.audio import resample
    from adaptitune.data.synth import SungNote, sing
    from adaptitune.pitch.trackers import PitchTrack

    s = sing([SungNote(57, 1.2, vowel="a")], seed=4, breath=0.02)
    tr = PitchTrack(np.arange(len(s.f0)) * s.hop_s, s.f0, (s.f0 > 0).astype(float), s.hop_s)
    p = plan(tr, CorrectionSettings(0.0), C)
    p.correction = np.where(np.isnan(p.midi), 0.0, 4.0)
    out = shift(s.samples, s.sr, p, tr.voicing, QualityConfig(force=route)).samples
    chip = resample(s.samples, s.sr, int(s.sr / 2 ** (4 / 12)))
    seg = slice(int(0.5 * s.sr), int(1.0 * s.sr))
    ref = _lpc_envelope_db(s.samples[seg], s.sr)
    d_shift = np.mean(np.abs(_lpc_envelope_db(out[seg], s.sr) - ref))
    d_chip = np.mean(np.abs(_lpc_envelope_db(chip[int(0.45 * s.sr):int(0.85 * s.sr)], s.sr) - ref))
    assert d_shift < 0.6 * d_chip, (d_shift, d_chip)


def test_pipeline_end_to_end(detuned, tracker):
    c = Corrector(tracker)
    an = c.analyze(detuned.samples, detuned.sr, key="C major")
    res = c.correct(an, REFERENCE)
    assert res.accuracy == 1.0
    assert 0.6 <= res.natural.score <= 1.0
    rows = res.note_report()
    assert len(rows) == 5
    assert all(abs(r["after_cents"]) <= 15 for r in rows)
    assert max(abs(r["before_cents"]) for r in rows) >= 30
    # cached analysis: a second setting only costs plan + synthesis
    res2 = c.correct(an, FIXED_CLASSIC, score=False)
    assert res2.seconds["shift"] < 2.0


def test_untouched_audio_passes_through(detuned, tracker):
    c = Corrector(tracker)
    an = c.analyze(detuned.samples, detuned.sr, key="C major")
    res = c.correct(an, CorrectionSettings(strength=0.0), score=False)
    assert np.allclose(res.audio.samples, an.audio.samples)


def test_rtf_command_writes_its_result(detuned, tmp_path):
    """`adaptitune rtf FILE --out` (what scripts/o2_laptop_speed.py runs for O5) times the whole
    correction path and writes the stages, the routing and the thread count."""
    import json

    from adaptitune.audio import Audio, save
    from adaptitune.cli import main

    wav = tmp_path / "take.wav"
    save(wav, Audio(detuned.samples, detuned.sr), "PCM_16")
    out = tmp_path / "rtf.json"
    main(["rtf", str(wav), "--tracker", "yin", "--repeats", "1", "--out", str(out)])
    r = json.loads(out.read_text())
    assert r["tracker"] == "yin" and r["rtf"] > 0 and r["threads"] >= 1
    assert set(r["stages_rtf"]) == {"analysis", "plan", "shift"}
    assert abs(sum(r["route_shares"].values()) - 1) < 1e-6


def test_world_f0_follows_the_tracker_where_dio_strays():
    """Frames where WORLD's own F0 is unvoiced or half an octave or more from the tracker's take
    the tracker's value; ordinary disagreement and frames the tracker calls unvoiced are left."""
    from adaptitune.correction.shifter import _fix_world_f0

    hint = np.array([200.0, 200.0, 200.0, 200.0, 0.0, 200.0])
    world = np.array([201.0, 400.0, 0.0, 150.0, 300.0, 100.0])
    assert _fix_world_f0(world, hint).tolist() == [False, True, True, False, False, True]


def test_world_analysis_keeps_to_the_hint(detuned, truth_track):
    """Even with a hint an octave above the truth, WORLD's F0 on hinted frames ends up within half
    an octave of the hint, so WORLD can never resynthesize a note an octave away from the plan."""
    from adaptitune.audio import SYNTH_SR, resample
    from adaptitune.correction.shifter import _world_analyze

    y = resample(detuned.samples, detuned.sr, SYNTH_SR)
    hint = truth_track.f0 * 2.0
    f0 = _world_analyze(y, SYNTH_SR, truth_track.hop_s * 1000, hint)[0]
    n = min(len(f0), len(hint))
    v = hint[:n] > 0
    assert v.sum() > 100
    assert np.all(f0[:n][v] > 0)
    assert np.max(np.abs(1200 * np.log2(f0[:n][v] / hint[:n][v]))) <= 600 + 1e-6
    # and with the true F0 as the hint, it stays WORLD's own estimate: within a few cents of the truth
    f0 = _world_analyze(y, SYNTH_SR, truth_track.hop_s * 1000, truth_track.f0)[0]
    v = truth_track.f0[:n] > 0
    assert np.median(np.abs(1200 * np.log2(f0[:n][v] / truth_track.f0[:n][v]))) < 10
