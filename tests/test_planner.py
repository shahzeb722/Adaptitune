import numpy as np
import pytest

from adaptitune import metrics as M
from adaptitune.correction.planner import plan, segment_notes
from adaptitune.correction.settings import (
    RETUNE_MS_LEVELS,
    CorrectionSettings,
    action_grid,
    action_index,
    action_matrix,
)
from adaptitune.music import Key

C = Key(0, "major")


def test_action_grid_is_504():
    g = action_grid()
    assert len(g) == 504 == len(set(g))
    assert action_matrix().shape == (504, 4)
    assert action_index(g[123]) == 123


def test_settings_normalized_roundtrip():
    s = CorrectionSettings(0.5, RETUNE_MS_LEVELS[3], "chromatic", 2 / 3)
    back = CorrectionSettings.from_normalized(s.normalized())
    assert back.strength == pytest.approx(0.5)
    assert back.retune_ms == pytest.approx(s.retune_ms)
    assert back.snap == "chromatic"


def test_segments_match_sung_notes(detuned):
    m = M.contour_midi(detuned.f0)
    segs = segment_notes(m, detuned.hop_s)
    assert len(segs) == len(detuned.notes)


def test_split_legato_notes_without_gap():
    from adaptitune.data.synth import SungNote, sing

    s = sing([SungNote(60, 0.5, gap_after=0), SungNote(64, 0.5, gap_after=0), SungNote(62, 0.5)], seed=3)
    segs = segment_notes(M.contour_midi(s.f0), s.hop_s)
    assert len(segs) == 3


def test_full_strength_hits_targets(truth_track, detuned):
    flat = plan(truth_track, CorrectionSettings(1.0, 5.0, "in_key", 0.0), C)
    assert [n.target for n in flat.notes] == [60, 62, 64, 65, 67]
    scores = M.score_notes(flat.corrected_midi, flat.hop_s, detuned.notes)
    assert max(abs(s.error_cents) for s in scores) < 2
    # with vibrato kept, the note-centre median wobbles a few cents but stays well under 25
    keep = plan(truth_track, CorrectionSettings(1.0, 5.0, "in_key", 1.0), C)
    scores = M.score_notes(keep.corrected_midi, keep.hop_s, detuned.notes)
    assert max(abs(s.error_cents) for s in scores) < 10


def test_zero_strength_is_identity(truth_track):
    p = plan(truth_track, CorrectionSettings(0.0, 20.0, "in_key", 0.0), C)
    assert np.allclose(p.correction, 0)


def test_vibrato_preservation(truth_track, detuned):
    keep = plan(truth_track, CorrectionSettings(1.0, 5.0, "in_key", 1.0), C)
    flat = plan(truth_track, CorrectionSettings(1.0, 5.0, "in_key", 0.0), C)
    orig = M.score_notes(M.contour_midi(detuned.f0), detuned.hop_s, detuned.notes)
    k = M.score_notes(keep.corrected_midi, keep.hop_s, detuned.notes)
    f = M.score_notes(flat.corrected_midi, flat.hop_s, detuned.notes)
    for o, a, b in zip(orig, k, f):
        if o.vib_extent > 20:
            assert a.vib_extent >= 0.7 * o.vib_extent  # O3 criterion
            assert abs(a.vib_rate - o.vib_rate) <= 0.1 * o.vib_rate
            assert b.vib_extent < 0.3 * o.vib_extent


def test_slow_retune_corrects_less_of_the_onset(truth_track):
    fast = plan(truth_track, CorrectionSettings(1.0, 5.0, "in_key", 1.0), C)
    slow = plan(truth_track, CorrectionSettings(1.0, 400.0, "in_key", 1.0), C)
    n = fast.notes[1]  # the scooped note
    first = slice(n.start, n.start + 10)
    assert np.mean(np.abs(slow.correction[first])) < np.mean(np.abs(fast.correction[first]))


def test_chromatic_vs_in_key():
    from adaptitune.data.synth import SungNote, sing
    from adaptitune.pitch.trackers import PitchTrack

    s = sing([SungNote(61, 0.6, offset_cents=-40)], seed=2)  # sung at 60.6
    tr = PitchTrack(np.arange(len(s.f0)) * s.hop_s, s.f0, (s.f0 > 0).astype(float), s.hop_s)
    chrom = plan(tr, CorrectionSettings(1.0, 5.0, "chromatic", 0.0), C)
    inkey = plan(tr, CorrectionSettings(1.0, 5.0, "in_key", 0.0), C)
    assert chrom.notes[0].target == 61
    assert inkey.notes[0].target == 60  # C# is out of C major; C is nearest in key


def test_low_voicing_left_untouched(truth_track):
    from adaptitune.pitch.trackers import PitchTrack

    tr = PitchTrack(truth_track.times, truth_track.f0, np.full(len(truth_track.f0), 0.1), truth_track.hop_s)
    p = plan(tr, CorrectionSettings(), C)
    assert np.allclose(p.correction, 0)
