import numpy as np
import pytest

from adaptitune import metrics as M
from adaptitune.data.synth import SungNote, sing


def test_vibrato_estimate():
    s = sing([SungNote(62, 1.5, vibrato_rate=5.5, vibrato_cents=50)], seed=0, jitter_cents=0)
    m = M.contour_midi(s.f0)
    a, b = int(0.45 / s.hop_s), int(1.6 / s.hop_s)
    rate, extent = M.vibrato(m[a:b], s.hop_s)
    assert rate == pytest.approx(5.5, rel=0.1)
    assert extent == pytest.approx(50, rel=0.2)


def test_in_key_accuracy():
    assert M.in_key_accuracy(np.array([60.1, 61.3]), np.array([60, 61])) == 0.5


def test_mcd_zero_for_identical(detuned):
    assert M.mel_cepstral_distortion(detuned.samples, detuned.samples, detuned.sr) == pytest.approx(0, abs=1e-6)


def test_new_jumps():
    a = np.full(100, 60.0)
    b = a.copy()
    b[50:] += 1.0
    assert M.new_pitch_jumps(a, a, 0.005) == 0
    assert M.new_pitch_jumps(a, b, 0.005) > 0


def test_rpa_perfect():
    t = np.arange(100) * 0.01
    f = np.full(100, 220.0)
    assert M.raw_pitch_accuracy(t, f, t, f)["rpa"] == 1.0
