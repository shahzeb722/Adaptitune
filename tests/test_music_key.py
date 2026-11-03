import numpy as np
import pytest

from adaptitune.data.synth import SungNote, sing
from adaptitune.music import Key, hz_to_midi, midi_to_hz, nearest_in_scale
from adaptitune.pitch.key import estimate_key


def test_hz_midi_roundtrip():
    assert hz_to_midi(440.0) == pytest.approx(69)
    assert midi_to_hz(60) == pytest.approx(261.6256, rel=1e-5)
    assert np.isnan(hz_to_midi(0.0))


@pytest.mark.parametrize("text,tonic,mode", [("A minor", 9, "minor"), ("Bb major", 10, "major"), ("F#m", 6, "minor"), ("C", 0, "major")])
def test_key_parse(text, tonic, mode):
    assert Key.parse(text) == Key(tonic, mode)


def test_nearest_in_scale():
    c = Key(0, "major")
    assert nearest_in_scale(60.6, c) == 60  # C# is not in C major; C is closer than D
    assert nearest_in_scale(61.6, c) == 62
    assert nearest_in_scale(60.6, None) == 61


def tonal_melody(rng, tonic, mode, n=24):
    """Random walk over the scale that leans on the tonic triad, like most real tunes."""
    k = Key(tonic, mode)
    triad = {tonic, (tonic + (4 if mode == "major" else 3)) % 12, (tonic + 7) % 12}
    pool = [m for m in range(55, 75) if m % 12 in k.pitch_classes]
    w = np.array([3.0 if m % 12 in triad else 1.0 for m in pool])
    mel = list(rng.choice(pool, size=n, p=w / w.sum()))
    root = min(m for m in pool if m % 12 == tonic)
    return [root] + mel + [root]


def test_key_estimation_accuracy_over_24_keys():
    """Same spirit as O2's key criterion (>= 75% correct); relative keys share a scale."""
    rng = np.random.default_rng(0)
    ok = 0
    for key in Key.all():
        mel = tonal_melody(rng, key.tonic, key.mode)
        s = sing([SungNote(int(m), 0.3, offset_cents=rng.uniform(-20, 20), gap_after=0.05) for m in mel], seed=key.tonic)
        est, _ = estimate_key(s.f0)
        ok += est == key
    assert ok / 24 >= 0.75, ok


