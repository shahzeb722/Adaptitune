import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore", category=UserWarning)

from adaptitune.data.synth import SungNote, sing  # noqa: E402
from adaptitune.pitch.trackers import PitchTrack  # noqa: E402

DETUNED = [
    SungNote(60, 0.6, offset_cents=35, vibrato_cents=40),
    SungNote(62, 0.5, offset_cents=-30, scoop_cents=-80),
    SungNote(64, 0.8, offset_cents=20, drift_cents=-30, vibrato_cents=50),
    SungNote(65, 0.5, offset_cents=-40),
    SungNote(67, 0.9, offset_cents=25, vibrato_cents=45),
]


@pytest.fixture(scope="session")
def detuned():
    return sing(DETUNED, seed=1)


@pytest.fixture(scope="session")
def truth_track(detuned):
    f0 = detuned.f0
    return PitchTrack(np.arange(len(f0)) * detuned.hop_s, f0, (f0 > 0).astype(float), detuned.hop_s, "truth")
