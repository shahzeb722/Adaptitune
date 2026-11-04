"""O2 key test on the Children's Song Dataset (docs/O2_PROTOCOL.md, B4): MIDI reading, the reference
keys, the excerpts and a full run on a fake CSD folder of synthetic singing."""

import json
import struct

import numpy as np
import pytest

from adaptitune.bench.key_csd import evaluate_csd, excerpt_window, find_csd, ks_key, read_midi_notes, reference_key, transposition
from adaptitune.music import Key

TWINKLE = [60, 60, 67, 67, 69, 69, 67, 65, 65, 64, 64, 62, 62, 60]  # C major, ends on C
MINOR = [69, 71, 72, 74, 76, 74, 72, 71, 69, 72, 71, 69]  # A minor with C natural, ends on A
QUARTER_S, LEAD_S, GAP_S = 0.6, 0.3, 0.06


def _varlen(v: int) -> bytes:
    out = [v & 0x7F]
    v >>= 7
    while v:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    return bytes(reversed(out))


def write_midi(path, pitches, division=480):
    """One note after another, a quarter note each at 100 bpm, after a 0.3 s lead-in. Every other
    note ends with a note-on of velocity 0 in running status, as many real files do."""
    tick_s = QUARTER_S / division
    ev = bytearray(_varlen(0) + b"\xff\x51\x03" + int(QUARTER_S * 1e6).to_bytes(3, "big"))
    delta = round(LEAD_S / tick_s)
    for i, m in enumerate(pitches):
        ev += _varlen(delta) + bytes([0x90, m, 90])
        ev += _varlen(division) + (bytes([0x80, m, 0]) if i % 2 else bytes([m, 0]))
        delta = round(GAP_S / tick_s)
    ev += _varlen(0) + b"\xff\x2f\x00"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"MThd" + struct.pack(">IHHH", 6, 0, 1, division) + b"MTrk" + struct.pack(">I", len(ev)) + bytes(ev))


def notes_of(pitches):
    t, out = LEAD_S, []
    for m in pitches:
        out.append((t, t + QUARTER_S, m))
        t += QUARTER_S + GAP_S
    return out


def test_midi_notes_are_read_in_seconds(tmp_path):
    write_midi(tmp_path / "x.mid", TWINKLE)
    got = read_midi_notes(tmp_path / "x.mid")
    assert [m for _, _, m in got] == TWINKLE
    np.testing.assert_allclose([(a, b) for a, b, _ in got], [(a, b) for a, b, _ in notes_of(TWINKLE)], atol=1e-6)


def test_reference_key_is_the_last_note_and_the_third():
    assert reference_key(notes_of(TWINKLE)) == Key(0, "major")
    assert reference_key(notes_of(MINOR)) == Key(9, "minor")
    assert reference_key(notes_of([p + 3 for p in MINOR])) == Key(0, "minor")
    assert ks_key(notes_of(TWINKLE)) == Key(0, "major")


def test_two_versions_differ_by_their_transposition():
    assert transposition(notes_of(TWINKLE), notes_of([p + 2 for p in TWINKLE])) == 2
    assert transposition(notes_of(MINOR), notes_of([p - 1 for p in MINOR][:-2])) == 11  # unequal lengths: histogram shift


def test_excerpt_stops_before_the_last_note():
    notes = notes_of(TWINKLE * 3)  # 42 notes, about 28 s
    start, end = excerpt_window(notes, max_s=25.0, min_s=8.0)
    assert start == notes[0][0]
    assert end - start <= 25.0 and end <= notes[-1][0]
    assert excerpt_window(notes_of(TWINKLE[:6]), min_s=8.0) is None  # too short to give an excerpt


@pytest.fixture(scope="module")
def fake_csd(tmp_path_factory):
    from adaptitune.audio import Audio, save
    from adaptitune.data.synth import SungNote, sing

    root = tmp_path_factory.mktemp("csd") / "CSD"
    songs = {"english/en001a": TWINKLE * 2, "english/en001b": [p + 2 for p in TWINKLE * 2],
             "korean/kr001a": MINOR * 2, "korean/kr001b": [p + 3 for p in MINOR * 2]}
    for i, (name, pitches) in enumerate(songs.items()):
        lang, rec = name.split("/")
        write_midi(root / lang / "mid" / f"{rec}.mid", pitches)
        s = sing([SungNote(midi=m, dur=QUARTER_S, gap_after=GAP_S, vibrato_cents=15) for m in pitches], lead_in=LEAD_S, seed=i)
        save(root / lang / "wav" / f"{rec}.wav", Audio(s.samples, s.sr), "PCM_16")
    return root


def test_finds_recordings_and_their_midi(fake_csd):
    items = find_csd(fake_csd)
    assert [(i["recording"], i["song"], i["version"], i["language"]) for i in items] == [
        ("en001a", "en001", "a", "english"), ("en001b", "en001", "b", "english"),
        ("kr001a", "kr001", "a", "korean"), ("kr001b", "kr001", "b", "korean")]


def test_csd_run_scores_every_recording(fake_csd, tmp_path):
    s = evaluate_csd(fake_csd, tmp_path / "out", tracker="yin")
    assert s["recordings"] == 4 and s["songs"] == 2 and s["no_reference"] == 0
    assert s["checks"]["song_pairs"] == 2 and s["checks"]["pairs_consistent_with_transposition"] == 2
    assert s["checks"]["midi_more_than_half_a_semitone_from_heard_pitch"] == 0
    assert s["excerpt"]["n"] == 4 and s["whole"]["n"] == 4
    assert s["whole"]["same_scale"] == 1.0  # clean synthetic singing: right scale on every song
    assert set(s["whole"]["by_language"]) == {"english", "korean"}
    assert json.loads((tmp_path / "out" / "summary.json").read_text())["recordings"] == 4
    rows = (tmp_path / "out" / "per_recording.csv").read_text().splitlines()
    assert len(rows) == 5 and "C major" in rows[1] and "A minor" in rows[3]
