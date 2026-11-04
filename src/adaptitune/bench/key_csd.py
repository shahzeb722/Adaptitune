"""O2 key test on the Children's Song Dataset (CSD), in place of the hand labels.

No two raters were available to label the MIR-1K excerpts by ear, and the MIR-1K karaoke backing
cannot stand in for them (the singer is often in a different key from the stored backing), so on
4 November 2026 the team chose a dataset whose keys are fixed by its own note files
(docs/O2_PROTOCOL.md, B4). The method below was written down before CSD was on disk.

CSD (Choi et al., ISMIR 2020 late-breaking demo; zenodo.org/records/4785016, CC BY-NC-SA 4.0):
50 Korean and 50 English children's songs, each sung twice in different keys ("a" and "b" at
the end of the file name) by one professional singer. Every recording has a MIDI file whose notes
were aligned to the singing by hand.

* **Reference key** of a recording, from its MIDI notes alone: the tonic is the pitch class of the
  last note, and the mode is major if the major third above the tonic is sung for longer than
  the minor third (the rule the Essen ground truth already uses for songs whose key field has no
  mode). Checks, reported but never used to change a reference: Krumhansl-Kessler key finding on
  the same notes (a different profile from the app's), the two versions of a song must differ by
  the transposition between their note lists, and the MIDI must sit within half a semitone of
  the pitch the tracker hears.
* **Test items:** each recording as a whole, and one excerpt per recording that starts at the
  first note and stops before the last note (at most 25 s, at least 8 s), so the final tonic is
  not given away, as with the Essen excerpts. The excerpt result is the O2 figure; the
  proposal's test set was excerpts.
* **Scoring:** exact key, with a Wilson interval and an interval that resamples songs (each song
  is sung twice), plus same-scale (relative) and MIREX scores, by language and by mode.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import numpy as np
import pandas as pd

from ..music import Key

EXCERPT_MAX_S = 25.0
EXCERPT_MIN_S = 8.0
EXCERPT_LEAD_S = 0.2  # audio kept before the first note's onset


# ----------------------------------------------------------------------------- MIDI


def _varlen(data: bytes, p: int) -> tuple[int, int]:
    v = 0
    while True:
        b = data[p]
        p += 1
        v = (v << 7) | (b & 0x7F)
        if not b & 0x80:
            return v, p


def read_midi_notes(path: str | Path) -> list[tuple[float, float, int]]:
    """Notes of a standard MIDI file as (onset s, offset s, MIDI number), sorted by onset. Tempo
    changes are followed; all tracks and channels are merged."""
    data = Path(path).read_bytes()
    if data[:4] != b"MThd":
        raise ValueError(f"{path} is not a MIDI file")
    hlen = int.from_bytes(data[4:8], "big")
    _fmt, n_tracks, division = struct.unpack(">HHH", data[8:14])
    if division & 0x8000:
        raise ValueError(f"{path}: SMPTE time division is not supported")
    pos = 8 + hlen
    events = []  # (tick, order, kind, value, channel); at one tick: tempo, then note-offs, then note-ons
    for _ in range(n_tracks):
        if data[pos:pos + 4] != b"MTrk":
            raise ValueError(f"{path}: bad track header")
        end = pos + 8 + int.from_bytes(data[pos + 4:pos + 8], "big")
        p, tick, status = pos + 8, 0, None
        pos = end
        while p < end:
            delta, p = _varlen(data, p)
            tick += delta
            b = data[p]
            if b == 0xFF:  # meta event
                kind = data[p + 1]
                n, p = _varlen(data, p + 2)
                if kind == 0x51:
                    events.append((tick, 0, "tempo", int.from_bytes(data[p:p + n], "big"), 0))
                p += n
                if kind == 0x2F:
                    break
                continue
            if b in (0xF0, 0xF7):  # sysex
                n, p = _varlen(data, p + 1)
                p += n
                continue
            if b & 0x80:
                status, p = b, p + 1
            elif status is None:
                raise ValueError(f"{path}: running status before any status byte")
            hi, ch = status & 0xF0, status & 0x0F
            if hi in (0xC0, 0xD0):
                p += 1
                continue
            d1, d2 = data[p], data[p + 1]
            p += 2
            if hi == 0x90 and d2 > 0:
                events.append((tick, 2, "on", d1, ch))
            elif hi == 0x80 or hi == 0x90:
                events.append((tick, 1, "off", d1, ch))
    events.sort(key=lambda e: (e[0], e[1]))
    tempo, last, secs = 500_000, 0, 0.0
    open_: dict[tuple[int, int], list[float]] = {}
    notes = []
    for tick, _, kind, value, ch in events:
        secs += (tick - last) * tempo / 1e6 / division
        last = tick
        if kind == "tempo":
            tempo = value
        elif kind == "on":
            open_.setdefault((value, ch), []).append(secs)
        elif open_.get((value, ch)):
            on = open_[(value, ch)].pop(0)
            if secs > on:
                notes.append((on, secs, value))
    return sorted(notes)


# ----------------------------------------------------------------------------- reference keys


def reference_key(notes: list[tuple[float, float, int]]) -> Key | None:
    """Tonic = the last note's pitch class; major if the major third above it lasts longer than
    the minor third (``bench.keys.mode_by_third``). None if the two thirds tie."""
    from .keys import mode_by_third

    if not notes:
        return None
    tonic = notes[-1][2] % 12
    mode = mode_by_third([(m, off - on) for on, off, m in notes], tonic)
    return Key(tonic, mode) if mode else None


def _pc_hist(notes: list[tuple[float, float, int]]) -> np.ndarray:
    h = np.zeros(12)
    for on, off, m in notes:
        h[m % 12] += off - on
    return h


def ks_key(notes: list[tuple[float, float, int]]) -> Key:
    """Krumhansl-Kessler key finding on the duration-weighted pitch classes (a cross-check only)."""
    from ..pitch.key import key_scores

    s = key_scores(_pc_hist(notes), "krumhansl")
    return max(s, key=s.get)


def transposition(a: list[tuple[float, float, int]], b: list[tuple[float, float, int]]) -> int:
    """Semitones that take version a of a song to version b (mod 12): the median pitch difference
    when the note lists match one to one, otherwise the best pitch-class histogram shift."""
    if len(a) == len(b) and len(a):
        return int(round(float(np.median([nb[2] - na[2] for na, nb in zip(a, b)])))) % 12
    ha, hb = _pc_hist(a), _pc_hist(b)
    return int(np.argmax([np.dot(np.roll(ha, s), hb) for s in range(12)]))


def excerpt_window(notes: list[tuple[float, float, int]], max_s: float = EXCERPT_MAX_S,
                   min_s: float = EXCERPT_MIN_S) -> tuple[float, float] | None:
    """(start, end) in seconds: from the first note's onset to the end of the last note that
    finishes within ``max_s``, never reaching the song's last note. None if that is under ``min_s``."""
    if len(notes) < 3:
        return None
    start = notes[0][0]
    final_on = notes[-1][0]
    ends = [off for on, off, _ in notes[:-1] if off - start <= max_s and off <= final_on]
    if not ends or max(ends) - start < min_s:
        return None
    return start, max(ends)


# ----------------------------------------------------------------------------- dataset


def find_csd(root: str | Path) -> list[dict]:
    """Every CSD recording below ``root`` with its MIDI file, matched by file name."""
    root = Path(root)
    mids = {p.stem: p for p in root.rglob("*.mid") if not p.name.startswith("._")}
    out = []
    for w in sorted(root.rglob("*.wav")):
        if w.name.startswith("._") or w.stem not in mids:
            continue
        m = re.match(r"^(.*?)([ab])$", w.stem)
        song, version = (m.group(1), m.group(2)) if m else (w.stem, "")
        lang = next((p.name for p in w.parents if p.name.lower() in ("english", "korean")), "")
        out.append({"recording": w.stem, "song": song, "version": version, "language": lang.lower(), "wav": w, "mid": mids[w.stem]})
    return out


def _midi_offset(notes: list[tuple[float, float, int]], f0: np.ndarray, hop_s: float) -> float:
    """Median of (heard pitch - MIDI pitch) in semitones, over the middle half of each note."""
    d = []
    for on, off, m in notes:
        a, b = int((on + 0.25 * (off - on)) / hop_s), int((off - 0.25 * (off - on)) / hop_s)
        seg = f0[a:b]
        seg = seg[seg > 0]
        if len(seg):
            d.append(float(np.median(69 + 12 * np.log2(seg / 440.0))) - m)
    return float(np.median(d)) if d else float("nan")


def _scores(df: pd.DataFrame, col: str) -> dict:
    from .keys import wilson
    from .trackers import bootstrap_ci

    s = df[df[col + "_key"].notna()]
    n = len(s)
    if not n:
        return {"n": 0}
    k = int(s[col + "_exact"].sum())
    ci_song = bootstrap_ci(s.assign(cluster=s.song, x=s[col + "_exact"].astype(float)), lambda d: d.x.mean(), n_boot=2000)
    by = lambda g: {str(name): round(float(x[col + "_exact"].mean()), 4) for name, x in s.groupby(g)}
    return {
        "n": n,
        "exact": round(k / n, 4),
        "exact_ci95": [round(v, 4) for v in wilson(k, n)],
        "exact_ci95_by_song": [round(float(v), 4) for v in ci_song],
        "same_scale": round(float(s[col + "_same_scale"].mean()), 4),
        "mirex": round(float(s[col + "_mirex"].mean()), 4),
        "by_language": by("language"),
        "by_reference_mode": by(s.reference.str.split().str[1]),
    }


def evaluate_csd(root: str | Path, out_dir: str | Path = "results/keys_csd", tracker: str = "auto",
                 limit: int | None = None) -> dict:
    """Key accuracy of the app's estimator on CSD (module docstring); writes per_recording.csv and
    summary.json to ``out_dir``."""
    from ..audio import load
    from ..pitch.key import analyze_key, mirex_key_score, same_key_or_relative
    from ..pitch.trackers import get_tracker
    from .trackers import _git_commit

    items = find_csd(root)[:limit] if limit else find_csd(root)
    if not items:
        raise FileNotFoundError(f"no CSD recordings with MIDI files below {root}")
    trk = get_tracker(tracker)
    rows, notes_of = [], {}
    for it in items:
        notes = read_midi_notes(it["mid"])
        notes_of[it["recording"]] = notes
        ref, ks = reference_key(notes), ks_key(notes)
        audio = load(it["wav"])
        tr = trk.track(audio.samples, audio.sr)
        r = {k: it[k] for k in ("recording", "song", "version", "language")}
        r.update(reference=ref.name if ref else None, krumhansl=ks.name, notes=len(notes),
                 seconds=round(audio.duration, 2), midi_vs_heard_semitones=round(_midi_offset(notes, tr.f0, tr.hop_s), 3))
        res = {"whole": analyze_key(tr.f0, tr.voicing, tr.hop_s)}
        win = excerpt_window(notes)
        if win:
            a0, a1 = int(max(0.0, win[0] - EXCERPT_LEAD_S) * audio.sr), int(win[1] * audio.sr)
            tr_e = trk.track(audio.samples[a0:a1], audio.sr)
            res["excerpt"] = analyze_key(tr_e.f0, tr_e.voicing, tr_e.hop_s)
            r.update(excerpt_start=round(win[0], 2), excerpt_end=round(win[1], 2))
        for kind in ("whole", "excerpt"):
            k = res.get(kind)
            r[kind + "_key"] = k.key.name if k else None
            r[kind + "_confidence"] = round(float(k.confidence), 4) if k else None
            r[kind + "_exact"] = bool(k and ref and k.key == ref)
            r[kind + "_same_scale"] = bool(k and ref and same_key_or_relative(k.key, ref))
            r[kind + "_mirex"] = mirex_key_score(k.key, ref) if k and ref else 0.0
        rows.append(r)
    df = pd.DataFrame(rows)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_recording.csv", index=False)

    # checks on the references (reported, never used to change one)
    pairs = consistent = 0
    by_song = {s: g.set_index("version") for s, g in df.groupby("song")}
    for s, g in by_song.items():
        if not {"a", "b"} <= set(g.index):
            continue
        ra, rb = g.loc["a", "reference"], g.loc["b", "reference"]
        if isinstance(ra, str) and isinstance(rb, str):  # a recording without a reference is NaN here
            ka, kb = Key.parse(ra), Key.parse(rb)
            t = transposition(notes_of[g.loc["a", "recording"]], notes_of[g.loc["b", "recording"]])
            pairs += 1
            consistent += int(ka.mode == kb.mode and (ka.tonic + t) % 12 == kb.tonic)
    scored = df[df.reference.notna()]
    off = scored.midi_vs_heard_semitones.abs() > 0.5
    agree = scored[scored.reference == scored.krumhansl]
    summary = {
        "dataset": "CSD (Children's Song Dataset)",
        "root": str(root),
        "tracker": trk.name,
        "recordings": len(df),
        "songs": int(df.song.nunique()),
        "no_reference": int(df.reference.isna().sum()),
        "reference_rule": "last note's pitch class; major if the major third above it is sung longer than the minor third",
        "checks": {
            "reference_equals_krumhansl": round(float((scored.reference == scored.krumhansl).mean()), 4),
            "song_pairs": pairs,
            "pairs_consistent_with_transposition": consistent,
            "midi_more_than_half_a_semitone_from_heard_pitch": int(off.sum()),
        },
        "excerpt": _scores(scored, "excerpt"),
        "whole": _scores(scored, "whole"),
        "excerpt_where_krumhansl_agrees": _scores(agree, "excerpt"),
        "git_commit": _git_commit(),
    }
    summary["O2_key_pass"] = bool(summary["excerpt"].get("exact", 0) >= 0.75)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
