"""Where the app's key differs from the CSD reference, and how sound those references are.

Reads the per-recording table of an ``adaptitune key-eval-csd`` run and the CSD MIDI files and
reports, without changing any reference or the O2 figure (docs/O2_PROTOCOL.md, B4):

* for every recording with a minor reference: the share of sung time on notes outside that minor
  key, even with its raised 6th and 7th, and the share inside the major key whose third is the
  reference's tonic (the key of a major-key song that ends on its third);
* for every excerpt the app got wrong: how its key relates to the reference, whether the whole
  recording was right, Krumhansl-Kessler's key, how many pitch classes the melody uses, and
  whether the other version of the song was missed too.

    python scripts/csd_misses.py results/keys_csd data/raw/CSD --out docs/results/o2/keys_csd/misses.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from adaptitune.bench.key_csd import _pc_hist, find_csd, read_midi_notes
from adaptitune.music import NOTE_NAMES, Key


def relation(ref: Key, est: Key) -> str:
    """How the estimated key relates to the reference key."""
    up = (est.tonic - ref.tonic) % 12
    if est.mode == ref.mode:
        return {7: "dominant", 5: "subdominant", 2: "whole tone above", 10: "whole tone below"}.get(up, f"{up} semitones up, same mode")
    if up == 0:
        return "parallel"
    if (ref.mode, up) in (("major", 9), ("minor", 3)):
        return "relative"
    if ref.mode == "minor" and up == 8:
        return "major key whose third is the reference tonic"
    return f"{up} semitones up, {est.mode}"


def share_in(hist, classes) -> float:
    return round(float(sum(hist[c] for c in classes) / hist.sum()), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="output folder of key-eval-csd (has per_recording.csv)")
    ap.add_argument("csd", help="the CSD root that run used")
    ap.add_argument("--out", default="results/keys_csd/misses.json")
    a = ap.parse_args()

    d = pd.read_csv(Path(a.run) / "per_recording.csv")
    items = {i["recording"]: i for i in find_csd(a.csd)}
    names = {}
    for m in Path(a.csd).rglob("metadata.json"):
        names.update({k: v.get("songname", "") for k, v in json.loads(m.read_text(encoding="utf-8")).items()})
    hist = {r: _pc_hist(read_midi_notes(items[r]["mid"])) for r in d.recording}
    last = {r: " ".join(NOTE_NAMES[m % 12] for _, _, m in read_midi_notes(items[r]["mid"])[-5:]) for r in d.recording}
    s = d[d.reference.notna()].copy()

    minor = []
    for _, r in s[s.reference.str.endswith("minor")].iterrows():
        ref, h = Key.parse(r.reference), hist[r.recording]
        major_below = Key((ref.tonic - 4) % 12, "major")
        minor.append(dict(recording=r.recording, song_name=names.get(r.recording, ""), reference=r.reference,
                          app_excerpt=r.excerpt_key, app_whole=r.whole_key, krumhansl=r.krumhansl,
                          share_outside_reference_minor=round(1 - share_in(h, ref.snap_classes), 3),
                          notes_outside_reference_minor=" ".join(NOTE_NAMES[c] for c in range(12) if h[c] > 0 and c not in ref.snap_classes),
                          major_key_below=major_below.name, share_inside_major_key_below=share_in(h, major_below.pitch_classes),
                          last_notes=last[r.recording]))

    missed = s[~s.excerpt_exact.astype(bool)]
    missed_songs = Counter(missed.song)
    rows = []
    for _, r in missed.iterrows():
        rows.append(dict(recording=r.recording, song_name=names.get(r.recording, ""), reference=r.reference,
                         app_excerpt=r.excerpt_key, relation=relation(Key.parse(r.reference), Key.parse(r.excerpt_key)),
                         whole_right=bool(r.whole_exact), krumhansl=r.krumhansl,
                         pitch_classes_used=int((hist[r.recording] > 0).sum()),
                         other_version_missed_too=missed_songs[r.song] == 2, last_notes=last[r.recording]))
    m = pd.DataFrame(rows, columns=["recording", "song_name", "reference", "app_excerpt", "relation", "whole_right",
                                    "krumhansl", "pitch_classes_used", "other_version_missed_too", "last_notes"])
    others = m[~m.reference.str.endswith("minor")]
    res = {
        "run": str(a.run),
        "scored": len(s),
        "minor_references": minor,
        "excerpt_misses": {
            "count": len(m),
            "with_a_minor_reference": int(m.reference.str.endswith("minor").sum()),
            "others_by_relation": dict(Counter(others.relation).most_common()),
            "others_right_on_the_whole_recording": int(others.whole_right.sum()),
            "others_where_the_other_version_was_missed_too": int(others.other_version_missed_too.sum()),
        },
        "excerpt_exact_on_major_references": round(float(s[s.reference.str.endswith("major")].excerpt_exact.mean()), 4),
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    m.to_csv(out.with_suffix(".csv"), index=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
