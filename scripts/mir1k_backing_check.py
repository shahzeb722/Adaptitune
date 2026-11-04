"""Can MIR-1K's karaoke backing stand in for hand-labelled keys? (docs/O2_PROTOCOL.md, B4)

MIR-1K stores the karaoke backing in the left channel beside each vocal. If the singer always sang in
the backing's key, the backing's key could serve as the reference for the key excerpts. This checks
that for each excerpt of ``adaptitune key-excerpts --source mir1k``: it compares the pitch classes
of the human-labelled vocal pitch (``PitchLabel/``) with the backing's chroma under every
transposition. An excerpt whose best match is a transposition other than 0 has a singer who is not
in the stored backing's key.

With Essentia installed (``pip install essentia``), it also reads each backing's key with five key
profiles and scores the app's key estimator (voice only) against their majority, overall and on the
excerpts where the voice clearly matches the backing untransposed (best shift 0 and correlation at
least 0.5). That is a rough check only: the profiles disagree on a third of the backings.

    python scripts/mir1k_backing_check.py key_excerpts data/raw/MIR-1K --out docs/results/o2/mir1k_backing_check.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


def wav_dir(root: Path) -> Path:
    return next(p for p in root.rglob("Wavfile") if p.is_dir())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("excerpts", help="folder with excerpts.json and audio/ (from key-excerpts --source mir1k)")
    ap.add_argument("mir1k", help="folder that contains MIR-1K's Wavfile/ and PitchLabel/")
    ap.add_argument("--out", default="results/mir1k_backing_check.json")
    a = ap.parse_args()
    import librosa

    ex = json.loads((Path(a.excerpts) / "excerpts.json").read_text())["excerpts"]
    wavs = wav_dir(Path(a.mir1k))
    pvs = wavs.parent / "PitchLabel"
    rows = []
    for e in ex:
        hv, hb, backing = np.zeros(12), np.zeros(12), []
        for c in e["clips"]:
            x, sr = sf.read(wavs / f"{c}.wav", dtype="float32")
            backing.append(x[:, 0])
            hb += librosa.feature.chroma_cqt(y=x[:, 0], sr=sr, hop_length=512).sum(axis=1)
            pv = np.loadtxt(pvs / f"{c}.pv")
            hv += np.bincount(np.round(pv[pv > 0]).astype(int) % 12, minlength=12)
        r = [float(np.corrcoef(np.roll(hv, s), hb)[0, 1]) for s in range(12)]  # voice moved up by s
        b = int(np.argmax(r))
        rows.append(dict(excerpt=e["excerpt"], song=e["song"], best_shift=b if b <= 6 else b - 12,
                         corr_unshifted=round(r[0], 3), corr_best=round(r[b], 3), backing=np.concatenate(backing), sr=sr))
    d = pd.DataFrame(rows)
    res = {
        "excerpts": len(d),
        "best_shift_counts": {str(k): v for k, v in sorted(Counter(d.best_shift).items())},
        "voice_transposed_against_backing": int((d.best_shift != 0).sum()),
        "clearly_transposed (best shift not 0, correlation gain over 0.3)": int(((d.best_shift != 0) & (d.corr_best - d.corr_unshifted > 0.3)).sum()),
    }
    aligned = (d.best_shift == 0) & (d.corr_unshifted >= 0.5)
    res["aligned (best shift 0, correlation at least 0.5)"] = int(aligned.sum())
    try:
        import essentia.standard as es
    except ImportError:
        es = None
    if es is not None:
        from adaptitune.audio import load
        from adaptitune.music import Key
        from adaptitune.pitch.key import analyze_key
        from adaptitune.pitch.trackers import get_tracker

        trk = get_tracker("auto")
        profiles = ["temperley", "krumhansl", "edma", "bgate", "shaath"]
        majority, votes, est = [], [], []
        for r in rows:
            y = es.Resample(inputSampleRate=r["sr"], outputSampleRate=44100)(r["backing"])
            keys = [Key.parse(" ".join(es.KeyExtractor(profileType=p, sampleRate=44100)(y)[:2])).name for p in profiles]
            k, n = Counter(keys).most_common(1)[0]
            majority.append(k)
            votes.append(n)
            au = load(Path(a.excerpts) / "audio" / f"{r['excerpt']}.wav")
            tr = trk.track(au.samples, au.sr)
            est.append(analyze_key(tr.f0, tr.voicing, tr.hop_s).key.name)
        d["backing_key"], d["backing_votes"], d["app_key_voice_only"] = majority, votes, est
        d["match"] = d.backing_key == d.app_key_voice_only
        res["essentia"] = {
            "profiles": profiles,
            "all_five_profiles_agree": int((d.backing_votes == 5).sum()),
            "app_matches_backing_key_all": f"{int(d.match.sum())}/{len(d)}",
            "app_matches_backing_key_aligned": f"{int(d.match[aligned].sum())}/{int(aligned.sum())}",
            "app_matches_backing_key_transposed": f"{int(d.match[d.best_shift != 0].sum())}/{int((d.best_shift != 0).sum())}",
            "tracker": trk.name,
        }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    d.drop(columns=["backing", "sr"]).to_csv(out.with_suffix(".csv"), index=False)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
