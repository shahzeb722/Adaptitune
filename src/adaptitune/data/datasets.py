"""Loaders for the public datasets named in the proposal (section 4.5).

None of these are redistributed with the code. Download them yourself and point the loaders
at the extracted folders:

* VocalSet (Wilkins et al. 2018), CC BY 4.0, https://zenodo.org/records/1442513
  Layout: ``.../<female1..9|male1..11>/<context>/<technique>/*.wav``. Clean solo singing,
  our main source of in-tune material for synthetic detuning.
* MIR-1K (Hsu & Jang 2010), http://mirlab.org/dataset/public/MIR-1K.zip
  Layout: ``Wavfile/*.wav`` (stereo, left = accompaniment, right = vocal) and
  ``PitchLabel/*.pv`` (one semitone value per 20 ms frame, first frame centred at 20 ms,
  0 = unvoiced). Used for the tracker benchmark.
* MDB-stem-synth (Salamon et al. 2017), https://zenodo.org/records/1481172
  Layout: ``audio_stem/*.RESYN.wav`` and ``annotation_stem/*.RESYN.csv`` (time, f0).
  Exact labels, the fairest tracker benchmark.
* MedleyDB vocal stems (Bittner et al. 2014), access on request, non-commercial.
  Layout: ``<Track>/<Track>_STEMS/*.wav`` plus the track's ``_METADATA.yaml``.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf

from ..audio import Audio, resample, to_mono
from ..music import midi_to_hz


@dataclass
class Item:
    id: str
    singer: str
    audio: Audio
    ref_times: np.ndarray | None = None  # reference F0 (for tracker benchmarks)
    ref_f0: np.ndarray | None = None
    meta: dict | None = None


def _read(path: Path, sr: int | None, channel: int | None = None) -> Audio:
    x, file_sr = sf.read(str(path), always_2d=True)
    x = x[:, channel] if channel is not None else to_mono(x)
    if sr and sr != file_sr:
        x = resample(x, file_sr, sr)
        file_sr = sr
    return Audio(np.asarray(x, np.float64), file_sr)


# ------------------------------------------------------------------------------ VocalSet

VOCALSET_TECHNIQUES = ("straight", "vibrato", "belt", "breathy", "vocal_fry", "spoken", "lip_trill", "trill", "trillo", "inhaled", "messa", "pp", "forte", "fast_forte", "fast_piano", "slow_forte", "slow_piano")
# Sustained, clearly pitched techniques that make good in-tune source material. Breathy and
# fry are kept out on purpose (they are what the quality check is for), and so are spoken,
# inhaled, lip trills and trills, which have no stable note centre.
VOCALSET_CLEAN_TECHNIQUES = ("straight", "vibrato", "slow_forte", "slow_piano", "messa", "forte", "pp", "belt", "fast_forte", "fast_piano")
VOCALSET_DEFAULT_TECHNIQUES = ("straight", "vibrato", "slow_forte", "slow_piano", "messa")
_SINGER_DIR = re.compile(r"^(female|male)(\d+)$")


def vocalset_files(root: str | Path, techniques: tuple[str, ...] | None = VOCALSET_DEFAULT_TECHNIQUES) -> list[dict]:
    """Every VocalSet recording under ``root`` with its singer, gender, context (arpeggios,
    scales, long_tones, excerpts) and technique, read from the folder names.

    Works whichever folder the zip was extracted to (``FULL/``, ``data_by_singer/`` or one
    level above), and lists each recording once even if the same file appears in several
    of VocalSet's alternative layouts.
    """
    out, seen = [], set()
    for wav in sorted(Path(root).rglob("*.wav")):
        parts = wav.parts
        idx = next((i for i, p in enumerate(parts) if _SINGER_DIR.match(p)), None)
        if idx is None or wav.name in seen:
            continue
        rest = parts[idx + 1 : -1]
        context = rest[0] if len(rest) >= 1 else ""
        technique = rest[1] if len(rest) >= 2 else ""
        if not technique:  # fall back to the file name, e.g. f1_scales_straight_a.wav
            technique = next((t for t in sorted(VOCALSET_TECHNIQUES, key=len, reverse=True) if f"_{t}" in wav.stem), "")
        if techniques and technique not in techniques:
            continue
        seen.add(wav.name)
        singer = parts[idx]
        out.append(dict(path=str(wav), singer=singer, gender="F" if singer.startswith("female") else "M", context=context, technique=technique))
    return out


def vocalset(root: str | Path, sr: int = 24_000, techniques: tuple[str, ...] | None = VOCALSET_DEFAULT_TECHNIQUES) -> Iterator[Item]:
    """Yield VocalSet takes (see :func:`vocalset_files` for which ones)."""
    for f in vocalset_files(root, techniques):
        wav = Path(f["path"])
        tid = f"vocalset/{f['singer']}/{f['context']}/{f['technique']}/{wav.stem}"
        yield Item(tid, f["singer"], _read(wav, sr), meta={**f, "source": "vocalset"})


# ------------------------------------------------------------------------------ MIR-1K


def mir1k(root: str | Path, sr: int = 16_000, mix_snr: float | None = None) -> Iterator[Item]:
    """MIR-1K clips with their reference pitch. By default only the right (vocal) channel is
    used. With ``mix_snr`` the left (accompaniment) channel is mixed back in at that
    vocal-to-accompaniment ratio in dB, the usual way to test robustness on MIR-1K."""
    root = Path(root)
    wav_dir = root / "Wavfile"
    pv_dir = root / "PitchLabel"
    for wav in sorted(wav_dir.glob("*.wav")):
        pv = pv_dir / (wav.stem + ".pv")
        if not pv.exists():
            continue
        semis = np.atleast_1d(np.loadtxt(pv))
        # 40 ms frames with a 20 ms hop: frame i covers [20i, 20i + 40) ms, centre 20 + 20i ms
        t = 0.02 + 0.02 * np.arange(len(semis))
        f0 = np.where(semis > 0, midi_to_hz(semis), 0.0)
        singer = wav.stem.split("_")[0]
        if mix_snr is None:
            audio = _read(wav, sr, channel=1)
        else:
            x, file_sr = sf.read(str(wav), always_2d=True)
            voc, acc = x[:, 1].astype(np.float64), x[:, 0].astype(np.float64)
            pv_, pa = float(np.mean(voc**2)), float(np.mean(acc**2))
            g = np.sqrt(pv_ / (pa * 10 ** (mix_snr / 10))) if pa > 0 else 0.0
            mix = voc + g * acc
            if sr and sr != file_sr:
                mix = resample(mix, file_sr, sr)
                file_sr = sr
            audio = Audio(mix, file_sr)
        yield Item(f"mir1k/{wav.stem}", singer, audio, t, f0, meta={"source": "mir1k", "mix_snr_db": mix_snr})


# ------------------------------------------------------------------------------ MDB-stem-synth


def mdb_stem_synth(root: str | Path, sr: int = 16_000) -> Iterator[Item]:
    root = Path(root)
    for wav in sorted((root / "audio_stem").glob("*.wav")):
        ann = root / "annotation_stem" / (wav.stem + ".csv")
        if not ann.exists():
            continue
        try:
            arr = np.loadtxt(ann, delimiter=",", ndmin=2)
        except ValueError:  # tolerate a header line
            arr = np.loadtxt(ann, delimiter=",", ndmin=2, skiprows=1)
        track = re.sub(r"_STEM.*", "", wav.stem)
        yield Item(f"mdb/{wav.stem}", track, _read(wav, sr), arr[:, 0], arr[:, 1], meta={"source": "mdb"})


def mdb_vocal_only(items: Iterator[Item], medleydb_metadata_csv: str | None = None) -> Iterator[Item]:
    """MDB-stem-synth includes every monophonic instrument; keep vocal stems only if a list
    of vocal stem names is given (one per line), else keep all."""
    if not medleydb_metadata_csv:
        yield from items
        return
    keep = {line.strip() for line in Path(medleydb_metadata_csv).read_text().splitlines() if line.strip()}
    for it in items:
        if any(k in it.id for k in keep):
            yield it


# ------------------------------------------------------------------------------ MedleyDB


def medleydb_vocals(root: str | Path, sr: int = 24_000) -> Iterator[Item]:
    """Isolated vocal stems from MedleyDB (needs PyYAML for the metadata)."""
    import yaml

    root = Path(root)
    for meta in sorted(root.rglob("*_METADATA.yaml")):
        info = yaml.safe_load(meta.read_text())
        stems_dir = meta.parent / info.get("stem_dir", "")
        for sid, stem in (info.get("stems") or {}).items():
            inst = str(stem.get("instrument", "")).lower()
            if "singer" in inst or "vocal" in inst:
                path = stems_dir / stem["filename"]
                if path.exists():
                    singer = f"medleydb_{info.get('artist', meta.parent.name)}"
                    yield Item(f"medleydb/{path.stem}", singer, _read(path, sr), meta={"source": "medleydb", "path": str(path), "gender": "?"})


# ------------------------------------------------------------------------------ splits


def split_by_singer(singers: list[str], seed: int = 0, ratios=(0.7, 0.15, 0.15)) -> dict[str, str]:
    """Singer-disjoint train/val/test assignment (proposal 5.2)."""
    rng = np.random.default_rng(seed)
    uniq = sorted(set(singers))
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr = max(1, int(round(ratios[0] * n)))
    n_va = max(1 if n >= 3 else 0, int(round(ratios[1] * n)))
    out = {}
    for i, s in enumerate(uniq):
        out[s] = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"
    return out


def plan_split(
    singer_hours: dict[str, float],
    singer_group: dict[str, str] | None = None,
    min_test_hours: float = 2.0,
    margin: float = 0.1,
    n_val: int = 2,
    min_train: int = 2,
    seed: int = 0,
    tries: int = 5000,
) -> dict:
    """Singer-disjoint train/val/test split that guarantees the O1 test-audio target.

    O1 asks for at least ``min_test_hours`` of unique test audio from singers not used in
    development. Clips from one singer must stay together, so this is a small packing
    problem: over many random singer orders, singers are added to the test set until it holds
    ``min_test_hours * (1 + margin)`` (the margin absorbs takes lost to QC), ``n_val`` of the
    rest become validation singers and the others training singers. The kept split
    (1) reaches the target, (2) uses the fewest test singers, so development keeps as many
    voices as possible, (3) overshoots least, and (4) balances the groups (gender) in test and
    validation best. Deterministic for a given seed.
    """
    singers = sorted(singer_hours)
    group = singer_group or {}
    target = min_test_hours * (1 + margin)
    total = float(sum(singer_hours.values()))
    rng = np.random.default_rng(seed)

    def imbalance(names):
        f = sum(group.get(n) == "F" for n in names)
        m = sum(group.get(n) == "M" for n in names)
        return abs(f - m)

    best, best_key = None, None
    for _ in range(tries):
        order = [str(x) for x in rng.permutation(singers)]
        test, hrs = [], 0.0
        while order and hrs < target:
            s_ = order.pop(0)
            test.append(s_)
            hrs += singer_hours[s_]
        # validation: prefer one of each group, in the shuffled order
        val = []
        for want in ("F", "M"):
            for s_ in order:
                if len(val) < n_val and s_ not in val and group.get(s_) == want:
                    val.append(s_)
                    break
        for s_ in order:
            if len(val) >= n_val:
                break
            if s_ not in val:
                val.append(s_)
        train = [s_ for s_ in order if s_ not in val]
        feasible = hrs >= min_test_hours and len(train) >= min_train and len(val) == min(n_val, len(order))
        key = (not feasible, len(test), round(hrs - target, 3), imbalance(test), imbalance(val))
        if best_key is None or key < best_key:
            best_key, best = key, (test, val, train, hrs)
    test, val, train, hrs = best
    assign = {s_: "test" for s_ in test} | {s_: "val" for s_ in val} | {s_: "train" for s_ in train}
    hours = {sp: float(sum(singer_hours[s_] for s_ in assign if assign[s_] == sp)) for sp in ("train", "val", "test")}
    return {
        "assignment": assign,
        "hours": hours,
        "singers": {sp: sorted(s_ for s_ in assign if assign[s_] == sp) for sp in ("train", "val", "test")},
        "target_test_hours": min_test_hours,
        "total_hours": total,
        "feasible": bool(hrs >= min_test_hours and len(train) >= min_train),
        "seed": seed,
    }


def read_key_labels(path: str | Path) -> list[dict]:
    """CSV for the ~100 hand-labelled excerpts: ``path,key_rater1,key_rater2`` plus optional
    ``adjudicated`` (see ``adaptitune key-agreement``)."""
    with open(path) as f:
        return list(csv.DictReader(f))
