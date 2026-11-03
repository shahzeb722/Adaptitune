"""Loaders for the public datasets named in the proposal (section 4.5).

None of these are redistributed with the code. Download them yourself and point the loaders
at the extracted folders:

* VocalSet (Wilkins et al. 2018), CC BY 4.0, https://zenodo.org/records/1442513
  Layout: ``.../<female1..9|male1..11>/<context>/<technique>/*.wav``. Clean solo singing,
  our main source of in-tune material for synthetic detuning.
* MIR-1K (Hsu & Jang 2010). The original http://mirlab.org/dataset/public/MIR-1K.rar now returns 404;
  a copy is at https://figshare.com/articles/dataset/MIR-1K_rar/5802891 (MD5 1810d01457ccbb84a0b41c4da53eee74)
  Layout: ``Wavfile/*.wav`` (stereo, left = accompaniment, right = vocal) and
  ``PitchLabel/*.pv`` (one semitone value per 20 ms frame, first frame centred at 20 ms,
  0 = unvoiced). Used for the tracker benchmark.
* MDB-stem-synth (Salamon et al. 2017), https://zenodo.org/records/1481172
  Layout: ``audio_stems/*.RESYN.wav`` and ``annotation_stems/*.RESYN.csv`` (time, f0). The
  singular folder names of some copies are accepted too.
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


def _junk(path: Path) -> bool:
    """macOS metadata that zips made on a Mac carry along: ``__MACOSX/`` and ``._name`` files.
    They have audio-like names but are not audio."""
    return path.name.startswith("._") or "__MACOSX" in path.parts


def _find_dir(root: str | Path, names: tuple[str, ...], max_depth: int = 3) -> Path | None:
    """The first folder named one of ``names`` at ``root`` or up to ``max_depth`` levels below
    it, searched breadth first in sorted order, so the dataset can be extracted anywhere."""
    level = [Path(root)]
    for _ in range(max_depth + 1):
        for d in level:
            for n in names:
                if (d / n).is_dir():
                    return d / n
        level = [c for d in level for c in sorted(d.iterdir()) if c.is_dir() and not _junk(c) and not c.name.startswith(".")]
    return None


# ------------------------------------------------------------------------------ VocalSet

VOCALSET_TECHNIQUES = ("straight", "vibrato", "belt", "breathy", "vocal_fry", "spoken", "lip_trill", "trill", "trillo", "inhaled", "messa", "pp", "forte", "fast_forte", "fast_piano", "slow_forte", "slow_piano")
# Sustained, clearly pitched techniques that make good in-tune source material. Breathy and
# fry are kept out on purpose (they are what the quality check is for), and so are spoken,
# inhaled, lip trills and trills, which have no stable note centre.
VOCALSET_CLEAN_TECHNIQUES = ("straight", "vibrato", "slow_forte", "slow_piano", "messa", "forte", "pp", "belt", "fast_forte", "fast_piano")
VOCALSET_DEFAULT_TECHNIQUES = ("straight", "vibrato", "slow_forte", "slow_piano", "messa")
_SINGER_DIR = re.compile(r"^(female|male)(\d+)$")
_NAME_SINGER = re.compile(r"^\s*([fm])(\d+)_")  # f3_..., m9_...: the singer a file name claims
# Folder-name slips in the released zips: six vibrato takes sit in a folder called "vibrado".
VOCALSET_TECHNIQUE_ALIASES = {"vibrado": "vibrato"}


def vocalset_files(root: str | Path, techniques: tuple[str, ...] | None = VOCALSET_DEFAULT_TECHNIQUES,
                   skipped: list | None = None) -> list[dict]:
    """Every VocalSet recording under ``root`` with its singer, gender, context (arpeggios,
    scales, long_tones, excerpts) and technique, read from the folder names.

    Works whichever folder the zip was extracted to (``FULL/``, ``data_by_singer/`` or one
    level above), and lists each recording once even if it appears in several of VocalSet's
    alternative layouts (the same path below the singer folder). macOS ``._`` files are
    ignored. A file whose name names a different singer than its folder (VocalSet 1.1 and 1.2
    both have ``male8/excerpts/vibrato/m9_caro_vibrato.wav``) is left out, because who sang
    it is uncertain and a wrong singer would leak a voice across the split; such files are
    appended to ``skipped`` with the reason. Files with no singer prefix (33 in VocalSet 1.1,
    e.g. ``female4/scales/straight/scales_straight_a.wav``) keep their folder's singer.
    """
    out, seen = [], set()
    for wav in sorted(Path(root).rglob("*.wav")):
        if _junk(wav):
            continue
        parts = wav.parts
        idx = next((i for i, p in enumerate(parts) if _SINGER_DIR.match(p)), None)
        if idx is None:
            continue
        key = "/".join(parts[idx:])
        if key in seen:
            continue
        rest = parts[idx + 1 : -1]
        context = rest[0] if len(rest) >= 1 else ""
        technique = VOCALSET_TECHNIQUE_ALIASES.get(rest[1], rest[1]) if len(rest) >= 2 else ""
        if not technique:  # fall back to the file name, e.g. f1_scales_straight_a.wav
            technique = next((t for t in sorted(VOCALSET_TECHNIQUES, key=len, reverse=True) if f"_{t}" in wav.stem), "")
        if techniques and technique not in techniques:
            continue
        seen.add(key)
        singer = parts[idx]
        m = _NAME_SINGER.match(wav.name)
        if m and f"{'female' if m.group(1) == 'f' else 'male'}{int(m.group(2))}" != singer:
            if skipped is not None:
                skipped.append(dict(path=str(wav), reason=f"file name says {m.group(1)}{m.group(2)}, folder says {singer}"))
            continue
        out.append(dict(path=str(wav), singer=singer, gender="F" if singer.startswith("female") else "M", context=context, technique=technique))
    return out


def vocalset(root: str | Path, sr: int = 24_000, techniques: tuple[str, ...] | None = VOCALSET_DEFAULT_TECHNIQUES) -> Iterator[Item]:
    """Yield VocalSet takes (see :func:`vocalset_files` for which ones)."""
    for f in vocalset_files(root, techniques):
        wav = Path(f["path"])
        tid = f"vocalset/{f['singer']}/{f['context']}/{f['technique']}/{wav.stem}"
        yield Item(tid, f["singer"], _read(wav, sr), meta={**f, "source": "vocalset"})


# ------------------------------------------------------------------------------ MIR-1K


def mir1k(root: str | Path, sr: int = 16_000, mix_snr: float | None = None, shard: tuple[int, int] | None = None) -> Iterator[Item]:
    """MIR-1K clips with their reference pitch. By default only the right (vocal) channel is
    used. With ``mix_snr`` the left (accompaniment) channel is mixed back in at that
    vocal-to-accompaniment ratio in dB, the usual way to test robustness on MIR-1K."""
    wav_dir = _find_dir(root, ("Wavfile",))
    if wav_dir is None:
        raise FileNotFoundError(f"no MIR-1K Wavfile/ folder under {root}")
    pv_dir = wav_dir.parent / "PitchLabel"
    pairs = [(w, pv_dir / (w.stem + ".pv")) for w in sorted(wav_dir.glob("*.wav")) if not _junk(w)]
    for i, (wav, pv) in enumerate(p for p in pairs if p[1].exists()):
        if shard and i % shard[1] != shard[0]:
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


def mdb_stem_synth(root: str | Path, sr: int = 16_000, shard: tuple[int, int] | None = None) -> Iterator[Item]:
    """MDB-stem-synth stems with their exact reference pitch. ``shard=(k, n)`` keeps every
    n-th stem starting at k (before any audio is read), to split a run across processes."""
    audio_dir = _find_dir(root, ("audio_stems", "audio_stem"))
    if audio_dir is None:
        raise FileNotFoundError(f"no MDB-stem-synth audio_stems/ folder under {root}")
    ann_dir = next((audio_dir.parent / n for n in ("annotation_stems", "annotation_stem") if (audio_dir.parent / n).is_dir()), audio_dir)
    pairs = [(w, ann_dir / (w.stem + ".csv")) for w in sorted(audio_dir.glob("*.wav")) if not _junk(w)]
    for i, (wav, ann) in enumerate(p for p in pairs if p[1].exists()):
        if shard and i % shard[1] != shard[0]:
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
