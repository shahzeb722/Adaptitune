"""Loaders for the public datasets named in the proposal (section 4.5).

None of these are redistributed with the code. Download them yourself and point the loaders
at the extracted folders:

* VocalSet (Wilkins et al. 2018), CC BY 4.0, https://zenodo.org/records/1442513
  Layout: ``FULL/<singer>/<context>/<technique>/*.wav``. Clean solo singing, our main
  source of in-tune material for synthetic detuning.
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

VOCALSET_CLEAN_TECHNIQUES = ("straight", "vibrato", "belt", "breathy", "vocal_fry", "spoken", "lip_trill", "trill", "trillo", "inhaled", "messa", "pp", "forte", "fast_forte", "fast_piano", "slow_forte", "slow_piano")


def vocalset(root: str | Path, sr: int = 24_000, techniques: tuple[str, ...] = ("straight", "vibrato", "slow_forte", "slow_piano", "messa")) -> Iterator[Item]:
    """Yield VocalSet takes. The default techniques are the sustained, pitched ones that make
    good in-tune source material; breathy/fry are kept out of the clean set on purpose (they
    are what the quality check is for)."""
    root = Path(root)
    base = root / "FULL" if (root / "FULL").exists() else root
    for wav in sorted(base.rglob("*.wav")):
        rel = wav.relative_to(base).parts
        singer = rel[0]
        if techniques and not any(t in wav.stem or t in rel for t in techniques):
            continue
        yield Item(f"vocalset/{'/'.join(rel)}", singer, _read(wav, sr), meta={"path": str(wav)})


# ------------------------------------------------------------------------------ MIR-1K


def mir1k(root: str | Path, sr: int = 16_000) -> Iterator[Item]:
    root = Path(root)
    wav_dir = root / "Wavfile"
    pv_dir = root / "PitchLabel"
    for wav in sorted(wav_dir.glob("*.wav")):
        pv = pv_dir / (wav.stem + ".pv")
        if not pv.exists():
            continue
        semis = np.loadtxt(pv)
        t = 0.02 + 0.02 * np.arange(len(semis))
        f0 = np.where(semis > 0, midi_to_hz(semis), 0.0)
        singer = wav.stem.split("_")[0]
        yield Item(f"mir1k/{wav.stem}", singer, _read(wav, sr, channel=1), t, f0)


# ------------------------------------------------------------------------------ MDB-stem-synth


def mdb_stem_synth(root: str | Path, sr: int = 16_000) -> Iterator[Item]:
    root = Path(root)
    for wav in sorted((root / "audio_stem").glob("*.wav")):
        ann = root / "annotation_stem" / (wav.stem + ".csv")
        if not ann.exists():
            continue
        arr = np.loadtxt(ann, delimiter=",")
        track = re.sub(r"_STEM.*", "", wav.stem)
        yield Item(f"mdb/{wav.stem}", track, _read(wav, sr), arr[:, 0], arr[:, 1])


def mdb_vocal_only(items: Iterator[Item], medleydb_metadata_csv: str | None = None) -> Iterator[Item]:
    """MDB-stem-synth includes every monophonic instrument; keep vocal stems only if a list
    of vocal stem names is given (one per line), else keep all."""
    if not medleydb_metadata_csv:
        yield from items
        return
    keep = {line.strip() for line in open(medleydb_metadata_csv) if line.strip()}
    for it in items:
        if any(k in it.id for k in keep):
            yield it


# ------------------------------------------------------------------------------ MedleyDB


def medleydb_vocals(root: str | Path, sr: int = 24_000) -> Iterator[Item]:
    """Isolated vocal stems from MedleyDB (needs PyYAML for the metadata)."""
    import yaml

    root = Path(root)
    for meta in sorted(root.rglob("*_METADATA.yaml")):
        info = yaml.safe_load(open(meta))
        stems_dir = meta.parent / info.get("stem_dir", "")
        for sid, stem in (info.get("stems") or {}).items():
            inst = str(stem.get("instrument", "")).lower()
            if "singer" in inst or "vocal" in inst:
                path = stems_dir / stem["filename"]
                if path.exists():
                    yield Item(f"medleydb/{path.stem}", info.get("artist", meta.parent.name), _read(path, sr))


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


def read_key_labels(path: str | Path) -> list[dict]:
    """CSV with columns path,key_rater1,key_rater2 for the ~100 hand-labelled excerpts."""
    with open(path) as f:
        return list(csv.DictReader(f))
