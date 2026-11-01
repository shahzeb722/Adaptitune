"""Synthetic detuning: paired out-of-tune / in-tune singing with exact per-note labels
(proposal 5.2, objective O1).

Pipeline for each clean take:

1. Get its F0 (reference annotation if available, else the tracker) and split it into notes.
2. Keep notes whose own centre is within 20 cents of a semitone; that semitone is the note's
   intended target. Other notes are left untouched and are not labelled.
3. Inject errors into each kept note: a constant offset of 10-100 cents (random sign), plus
   optionally slow drift, an onset scoop and exaggerated or suppressed vibrato.
4. Render with WORLD for half of the takes and PSOLA for the other half (alternating), so
   neither shifter can win by undoing its own artifacts. The route is stored per take.
5. Notes with |offset| > 50 cents go in a separate "highly detuned" set, because a perfect
   corrector may legitimately pick the neighbouring semitone for them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .. import metrics as M
from ..audio import Audio, save
from ..correction.planner import Plan, _robust_center, _smooth, _vibrato_window, segment_notes, VIBRATO_WINDOW_S
from ..correction.shifter import PSOLA, WORLD, QualityConfig, shift
from ..pitch.trackers import PitchTrack, Tracker, get_tracker

KEEP_WITHIN_CENTS = 20
HIGH_DETUNE_CENTS = 50


@dataclass
class DetuneSpec:
    offset_cents: float
    drift_cents: float = 0.0
    scoop_cents: float = 0.0
    scoop_s: float = 0.08
    vibrato_factor: float = 1.0  # >1 exaggerate, <1 suppress


@dataclass
class NoteLabel:
    take: str
    index: int
    start: float
    end: float
    target_midi: int
    source_center_midi: float
    offset_cents: float
    drift_cents: float
    scoop_cents: float
    vibrato_factor: float
    highly_detuned: bool
    route: str
    singer: str
    split: str = ""


def sample_spec(rng: np.random.Generator, p_drift=0.35, p_scoop=0.35, p_vib=0.3) -> DetuneSpec:
    off = rng.uniform(10, 100) * rng.choice([-1, 1])
    spec = DetuneSpec(offset_cents=float(off))
    if rng.random() < p_drift:
        spec.drift_cents = float(rng.uniform(15, 40) * rng.choice([-1, 1]))
    if rng.random() < p_scoop:
        spec.scoop_cents = float(-rng.uniform(40, 150))  # scoops come from below
        spec.scoop_s = float(rng.uniform(0.05, 0.15))
    if rng.random() < p_vib:
        spec.vibrato_factor = float(rng.choice([rng.uniform(1.5, 2.2), rng.uniform(0.0, 0.5)]))
    return spec


def injection_curve(midi: np.ndarray, hop_s: float, notes: list[tuple[int, int]], specs: list[DetuneSpec | None]) -> np.ndarray:
    """Per-frame semitone offsets that turn the clean take into the detuned one."""
    inj = np.zeros(len(midi))
    default_win = max(1, int(VIBRATO_WINDOW_S / hop_s))
    for (a, b), sp in zip(notes, specs):
        if sp is None:
            continue
        seg = midi[a:b]
        t = np.arange(b - a) * hop_s
        dur = max((b - a) * hop_s, hop_s)
        vib = seg - _smooth(seg, _vibrato_window(seg, hop_s, default_win))
        c = np.full(b - a, sp.offset_cents / 100)
        c += (sp.drift_cents / 100) * t / dur
        if sp.scoop_cents:
            c += (sp.scoop_cents / 100) * np.exp(-t / max(sp.scoop_s / 3, 1e-3))
        c += (sp.vibrato_factor - 1) * np.nan_to_num(vib)
        # 10 ms fades so note boundaries do not click
        k = min(len(c) // 2, max(1, int(0.01 / hop_s)))
        ramp = np.ones(len(c))
        ramp[:k] = np.linspace(0, 1, k)
        ramp[-k:] = np.minimum(ramp[-k:], np.linspace(1, 0, k))
        if sp.scoop_cents:
            ramp[:k] = 1.0  # a scoop starts away from the note by definition
        inj[a:b] = c * ramp
    inj[np.isnan(midi)] = 0.0
    return inj


def detune_take(
    take_id: str,
    audio: Audio,
    track: PitchTrack,
    route: str,
    rng: np.random.Generator,
    singer: str = "",
) -> tuple[Audio, list[NoteLabel]]:
    hop = track.hop_s
    midi = M.contour_midi(track.f0)
    notes = segment_notes(midi, hop)
    specs: list[DetuneSpec | None] = []
    labels: list[NoteLabel] = []
    for i, (a, b) in enumerate(notes):
        center = _robust_center(midi[a:b])
        target = int(round(center))
        if abs(100 * (center - target)) > KEEP_WITHIN_CENTS or (b - a) * hop < 0.12:
            specs.append(None)
            continue
        sp = sample_spec(rng)
        specs.append(sp)
        labels.append(
            NoteLabel(
                take_id, i, round(a * hop, 4), round(b * hop, 4), target, round(float(center), 4),
                round(sp.offset_cents, 2), round(sp.drift_cents, 2), round(sp.scoop_cents, 2),
                round(sp.vibrato_factor, 3), abs(sp.offset_cents) > HIGH_DETUNE_CENTS, route, singer,
            )
        )
    inj = injection_curve(midi, hop, notes, specs)
    p = Plan(hop, midi, [], inj)
    out = shift(audio.samples, audio.sr, p, track.voicing, QualityConfig(force=route), work_sr=audio.sr)
    return Audio(out.samples, out.sr), labels


# ------------------------------------------------------------------------------ synthetic singers


@dataclass
class SynthSinger:
    name: str
    low: int
    high: int
    vowel_pool: tuple[str, ...]
    vib_rate: float
    vib_cents: float
    breath: float
    jitter: float


def make_singers(n: int, seed: int = 0) -> list[SynthSinger]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        base = int(rng.choice([45, 50, 55, 57, 60, 62]))  # bass .. soprano-ish
        vowels = tuple(rng.choice(list("aeiou"), size=int(rng.integers(2, 5)), replace=False))
        out.append(
            SynthSinger(
                f"synth{i:02d}", base, base + int(rng.integers(10, 16)), vowels,
                float(rng.uniform(4.5, 6.8)), float(rng.choice([0, rng.uniform(20, 70)])),
                float(rng.uniform(0.02, 0.12)), float(rng.uniform(1, 5)),
            )
        )
    return out


def synth_clean_take(singer: SynthSinger, rng: np.random.Generator, sr: int = 24_000):
    """An in-tune take (small natural deviations < 10 cents) with exact F0."""
    from .synth import SungNote, random_melody, sing

    tonic = int(rng.integers(0, 12))
    mode = str(rng.choice(["major", "minor"]))
    mel = random_melody(rng, int(rng.integers(6, 12)), tonic, mode, singer.low, singer.high)
    notes = [
        SungNote(
            m, float(rng.uniform(0.3, 1.2)), offset_cents=float(rng.normal(0, 4)),
            vibrato_rate=singer.vib_rate + float(rng.normal(0, 0.2)),
            vibrato_cents=singer.vib_cents * float(rng.uniform(0.7, 1.2)),
            vowel=str(rng.choice(singer.vowel_pool)), gap_after=float(rng.uniform(0.06, 0.25)),
        )
        for m in mel
    ]
    s = sing(notes, sr=sr, breath=singer.breath, jitter_cents=singer.jitter, seed=int(rng.integers(1 << 30)))
    track = PitchTrack(np.arange(len(s.f0)) * s.hop_s, s.f0, (s.f0 > 0).astype(float), s.hop_s, "truth")
    return Audio(s.samples, sr), track, f"{tonic}:{mode}"


# ------------------------------------------------------------------------------ corpus builder


def build_corpus(
    out_dir: str | Path,
    source: str = "synthetic",
    root: str | None = None,
    n_singers: int = 20,
    takes_per_singer: int = 6,
    tracker: Tracker | str = "auto",
    seed: int = 0,
    sr: int = 24_000,
) -> Path:
    """Write ``clean/``, ``detuned/``, ``labels.jsonl`` and ``manifest.csv``.

    ``source`` is ``synthetic`` (no download needed), ``vocalset`` or ``medleydb``.
    """
    from .datasets import medleydb_vocals, split_by_singer, vocalset

    out = Path(out_dir)
    (out / "clean").mkdir(parents=True, exist_ok=True)
    (out / "detuned").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    from ..music import Key
    from ..pitch.key import estimate_key

    takes = []  # (id, singer, audio, track, key name)
    if source == "synthetic":
        for sg in make_singers(n_singers, seed):
            for k in range(takes_per_singer):
                a, tr, key = synth_clean_take(sg, rng, sr)
                tonic, mode = key.split(":")
                takes.append((f"{sg.name}_t{k:02d}", sg.name, a, tr, Key(int(tonic), mode).name))
    else:
        trk = get_tracker(tracker) if isinstance(tracker, str) else tracker
        it = vocalset(root, sr) if source == "vocalset" else medleydb_vocals(root, sr)
        for item in it:
            tr = trk.track(item.audio.samples, item.audio.sr)
            # the key a user would type in, estimated from the clean (in-tune) take
            takes.append((item.id.replace("/", "__"), item.singer, item.audio, tr, estimate_key(tr.f0, tr.voicing)[0].name))

    split = split_by_singer([t[1] for t in takes], seed)
    all_labels = []
    with open(out / "manifest.csv", "w") as man:
        man.write("take,singer,split,route,key,clean,detuned,duration\n")
        for i, (tid, singer, a, tr, key) in enumerate(takes):
            route = WORLD if i % 2 == 0 else PSOLA
            det, labels = detune_take(tid, a, tr, route, rng, singer)
            for lb in labels:
                lb.split = split[singer]
            save(out / "clean" / f"{tid}.wav", a)
            save(out / "detuned" / f"{tid}.wav", det)
            all_labels += labels
            man.write(f"{tid},{singer},{split[singer]},{route},{key},clean/{tid}.wav,detuned/{tid}.wav,{a.duration:.3f}\n")
    with open(out / "labels.jsonl", "w") as f:
        for lb in all_labels:
            f.write(json.dumps(asdict(lb)) + "\n")
    return out


def load_labels(corpus_dir: str | Path) -> list[NoteLabel]:
    with open(Path(corpus_dir) / "labels.jsonl") as f:
        return [NoteLabel(**json.loads(line)) for line in f]


def verify_labels(corpus_dir: str | Path, tracker: Tracker | str = "pyin", max_takes: int | None = None, tol_cents: float = 15) -> dict:
    """O1 check: every injected offset can be recovered from the stored labels.

    For each labelled note, (centre of detuned note) - (centre of clean note) should equal the
    stored offset (plus mean drift) within ``tol_cents``.
    """
    import pandas as pd

    from ..audio import load

    trk = get_tracker(tracker) if isinstance(tracker, str) else tracker
    corpus = Path(corpus_dir)
    labels = load_labels(corpus)
    df = pd.DataFrame([asdict(lb) for lb in labels])
    rows = []
    for k, (take, g) in enumerate(df.groupby("take")):
        if max_takes and k >= max_takes:
            break
        clean = load(corpus / "clean" / f"{take}.wav", normalize=False)
        det = load(corpus / "detuned" / f"{take}.wav", normalize=False)
        mc = M.contour_midi(trk.track(clean.samples, clean.sr).f0)
        md = M.contour_midi(trk.track(det.samples, det.sr).f0)
        hop = 0.005
        for _, r in g.iterrows():
            a, b = int(r.start / hop), int(r.end / hop)
            # compare the middle of the note, away from scoops
            mid = slice(a + (b - a) // 3, b - (b - a) // 6)
            dc = np.nanmedian(md[mid]) - np.nanmedian(mc[mid])
            expected = r.offset_cents + r.drift_cents * (0.5 * (1 / 3 + 5 / 6))
            rows.append(dict(take=take, index=r["index"], measured=100 * dc, expected=expected, route=r.route))
    res = pd.DataFrame(rows)
    res["abs_err"] = (res.measured - res.expected).abs()
    return {
        "notes_checked": int(len(res)),
        "recovered_share": float((res.abs_err <= tol_cents).mean()) if len(res) else 0.0,
        "median_abs_err_cents": float(res.abs_err.median()) if len(res) else float("nan"),
        "by_route": res.groupby("route").abs_err.median().to_dict() if len(res) else {},
    }
