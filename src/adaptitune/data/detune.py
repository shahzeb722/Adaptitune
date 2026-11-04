"""Synthetic detuning: paired out-of-tune / in-tune singing with exact per-note labels
(proposal 5.2, objective O1).

Pipeline for each clean take:

1. Get its F0 (the exact contour for synthetic takes, otherwise the analysis tracker) and
   split it into notes.
2. Keep notes that last at least 120 ms and whose own centre is within 20 cents of a
   semitone; that semitone is the note's intended target. Other notes are left untouched
   and are not labelled.
3. Inject errors into each kept note: a constant offset of 10-100 cents (random sign), plus
   optionally slow drift, an onset scoop and exaggerated or suppressed vibrato. The exact
   per-frame injection curve is stored next to the audio.
4. Render the same injection with **both** WORLD and PSOLA, so half of the detuned copies come
   from each shifter, every note can be compared across them, and neither shifter can win by
   undoing its own artifacts.
5. **QC gate.** Re-track the clean take and each detuned copy with the QC tracker and compare
   the measured pitch difference with the stored curve, note by note. A label is kept for
   evaluation only if the median difference is within ``QC_TOL_CENTS`` and enough of the
   note was tracked. Failed labels stay in the file with the reason, and the share is
   reported, so "every injected offset can be recovered from the stored labels" holds for
   every label used, and the cost of the gate is visible.
6. Notes with |offset| > 50 cents go in a separate "highly detuned" set, because a perfect
   corrector may legitimately pick the neighbouring semitone for them.

Each take gets a random generator seeded from the corpus seed and the take id, so a take's
detuning does not depend on which other takes are built, in what order or in how many
processes. ``docs/O1_CORPUS.md`` is the datasheet.
"""

from __future__ import annotations

import hashlib
import json
import time
import zlib
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from .. import metrics as M
from ..audio import Audio, save
from ..correction.planner import Plan, _robust_center, _smooth, _vibrato_window, segment_notes, VIBRATO_WINDOW_S
from ..correction.shifter import PSOLA, WORLD, QualityConfig, apply, prepare, shift
from ..pitch.trackers import PitchTrack, Tracker, get_tracker

SCHEMA_VERSION = 2
KEEP_WITHIN_CENTS = 20
HIGH_DETUNE_CENTS = 50
MIN_LABEL_NOTE_S = 0.12
ROUTES = (WORLD, PSOLA)
QC_TOL_CENTS = 10.0  # well under the 25-cent O3 threshold and near the pitch JND
QC_MIN_COVERAGE = 0.6  # share of the note's core that both contours must cover
QC_EDGE_S = 0.02  # ignore the first and last 20 ms (crossfades, tracker onsets)
HOP_S = 0.005


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
    # schema 2
    schema: int = SCHEMA_VERSION
    source: str = ""
    gender: str = ""
    scoop_s: float = 0.0
    source_dev_cents: float = 0.0  # the clean note's own distance from its target
    injected_mean_cents: float = 0.0  # mean of the per-frame injection over the note
    qc_pass: bool = True
    qc_reason: str = ""
    qc_coverage: float = float("nan")
    qc_residual_cents: float = float("nan")  # median(measured difference - stored curve)
    recovered_offset_cents: float = float("nan")  # offset implied by the measurement


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


def take_rng(seed: int, take_id: str, stream: int = 0) -> np.random.Generator:
    """Independent, reproducible random stream per take."""
    return np.random.default_rng([int(seed), zlib.crc32(take_id.encode()), int(stream)])


def design_take(midi: np.ndarray, hop: float, rng: np.random.Generator):
    """Notes, a spec per note (None = left untouched) and the injection curve (semitones)."""
    notes = segment_notes(midi, hop)
    specs: list[DetuneSpec | None] = []
    for a, b in notes:
        center = _robust_center(midi[a:b])
        target = int(round(center))
        if np.isnan(center) or abs(100 * (center - target)) > KEEP_WITHIN_CENTS or (b - a) * hop < MIN_LABEL_NOTE_S:
            specs.append(None)
        else:
            specs.append(sample_spec(rng))
    return notes, specs, injection_curve(midi, hop, notes, specs)


def render_routes(audio: Audio, voicing: np.ndarray, midi: np.ndarray, inj: np.ndarray, hop: float, routes=ROUTES) -> dict[str, Audio]:
    """Render one injection curve with each shifter. WORLD analysis is done once and shared."""
    plan = Plan(hop, midi, [], inj)
    prep = prepare(audio.samples, audio.sr, midi, voicing, hop, QualityConfig(force=WORLD), work_sr=audio.sr)
    out = {}
    for route in routes:
        for s in prep.segments:
            s.route = route
        res = apply(prep, plan)
        out[route] = Audio(res.samples, res.sr)
    return out


def note_qc(mc: np.ndarray, md: np.ndarray, inj_cents: np.ndarray, a: int, b: int, hop: float,
            tol: float = QC_TOL_CENTS, min_cov: float = QC_MIN_COVERAGE) -> dict:
    """Compare the measured clean-to-detuned pitch difference with the stored curve on one note."""
    edge = int(round(QC_EDGE_S / hop))
    lo, hi = a + edge, b - edge
    if hi - lo < 3:
        lo, hi = a, b
    c, d, i = mc[lo:hi], md[lo:hi], inj_cents[lo:hi]
    ok = ~np.isnan(c) & ~np.isnan(d)
    cov = float(ok.mean()) if len(ok) else 0.0
    if ok.sum() < 3:
        return dict(qc_pass=False, qc_reason="low_coverage", qc_coverage=cov, qc_residual_cents=float("nan"))
    resid = float(np.median(100 * (d[ok] - c[ok]) - i[ok]))
    if cov < min_cov:
        reason = "low_coverage"
    elif abs(resid) > 600:
        reason = "octave_error"
    elif abs(resid) > tol:
        reason = "offset_mismatch"
    else:
        reason = ""
    return dict(qc_pass=not reason, qc_reason=reason, qc_coverage=round(cov, 3), qc_residual_cents=round(resid, 2))


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    return x[:n] if len(x) >= n else np.concatenate([x, np.full(n - len(x), np.nan)])


def detune_take(take_id: str, audio: Audio, track: PitchTrack, route: str, rng: np.random.Generator, singer: str = "") -> tuple[Audio, list[NoteLabel]]:
    """One detuned copy with one shifter, without QC (kept for scripts that need a single
    copy; the corpus builder uses :func:`build_corpus`)."""
    hop = track.hop_s
    midi = M.contour_midi(track.f0)
    notes, specs, inj = design_take(midi, hop, rng)
    labels = [_label(take_id, i, a, b, midi, sp, hop, route, singer) for i, ((a, b), sp) in enumerate(zip(notes, specs)) if sp is not None]
    p = Plan(hop, midi, [], inj)
    out = shift(audio.samples, audio.sr, p, track.voicing, QualityConfig(force=route), work_sr=audio.sr)
    return Audio(out.samples, out.sr), labels


def _label(take, i, a, b, midi, sp: DetuneSpec, hop, route, singer, inj=None, **extra) -> NoteLabel:
    center = _robust_center(midi[a:b])
    target = int(round(center))
    lb = NoteLabel(
        take, i, round(a * hop, 4), round(b * hop, 4), target, round(float(center), 4),
        round(sp.offset_cents, 2), round(sp.drift_cents, 2), round(sp.scoop_cents, 2),
        round(sp.vibrato_factor, 3), abs(sp.offset_cents) > HIGH_DETUNE_CENTS, route, singer,
        scoop_s=round(sp.scoop_s, 3) if sp.scoop_cents else 0.0,
        source_dev_cents=round(100 * (float(center) - target), 2),
        injected_mean_cents=round(float(np.mean(inj[a:b]) * 100), 2) if inj is not None else 0.0,
    )
    for k, v in extra.items():
        setattr(lb, k, v)
    return lb


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

    @property
    def gender(self) -> str:
        return "M" if self.low < 55 else "F"


def make_singers(n: int, seed: int = 0) -> list[SynthSinger]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        base = int(rng.choice([45, 50, 55, 57, 60, 62]))  # bass .. soprano-ish
        vowels = tuple(str(v) for v in rng.choice(list("aeiou"), size=int(rng.integers(2, 5)), replace=False))
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

_TRACKERS: dict[str, Tracker] = {}


def _tracker(name: str) -> Tracker:
    if name not in _TRACKERS:
        _TRACKERS[name] = get_tracker(name)
    return _TRACKERS[name]


def _take_descriptors(source: str, root: str | None, n_singers: int, takes_per_singer: int, seed: int,
                      techniques: tuple[str, ...] | None, skipped: list | None = None) -> list[dict]:
    if source == "synthetic":
        out = []
        for sg in make_singers(n_singers, seed):
            for k in range(takes_per_singer):
                out.append(dict(take=f"{sg.name}_t{k:02d}", singer=sg.name, gender=sg.gender, source="synthetic", synth=asdict(sg)))
        return out
    import soundfile as sf

    if source == "vocalset":
        from .datasets import VOCALSET_DEFAULT_TECHNIQUES, vocalset_files

        files = vocalset_files(root, techniques or VOCALSET_DEFAULT_TECHNIQUES, skipped=skipped)
        out = []
        for f in files:
            stem = Path(f["path"]).stem
            out.append(dict(take=f"vocalset__{f['singer']}__{f['context']}__{f['technique']}__{stem}", singer=f["singer"],
                            gender=f["gender"], source="vocalset", path=f["path"], context=f["context"], technique=f["technique"],
                            duration=float(sf.info(f["path"]).duration)))
        return out
    if source == "medleydb":
        import yaml

        out = []
        for meta in sorted(Path(root).rglob("*_METADATA.yaml")):
            info = yaml.safe_load(meta.read_text())
            stems_dir = meta.parent / info.get("stem_dir", "")
            for stem in (info.get("stems") or {}).values():
                inst = str(stem.get("instrument", "")).lower()
                path = stems_dir / str(stem.get("filename", ""))
                if ("singer" in inst or "vocal" in inst) and path.is_file():
                    singer = f"medleydb_{info.get('artist', meta.parent.name)}"
                    out.append(dict(take=f"medleydb__{path.stem}", singer=singer, gender="?", source="medleydb", path=str(path),
                                    context="song", technique=inst, duration=float(sf.info(str(path)).duration)))
        return out
    raise ValueError(f"unknown source {source!r}")


def _process_take(desc: dict, cfg: dict) -> dict:
    """Build one take: clean + one detuned copy per route + labels, with the QC gate."""
    from ..music import Key
    from ..pitch.key import analyze_key, fit_key
    from .datasets import _read

    out = Path(cfg["out"])
    tid = desc["take"]
    rec_path = out / "labels" / f"{tid}.json"
    if cfg.get("resume") and rec_path.exists():
        return json.loads(rec_path.read_text())
    sr = cfg["sr"]
    if desc["source"] == "synthetic":
        sg = SynthSinger(**{**desc["synth"], "vowel_pool": tuple(desc["synth"]["vowel_pool"])})
        audio, track, key_txt = synth_clean_take(sg, take_rng(cfg["seed"], tid, 0), sr)
        tonic, mode = key_txt.split(":")
        given_key = Key(int(tonic), mode)
    else:
        audio = _read(Path(desc["path"]), sr)
        track = _tracker(cfg["tracker"]).track(audio.samples, audio.sr, HOP_S)
        given_key = None
    track = track.resampled(HOP_S)
    hop = HOP_S
    midi = M.contour_midi(track.f0)
    n = len(midi)
    notes, specs, inj = design_take(midi, hop, take_rng(cfg["seed"], tid, 1))
    inj_cents = inj * 100
    copies = render_routes(audio, track.voicing, midi, inj, hop, cfg["routes"])

    # level: the same gain for the clean take and its copies, so nothing clips at 16 bit
    peak = max([np.max(np.abs(audio.samples))] + [np.max(np.abs(c.samples)) for c in copies.values()])
    gain = 0.99 / peak if peak > 0.99 else 1.0

    # QC: measure clean -> detuned pitch difference with the QC tracker
    qtrk = _tracker(cfg["qc_tracker"])
    mc = _fit(M.contour_midi(qtrk.track(audio.samples, audio.sr, hop).f0), n)
    measured = {}
    for route, c in copies.items():
        md = _fit(M.contour_midi(qtrk.track(c.samples, c.sr, hop).f0), n)
        measured[route] = md

    note_index = np.full(n, -1, np.int32)
    labels, manifest = [], []
    for route in cfg["routes"]:
        for i, ((a, b), sp) in enumerate(zip(notes, specs)):
            if sp is None:
                continue
            note_index[a:b] = i
            qc = note_qc(mc, measured[route], inj_cents, a, b, hop, cfg["qc_tol"])
            lb = _label(tid, i, a, b, midi, sp, hop, route, desc["singer"], inj, split=desc["split"], source=desc["source"], gender=desc.get("gender", ""), **qc)
            lb.recovered_offset_cents = round(lb.offset_cents + qc["qc_residual_cents"], 2) if not np.isnan(qc["qc_residual_cents"]) else float("nan")
            labels.append(asdict(lb))

    # the key a singer would type in: the best-ranked key containing every labelled target
    if given_key is not None:
        key = given_key
    else:
        ranked = analyze_key(track.f0, track.voicing, hop).ranked()
        targets = {lb["target_midi"] % 12 for lb in labels}
        key = fit_key(ranked, targets) if ranked else Key(0, "major")
    in_scale = [lb["target_midi"] % 12 in key.pitch_classes for lb in labels]
    centres = np.array([_robust_center(midi[a:b]) for a, b in notes]) if notes else np.zeros(0)
    durs = np.array([(b - a) * hop for a, b in notes]) if notes else np.zeros(0)
    from ..music import tuning_offset_cents

    tuning = tuning_offset_cents(centres, durs) if len(centres) else 0.0

    save(out / "clean" / f"{tid}.wav", Audio(audio.samples * gain, sr), "PCM_16")
    for route, c in copies.items():
        save(out / "detuned" / route / f"{tid}.wav", Audio(c.samples * gain, sr), "PCM_16")
    np.savez_compressed(
        out / "labels" / f"{tid}.npz",
        hop_s=hop, clean_midi=midi.astype(np.float32), voicing=track.voicing[:n].astype(np.float32),
        injection_cents=inj_cents.astype(np.float32), note_index=note_index,
        qc_clean_midi=mc.astype(np.float32), **{f"qc_{r}_midi": m.astype(np.float32) for r, m in measured.items()},
    )
    for route in cfg["routes"]:
        lbs = [lb for lb in labels if lb["route"] == route]
        manifest.append(dict(
            take=tid, singer=desc["singer"], gender=desc.get("gender", ""), source=desc["source"], split=desc["split"], route=route,
            key=key.name, clean=f"clean/{tid}.wav", detuned=f"detuned/{route}/{tid}.wav", labels=f"labels/{tid}.npz",
            duration=round(audio.duration, 3), n_notes=len(notes), n_labelled=len(lbs), n_qc_pass=sum(lb["qc_pass"] for lb in lbs),
            key_covers_targets=float(np.mean(in_scale)) if in_scale else 1.0, tuning_cents=round(float(tuning), 2), gain=round(gain, 4),
            context=desc.get("context", ""), technique=desc.get("technique", ""), source_path=desc.get("path", ""),
        ))
    rec = _nan_to_none(dict(take=tid, labels=labels, manifest=manifest))
    rec_path.write_text(json.dumps(rec))
    return rec


def _nan_to_none(obj):
    """Strict JSON: NaN becomes null."""
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, np.generic):
        return _nan_to_none(obj.item())
    return obj


def _process_take_safe(desc: dict, cfg: dict) -> dict:
    try:
        return _process_take(desc, cfg)
    except Exception as e:  # one bad file must not stop a multi-hour build
        return dict(take=desc["take"], error=f"{type(e).__name__}: {e}", labels=[], manifest=[])


def build_corpus(
    out_dir: str | Path,
    source: str = "synthetic",
    root: str | None = None,
    n_singers: int = 20,
    takes_per_singer: int = 6,
    tracker: str = "auto",
    seed: int = 0,
    sr: int = 24_000,
    routes: tuple[str, ...] = ROUTES,
    qc_tracker: str = "pyin",
    qc_tol_cents: float = QC_TOL_CENTS,
    min_test_hours: float | None = None,
    techniques: tuple[str, ...] | None = None,
    limit: int | None = None,
    workers: int = 1,
    resume: bool = False,
    verbose: bool = False,
) -> Path:
    """Write ``clean/``, ``detuned/<route>/``, ``labels/`` (per-take curves), ``labels.jsonl``,
    ``manifest.csv``, ``split.json``, ``build_info.json`` and ``checksums.sha256``.

    ``source`` is ``synthetic`` (no download needed), ``vocalset`` or ``medleydb``. For real
    sources the singer split is planned so the test set holds at least ``min_test_hours``
    (default 2 h, the O1 target) of unique clean audio; synthetic corpora use a 70/15/15 split.
    ``workers > 1`` builds takes in parallel; the result is identical to a serial build.
    """
    from .datasets import plan_split, split_by_singer

    t0 = time.time()
    out = Path(out_dir)
    for d in ("clean", "labels", *[f"detuned/{r}" for r in routes]):
        (out / d).mkdir(parents=True, exist_ok=True)
    skipped: list[dict] = []
    descs = _take_descriptors(source, root, n_singers, takes_per_singer, seed, techniques, skipped)
    if not descs:
        raise FileNotFoundError(f"no {source} takes found under {root}")

    groups = {d["singer"]: d.get("gender", "") for d in descs}
    if source == "synthetic":  # durations are only known after synthesis; use fixed ratios
        assign = split_by_singer(list(groups), seed)
        split_info = {"method": "ratio 0.7/0.15/0.15", "assignment": assign}
    else:
        hours: dict[str, float] = {}
        for d in descs:
            hours[d["singer"]] = hours.get(d["singer"], 0.0) + d.get("duration", 0.0) / 3600
        split_info = plan_split(hours, groups, 2.0 if min_test_hours is None else min_test_hours, seed=seed)
        split_info["method"] = "plan_split (fewest test singers reaching the target)"
        assign = split_info["assignment"]
    for d in descs:
        d["split"] = assign[d["singer"]]
    if limit:
        descs = descs[:limit]
    (out / "split.json").write_text(json.dumps(split_info, indent=2, default=float))

    cfg = dict(out=str(out), sr=sr, routes=list(routes), tracker=tracker, qc_tracker=qc_tracker, qc_tol=qc_tol_cents, seed=seed, resume=resume)
    records = []
    if workers > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            for k, rec in enumerate(ex.map(_process_take_safe, descs, [cfg] * len(descs), chunksize=1)):
                records.append(rec)
                if verbose:
                    print(f"[{k + 1}/{len(descs)}] {rec['take']} {rec.get('error', '')}", flush=True)
    else:
        for k, d in enumerate(descs):
            rec = _process_take_safe(d, cfg)
            records.append(rec)
            if verbose:
                print(f"[{k + 1}/{len(descs)}] {rec['take']} {rec.get('error', '')}", flush=True)

    import pandas as pd

    labels = [lb for r in records for lb in r["labels"]]
    with open(out / "labels.jsonl", "w") as f:
        for lb in labels:
            f.write(json.dumps(lb) + "\n")
    man = pd.DataFrame([m for r in records for m in r["manifest"]])
    man.to_csv(out / "manifest.csv", index=False)
    errors = {r["take"]: r["error"] for r in records if r.get("error")}
    info = dict(
        schema=SCHEMA_VERSION, source=source, root=str(root) if root else None, seed=seed, sr=sr, routes=list(routes),
        tracker=tracker, qc_tracker=qc_tracker, qc_tol_cents=qc_tol_cents, qc_min_coverage=QC_MIN_COVERAGE,
        keep_within_cents=KEEP_WITHIN_CENTS, high_detune_cents=HIGH_DETUNE_CENTS, min_label_note_s=MIN_LABEL_NOTE_S,
        takes=len(records), failed_takes=errors, source_files_skipped=skipped, labels=len(labels), seconds=round(time.time() - t0, 1),
        versions=_versions(), git_commit=_git_commit(), created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
    (out / "build_info.json").write_text(json.dumps(info, indent=2, default=str))
    write_checksums(out)
    corpus_stats(out)
    return out


def _versions() -> dict:
    from importlib import metadata

    vs = {}
    for p in ("adaptitune", "numpy", "scipy", "librosa", "soundfile", "pyworld", "praat-parselmouth", "torch",
              "onnxruntime", "torchfcpe", "pesto-pitch", "torchcrepe"):
        try:
            vs[p] = metadata.version(p)
        except Exception:
            vs[p] = None
    return vs


def _git_commit() -> str | None:
    import subprocess

    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None
    except Exception:
        return None


# ------------------------------------------------------------------------------ integrity


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _corpus_files(corpus: Path) -> list[Path]:
    files = [p for d in ("clean", "detuned", "labels") for p in sorted((corpus / d).rglob("*")) if p.is_file() and p.suffix in (".wav", ".npz")]
    return files + [corpus / f for f in ("labels.jsonl", "manifest.csv") if (corpus / f).exists()]


def write_checksums(corpus: str | Path) -> Path:
    corpus = Path(corpus)
    lines = [f"{_sha256(p)}  {p.relative_to(corpus).as_posix()}" for p in _corpus_files(corpus)]
    (corpus / "checksums.sha256").write_text("\n".join(lines) + "\n")
    return corpus / "checksums.sha256"


def verify_checksums(corpus: str | Path) -> dict:
    corpus = Path(corpus)
    f = corpus / "checksums.sha256"
    if not f.exists():
        return {"checked": 0, "missing_file": True, "bad": []}
    bad = []
    n = 0
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(None, 1)
        p = corpus / rel.strip()
        n += 1
        if not p.exists() or _sha256(p) != digest:
            bad.append(rel.strip())
    return {"checked": n, "bad": bad}


# ------------------------------------------------------------------------------ reading back


def load_labels(corpus_dir: str | Path, qc_only: bool = False) -> list[NoteLabel]:
    names = {f.name for f in fields(NoteLabel)}
    out = []
    with open(Path(corpus_dir) / "labels.jsonl") as f:
        for line in f:
            d = json.loads(line)
            lb = NoteLabel(**{k: (float("nan") if v is None and k not in ("qc_reason", "split", "source", "gender") else v) for k, v in d.items() if k in names})
            for k in ("qc_reason", "split", "source", "gender"):
                if getattr(lb, k) is None:
                    setattr(lb, k, "")
            if qc_only and not lb.qc_pass:
                continue
            out.append(lb)
    return out


def load_manifest(corpus_dir: str | Path):
    import pandas as pd

    man = pd.read_csv(Path(corpus_dir) / "manifest.csv", keep_default_na=False)
    if "detuned" not in man:  # schema 1
        man["detuned"] = "detuned/" + man["take"] + ".wav"
    return man


def verify_labels(corpus_dir: str | Path, tracker: Tracker | str = "auto", max_takes: int | None = None,
                  tol_cents: float = QC_TOL_CENTS, split: str | None = None, check_files: bool = True) -> dict:
    """O1 check: every injected offset can be recovered from the stored labels.

    Re-tracks the clean take and every detuned copy with ``tracker`` and compares the measured
    pitch difference with the stored per-frame injection curve on every QC-passed note. With a
    tracker other than the build's QC tracker this is an independent re-measurement; with the
    same one it is a self-check. Also verifies the checksums of every corpus file.
    """
    import pandas as pd

    from ..audio import load

    corpus = Path(corpus_dir)
    info = json.loads((corpus / "build_info.json").read_text()) if (corpus / "build_info.json").exists() else {}
    trk = get_tracker(tracker) if isinstance(tracker, str) else tracker
    man = load_manifest(corpus)
    labels = pd.DataFrame([asdict(lb) for lb in load_labels(corpus)])
    if split:
        labels = labels[labels.split == split]
    all_kept = labels[labels.qc_pass]
    rows = []
    for k, (take, g) in enumerate(all_kept.groupby("take")):
        if max_takes and k >= max_takes:
            break
        npz_path = corpus / "labels" / f"{take}.npz"
        clean = load(corpus / "clean" / f"{take}.wav", normalize=False)
        mc = M.contour_midi(trk.track(clean.samples, clean.sr, HOP_S).f0)
        if npz_path.exists():
            z = np.load(npz_path)
            inj = z["injection_cents"].astype(float)
        else:  # schema 1: rebuild the constant part only
            inj = None
        for route, gr in g.groupby("route"):
            row = man[(man["take"] == take) & (man["route"] == route)]
            det_path = corpus / (row.detuned.iloc[0] if len(row) else f"detuned/{take}.wav")
            det = load(det_path, normalize=False)
            md = M.contour_midi(trk.track(det.samples, det.sr, HOP_S).f0)
            n = len(inj) if inj is not None else len(mc)
            mc_, md_ = _fit(mc, n), _fit(md, n)
            for _, r in gr.iterrows():
                a, b = int(round(r.start / HOP_S)), int(round(r.end / HOP_S))
                if inj is None:
                    curve = np.full(n, 0.0)
                    curve[a:b] = r.offset_cents + r.drift_cents * np.linspace(0, 1, b - a)
                else:
                    curve = inj
                qc = note_qc(mc_, md_, curve, a, b, HOP_S, tol_cents)
                rows.append(dict(take=take, index=r["index"], route=route, residual=qc["qc_residual_cents"], coverage=qc["qc_coverage"],
                                 ok=qc["qc_pass"], reason=qc["qc_reason"], highly_detuned=r.highly_detuned))
    res = pd.DataFrame(rows)
    qc_tracker = info.get("qc_tracker")
    name = getattr(trk, "name", str(tracker))
    out = {
        "tracker": name,
        "independent_of_build_qc": bool(qc_tracker and qc_tracker != name),
        "labels_total": int(len(labels)),
        "labels_failed_build_qc": int((~labels.qc_pass).sum()),
        "build_qc_pass_share": float(labels.qc_pass.mean()) if len(labels) else float("nan"),
        "notes_checked": int(len(res)),
        "recovered_share": float(res.ok.mean()) if len(res) else 0.0,
        "median_abs_residual_cents": float(res.residual.abs().median()) if len(res) else float("nan"),
        "p95_abs_residual_cents": float(res.residual.abs().quantile(0.95)) if len(res) else float("nan"),
        "by_route": {r: {"recovered_share": float(g.ok.mean()), "median_abs_residual_cents": float(g.residual.abs().median())} for r, g in res.groupby("route")} if len(res) else {},
        "failures": res[~res.ok][["take", "index", "route", "residual", "coverage", "reason"]].to_dict("records")[:50] if len(res) else [],
        "tolerance_cents": tol_cents,
    }
    if check_files:
        cs = verify_checksums(corpus)
        out["files_checked"] = cs["checked"]
        out["files_bad"] = cs["bad"][:50]
    return out


def corpus_stats(corpus_dir: str | Path, write: bool = True) -> dict:
    """Composition of the corpus and the O1 checks that can be read off the files."""
    import pandas as pd

    corpus = Path(corpus_dir)
    man = load_manifest(corpus)
    labels = pd.DataFrame([asdict(lb) for lb in load_labels(corpus)])
    takes = man.drop_duplicates("take")
    hours = takes.groupby("split").duration.sum().div(3600).to_dict()
    singers = takes.groupby("split").singer.apply(lambda s: sorted(set(s))).to_dict()
    overlap = sorted(set(singers.get("test", [])) & (set(singers.get("train", [])) | set(singers.get("val", []))))
    s: dict = {
        "takes": int(len(takes)),
        "routes": sorted(man.route.unique().tolist()),
        "unique_hours": {k: round(v, 3) for k, v in hours.items()},
        "unique_hours_total": round(float(takes.duration.sum() / 3600), 3),
        "singers": {k: len(v) for k, v in singers.items()},
        "gender": {sp: g.drop_duplicates("singer").gender.value_counts().to_dict() for sp, g in takes.groupby("split")},
        "singer_overlap_test_vs_dev": overlap,
    }
    if len(labels):
        per_note = labels.drop_duplicates(["take", "index"])
        s["labelled_notes"] = int(len(per_note))
        s["labels"] = int(len(labels))
        s["labelled_note_hours"] = {sp: round(float((g.end - g.start).sum() / 3600), 3) for sp, g in per_note.groupby("split")}
        s["kept_note_share"] = round(float(takes.n_labelled.sum() / max(takes.n_notes.sum(), 1)), 3)
        s["highly_detuned_share"] = round(float(per_note.highly_detuned.mean()), 3)
        s["qc_pass_share"] = round(float(labels.qc_pass.mean()), 4)
        s["qc_pass_share_by_route"] = {r: round(float(g.qc_pass.mean()), 4) for r, g in labels.groupby("route")}
        s["qc_fail_reasons"] = labels[~labels.qc_pass].qc_reason.value_counts().to_dict()
        s["qc_abs_residual_cents"] = {
            "median": round(float(labels.qc_residual_cents.abs().median()), 2),
            "p95": round(float(labels.qc_residual_cents.abs().quantile(0.95)), 2),
        }
        off = per_note.offset_cents.abs()
        s["offset_abs_cents"] = {"min": round(float(off.min()), 1), "median": round(float(off.median()), 1), "max": round(float(off.max()), 1)}
        s["offset_sign_positive_share"] = round(float((per_note.offset_cents > 0).mean()), 3)
        s["drift_share"] = round(float((per_note.drift_cents != 0).mean()), 3)
        s["scoop_share"] = round(float((per_note.scoop_cents != 0).mean()), 3)
        s["vibrato_changed_share"] = round(float((per_note.vibrato_factor != 1).mean()), 3)
        s["source_dev_abs_cents_median"] = round(float(per_note.source_dev_cents.abs().median()), 2)
        test_kept = labels[(labels.split == "test") & labels.qc_pass]
        s["test_labels_kept"] = int(len(test_kept))
    if "technique" in takes and takes.technique.astype(str).str.len().any():
        s["hours_by_technique"] = {k: round(v / 3600, 3) for k, v in takes.groupby("technique").duration.sum().items()}
    s["O1_test_hours_ok"] = bool(hours.get("test", 0.0) >= 2.0)
    s["O1_singers_disjoint"] = not overlap
    if write:
        (corpus / "stats.json").write_text(json.dumps(s, indent=2, default=float))
    return s
