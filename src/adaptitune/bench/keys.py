"""Key-estimation validation on real melodies (part of O2).

O2 is judged on about 100 hand-labelled excerpts of real singing (``adaptitune key-excerpts``,
``tools/key_rater.html``, ``adaptitune key-agreement``, ``adaptitune eval-keys``). Before those
labels exist, this module checks and tunes the estimator on the **Essen folksong collection**
that ships with music21 (8,000+ monophonic songs):

* **Ground truth.** Each song's ABC ``K:`` field gives its tonic (in this encoding every
  accidental is written out in the notes, so ``K:`` names the tonic, not a key signature; we
  check that the songs end on it). The mode comes from the third above the tonic in the
  score: major if the major third is sung for longer than the minor third, minor if the
  reverse, and the song is left out if neither occurs. This is symbolic information the
  estimator never sees.
* **Singing.** Every song is transposed to a random key in a comfortable range and rendered
  as an F0 contour with the same pitch model as the synthetic singer (vibrato, scoops, drift,
  jitter, gaps), in three conditions: ``in_tune`` (professional), ``amateur`` (global tuning
  error plus 25-cent note scatter) and ``corpus`` (the O1 detuning: 10-100 cent offsets).
  Both whole songs and 8-20 s excerpts that stop before the last note (so the final tonic is
  not given away) are tested.
* **Selection.** Collections are split into training and test volumes. The configuration
  with the best mode-balanced accuracy (the mean of major and minor accuracy, so the many
  major songs cannot outvote the minor ones) on the training volumes (in-tune and amateur,
  songs and excerpts) becomes ``pitch.key.DEFAULT_CONFIG``; Aarden's profiles are derived from Essen
  itself, so they are reported but never chosen. Test accuracy is reported with Wilson 95 %
  intervals, by condition and for the Chinese (Han) volumes separately.
* **Audio check.** A sample of test excerpts is synthesized to audio and tracked, to confirm
  that tracker errors do not change the picture.
"""

from __future__ import annotations

import gzip
import json
import re
import zlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..music import NOTE_NAMES, Key
from ..pitch.key import (
    FAIR_PROFILES,
    KEYS,
    KeyConfig,
    mirex_key_score,
    note_events,
    pitch_class_histogram,
    same_key_or_relative,
    soft_histogram,
)

ESSEN_TRAIN = ("altdeu10", "ballad10", "ballad30", "ballad50", "ballad70", "boehme10", "erk10", "erk30", "fink0", "han1", "kinder0", "lot", "zuccal0")
ESSEN_TEST = ("altdeu20", "ballad20", "ballad40", "ballad60", "ballad80", "boehme20", "dva0", "erk20", "erk5", "han2", "irl", "lux", "folkHaydn")
CONDITIONS = ("in_tune", "amateur", "corpus")
KINDS = ("excerpt", "full")
SELECTION_CONDITIONS = ("in_tune", "amateur")
HOP_S = 0.005

_MINOR_MODES = ("m", "min", "minor", "aeo", "aeolian", "dor", "dorian", "phr", "phrygian", "loc", "locrian")
_MAJOR_MODES = ("maj", "major", "ion", "ionian", "mix", "mixolydian", "lyd", "lydian")


# ----------------------------------------------------------------------------- ground truth


def parse_key_field(k: str) -> tuple[int | None, str | None]:
    """'G' -> (7, None), 'Bb' -> (10, None), 'Dm' -> (2, 'minor'), 'Ador' -> (9, 'minor')."""
    k = k.strip()
    m = re.match(r"^([A-Ga-g])([#b]?)\s*([A-Za-z]*)", k)
    if not m:
        return None, None
    pc = NOTE_NAMES.index(m.group(1).upper())
    pc = (pc + (1 if m.group(2) == "#" else -1 if m.group(2) == "b" else 0)) % 12
    suffix = m.group(3).lower()
    mode = None
    if suffix in _MINOR_MODES or suffix.startswith(("min", "dor", "phr", "aeo", "loc")):
        mode = "minor"
    elif suffix in _MAJOR_MODES or suffix.startswith(("maj", "mix", "lyd", "ion")):
        mode = "major"
    return pc, mode


def mode_by_third(notes: list[tuple[int | None, float]], tonic: int) -> str | None:
    """Major if the major third above the tonic is sung for longer than the minor third."""
    m3 = sum(d for m, d in notes if m is not None and (m - tonic) % 12 == 3)
    M3 = sum(d for m, d in notes if m is not None and (m - tonic) % 12 == 4)
    if m3 == M3:
        return None
    return "major" if M3 > m3 else "minor"


def _parse_file(path: str) -> list[dict]:
    """Songs of one Essen ABC file: notes as (midi or None for rests, quarter lengths), ties merged."""
    from music21 import converter, note

    text = Path(path).read_text(encoding="latin-1")
    blocks = re.split(r"(?m)^(?=X:)", text)
    blocks = [b for b in blocks if b.startswith("X:")]
    try:
        op = converter.parse(path, format="abc")
        scores = list(op.scores) if hasattr(op, "scores") else [op]
    except Exception:
        return []
    out = []
    region = re.sub(r"\d+$", "", Path(path).stem)
    if len(scores) != len(blocks):  # cannot align scores and headers safely
        return []
    for i, (sc, blk) in enumerate(zip(scores, blocks)):
        km = re.search(r"(?m)^K:\s*(.+)$", blk)
        if not km:
            continue
        tonic, mode = parse_key_field(km.group(1))
        if tonic is None:
            continue
        notes: list[list] = []
        for el in sc.flatten().notesAndRests:
            ql = float(el.quarterLength)
            if ql <= 0:
                continue  # grace notes
            if isinstance(el, note.Note):
                tie = el.tie.type if el.tie is not None else None
                if tie in ("stop", "continue") and notes and notes[-1][0] == el.pitch.midi:
                    notes[-1][1] += ql
                else:
                    notes.append([int(el.pitch.midi), ql])
            elif isinstance(el, note.Rest):
                if notes and notes[-1][0] is None:
                    notes[-1][1] += ql
                else:
                    notes.append([None, ql])
        pitched = [n for n in notes if n[0] is not None]
        if len(pitched) < 8:
            continue
        explicit = mode is not None
        if mode is None:
            mode = mode_by_third([tuple(n) for n in notes], tonic)
        if mode is None:
            continue
        title = re.search(r"(?m)^T:\s*(.+)$", blk)
        out.append(dict(
            id=f"{Path(path).stem}/{i + 1}", file=Path(path).stem, region=region, title=title.group(1).strip() if title else "",
            tonic=tonic, mode=mode, explicit_mode=explicit, ends_on_tonic=bool(pitched[-1][0] % 12 == tonic), notes=notes,
        ))
    return out


def parse_essen(cache: str | Path | None = "results/keys/essen_melodies.json.gz", workers: int = 4) -> list[dict]:
    """All usable Essen songs from the music21 corpus (parsed once, then cached)."""
    if cache and Path(cache).exists():
        with gzip.open(cache, "rt") as f:
            return json.load(f)
    import music21

    root = Path(music21.__file__).parent / "corpus" / "essenFolksong"
    files = [str(root / f"{n}.abc") for n in ESSEN_TRAIN + ESSEN_TEST if (root / f"{n}.abc").exists()]
    if workers > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
            parts = list(ex.map(_parse_file, files))
    else:
        parts = [_parse_file(f) for f in files]
    songs = [s for p in parts for s in p]
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(cache, "wt") as f:
            json.dump(songs, f)
    return songs


# ----------------------------------------------------------------------------- rendering


def _transpose(notes: list, tonic: int, rng: np.random.Generator) -> tuple[list, int]:
    pitches = [m for m, _ in notes if m is not None]
    med = float(np.median(pitches))
    shifts = [s for s in range(-12, 13) if 55 <= med + s <= 70]
    s = int(rng.choice(shifts)) if shifts else int(round(62 - med))
    return [[None if m is None else m + s, d] for m, d in notes], (tonic + s) % 12


def _window(events: list, kind: str, rng: np.random.Generator) -> list | None:
    """events: [midi or None, seconds]. ``excerpt`` drops the final note and takes 8-20 s."""
    if kind == "full":
        out, t = [], 0.0
        for e in events:
            if t > 60:
                break
            out.append(e)
            t += e[1]
        return out
    last = max(i for i, e in enumerate(events) if e[0] is not None)
    body = events[:last]
    total = sum(d for _, d in body)
    if total < 4.0:
        return None
    w = float(rng.uniform(8, 20))
    if total <= w:
        return body
    starts = [i for i, e in enumerate(body) if e[0] is not None]
    cum = np.concatenate([[0], np.cumsum([d for _, d in body])])
    ok = [i for i in starts if cum[-1] - cum[i] >= w]
    i0 = int(rng.choice(ok)) if ok else 0
    out, t = [], 0.0
    for e in body[i0:]:
        if t >= w:
            break
        out.append(e)
        t += e[1]
    return out


def sung_notes(events: list, condition: str, rng: np.random.Generator):
    """Turn (midi or None, seconds) events into SungNotes for one singing condition."""
    from ..data.detune import sample_spec
    from ..data.synth import SungNote

    glob = {"in_tune": rng.normal(0, 8), "amateur": rng.normal(0, 20), "corpus": rng.normal(0, 8)}[condition]
    vib_rate = float(rng.uniform(4.8, 6.5))
    vib_ext = float(rng.uniform(15, 55))
    out = []
    for m, d in events:
        if m is None:
            if out:
                out[-1].gap_after += d
            continue
        gap = float(min(rng.uniform(0.04, 0.10), 0.3 * d))
        dur = d - gap
        n = SungNote(float(m), dur, vibrato_rate=vib_rate + float(rng.normal(0, 0.2)),
                     vibrato_cents=vib_ext if dur >= 0.35 else 0.0, gap_after=gap)
        if condition == "in_tune":
            n.offset_cents = glob + float(rng.normal(0, 6))
            if rng.random() < 0.15:
                n.scoop_cents, n.scoop_s = -float(rng.uniform(30, 80)), float(rng.uniform(0.05, 0.1))
        elif condition == "amateur":
            n.offset_cents = glob + float(rng.normal(0, 25))
            if rng.random() < 0.3:
                n.drift_cents = float(rng.uniform(-25, 25))
            if rng.random() < 0.35:
                n.scoop_cents, n.scoop_s = -float(rng.uniform(40, 150)), float(rng.uniform(0.05, 0.15))
        else:  # the O1 corpus detuning
            sp = sample_spec(rng)
            n.offset_cents = glob + sp.offset_cents
            n.drift_cents = sp.drift_cents
            if sp.scoop_cents:
                n.scoop_cents, n.scoop_s = sp.scoop_cents, sp.scoop_s
            n.vibrato_cents *= sp.vibrato_factor
        out.append(n)
    return out


def contour(notes, rng: np.random.Generator, hop: float = HOP_S, lead_in: float = 0.15, jitter_cents: float = 3.0) -> np.ndarray:
    """F0 contour (Hz, 0 = unvoiced) with the synthetic singer's pitch model, without audio."""
    from ..data.synth import note_contour
    from ..music import midi_to_hz

    total = lead_in + sum(n.dur + n.gap_after for n in notes)
    nfr = int(total / hop) + 1
    midi = np.full(nfr, np.nan)
    t0 = lead_in
    for n in notes:
        a, b = int(round(t0 / hop)), int(round((t0 + n.dur) / hop))
        tt = np.arange(b - a) * hop
        if len(tt):
            seg = note_contour(n, tt, rng)
            att, rel = min(0.03, n.dur / 4), min(0.05, n.dur / 4)
            env = np.minimum(1, np.minimum(tt / max(att, 1e-6), (n.dur - tt) / max(rel, 1e-6)))
            seg = np.where(env >= 0.3, seg, np.nan)  # as in sing(): quiet edges are unvoiced
            midi[a:b] = seg
        t0 += n.dur + n.gap_after
    jit = rng.standard_normal(nfr // 40 + 2)
    jit = np.interp(np.arange(nfr) / 40, np.arange(len(jit)), jit) * (jitter_cents / 100)
    return np.where(np.isnan(midi), 0.0, midi_to_hz(np.nan_to_num(midi + jit)))


def make_items(songs: list[dict], conditions=CONDITIONS, kinds=KINDS, seed: int = 0, keep_sung: bool = False) -> list[dict]:
    """One rendered contour per (song, condition, kind), with its transposed ground truth and
    its features. The contour itself is dropped after feature extraction to save memory;
    ``keep_sung`` keeps the note list for excerpts so they can be synthesized to audio."""
    items = []
    for s in songs:
        rng = np.random.default_rng([seed, zlib.crc32(s["id"].encode())])
        q = float(rng.uniform(0.35, 0.75))  # seconds per quarter note
        notes, tonic = _transpose(s["notes"], s["tonic"], rng)
        events = [[m, d * q] for m, d in notes]
        for kind in kinds:
            win = _window(events, kind, rng)
            if not win or sum(1 for m, _ in win if m is not None) < 4:
                continue
            for cond in conditions:
                sn = sung_notes(win, cond, rng)
                pcd = np.zeros(12)
                for m, d in win:
                    if m is not None:
                        pcd[int(m) % 12] += d
                it = dict(song=s["id"], file=s["file"], region=s["region"], kind=kind, condition=cond,
                          ref=Key(tonic, s["mode"]), pc_dur=pcd, feat=_features(contour(sn, rng)))
                if keep_sung and kind == "excerpt" and cond in SELECTION_CONDITIONS:
                    it["sung"] = sn
                items.append(it)
    return items


def _make_items_job(args) -> list[dict]:
    return make_items(*args)


def make_items_parallel(songs: list[dict], conditions, kinds, seed: int, keep_sung: bool, workers: int) -> list[dict]:
    if workers <= 1 or len(songs) < 50:
        return make_items(songs, conditions, kinds, seed, keep_sung)
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    chunks = [songs[i::workers * 4] for i in range(workers * 4)]
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        parts = list(ex.map(_make_items_job, [(c, conditions, kinds, seed, keep_sung) for c in chunks]))
    order = {s["id"]: i for i, s in enumerate(songs)}
    items = [it for p in parts for it in p]
    items.sort(key=lambda it: (order[it["song"]], it["kind"], it["condition"]))
    return items


# ----------------------------------------------------------------------------- fast sweep


def _features(f0: np.ndarray) -> dict:
    from ..music import hz_to_midi, tuning_offset_cents

    centres, durs = note_events(f0, HOP_S)
    tau = tuning_offset_cents(centres, durs) if len(centres) >= 3 else 0.0
    m = hz_to_midi(np.where(f0 > 0, f0, np.nan))
    f_t = np.where(np.isnan(m), 0.0, 440 * 2 ** ((m - tau / 100 - 69) / 12))
    return dict(
        centres=centres, durs=durs, tau=tau,
        frames={False: pitch_class_histogram(f0), True: pitch_class_histogram(f_t)},
        final={False: int(np.round(centres[-1])) % 12 if len(centres) else -1,
               True: int(np.round(centres[-1] - tau / 100)) % 12 if len(centres) else -1},
    )


def _hist(feat: dict, cfg: KeyConfig) -> np.ndarray:
    if cfg.unit == "frames":
        return feat["frames"][cfg.tuning]
    c = feat["centres"] - (feat["tau"] / 100 if cfg.tuning else 0.0)
    return soft_histogram(c, feat["durs"] ** cfg.alpha, cfg.soft_width)


def _profile_rows(profile: str, scoring: str) -> np.ndarray:
    from ..pitch.key import _profile_matrix

    return _profile_matrix(profile, scoring)


def predict(feats: list[dict], cfg: KeyConfig, cache: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized: (index into KEYS of the estimate, scale confidence) for every item. ``cache``
    keeps the histograms between configurations that share them."""
    hkey = (cfg.unit, cfg.alpha if cfg.unit == "notes" else None, cfg.tuning, cfg.soft_width)
    if cache is not None and hkey in cache:
        H = cache[hkey]
    else:
        H = np.array([_hist(f, cfg) for f in feats])
        if cache is not None:
            cache[hkey] = H
    P = _profile_rows(cfg.profile, cfg.scoring)
    if cfg.scoring == "corr":
        Hc = H - H.mean(axis=1, keepdims=True)
        Pc = P - P.mean(axis=1, keepdims=True)
        den = np.sqrt((Hc**2).sum(1, keepdims=True) * (Pc**2).sum(1)[None, :])
        S = (Hc @ Pc.T) / np.where(den > 0, den, 1)
    else:
        Hn = H / np.maximum(H.sum(1, keepdims=True), 1e-12)
        S = Hn @ P.T
    if cfg.final_bonus:
        fin = np.array([f["final"][cfg.tuning] for f in feats])
        tonics = np.array([k.tonic for k in KEYS])
        S = S + cfg.final_bonus * (fin[:, None] == tonics[None, :])
    best = S.argmax(1)
    order = np.argsort(-S, axis=1)
    scale_sets = [frozenset(k.pitch_classes) for k in KEYS]
    conf = np.zeros(len(feats))
    for i in range(len(feats)):
        b = order[i, 0]
        other = next(j for j in order[i, 1:] if scale_sets[j] != scale_sets[b])
        spread = S[i, b] - np.median(S[i]) or 1.0
        conf[i] = np.clip((S[i, b] - S[i, other]) / spread, 0, 1)
    return best, conf


def config_grid(include_aarden: bool = True) -> list[KeyConfig]:
    out = []
    profiles = list(FAIR_PROFILES) + (["aarden"] if include_aarden else [])
    for p in profiles:
        for sc in ("corr", "loglik"):
            for tuning in (False, True):
                for bonus in (0.0, 0.05, 0.1, 0.2):
                    out.append(KeyConfig(p, sc, "frames", 1.0, tuning, bonus))
                    for alpha in (0.5, 1.0):
                        out.append(KeyConfig(p, sc, "notes", alpha, tuning, bonus))
    return out


def _cfg_name(c: KeyConfig) -> str:
    unit = "frames" if c.unit == "frames" else f"notes^{c.alpha:g}"
    return f"{c.profile}/{c.scoring}/{unit}/tuning={'on' if c.tuning else 'off'}/final+{c.final_bonus:g}"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z**2 / n
    c = p + z**2 / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n * n))
    return float((c - h) / d), float((c + h) / d)


def _score_rows(items: list[dict], est_idx: np.ndarray, conf: np.ndarray, name: str) -> pd.DataFrame:
    rows = []
    for it, e, c in zip(items, est_idx, conf):
        est = KEYS[int(e)]
        ref = it["ref"]
        pcd = it["pc_dur"]
        fit = float(sum(pcd[p] for p in est.pitch_classes) / max(pcd.sum(), 1e-9))
        rows.append(dict(song=it["song"], file=it["file"], region=it["region"], kind=it["kind"], condition=it["condition"],
                         config=name, ref=ref.name, est=est.name, exact=est == ref, scale_equiv=same_key_or_relative(est, ref),
                         mirex=mirex_key_score(est, ref), conf=float(c), scale_fit=fit, melody_in_scale=fit >= 0.999))
    return pd.DataFrame(rows)


def evaluate_essen(out_dir: str | Path = "results/keys", n_audio: int = 150, seed: int = 0, workers: int = 4,
                   max_songs: int | None = None, audio_trackers: tuple[str, ...] = ("pyin", "auto")) -> dict:
    """Run the whole validation and write ``essen_sweep.csv`` (train accuracy of every
    configuration), ``essen_test.csv`` (per item, chosen config, baseline and Aarden),
    ``essen_audio.csv`` and ``essen_summary.json``/``.md``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    songs = parse_essen(out / "essen_melodies.json.gz", workers)
    if max_songs:
        rng = np.random.default_rng(seed)
        songs = [songs[i] for i in sorted(rng.choice(len(songs), min(max_songs, len(songs)), replace=False))]
    # The Han (Chinese) volumes name the pentatonic "gong" note in K:, not the final (only ~16 %
    # of them end on it, against 70-98 % for the European volumes), so their major/minor label
    # is not meaningful. They are kept out of the selection and only scored up to scale.
    train_songs = [s for s in songs if s["file"] in ESSEN_TRAIN and s["region"] != "han"]
    test_songs = [s for s in songs if s["file"] in ESSEN_TEST]
    train = make_items_parallel(train_songs, SELECTION_CONDITIONS, KINDS, seed, False, workers)
    test = make_items_parallel(test_songs, CONDITIONS, KINDS, seed + 1, n_audio > 0, workers)
    ftrain = [it["feat"] for it in train]
    ftest = [it["feat"] for it in test]

    # 1. sweep on the training volumes. Objective: exact accuracy averaged over the two modes
    # (major is ~89 % of Essen, so plain accuracy would reward a major bias) and over the four
    # condition x kind cells.
    refs = np.array([KEYS.index(it["ref"]) for it in train])
    cells = pd.DataFrame(dict(kind=[it["kind"] for it in train], cond=[it["condition"] for it in train], mode=[it["ref"].mode for it in train]))
    sweep = []
    cache: dict = {}
    for cfg in config_grid():
        idx, _ = predict(ftrain, cfg, cache)
        cells["ok"] = idx == refs
        bal = cells.groupby(["cond", "kind", "mode"]).ok.mean().groupby(["cond", "kind"]).mean()
        plain = cells.groupby(["cond", "kind"]).ok.mean()
        sweep.append(dict(config=_cfg_name(cfg), **asdict(cfg), train_balanced=float(bal.mean()), train_exact=float(plain.mean()),
                          train_major=float(cells[cells["mode"] == "major"].ok.mean()), train_minor=float(cells[cells["mode"] == "minor"].ok.mean()),
                          **{f"train_{c}_{k}": float(v) for (c, k), v in plain.items()}))
    sweep = pd.DataFrame(sweep).sort_values("train_balanced", ascending=False)
    sweep.to_csv(out / "essen_sweep.csv", index=False)
    fair = sweep[sweep.profile.isin(FAIR_PROFILES)]
    top = fair.iloc[0]
    chosen = KeyConfig(top.profile, top.scoring, top.unit, float(top.alpha), bool(top.tuning), float(top.final_bonus))
    baseline = KeyConfig("krumhansl", "corr", "frames", 1.0, False, 0.0)  # the estimator before this work
    best_aarden = sweep[sweep.profile == "aarden"].iloc[0]
    aarden = KeyConfig("aarden", best_aarden.scoring, best_aarden.unit, float(best_aarden.alpha), bool(best_aarden.tuning), float(best_aarden.final_bonus))

    # 2. test volumes
    frames = []
    for name, cfg in (("chosen", chosen), ("baseline_v1", baseline), ("aarden_reference", aarden)):
        idx, conf = predict(ftest, cfg)
        frames.append(_score_rows(test, idx, conf, name))
    test_df = pd.concat(frames)
    test_df["mode"] = test_df.ref.str.split().str[-1]
    test_df.to_csv(out / "essen_test.csv", index=False)

    def cell(df):
        k, n = int(df.exact.sum()), len(df)
        lo, hi = wilson(k, n)
        return dict(n=n, exact=k / n if n else float("nan"), exact_ci95=[lo, hi], scale_equiv=float(df.scale_equiv.mean()) if n else float("nan"),
                    mirex=float(df.mirex.mean()) if n else float("nan"), melody_in_scale=float(df.melody_in_scale.mean()) if n else float("nan"))

    euro_songs = [s for s in songs if s["region"] != "han"]
    summary = {
        "songs": {"train": len(train_songs), "test_european": sum(s["region"] != "han" for s in test_songs), "test_han": sum(s["region"] == "han" for s in test_songs)},
        "ends_on_tonic_share_european": float(np.mean([s["ends_on_tonic"] for s in euro_songs])),
        "ends_on_tonic_share_han": float(np.mean([s["ends_on_tonic"] for s in songs if s["region"] == "han"])) if any(s["region"] == "han" for s in songs) else float("nan"),
        "minor_share_european": float(np.mean([s["mode"] == "minor" for s in euro_songs])),
        "chosen_config": asdict(chosen),
        "chosen_name": _cfg_name(chosen),
        "train_balanced_chosen": float(top.train_balanced),
        "train_exact_chosen": float(top.train_exact),
        "test": {},
    }
    for name, g in test_df.groupby("config"):
        res = {}
        eu = g[g.region != "han"]
        for (cond, kind), gg in eu.groupby(["condition", "kind"]):
            res[f"{cond}/{kind}"] = cell(gg)
        sel = eu[eu.condition.isin(SELECTION_CONDITIONS)]
        res["european_major"] = cell(sel[sel["mode"] == "major"])
        res["european_minor"] = cell(sel[sel["mode"] == "minor"])
        han = g[(g.region == "han") & g.condition.isin(SELECTION_CONDITIONS)]
        res["han_chinese"] = cell(han) if len(han) else {}
        # selective accuracy: if the app asked the user whenever confidence is in the lowest 30 %
        thr = sel.conf.quantile(0.3)
        res["top70pct_confidence"] = cell(sel[sel.conf >= thr])
        summary["test"][name] = res

    # 3. audio check on a sample of test excerpts (in-tune and amateur)
    if n_audio:
        summary["audio"] = _audio_check(test, chosen, n_audio, seed, audio_trackers, out)
    (out / "essen_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    (out / "essen_summary.md").write_text(summary_markdown(summary))
    return summary


def _audio_check(items: list[dict], cfg: KeyConfig, n: int, seed: int, trackers: tuple[str, ...], out: Path) -> dict:
    from ..data.synth import sing
    from ..pitch.key import analyze_key
    from ..pitch.trackers import get_tracker

    rng = np.random.default_rng(seed + 7)
    # European excerpts only: the Han K: field is not the tonic (see evaluate_essen)
    pool = [i for i, it in enumerate(items) if "sung" in it and it["region"] != "han"]
    pick = sorted(rng.choice(pool, min(n, len(pool)), replace=False))
    rows = []
    trks = {}
    for t in trackers:
        try:
            trks[t] = get_tracker(t)
        except Exception as e:  # report and carry on with the trackers that load
            print(f"audio check: tracker {t!r} unavailable ({type(e).__name__}: {e})")
    for i in pick:
        it = items[i]
        s = sing(it["sung"], sr=16_000, seed=int(i))
        contour_est = analyze_key(s.f0, None, s.hop_s, cfg).key
        row = dict(song=it["song"], condition=it["condition"], ref=it["ref"].name, contour=contour_est.name, contour_exact=contour_est == it["ref"])
        for name, trk in trks.items():
            tr = trk.track(s.samples, s.sr)
            est = analyze_key(tr.f0, tr.voicing, tr.hop_s, cfg).key
            row[f"{trk.name}"] = est.name
            row[f"{trk.name}_exact"] = est == it["ref"]
            row[f"{trk.name}_scale_equiv"] = same_key_or_relative(est, it["ref"])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "essen_audio.csv", index=False)
    res = {"excerpts": len(df), "true_contour_exact": float(df.contour_exact.mean())}
    for c in df.columns:
        if c.endswith("_exact") and c != "contour_exact":
            k = int(df[c].sum())
            res[c] = k / len(df)
            res[c + "_ci95"] = wilson(k, len(df))
        if c.endswith("_scale_equiv"):
            res[c] = float(df[c].mean())
    return res


def summary_markdown(s: dict) -> str:
    lines = [
        f"Essen songs: {s['songs']['train']} training and {s['songs']['test_european']} test (European volumes), plus "
        f"{s['songs']['test_han']} Han Chinese test songs scored up to scale. {100 * s['ends_on_tonic_share_european']:.1f}% of the "
        f"European songs end on the K: tonic; {100 * s['minor_share_european']:.1f}% are minor.",
        f"Chosen configuration: `{s['chosen_name']}` (training accuracy {100 * s['train_exact_chosen']:.1f}%, mode-balanced {100 * s['train_balanced_chosen']:.1f}%).",
        "",
        "| Condition | Chosen: exact (95% CI) | Scale-equivalent | Melody fits scale | MIREX | Before (KK, frames): exact | Aarden (reference) |",
        "|---|---|---|---|---|---|---|",
    ]
    ch, base, aa = s["test"].get("chosen", {}), s["test"].get("baseline_v1", {}), s["test"].get("aarden_reference", {})
    for key in sorted(k for k, v in ch.items() if isinstance(v, dict) and v and k != "han_chinese"):
        c = ch[key]
        lo, hi = c["exact_ci95"]
        lines.append(f"| {key} (n={c['n']}) | {100 * c['exact']:.1f}% ({100 * lo:.1f}-{100 * hi:.1f}) | {100 * c['scale_equiv']:.1f}% | {100 * c['melody_in_scale']:.1f}% | {c['mirex']:.3f} | "
                     f"{100 * base.get(key, {}).get('exact', float('nan')):.1f}% | {100 * aa.get(key, {}).get('exact', float('nan')):.1f}% |")
    han = {k: v.get("han_chinese", {}) for k, v in s["test"].items()}
    if han.get("chosen"):
        lines += ["", f"Han Chinese test songs (pentatonic, so only the scale is scored): the estimated key's scale contains every "
                  f"note of the melody in {100 * han['chosen']['melody_in_scale']:.1f}% of cases, against "
                  f"{100 * han.get('baseline_v1', {}).get('melody_in_scale', float('nan')):.1f}% before."]
    if "audio" in s:
        a = s["audio"]
        lines += ["", f"Audio check on {a['excerpts']} synthesized test excerpts: " + ", ".join(
            f"{k.replace('_exact', '')} {100 * v:.1f}%" for k, v in a.items() if k.endswith("_exact") and not isinstance(v, (list, tuple)))]
    return "\n".join(lines) + "\n"
