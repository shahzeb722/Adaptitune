"""Hand-labelled key excerpts for O2 ("key estimation is correct on at least 75 % of about 100
hand-labelled excerpts"; proposal 5.3: two of us label each excerpt separately).

Workflow (``docs/O2_PROTOCOL.md``):

1. ``adaptitune key-excerpts PATH --out key_excerpts`` writes about 100 excerpts, their
   ``excerpts.json`` and a blank label template. With ``--source mir1k`` (recommended) PATH is
   MIR-1K and every excerpt is a different song: the vocal channel of one singer's consecutive
   clips, 12-30 s. With ``--source corpus`` PATH is an O1 corpus and the excerpts come from the
   **test-split** clean takes (songs first, then arpeggios and scales, never single long
   tones), spread over all test singers. VocalSet has only three songs, so that set repeats
   the same tunes; its accuracy is reported with an interval that resamples songs. The
   estimator's output is not written anywhere the raters can see it.
2. Each rater opens ``tools/key_rater.html`` in a browser, loads the folder, labels every
   excerpt on their own (a tonic drone and a scale player help) and exports a CSV.
3. ``adaptitune key-agreement rater1.csv rater2.csv --out key_labels.csv`` reports raw
   agreement and Cohen's kappa, and writes the merged file with an ``adjudicated`` column that
   is pre-filled where the raters agree. The raters settle the rest together.
4. ``adaptitune eval-keys key_labels.csv --audio-root key_excerpts`` scores the estimator.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ..music import Key

NONE_LABELS = {"", "none", "no clear key", "no key", "atonal", "unsure", "skip", "?"}
CONTEXT_PRIORITY = {"excerpts": 0, "song": 0, "arpeggios": 1, "scales": 2}  # long_tones never


def parse_label(text: str | None) -> Key | None:
    t = (text or "").strip()
    if t.lower() in NONE_LABELS:
        return None
    return Key.parse(t)


def cohens_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa for two raters over nominal categories."""
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] / n * cb[c] / n for c in set(ca) | set(cb))
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


# ----------------------------------------------------------------------------- excerpts


def select_excerpts(corpus_dir: str | Path, out_dir: str | Path, n: int = 100, split: str = "test",
                    min_s: float = 8.0, max_s: float = 25.0, seed: int = 0) -> pd.DataFrame:
    """Cut ``n`` excerpts from the clean takes of ``split`` and write them with ``excerpts.json``
    and ``key_labels_template.csv``. Takes are drawn round-robin over singers, songs first."""
    from ..audio import Audio, load, save
    from ..data.detune import load_manifest

    corpus, out = Path(corpus_dir), Path(out_dir)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    man = load_manifest(corpus).drop_duplicates("take")
    man = man[man["split"] == split] if split else man
    ctx = man["context"].astype(str) if "context" in man else pd.Series([""] * len(man), index=man.index)
    man = man.assign(_prio=[CONTEXT_PRIORITY.get(c, 3 if c and c != "long_tones" else (9 if c == "long_tones" else 3)) for c in ctx])
    man = man[man._prio < 9]
    if man.empty:
        raise ValueError(f"no usable {split} takes in {corpus}")
    rng = np.random.default_rng(seed)
    queues = {}
    for singer, g in man.groupby("singer"):
        g = g.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).sort_values("_prio", kind="stable")
        queues[singer] = list(g.itertuples(index=False))
    order = sorted(queues)
    picked = []
    while len(picked) < n and any(queues.values()):
        for singer in order:
            if queues[singer] and len(picked) < n:
                picked.append(queues[singer].pop(0))
    rows = []
    for k, r in enumerate(picked):
        a = load(corpus / r.clean, normalize=False)
        dur = a.duration
        length = min(dur, float(rng.uniform(min_s, max_s)))
        start = float(rng.uniform(0, max(dur - length, 0)))
        seg = a.samples[int(start * a.sr): int((start + length) * a.sr)]
        ex_id = f"ex{k + 1:03d}"
        save(out / "audio" / f"{ex_id}.wav", Audio(seg, a.sr), "PCM_16")
        rows.append(dict(excerpt=ex_id, path=f"audio/{ex_id}.wav", source="corpus", take=r.take, singer=r.singer,
                         context=getattr(r, "context", ""), song=_song_of(r), start=round(start, 3), end=round(start + length, 3)))
    df = pd.DataFrame(rows)
    (out / "excerpts.json").write_text(json.dumps({"excerpts": rows, "seed": seed, "split": split, "source": "corpus"}, indent=2))
    tpl = df[["excerpt", "path"]].assign(key_rater1="", key_rater2="", adjudicated="", notes="")
    tpl.to_csv(out / "key_labels_template.csv", index=False)
    return df


def _song_of(r) -> str:
    """The tune an O1 take sings: VocalSet song excerpts are named like ``f3_caro_vibrato``,
    so the word after the singer prefix names the song; other takes are their own song."""
    ctx = str(getattr(r, "context", ""))
    src = getattr(r, "source_path", "")
    src = "" if src is None or (isinstance(src, float) and np.isnan(src)) else str(src)
    # the source file's own name, e.g. f3_caro_vibrato (take ids look like vocalset__female3__...__f3_caro_vibrato)
    stem = Path(src).stem if src else str(r.take).split("__")[-1]
    if ctx in ("excerpts", "song"):
        parts = stem.split("_")
        if len(parts) >= 2 and re.fullmatch(r"[fm]\d+", parts[0]):
            return f"vocalset/{parts[1]}"
    return str(r.take)


_MIR1K_NAME = re.compile(r"^(?P<singer>[A-Za-z]+)_(?P<song>\d+)_(?P<clip>\d+)$")


def select_mir1k_excerpts(root: str | Path, out_dir: str | Path, n: int = 100, min_s: float = 12.0,
                          max_s: float = 30.0, gap_s: float = 0.3, seed: int = 0) -> pd.DataFrame:
    """One excerpt per song from MIR-1K (amateur singers, ~110 songs): the vocal (right)
    channel of one singer's consecutive clips of a song, joined with short silences until the
    next clip would pass ``max_s``. Songs shorter than ``min_s`` are skipped. Songs are drawn
    round-robin over singers, and the excerpt numbers are shuffled so they do not reveal the
    singer."""
    import soundfile as sf

    from ..audio import Audio, save

    root, out = Path(root), Path(out_dir)
    wav_dir = root / "Wavfile" if (root / "Wavfile").is_dir() else root
    groups: dict[tuple[str, int], list[tuple[int, Path]]] = {}
    for p in sorted(wav_dir.rglob("*.wav")):
        m = _MIR1K_NAME.match(p.stem)
        if m:
            groups.setdefault((m["singer"], int(m["song"])), []).append((int(m["clip"]), p))
    if not groups:
        raise ValueError(f"no MIR-1K clips (singer_song_clip.wav) under {wav_dir}")
    cands: dict[str, list] = {}
    for (singer, song), clips in sorted(groups.items()):
        use, total = [], 0.0
        for _, p in sorted(clips):
            d = sf.info(str(p)).duration
            extra = d + (gap_s if use else 0.0)
            if use and total + extra > max_s:
                break
            use.append(p)
            total += extra
        if total >= min_s:
            cands.setdefault(singer, []).append((song, use, total))
    rng = np.random.default_rng(seed)
    order = [str(x) for x in rng.permutation(sorted(cands))]
    for singer in order:
        cands[singer] = [cands[singer][i] for i in rng.permutation(len(cands[singer]))]
    picked = []
    while len(picked) < n and any(cands.values()):
        for singer in order:
            if cands[singer] and len(picked) < n:
                picked.append((singer, *cands[singer].pop(0)))
    (out / "audio").mkdir(parents=True, exist_ok=True)
    rows = []
    for k, i in enumerate(rng.permutation(len(picked))):
        singer, song, use, _ = picked[i]
        parts, sr = [], None
        for j, p in enumerate(use):
            x, sr = sf.read(str(p), always_2d=True)
            if j:
                parts.append(np.zeros(int(gap_s * sr)))
            parts.append(x[:, 1] if x.shape[1] > 1 else x[:, 0])  # right channel = vocal
        y = np.concatenate(parts)
        ex_id = f"ex{k + 1:03d}"
        save(out / "audio" / f"{ex_id}.wav", Audio(y, sr), "PCM_16")
        rows.append(dict(excerpt=ex_id, path=f"audio/{ex_id}.wav", source="mir1k", singer=singer, song=f"mir1k/{singer}_{song}",
                         clips=[p.stem for p in use], duration=round(len(y) / sr, 2)))
    rows.sort(key=lambda r: r["excerpt"])
    df = pd.DataFrame(rows)
    (out / "excerpts.json").write_text(json.dumps({"excerpts": rows, "seed": seed, "source": "mir1k"}, indent=2))
    df[["excerpt", "path"]].assign(key_rater1="", key_rater2="", adjudicated="", notes="").to_csv(out / "key_labels_template.csv", index=False)
    return df


# ----------------------------------------------------------------------------- agreement


def _read_rater(path: str | Path) -> dict[str, dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        ex = r.get("excerpt") or Path(r.get("path", "")).stem
        out[ex] = r
    return out


def merge_raters(rater1_csv: str | Path, rater2_csv: str | Path, out_csv: str | Path) -> dict:
    """Merge two raters' exports, report agreement and write the adjudication file."""
    r1, r2 = _read_rater(rater1_csv), _read_rater(rater2_csv)
    ids = sorted(set(r1) & set(r2))
    rows, la, lb = [], [], []
    for ex in ids:
        k1, k2 = parse_label(r1[ex].get("key")), parse_label(r2[ex].get("key"))
        n1, n2 = (k1.name if k1 else "none"), (k2.name if k2 else "none")
        la.append(n1)
        lb.append(n2)
        scale = bool(k1 and k2 and set(k1.pitch_classes) == set(k2.pitch_classes))
        rows.append(dict(excerpt=ex, path=r1[ex].get("path") or r2[ex].get("path") or f"audio/{ex}.wav", key_rater1=n1, key_rater2=n2,
                         agree=n1 == n2, same_scale=scale or n1 == n2, adjudicated=n1 if n1 == n2 else "",
                         conf_rater1=r1[ex].get("confidence", ""), conf_rater2=r2[ex].get("confidence", ""),
                         notes=" | ".join(x for x in (r1[ex].get("notes", ""), r2[ex].get("notes", "")) if x)))
    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    res = {
        "excerpts_in_both": len(ids),
        "only_rater1": sorted(set(r1) - set(r2)),
        "only_rater2": sorted(set(r2) - set(r1)),
        "raw_agreement": float(df.agree.mean()) if len(df) else float("nan"),
        "scale_agreement": float(df.same_scale.mean()) if len(df) else float("nan"),
        "cohens_kappa": cohens_kappa(la, lb),
        "to_adjudicate": df.loc[~df.agree, "excerpt"].tolist() if len(df) else [],
    }
    Path(out_csv).with_suffix(".agreement.json").write_text(json.dumps(res, indent=2))
    return res


# ----------------------------------------------------------------------------- evaluation


def evaluate_key_labels(label_csv: str | Path, tracker: str = "auto", audio_root: str | Path = ".", out_csv: str | Path | None = None) -> dict:
    """Key accuracy on the hand-labelled excerpts (O2: exact accuracy >= 75 %).

    The reference is the ``adjudicated`` key where present, otherwise rater 1's key where both
    raters agree; excerpts whose reference is "none" (no clear key) are counted but not
    scored. Accuracy is reported on all excerpts with a reference (the O2 figure) and on the
    subset where the raters agreed from the start.
    """
    from ..audio import load
    from ..bench.keys import wilson
    from ..pitch.key import analyze_key, mirex_key_score, same_key_or_relative
    from ..pitch.trackers import get_tracker

    with open(label_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    trk = get_tracker(tracker)
    out, n_none, n_unresolved = [], 0, 0
    l1, l2 = [], []
    for r in rows:
        k1, k2 = parse_label(r.get("key_rater1")), parse_label(r.get("key_rater2") or r.get("key_rater1"))
        l1.append(k1.name if k1 else "none")
        l2.append(k2.name if k2 else "none")
        agree = (k1 == k2)
        adj = (r.get("adjudicated") or "").strip()
        if adj:
            ref = parse_label(adj)
        elif agree:
            ref = k1
        else:
            n_unresolved += 1
            continue
        if ref is None:
            n_none += 1
            continue
        audio = load(Path(audio_root) / r["path"])
        tr = trk.track(audio.samples, audio.sr)
        res = analyze_key(tr.f0, tr.voicing, tr.hop_s)
        est = res.key
        out.append(dict(excerpt=r.get("excerpt", Path(r["path"]).stem), path=r["path"], ref=ref.name, est=est.name,
                        agreed_initially=agree, exact=est == ref, scale_equiv=same_key_or_relative(est, ref),
                        mirex=mirex_key_score(est, ref), confidence=res.confidence))
    df = pd.DataFrame(out)
    songs = _excerpt_songs(Path(audio_root))
    if len(df):
        df["song"] = [songs.get(e, e) for e in df.excerpt]
    if out_csv:
        df.to_csv(out_csv, index=False)
    k = int(df.exact.sum()) if len(df) else 0
    lo, hi = wilson(k, len(df)) if len(df) else (float("nan"), float("nan"))
    agreed = df[df.agreed_initially] if len(df) else df
    n_songs = int(df.song.nunique()) if len(df) else 0
    if len(df) and n_songs < len(df):  # repeated tunes: resample songs, not excerpts
        from .trackers import bootstrap_ci

        song_ci = list(bootstrap_ci(df.assign(cluster=df.song, x=df.exact.astype(float)), lambda d: d.x.mean(), n_boot=2000))
    else:
        song_ci = [lo, hi]
    return {
        "excerpts": len(rows),
        "scored": len(df),
        "no_clear_key": n_none,
        "unresolved_disagreements": n_unresolved,
        "rater_raw_agreement": float(np.mean([a == b for a, b in zip(l1, l2)])) if rows else float("nan"),
        "rater_cohens_kappa": cohens_kappa(l1, l2),
        "accuracy_exact": k / len(df) if len(df) else float("nan"),
        "accuracy_exact_ci95": [lo, hi],
        "distinct_songs": n_songs,
        "accuracy_exact_ci95_by_song": song_ci,
        "accuracy_scale_equivalent": float(df.scale_equiv.mean()) if len(df) else float("nan"),
        "mirex_score": float(df.mirex.mean()) if len(df) else float("nan"),
        "accuracy_exact_agreed_only": float(agreed.exact.mean()) if len(agreed) else float("nan"),
        "tracker": trk.name,
        "O2_key_pass": bool(len(df) and k / len(df) >= 0.75),
    }


def _excerpt_songs(audio_root: Path) -> dict[str, str]:
    """excerpt id -> song id, from the ``excerpts.json`` that ``key-excerpts`` wrote."""
    f = audio_root / "excerpts.json"
    if not f.exists():
        return {}
    try:
        return {e["excerpt"]: e.get("song") or e.get("take") or e["excerpt"] for e in json.loads(f.read_text())["excerpts"]}
    except (KeyError, TypeError, ValueError):
        return {}
