"""Tracker benchmark (objective O2, proposal 5.3).

For every installed tracker this measures, per item and in total:

* raw pitch accuracy (RPA) and raw chroma accuracy (RCA) at 50 cents, voicing-independent as
  in the CREPE, PESTO and FCPE papers, plus RPA at 25 and 10 cents (finer tolerances matter
  for pitch correction), overall accuracy, voicing recall and false alarm, the median error
  of correct frames and the octave-error rate (:func:`adaptitune.metrics.pitch_metrics`);
* CPU real-time factor (tracking time / audio duration) with a fixed number of threads,
  after a warm-up call so model loading and JIT compilation are not counted;
* peak resident memory, and the memory the model itself adds.

Every tracker that takes a search range gets the same one, C1-B6 (32.7-1975.5 Hz, the native
range of CREPE and FCPE), because MDB-stem-synth has bass and flute stems outside the singing
range the app uses (65-1100 Hz). PESTO and its ONNX export have no range setting. RPA is also
reported on the reference frames inside 65-1100 Hz (``rpa_sing``), which is what the app needs.

Each tracker runs in a fresh interpreter, so the memory numbers are that tracker's alone and
one crash cannot sink the whole run. Item-level 95 % confidence intervals come from a
bootstrap that resamples singers (MIR-1K) or tracks (MDB-stem-synth) rather than clips,
because clips from the same voice are not independent.

The selection rule in :func:`choose_tracker` is fixed in ``docs/O2_PROTOCOL.md`` before any
real-data run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from .. import metrics as M

TORCH_TRACKERS = {"pesto", "fcpe", "crepe"}
DATASETS = ("mir1k", "mdb", "synthetic")
OFFICIAL_DATASETS = ("mir1k", "mdb")  # O2 is judged on these two
AUDIT_SHIFTS_MS = tuple(range(-30, 31, 5))

# Pre-registered selection rule (docs/O2_PROTOCOL.md). Do not change after seeing real results.
MAX_TRACKER_RTF = 0.25  # at most half of the 0.5 full-pipeline budget (O5)
TIE_PP = 1.0  # trackers within 1 RPA point of the best count as tied; the faster one wins
O2_MIN_TRACKERS = 4
O2_MIN_MDB_RPA = 0.90
# Benchmark sets a released model has already seen in training (audio, with or without labels).
# Its accuracy there is reported but flagged, and left out of its selection score.
TRAINED_ON: dict[str, tuple[str, ...]] = {
    # the released CREPE weights were trained on MIR-1K, Bach10, RWC-synth, MedleyDB,
    # MDB-stem-synth and NSynth (github.com/marl/crepe README)
    "crepe": ("mir1k", "mdb"),
    # the only checkpoint PESTO ships was trained (self-supervised, no labels) on MIR-1K audio
    # (pesto-pitch README)
    "pesto": ("mir1k",),
    "pesto-onnx": ("mir1k",),
    # FCPE was trained on M4Singer and VCTK resynthesized with DDSP; MIR-1K is only a test set
    # (Luo et al. 2025, arXiv:2509.15140)
    "fcpe": (),
    "fcpe-onnx": (),  # the same model, exported for the app after O2 chose FCPE
    "pyin": (),
    "yin": (),
}


# ----------------------------------------------------------------------------- environment


def _cpu_name() -> str:
    import platform

    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        elif sys.platform == "win32":
            import winreg

            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def _version(pkg: str) -> str | None:
    from importlib import metadata

    try:
        return metadata.version(pkg)
    except Exception:
        return None


def _git_commit() -> str | None:
    try:
        here = Path(__file__).resolve().parent
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def machine_info(threads: int) -> dict:
    """Everything needed to say which machine and software produced a speed number."""
    import platform

    info = {
        "cpu": _cpu_name(),
        "logical_cores": os.cpu_count(),
        "threads": threads,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git_commit": _git_commit(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        import psutil

        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        pass
    info["versions"] = {
        p: _version(p)
        for p in ("numpy", "scipy", "librosa", "mir_eval", "torch", "onnxruntime", "pesto-pitch", "torchfcpe", "torchcrepe")
    }
    return info


def thread_env(threads: int) -> dict:
    """Environment that pins every numeric library to ``threads`` threads."""
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "ADAPTITUNE_THREADS"):
        env[var] = str(threads)
    return env


def _rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 2**20
    except Exception:
        pass
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return float("nan")


def _peak_rss_mb() -> float:
    try:
        import resource  # Unix only

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 2**20 if sys.platform == "darwin" else peak / 1024  # bytes on macOS, KiB on Linux
    except Exception:
        pass
    try:
        import psutil

        mi = psutil.Process().memory_info()
        return getattr(mi, "peak_wset", mi.rss) / 2**20  # Windows peak working set
    except Exception:
        return float("nan")


# ----------------------------------------------------------------------------- items


def dataset_label(dataset: str, noise_snr: float | None = None, mix_snr: float | None = None) -> str:
    label = dataset
    if mix_snr is not None:
        label += f"_mix{mix_snr:g}dB"
    if noise_snr is not None:
        label += f"_noise{noise_snr:g}dB"
    return label


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f noise by shaping white noise in the frequency domain."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spec))
    spec[1:] /= np.sqrt(f[1:])
    spec[0] = 0
    x = np.fft.irfft(spec, n)
    return x / (np.std(x) + 1e-12)


def active_power(x: np.ndarray, sr: int, floor_db: float = -40.0) -> float:
    """Mean power over 20 ms frames within ``floor_db`` of the loudest frame, so long
    silences in a stem do not inflate the SNR."""
    hop = max(1, int(0.02 * sr))
    n = len(x) // hop
    if n == 0:
        return float(np.mean(x**2))
    fr = x[: n * hop].reshape(n, hop)
    p = np.mean(fr**2, axis=1)
    keep = p >= p.max() * 10 ** (floor_db / 10)
    return float(np.mean(p[keep])) if keep.any() else float(np.mean(x**2))


def add_noise(x: np.ndarray, sr: int, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = pink_noise(len(x), rng)
    noise *= np.sqrt(active_power(x, sr) / (np.mean(noise**2) * 10 ** (snr_db / 10)))
    return x + noise


def parse_shard(shard: str | None) -> tuple[int, int] | None:
    """``"k/n"`` -> ``(k, n)``: run the k-th of n interleaved parts of the dataset."""
    if not shard:
        return None
    k, n = (int(v) for v in str(shard).split("/"))
    if n < 1 or not 0 <= k < n:
        raise ValueError(f"bad shard {shard!r}: need k/n with 0 <= k < n")
    return k, n


def _items(dataset: str, root: str | None, limit: int | None, seed: int = 0, noise_snr: float | None = None, mix_snr: float | None = None,
           shard: str | None = None):
    from ..audio import Audio
    from ..data import datasets as D

    root = root or None
    sh = parse_shard(shard)
    if dataset == "mir1k":
        it = D.mir1k(root, mix_snr=mix_snr, shard=sh)
    elif dataset == "mdb":
        it = D.mdb_stem_synth(root, shard=sh)
    elif dataset == "synthetic":
        it = (x for i, x in enumerate(_synthetic_items(limit or 40, seed)) if not sh or i % sh[1] == sh[0])
    else:
        raise ValueError(dataset)
    for i, item in enumerate(it):
        if limit and i >= limit:
            break
        if noise_snr is not None:
            x = add_noise(item.audio.samples, item.audio.sr, noise_snr, zlib.crc32(item.id.encode()) + seed)
            item.audio = Audio(x, item.audio.sr)
            item.meta = {**(item.meta or {}), "noise_snr_db": noise_snr}
        yield item


def _synthetic_items(n: int, seed: int):
    from ..audio import Audio
    from ..data.datasets import Item
    from ..data.detune import make_singers, synth_clean_take

    rng = np.random.default_rng(seed)
    singers = make_singers(max(4, n // 4), seed + 1)
    for i in range(n):
        a, tr, _ = synth_clean_take(singers[i % len(singers)], rng, 16_000)
        # background noise at 0-20 dB SNR, otherwise every tracker scores ~100 %
        snr = float(rng.uniform(0, 20))
        x = add_noise(a.samples, a.sr, snr, seed * 7919 + i)
        yield Item(f"synth/{i}", singers[i % len(singers)].name, Audio(x, a.sr), tr.times, tr.f0, meta={"snr_db": snr})


# ----------------------------------------------------------------------------- worker


def _worker(name: str, dataset: str, root: str | None, limit: int | None, out_path: str, threads: int = 4,
            noise_snr: float | None = None, mix_snr: float | None = None, audit: bool = False, seed: int = 0,
            shard: str | None = None):
    import warnings

    warnings.filterwarnings("ignore")
    if name in TORCH_TRACKERS:
        try:
            import torch

            torch.set_num_threads(threads)
            torch.set_num_interop_threads(1)
        except Exception:
            pass
    from ..pitch.trackers import FMAX, FMIN

    rows, audit_rows = [], []
    res: dict = {}
    try:
        base_mb = _rss_mb()
        trk = bench_tracker(name)
        first = True
        loaded_mb = float("nan")
        for item in _items(dataset, root, limit, seed, noise_snr, mix_snr, shard):
            if first:  # warm-up (model load, JIT) is not part of the steady-state RTF
                trk.track(item.audio.samples[: item.audio.sr], item.audio.sr)
                loaded_mb = _rss_mb()
                first = False
            t0 = time.perf_counter()
            tr = trk.track(item.audio.samples, item.audio.sr)
            el = time.perf_counter() - t0
            acc = M.pitch_metrics(item.ref_times, item.ref_f0, tr.times, tr.f0, tr.raw_f0)
            sing = (item.ref_f0 >= FMIN) & (item.ref_f0 <= FMAX)
            rpa_sing = float("nan")
            if sing.any():  # the same accuracy over the reference frames inside the app's range
                ref_sing = np.where(sing, item.ref_f0, 0.0)
                rpa_sing = M.pitch_metrics(item.ref_times, ref_sing, tr.times, tr.f0, tr.raw_f0, tolerances=(50,))["rpa"]
            rows.append(dict(tracker=name, item=item.id, cluster=item.singer, duration=item.audio.duration, seconds=el, **acc,
                             rpa_sing=rpa_sing, ref_sing_share=float(sing.sum() / max(int((item.ref_f0 > 0).sum()), 1))))
            if audit:
                for sh in AUDIT_SHIFTS_MS:
                    a2 = M.pitch_metrics(item.ref_times, item.ref_f0, tr.times + sh / 1000, tr.f0, tr.raw_f0, tolerances=(50, 10))
                    audit_rows.append(dict(tracker=name, item=item.id, shift_ms=sh, oa=a2["oa"], rpa=a2["rpa"], rpa_10=a2["rpa_10"]))
        res = dict(ok=True, rows=rows, audit=audit_rows, peak_mb=_peak_rss_mb(), model_mb=loaded_mb - base_mb, base_mb=base_mb)
    except Exception as e:  # pragma: no cover - reported, not raised
        res = dict(ok=False, error=f"{type(e).__name__}: {e}", rows=rows, audit=audit_rows)
    with open(out_path, "w") as f:
        json.dump(res, f)


def bench_tracker(name: str):
    """The tracker as benchmarked: the common C1-B6 search range for every tracker that takes a
    range; PESTO and the ONNX export have no range setting and keep their own."""
    import inspect

    from ..pitch.trackers import BENCH_FMAX, BENCH_FMIN, REGISTRY, get_tracker

    cls = REGISTRY.get(name)
    if cls is not None and "fmin" in inspect.signature(cls.__init__).parameters:
        return cls(fmin=BENCH_FMIN, fmax=BENCH_FMAX)
    return get_tracker(name)


def _run_isolated(name: str, dataset: str, root: str | None, limit: int | None, threads: int = 4, **kw) -> dict:
    """Fresh interpreter per tracker, so peak memory is that tracker's alone."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        out = tf.name
    args = dict(name=name, dataset=dataset, root=root or None, limit=limit, out_path=out, threads=threads, **kw)
    code = "import sys, json; from adaptitune.bench.trackers import _worker; _worker(**json.loads(sys.argv[1]))"
    proc = subprocess.run([sys.executable, "-c", code, json.dumps(args)], capture_output=True, text=True, env=thread_env(threads))
    try:
        with open(out) as f:
            return json.load(f)
    except Exception:
        return dict(ok=False, error=f"worker exited {proc.returncode}: {proc.stderr[-800:]}", rows=[])
    finally:
        try:
            os.unlink(out)
        except OSError:
            pass


# ----------------------------------------------------------------------------- statistics


def bootstrap_ci(df: pd.DataFrame, stat, cluster: str | None = "cluster", n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap CI of ``stat(df)``, resampling whole clusters (singers or tracks)."""
    if len(df) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby(cluster)] if cluster and cluster in df else [df.iloc[[i]] for i in range(len(df))]
    if len(groups) < 2:
        v = float(stat(df))
        return v, v
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        vals.append(stat(pd.concat([groups[i] for i in pick])))
    lo, hi = np.percentile(vals, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def _rtf(df: pd.DataFrame) -> float:
    return float(df.seconds.sum() / df.duration.sum())


def summarize(df: pd.DataFrame, label: str, mem: dict | None = None, n_boot: int = 2000) -> dict:
    """One summary row: item-averaged metrics with cluster-bootstrap CIs, pooled-frame RPA,
    total RTF and memory."""
    mem = mem or {}
    s = dict(tracker=df.tracker.iloc[0], dataset=label, items=len(df), clusters=int(df.cluster.nunique()))
    s["rpa"] = float(df.rpa.mean())
    s["rpa_lo"], s["rpa_hi"] = bootstrap_ci(df, lambda d: d.rpa.mean(), n_boot=n_boot)
    for k in ("rca", "rpa_25", "rpa_10", "oa", "vr", "vfa", "median_err_cents", "octave_err", "rpa_sing", "ref_sing_share"):
        if k in df:
            s[k] = float(df[k].mean()) if k != "median_err_cents" else float(df[k].median())
    w = df.n_ref_voiced.clip(lower=0)
    s["rpa_pooled"] = float((df.rpa * w).sum() / max(w.sum(), 1))
    s["rtf"] = _rtf(df)
    s["rtf_lo"], s["rtf_hi"] = bootstrap_ci(df, _rtf, n_boot=n_boot)
    s["peak_mb"] = mem.get("peak_mb", float("nan"))
    s["model_mb"] = mem.get("model_mb", float("nan"))
    return s


def summary_markdown(s: pd.DataFrame) -> str:
    if s.empty:
        return "no results\n"
    rows = []
    for _, r in s.sort_values("rpa", ascending=False).iterrows():
        rows.append({
            "Tracker": r.tracker,
            "Items": int(r["items"]),
            "RPA (95% CI)": f"{r.rpa:.3f} ({r.rpa_lo:.3f}-{r.rpa_hi:.3f})",
            "RCA": f"{r.rca:.3f}",
            "RPA@25": f"{r.rpa_25:.3f}",
            "RPA@10": f"{r.rpa_10:.3f}",
            "RPA 65-1100 Hz": f"{r.rpa_sing:.3f}" if "rpa_sing" in r and pd.notna(r.rpa_sing) else "",
            "OA": f"{r.oa:.3f}",
            "VR": f"{r.vr:.3f}",
            "VFA": f"{r.vfa:.3f}",
            "Median err (c)": f"{r.median_err_cents:.1f}",
            "Octave err": f"{r.octave_err:.3f}",
            "RTF (95% CI)": f"{r.rtf:.3f} ({r.rtf_lo:.3f}-{r.rtf_hi:.3f})",
            "Peak MB": f"{r.peak_mb:.0f}",
            "Model MB": f"{r.model_mb:.0f}",
        })
    return pd.DataFrame(rows).to_markdown(index=False) + "\n"


# ----------------------------------------------------------------------------- run


def run(
    dataset: str = "synthetic",
    root: str | None = None,
    trackers: list[str] | None = None,
    limit: int | None = None,
    out_dir: str | Path = "results/trackers",
    threads: int = 4,
    noise_snr: float | None = None,
    mix_snr: float | None = None,
    audit: bool = False,
    n_boot: int = 2000,
    seed: int = 0,
    shard: str | None = None,
) -> pd.DataFrame:
    """Benchmark ``trackers`` (default: every installed one) on one dataset condition and write
    ``<label>_per_item.csv``, ``<label>_summary.csv/.md``, ``<label>_run.json`` (machine,
    threads, versions, git commit) and, with ``audit``, ``<label>_alignment.csv``."""
    from ..pitch.trackers import available_trackers

    trackers = trackers or available_trackers()
    label = dataset_label(dataset, noise_snr, mix_snr)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_item, summary, audits, failures = [], [], [], {}
    for name in trackers:
        res = _run_isolated(name, dataset, root, limit, threads, noise_snr=noise_snr, mix_snr=mix_snr, audit=audit, seed=seed, shard=shard)
        if not res.get("ok"):
            print(f"[{name}] failed: {res.get('error')}")
            failures[name] = res.get("error")
            continue
        df = pd.DataFrame(res["rows"])
        if df.empty:
            failures[name] = "no items"
            continue
        df["peak_mb"], df["model_mb"] = res.get("peak_mb", float("nan")), res.get("model_mb", float("nan"))
        per_item.append(df)
        summary.append(summarize(df, label, res, n_boot))
        if res.get("audit"):
            audits.append(pd.DataFrame(res["audit"]))
    s = pd.DataFrame(summary)
    if per_item:
        pd.concat(per_item).to_csv(out / f"{label}_per_item.csv", index=False)
    s.to_csv(out / f"{label}_summary.csv", index=False)
    (out / f"{label}_summary.md").write_text(summary_markdown(s))
    if audits:
        a_items = pd.concat(audits)
        a_items.to_csv(out / f"{label}_alignment_per_item.csv", index=False)
        a = a_items.groupby(["tracker", "shift_ms"])[["oa", "rpa", "rpa_10"]].mean().reset_index()
        a.to_csv(out / f"{label}_alignment.csv", index=False)
    from ..pitch.trackers import BENCH_FMAX, BENCH_FMIN, SEGMENT_CONTEXT_S, SEGMENT_S

    info = dict(dataset=dataset, label=label, root=str(root) if root else None, limit=limit, noise_snr=noise_snr,
                mix_snr=mix_snr, trackers=trackers, search_range_hz=[BENCH_FMIN, BENCH_FMAX],
                segment_s=SEGMENT_S, segment_context_s=SEGMENT_CONTEXT_S, shard=shard, failures=failures,
                machine=machine_info(threads))
    (out / f"{label}_run.json").write_text(json.dumps(json_safe(info), indent=2))
    return s


def merge_shards(shard_dirs: list[str | Path], out_dir: str | Path, label: str, n_boot: int = 2000) -> pd.DataFrame:
    """Join the per-item results of ``bench-trackers --shard k/n`` runs into the files an
    unsharded run writes. Every metric is a mean over items (or a sum, for RTF), so the
    merged table is the same as one run over all items; peak memory is the largest of any
    shard. Fails if a shard is missing, a stem appears twice, or the shards' settings differ."""
    dirs = [Path(d) for d in shard_dirs]
    infos = [json.loads((d / f"{label}_run.json").read_text()) for d in dirs]
    same = ("dataset", "noise_snr", "mix_snr", "search_range_hz", "segment_s", "segment_context_s", "limit")
    for i in infos[1:]:
        diff = [k for k in same if i.get(k) != infos[0].get(k)]
        if diff:
            raise ValueError(f"shards differ in {diff}")
    parts, cover = [], {}
    for d, i in zip(dirs, infos):
        part = pd.read_csv(d / f"{label}_per_item.csv") if (d / f"{label}_per_item.csv").exists() else pd.DataFrame(columns=["tracker"])
        parts.append(part)
        for t in part.tracker.unique():  # a directory may hold one tracker or several
            cover.setdefault(t, []).append(parse_shard(i.get("shard")) or (0, 1))
    n = max((k_n[1] for v in cover.values() for k_n in v), default=1)
    for t, ks in cover.items():
        if sorted(ks) != [(k, n) for k in range(n)]:
            raise ValueError(f"{t}: need every shard 0..{n - 1} of {n} exactly once, got {sorted(ks)}")
    df = pd.concat(parts, ignore_index=True)
    if df.duplicated(["tracker", "item"]).any():
        raise ValueError("an item appears in more than one shard")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, g in df.groupby("tracker", sort=False):
        mem = dict(peak_mb=float(g.peak_mb.max()), model_mb=float(g.model_mb.median()))
        summary.append(summarize(g.reset_index(drop=True), label, mem, n_boot))
    s = pd.DataFrame(summary)
    df.to_csv(out / f"{label}_per_item.csv", index=False)
    s.to_csv(out / f"{label}_summary.csv", index=False)
    (out / f"{label}_summary.md").write_text(summary_markdown(s))
    audit_files = [d / f"{label}_alignment_per_item.csv" for d in dirs]
    if all(f.exists() for f in audit_files):
        a_items = pd.concat([pd.read_csv(f) for f in audit_files], ignore_index=True)
        a_items.to_csv(out / f"{label}_alignment_per_item.csv", index=False)
        a_items.groupby(["tracker", "shift_ms"])[["oa", "rpa", "rpa_10"]].mean().reset_index().to_csv(out / f"{label}_alignment.csv", index=False)
    failures = {k: v for i in infos for k, v in (i.get("failures") or {}).items()}
    info = {**infos[0], "shard": None, "merged_from": [str(d) for d in dirs], "shards": n, "failures": failures,
            "machines": [i.get("machine") for i in infos]}
    (out / f"{label}_run.json").write_text(json.dumps(json_safe(info), indent=2))
    return s


def alignment_report(alignment_csv: str | Path) -> pd.DataFrame:
    """Best time shift per tracker from an ``--audit`` run. If every tracker prefers the same
    non-zero shift, the reference labels are offset (e.g. a wrong frame-time convention)."""
    a = pd.read_csv(alignment_csv)
    best = a.loc[a.groupby("tracker").oa.idxmax(), ["tracker", "shift_ms", "oa"]].rename(columns={"shift_ms": "best_shift_ms_oa"})
    best10 = a.loc[a.groupby("tracker").rpa_10.idxmax(), ["tracker", "shift_ms"]].rename(columns={"shift_ms": "best_shift_ms_rpa10"})
    return best.merge(best10, on="tracker")


# ----------------------------------------------------------------------------- selection


def _load_condition(out_dir: Path, label: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    s, p = out_dir / f"{label}_summary.csv", out_dir / f"{label}_per_item.csv"
    return (pd.read_csv(s) if s.exists() else None), (pd.read_csv(p) if p.exists() else None)


def paired_diff_ci(per_item: pd.DataFrame, a: str, b: str, metric: str = "rpa", n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean per-item difference a - b on the items both trackers ran, with a cluster bootstrap CI."""
    wide = per_item[per_item.tracker.isin([a, b])].pivot_table(index=["item", "cluster"], columns="tracker", values=metric).dropna().reset_index()
    if wide.empty or a not in wide or b not in wide:
        return float("nan"), float("nan"), float("nan")
    wide["d"] = wide[a] - wide[b]
    lo, hi = bootstrap_ci(wide, lambda d: d.d.mean(), n_boot=n_boot, seed=seed)
    return float(wide.d.mean()), lo, hi


def choose_tracker(
    out_dir: str | Path = "results/trackers",
    datasets: tuple[str, ...] = OFFICIAL_DATASETS,
    max_rtf: float = MAX_TRACKER_RTF,
    tie_pp: float = TIE_PP,
    trained_on: dict[str, tuple[str, ...]] | None = None,
    speed_dir: str | Path | None = None,
) -> dict:
    """Apply the pre-registered selection rule (docs/O2_PROTOCOL.md) to saved results.

    ``speed_dir`` takes the RTF of every tracker and dataset from another run, for example a
    timing run on the reference laptop, while accuracy comes from ``out_dir``. A tracker or
    dataset missing from the speed run makes the choice fail rather than mix machines.

    1. Eligible: tracker RTF <= ``max_rtf`` on every dataset.
    2. Score: mean RPA (50 cents, voicing-independent) over the datasets the tracker was not
       trained on (``TRAINED_ON``). A tracker trained on every dataset gets no score and is
       reported but not eligible.
    3. Chosen: the best-scoring eligible tracker, unless another eligible tracker is within
       ``tie_pp`` RPA points and faster, in which case the fastest of those wins.
    4. O2 passes when at least four trackers are reported on both MIR-1K and MDB-stem-synth and
       the chosen one reaches RPA >= 0.90 on MDB-stem-synth.

    The choice the same rule would make if training overlap were ignored is reported too, as a
    sensitivity check. If MIR-1K or MDB-stem-synth results are missing, whatever conditions
    exist are used and the result is marked ``official: false``.
    """
    trained_on = TRAINED_ON if trained_on is None else trained_on
    out = Path(out_dir)
    sums, items = {}, {}
    for ds in datasets:
        s, p = _load_condition(out, ds)
        if s is not None and len(s):
            sums[ds], items[ds] = s, p
    official = all(ds in sums for ds in OFFICIAL_DATASETS)
    if not sums:  # fall back to any clean summary in the folder (e.g. synthetic)
        for f in sorted(out.glob("*_summary.csv")):
            label = f.name[: -len("_summary.csv")]
            if "_noise" in label or "_mix" in label:
                continue
            s, p = _load_condition(out, label)
            if s is not None and len(s):
                sums[label], items[label] = s, p
    if not sums:
        raise FileNotFoundError(f"no benchmark summaries in {out}")
    table = None
    for ds, s in sums.items():
        part = s.set_index("tracker")[["rpa", "rtf"]].rename(columns={"rpa": f"rpa_{ds}", "rtf": f"rtf_{ds}"})
        table = part if table is None else table.join(part, how="inner")
    speed_from = None
    if speed_dir is not None:
        speed_from = {}
        for ds in sums:
            sp, _ = _load_condition(Path(speed_dir), ds)
            if sp is None:
                raise FileNotFoundError(f"no {ds} timing run in {speed_dir}")
            sp = sp.set_index("tracker")
            missing = [t for t in table.index if t not in sp.index]
            if missing:
                raise ValueError(f"the timing run in {speed_dir} has no {ds} result for {missing}")
            table[f"rtf_{ds}"] = sp.loc[table.index, "rtf"].values
            run = Path(speed_dir) / f"{ds}_run.json"
            speed_from[ds] = json.loads(run.read_text()).get("machine") if run.exists() else None
    rpa_cols = [c for c in table if c.startswith("rpa_")]
    rtf_cols = [c for c in table if c.startswith("rtf_")]

    def unseen(tracker: str) -> list[str]:
        return [ds for ds in sums if ds not in trained_on.get(tracker, ())]

    table["trained_on"] = [",".join(ds for ds in sums if ds in trained_on.get(t, ())) for t in table.index]
    table["score_datasets"] = [",".join(unseen(t)) for t in table.index]
    table["score"] = [float(table.loc[t, [f"rpa_{d}" for d in unseen(t)]].mean()) if unseen(t) else float("nan") for t in table.index]
    table["score_all"] = table[rpa_cols].mean(axis=1)
    table["rtf_max"] = table[rtf_cols].max(axis=1)
    table["rtf_mean"] = table[rtf_cols].mean(axis=1)
    table["eligible"] = (table.rtf_max <= max_rtf) & table.score.notna()
    chosen, best, pool, note = _apply_rule(table, "score", max_rtf, tie_pp)
    chosen_all, _, _, _ = _apply_rule(table.assign(eligible=table.rtf_max <= max_rtf), "score_all", max_rtf, tie_pp)
    runner_up = pool.drop(index=chosen).score.idxmax() if len(pool.dropna(subset=["score"])) > 1 else None
    diffs = {}
    if runner_up is not None:
        for ds, p in items.items():
            if p is not None:
                d, lo, hi = paired_diff_ci(p, chosen, runner_up)
                diffs[ds] = {"mean_rpa_difference": d, "ci95": [lo, hi]}
    n_both = int(len(table)) if official else 0
    mdb_rpa = float(table.loc[chosen, "rpa_mdb"]) if "rpa_mdb" in table else float("nan")
    res = {
        "tracker": str(chosen),
        "official": official,
        "datasets": list(sums),
        "rule": (f"eligible if tracker RTF <= {max_rtf} on every dataset; score = mean RPA over the datasets the tracker "
                 f"was not trained on; highest score wins; ties within {tie_pp} RPA points go to the faster tracker"),
        "best_by_accuracy": str(best),
        "runner_up": None if runner_up is None else str(runner_up),
        "paired_difference_vs_runner_up": diffs,
        "chosen_if_training_overlap_ignored": str(chosen_all),
        "table": json.loads(table.reset_index().to_json(orient="records")),
        "trackers_reported_on_both": n_both,
        "chosen_rpa_mdb": mdb_rpa,
        "speed_from": None if speed_dir is None else {"dir": str(speed_dir), "machine": speed_from},
        "O2_tracker_pass": bool(official and n_both >= O2_MIN_TRACKERS and mdb_rpa >= O2_MIN_MDB_RPA),
    }
    if note:
        res["note"] = note
    return res


def _apply_rule(table: pd.DataFrame, score: str, max_rtf: float, tie_pp: float):
    """(chosen, most accurate, pool, note) under the selection rule, scoring by ``score``."""
    pool = table[table.eligible & table[score].notna()]
    note = ""
    if pool.empty:
        pool = table[table[score].notna()]
        note = f"no tracker met RTF <= {max_rtf}; chose among all"
    if pool.empty:
        pool, note = table.assign(**{score: table["score_all"]}), "no tracker has unseen data; scored on all datasets"
    best = pool[score].idxmax()
    tied = pool[pool[score] >= pool[score].max() - tie_pp / 100]
    return tied.rtf_mean.idxmin(), best, pool, note


def json_safe(obj):
    """NaN and numpy scalars to plain JSON (NaN becomes null)."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_choice(choice: dict | str, path: str | Path = "results/chosen_tracker.json") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if isinstance(choice, str):
        choice = {"tracker": choice}
    Path(path).write_text(json.dumps(json_safe(choice), indent=2))
