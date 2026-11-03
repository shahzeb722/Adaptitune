"""Command-line entry point.

    adaptitune correct in.wav out.wav [--key "A minor"] [--strength 0.8 ...]
    adaptitune build-corpus results/corpus --source synthetic|vocalset|medleydb [--root DIR] [--workers 4]
    adaptitune verify-corpus results/corpus [--tracker auto]
    adaptitune corpus-stats results/corpus
    adaptitune bench-trackers --dataset synthetic|mir1k|mdb [--root DIR] [--threads 4] [--audit]
    adaptitune choose-tracker [--results results/trackers]
    adaptitune eval-corrector results/corpus [--key-source user|estimated]
    adaptitune key-eval [--essen] [--audio 150]
    adaptitune key-excerpts data/raw/MIR-1K --source mir1k --out key_excerpts
    adaptitune key-agreement rater1.csv rater2.csv --out key_labels.csv
    adaptitune eval-keys key_labels.csv --audio-root key_excerpts
    adaptitune calibrate clean1.wav clean2.wav ...
    adaptitune simulate [--singers 100 --horizon 20]
    adaptitune rtf in.wav [--tracker auto]
    adaptitune study-plan p01 p02 ... --stimuli a.wav b.wav ... --out plan.json
    adaptitune study-analyze trials.csv [--manual manual.csv]
    adaptitune app
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path


def _settings(a):
    from .correction.settings import CorrectionSettings

    return CorrectionSettings(a.strength, a.retune_ms, a.snap, a.vibrato)


def main(argv=None):
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(prog="adaptitune")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("correct", help="correct one file with fixed settings")
    c.add_argument("inp")
    c.add_argument("out")
    c.add_argument("--key")
    c.add_argument("--tracker", default="auto")
    c.add_argument("--strength", type=float, default=0.75)
    c.add_argument("--retune-ms", type=float, default=44.7)
    c.add_argument("--snap", default="in_key", choices=["in_key", "chromatic"])
    c.add_argument("--vibrato", type=float, default=2 / 3)

    b = sub.add_parser("build-corpus", help="O1: clean + detuned copies (WORLD and PSOLA) with QC-checked labels")
    b.add_argument("out")
    b.add_argument("--source", default="synthetic", choices=["synthetic", "vocalset", "medleydb"])
    b.add_argument("--root")
    b.add_argument("--singers", type=int, default=20, help="synthetic only")
    b.add_argument("--takes", type=int, default=6, help="synthetic only: takes per singer")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--tracker", default="auto", help="analysis tracker for real takes")
    b.add_argument("--qc-tracker", default="pyin")
    b.add_argument("--qc-tol", type=float, default=10.0, help="cents")
    b.add_argument("--routes", nargs="+", default=["world", "psola"], choices=["world", "psola"])
    b.add_argument("--min-test-hours", type=float, default=2.0, help="real sources: test-split target (O1)")
    b.add_argument("--techniques", nargs="*", help="VocalSet techniques to use (default: sustained, pitched ones)")
    b.add_argument("--limit", type=int, help="build only the first N takes (quick check)")
    b.add_argument("--workers", type=int, default=1)
    b.add_argument("--resume", action="store_true", help="skip takes already built")

    v = sub.add_parser("verify-corpus", help="O1: re-measure every kept label and check file checksums")
    v.add_argument("corpus")
    v.add_argument("--tracker", default="auto", help="use one other than the build's QC tracker for an independent check")
    v.add_argument("--max-takes", type=int)
    v.add_argument("--split")
    v.add_argument("--tol", type=float, default=10.0)

    cs = sub.add_parser("corpus-stats", help="O1: hours per split, singers, QC and offset statistics")
    cs.add_argument("corpus")

    t = sub.add_parser("bench-trackers", help="O2: accuracy, speed and memory of every installed tracker")
    t.add_argument("--dataset", default="synthetic", choices=["synthetic", "mir1k", "mdb"])
    t.add_argument("--root")
    t.add_argument("--trackers", nargs="*")
    t.add_argument("--limit", type=int)
    t.add_argument("--threads", type=int, default=4)
    t.add_argument("--noise-snr", type=float, help="add pink noise at this SNR in dB")
    t.add_argument("--mix-snr", type=float, help="MIR-1K: mix the accompaniment back in at this SNR in dB")
    t.add_argument("--audit", action="store_true", help="also test label time alignment (shifts of -30..30 ms)")
    t.add_argument("--out", default="results/trackers")
    t.add_argument("--shard", help="k/n: run only the k-th of n interleaved parts (merge them with bench-merge)")
    bm = sub.add_parser("bench-merge", help="O2: join bench-trackers --shard runs into one result")
    bm.add_argument("dirs", nargs="+")
    bm.add_argument("--label", required=True, help="e.g. mdb or mir1k")
    bm.add_argument("--out", default="results/trackers")

    ct = sub.add_parser("choose-tracker", help="O2: apply the pre-registered selection rule to saved results")
    ct.add_argument("--results", default="results/trackers")
    ct.add_argument("--out", default="results/chosen_tracker.json")

    e = sub.add_parser("eval-corrector")
    e.add_argument("corpus")
    e.add_argument("--tracker", default="auto")
    e.add_argument("--split", default="test")
    e.add_argument("--key-source", default="user", choices=["user", "estimated"])
    e.add_argument("--out", default="results/corrector")
    e.add_argument("--max-takes", type=int)

    k = sub.add_parser("eval-keys", help="O2: key accuracy on the hand-labelled excerpts")
    k.add_argument("labels")
    k.add_argument("--audio-root", default=".")
    k.add_argument("--tracker", default="auto")
    k.add_argument("--out", help="per-excerpt CSV")

    ke = sub.add_parser("key-eval", help="validate and tune the key estimator on the Essen folk songs (music21 corpus)")
    ke.add_argument("--out", default="results/keys")
    ke.add_argument("--audio", type=int, default=150, help="test excerpts to synthesize and track")
    ke.add_argument("--max-songs", type=int)
    ke.add_argument("--workers", type=int, default=4)
    ke.add_argument("--seed", type=int, default=0)

    kx = sub.add_parser("key-excerpts", help="O2: cut ~100 excerpts for hand labelling")
    kx.add_argument("path", help="MIR-1K folder (--source mir1k) or an O1 corpus (--source corpus)")
    kx.add_argument("--source", choices=["mir1k", "corpus"], default="corpus",
                    help="mir1k: one excerpt per song (recommended); corpus: test-split clean takes")
    kx.add_argument("--out", default="key_excerpts")
    kx.add_argument("-n", type=int, default=100)
    kx.add_argument("--split", default="test", help="corpus only")
    kx.add_argument("--seed", type=int, default=0)

    ka = sub.add_parser("key-agreement", help="O2: merge two raters' labels, agreement and Cohen's kappa")
    ka.add_argument("rater1")
    ka.add_argument("rater2")
    ka.add_argument("--out", default="key_labels.csv")

    cal = sub.add_parser("calibrate", help="95th-percentile WORLD copy-synthesis MCD on clean vocals")
    cal.add_argument("files", nargs="+")

    s = sub.add_parser("simulate")
    s.add_argument("--singers", type=int, default=100)
    s.add_argument("--horizon", type=int, default=20)
    s.add_argument("--out", default="results/agent")

    r = sub.add_parser("rtf", help="full-pipeline real-time factor (NFR2)")
    r.add_argument("inp", nargs="?")
    r.add_argument("--tracker", default="auto")
    r.add_argument("--repeats", type=int, default=3)

    sp = sub.add_parser("study-plan")
    sp.add_argument("participants", nargs="+")
    sp.add_argument("--stimuli", nargs="+", required=True)
    sp.add_argument("--out", default="study_plan.json")

    sa = sub.add_parser("study-analyze")
    sa.add_argument("trials")
    sa.add_argument("--manual")

    sub.add_parser("app")

    a = ap.parse_args(argv)

    if a.cmd == "correct":
        from .pipeline import correct_file

        res = correct_file(a.inp, a.out, _settings(a), a.key, a.tracker)
        for row in res.note_report():
            print(row)
        print(f"key {res.key.name if res.key else '-'}  rtf {res.rtf:.2f}")
    elif a.cmd == "build-corpus":
        from .data.detune import build_corpus, corpus_stats

        out = build_corpus(
            a.out, a.source, a.root, a.singers, a.takes, tracker=a.tracker, seed=a.seed, routes=tuple(a.routes),
            qc_tracker=a.qc_tracker, qc_tol_cents=a.qc_tol, min_test_hours=a.min_test_hours,
            techniques=tuple(a.techniques) if a.techniques else None, limit=a.limit, workers=a.workers, resume=a.resume, verbose=True,
        )
        print(json.dumps(corpus_stats(out, write=False), indent=2, default=float))
    elif a.cmd == "verify-corpus":
        from .data.detune import verify_labels

        print(json.dumps(verify_labels(a.corpus, a.tracker, a.max_takes, a.tol, a.split), indent=2, default=float))
    elif a.cmd == "corpus-stats":
        from .data.detune import corpus_stats

        print(json.dumps(corpus_stats(a.corpus), indent=2, default=float))
    elif a.cmd == "bench-trackers":
        from .bench.trackers import OFFICIAL_DATASETS, choose_tracker, dataset_label, run, save_choice

        s_ = run(a.dataset, a.root, a.trackers, a.limit, a.out, threads=a.threads, noise_snr=a.noise_snr, mix_snr=a.mix_snr, audit=a.audit,
                 shard=a.shard)
        print((Path(a.out) / f"{dataset_label(a.dataset, a.noise_snr, a.mix_snr)}_summary.md").read_text())
        if not a.shard and all((Path(a.out) / f"{d}_summary.csv").exists() for d in OFFICIAL_DATASETS):
            choice = choose_tracker(a.out)
            save_choice(choice, Path(a.out) / "chosen_tracker.json")
            print("chosen (pre-registered rule):", choice["tracker"], " O2 tracker criterion met:", choice["O2_tracker_pass"])
    elif a.cmd == "bench-merge":
        from .bench.trackers import merge_shards

        merge_shards(a.dirs, a.out, a.label)
        print((Path(a.out) / f"{a.label}_summary.md").read_text())
    elif a.cmd == "choose-tracker":
        from .bench.trackers import choose_tracker, json_safe, save_choice

        choice = choose_tracker(a.results)
        save_choice(choice, a.out)
        print(json.dumps(json_safe({k: v for k, v in choice.items() if k != "table"}), indent=2))
    elif a.cmd == "eval-corrector":
        from .bench.corrector import evaluate

        print(json.dumps(evaluate(a.corpus, a.tracker, split=a.split or None, out_dir=a.out, max_takes=a.max_takes, key_source=a.key_source), indent=2))
    elif a.cmd == "eval-keys":
        from .bench.key_labels import evaluate_key_labels

        print(json.dumps(evaluate_key_labels(a.labels, a.tracker, a.audio_root, a.out), indent=2, default=float))
    elif a.cmd == "key-eval":
        from .bench.keys import evaluate_essen

        s_ = evaluate_essen(a.out, a.audio, a.seed, a.workers, a.max_songs)
        print((Path(a.out) / "essen_summary.md").read_text())
    elif a.cmd == "key-excerpts":
        from .bench.key_labels import select_excerpts, select_mir1k_excerpts

        if a.source == "mir1k":
            df = select_mir1k_excerpts(a.path, a.out, a.n, seed=a.seed)
        else:
            df = select_excerpts(a.path, a.out, a.n, a.split, seed=a.seed)
        print(f"{len(df)} excerpts of {df.song.nunique()} songs by {df.singer.nunique()} singers in {a.out}; "
              "open tools/key_rater.html and load that folder")
    elif a.cmd == "key-agreement":
        from .bench.key_labels import merge_raters

        print(json.dumps(merge_raters(a.rater1, a.rater2, a.out), indent=2, default=float))
    elif a.cmd == "calibrate":
        from .bench.corrector import calibrate_mcd_threshold

        print(f"mcd_threshold = {calibrate_mcd_threshold(a.files):.2f} dB")
    elif a.cmd == "simulate":
        from .agent.experiment import run, train_population

        _, s_ = run(singers_per_model=a.singers, horizon=a.horizon, population=train_population(), out_dir=a.out)
        print(json.dumps(s_, indent=2, default=float))
    elif a.cmd == "rtf":
        from .bench.rtf import measure

        print(json.dumps(measure(a.inp, a.tracker, a.repeats), indent=2))
    elif a.cmd == "study-plan":
        from .study.protocol import write_plan

        print(write_plan(a.participants, a.stimuli, a.out))
    elif a.cmd == "study-analyze":
        from .study.analysis import analyze

        print(json.dumps(analyze(a.trials, a.manual), indent=2, default=float))
    elif a.cmd == "app":
        from .app.main import main as app_main

        return app_main([sys.argv[0]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
