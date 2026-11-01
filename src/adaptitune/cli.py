"""Command-line entry point.

    adaptitune correct in.wav out.wav [--key "A minor"] [--strength 0.8 ...]
    adaptitune build-corpus results/corpus --source synthetic|vocalset|medleydb [--root DIR]
    adaptitune verify-corpus results/corpus
    adaptitune bench-trackers --dataset synthetic|mir1k|mdb [--root DIR]
    adaptitune eval-corrector results/corpus [--key-source user|estimated]
    adaptitune eval-keys labels.csv --audio-root DIR
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

    b = sub.add_parser("build-corpus")
    b.add_argument("out")
    b.add_argument("--source", default="synthetic", choices=["synthetic", "vocalset", "medleydb"])
    b.add_argument("--root")
    b.add_argument("--singers", type=int, default=20)
    b.add_argument("--takes", type=int, default=6)
    b.add_argument("--seed", type=int, default=0)

    v = sub.add_parser("verify-corpus")
    v.add_argument("corpus")
    v.add_argument("--tracker", default="auto")
    v.add_argument("--max-takes", type=int)

    t = sub.add_parser("bench-trackers")
    t.add_argument("--dataset", default="synthetic", choices=["synthetic", "mir1k", "mdb"])
    t.add_argument("--root")
    t.add_argument("--trackers", nargs="*")
    t.add_argument("--limit", type=int)
    t.add_argument("--out", default="results/trackers")

    e = sub.add_parser("eval-corrector")
    e.add_argument("corpus")
    e.add_argument("--tracker", default="auto")
    e.add_argument("--split", default="test")
    e.add_argument("--key-source", default="user", choices=["user", "estimated"])
    e.add_argument("--out", default="results/corrector")
    e.add_argument("--max-takes", type=int)

    k = sub.add_parser("eval-keys")
    k.add_argument("labels")
    k.add_argument("--audio-root", default=".")
    k.add_argument("--tracker", default="auto")

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
        from .data.detune import build_corpus

        print(build_corpus(a.out, a.source, a.root, a.singers, a.takes, seed=a.seed))
    elif a.cmd == "verify-corpus":
        from .data.detune import verify_labels

        print(json.dumps(verify_labels(a.corpus, a.tracker, a.max_takes), indent=2))
    elif a.cmd == "bench-trackers":
        from .bench.trackers import pick_tracker, run, save_choice

        s_ = run(a.dataset, a.root, a.trackers, a.limit, a.out)
        print(s_.round(4).to_string(index=False))
        if len(s_):
            best = pick_tracker([s_])
            save_choice(best, f"{a.out}/chosen_tracker.json")
            print("chosen:", best)
    elif a.cmd == "eval-corrector":
        from .bench.corrector import evaluate

        print(json.dumps(evaluate(a.corpus, a.tracker, split=a.split or None, out_dir=a.out, max_takes=a.max_takes, key_source=a.key_source), indent=2))
    elif a.cmd == "eval-keys":
        from .bench.corrector import evaluate_keys

        print(json.dumps(evaluate_keys(a.labels, a.tracker, a.audio_root), indent=2))
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
