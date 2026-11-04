"""O6 dry run: the whole five-session study with simulated participants on real singing.

Each simulated participant is one MIR-1K singer (amateur singers, so the takes carry real pitch
errors). Every session goes through ``adaptitune.study.runner.StudySession``, the code the app's
study window runs: three of the singer's clips are the session's takes, the blind trial list is
the protocol's, C1/C2/C4 are rendered with their fixed settings and C3 by the agent, which learns
from the C3 judgments. Then the export goes through ``adaptitune.study.analysis.analyze``.

What is simulated is only the person: keep or discard comes from the O4 simulator's preference
models (with drift between sessions and fatigue within one), anchored to the singer's measured
context; the naturalness rating is the pipeline's own naturalness score of the version heard,
plus rating noise; the hand-tuning task lands near the simulated singer's ideal setting, off by
``--manual-noise`` (standard deviation per normalised setting). So the acceptance numbers follow
the simulator's assumptions and are not a result about people. The objective metrics (pitch
error, in-key accuracy, naturalness score) are measured on the real audio of every version and
do not depend on the simulated people.

Only the hand-tuned setting draws on ``--manual-noise``, and every simulated singer makes the same
number of random draws whatever it is, so runs that differ only in it judge every C1, C2 and C3
version identically. Only the C4 judgments change, and with them the consistency flag of a
session that repeats a C4 version.

    python scripts/o6_dry_run.py data/raw/MIR-1K --participants 15 --out results/o6_dry_run --workers 4
    python scripts/o6_dry_run.py data/raw/MIR-1K --manual-noise 0.15 --out results/o6_dry_run_noise015
"""

from __future__ import annotations

import os

from adaptitune.machine import THREAD_VARS

for _v in THREAD_VARS:  # one thread per worker; the workers run side by side
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

N_SESSIONS = 5
CLIPS_PER_SESSION = 3


def mir1k_singers(root: Path, need: int) -> dict[str, list[Path]]:
    """Singers with enough clips for five sessions, most clips first, clips in name order."""
    from adaptitune.data.datasets import _find_dir, _junk

    wav_dir = _find_dir(root, ("Wavfile",))
    by = defaultdict(list)
    for w in sorted(wav_dir.glob("*.wav")):
        if not _junk(w):
            by[w.stem.split("_")[0]].append(w)
    ok = {s: v for s, v in by.items() if len(v) >= N_SESSIONS * CLIPS_PER_SESSION}
    return dict(sorted(ok.items(), key=lambda kv: (-len(kv[1]), kv[0].lower()))[:need])


def read_take(path: Path):
    from adaptitune.audio import SYNTH_SR, Audio, normalize_loudness
    from adaptitune.data.datasets import _read

    a = _read(path, SYNTH_SR, channel=1)  # the vocal channel
    return Audio(normalize_loudness(a.samples), a.sr)


def sim_participant(rng, model: str, context: np.ndarray):
    """The O4 simulator's singer, with its ideal placed around what the population wants for
    this singer's real takes."""
    from adaptitune.agent.simulator import SimSinger, population_ideal

    s = SimSinger.sample(rng, model)
    s.context = context
    ideal = population_ideal(context) + rng.normal(0, [0.2, 0.2, 0.0, 0.2])
    ideal[2] = 1.0 if rng.random() < 0.8 else 0.0
    s.ideal = np.clip(ideal, 0, 1)
    return s


def stars(p: float, rng, sd: float = 0.5) -> int:
    return int(np.clip(np.round(1 + 4 * p + rng.normal(0, sd)), 1, 5))


def run_participant(job: dict) -> dict:
    from adaptitune.agent.features import context_from_analysis
    from adaptitune.agent.profile import ProfileStore
    from adaptitune.agent.session import AdaptiveSession
    from adaptitune.correction.settings import RETUNE_MS_LEVELS, CorrectionSettings
    from adaptitune.pipeline import Corrector
    from adaptitune.study.runner import StudySession

    code, idx, clips, out = job["code"], job["index"], [Path(c) for c in job["clips"]], Path(job["out"])
    rng = np.random.default_rng(1000 + idx)
    store = ProfileStore(out / "profiles" / f"{code}.sqlite")
    corrector = Corrector("auto")
    singer, log = None, []
    t0 = time.time()
    for s in range(1, N_SESSIONS + 1):
        api = AdaptiveSession(code, store, corrector, seed=100 * idx + s)
        run = StudySession(code, s, api, root=out / "study", keep_audio=False, build=job["build"])
        for c in clips[(s - 1) * CLIPS_PER_SESSION: s * CLIPS_PER_SESSION]:
            run.add_take(read_take(c))
        if singer is None:
            ctx = np.mean([context_from_analysis(t.analysis) for t in run.takes.values()], axis=0)
            singer = sim_participant(rng, job["model"], ctx)
        else:
            singer.new_session()  # drift between sessions, fatigue starts again
        run.start()
        for i in range(len(run.trials)):
            res = run.render(i)
            keep = int(singer.judge(res.settings.normalized(), "keep"))
            nat = stars(res.natural.score if res.natural else 1.0, rng)
            run.judge(i, keep, nat, seconds=float(rng.uniform(6, 20)))
        # the hand-tuning task: the singer ends near their ideal, on the sliders' own steps
        sd = job["manual_noise"]
        v = np.clip(singer.ideal + rng.normal(0, [sd, sd, 0.0, sd]), 0, 1)
        m = CorrectionSettings.from_normalized(v)
        m = CorrectionSettings(round(m.strength, 2), float(min(RETUNE_MS_LEVELS, key=lambda r: abs(np.log(r / m.retune_ms)))),
                               m.snap, round(m.vibrato, 2))
        run.manual(m)
        run.finish_manual(m, seconds=float(np.clip(rng.normal(160, 50), 45, 300)), adjustments=int(3 + rng.poisson(12)),
                          rating=stars(float(singer.p_accept(m.normalized())[0]), rng))
        run.finish()
        log.append({"session": s, "ideal": singer.ideal.round(4).tolist()})
    return {"code": code, "model": job["model"], "singer": job["singer"], "tracker": corrector.tracker.name,
            "minutes": round((time.time() - t0) / 60, 1), "ideal_by_session": log}


def describe(out: Path, participants: list[dict]) -> dict:
    """The protocol's analysis plus the descriptive tables the O6 report needs."""
    from adaptitune.correction.settings import CorrectionSettings
    from adaptitune.study.analysis import analyze
    from adaptitune.study.runner import export

    paths = export(out / "study", out / "export")
    summary = analyze(str(paths["trials"]), str(paths["manual"]))
    df = pd.read_csv(paths["trials"])
    man = pd.read_csv(paths["manual"])
    clean = df[~df.flagged.astype(bool)]
    by_session = clean.groupby(["session", "condition"]).keep.mean().unstack().round(3)
    originals = df[df.repeat_of.isna()]
    objective = originals.groupby("condition").agg(
        versions=("keep", "size"), before_cents=("before_cents_median", "median"), after_cents=("after_cents_median", "median"),
        in_key_accuracy=("accuracy", "mean"), naturalness_score=("natural_score", "mean"), vibrato_penalty=("vib_penalty", "mean"),
        naturalness_rating=("naturalness", "mean")).round(3)

    def norm(r):
        return CorrectionSettings(float(r.strength), float(r.retune_ms), r.snap, float(r.vibrato)).normalized()

    # how close the agent's C3 choices come to the setting the singer saved by hand that session
    saved = {(r.participant, r.session): norm(r) for r in man.itertuples()}
    ideal = {(p["code"], e["session"]): np.array(e["ideal"]) for p in participants for e in p["ideal_by_session"]}
    dist = []
    for r in originals[originals.condition == "C3"].itertuples():
        x = norm(r)
        dist.append({"session": r.session, "to_saved": float(np.linalg.norm(x - saved[(r.participant, r.session)])),
                     "to_true_ideal": float(np.linalg.norm(x - ideal[(r.participant, r.session)]))})
    dist = pd.DataFrame(dist).groupby("session").mean().round(3)
    by_model = clean[clean.session >= 3].merge(pd.DataFrame(participants)[["code", "model"]], left_on="participant", right_on="code")
    by_model = by_model.groupby(["model", "condition"]).keep.mean().unstack().round(3)
    by_session.to_csv(out / "acceptance_by_session.csv")
    objective.to_csv(out / "objective_by_condition.csv")
    dist.to_csv(out / "agent_distance_by_session.csv")
    by_model.to_csv(out / "acceptance_by_model_s3_5.csv")
    summary.update({
        "acceptance_by_session": by_session.to_dict("index"),
        "objective_by_condition": objective.to_dict("index"),
        "agent_distance_by_session": dist.to_dict("index"),
        "acceptance_by_model_s3_5": by_model.to_dict("index"),
        "manual_seconds_by_session": man.groupby("session").seconds.mean().round(1).to_dict(),
        "participants_detail": [{k: p[k] for k in ("code", "model", "singer", "tracker", "minutes")} for p in participants],
    })
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mir1k")
    ap.add_argument("--participants", type=int, default=15)
    ap.add_argument("--out", default="results/o6_dry_run")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--manual-noise", type=float, default=0.06,
                    help="how far the hand-tuned setting lands from the singer's ideal (sd per normalised setting)")
    a = ap.parse_args()
    from adaptitune.agent.simulator import MODELS
    from adaptitune.machine import build_info, machine_info

    out = Path(a.out)
    (out / "study").mkdir(parents=True, exist_ok=True)
    (out / "profiles").mkdir(exist_ok=True)
    singers = mir1k_singers(Path(a.mir1k), a.participants)
    if len(singers) < a.participants:
        raise SystemExit(f"only {len(singers)} MIR-1K singers have {N_SESSIONS * CLIPS_PER_SESSION} clips")
    build = build_info().get("git_commit")
    jobs = [{"code": f"P{i + 1:02d}", "index": i, "singer": s, "model": MODELS[i % len(MODELS)], "out": str(out), "build": build,
             "manual_noise": a.manual_noise, "clips": [str(c) for c in clips[: N_SESSIONS * CLIPS_PER_SESSION]]}
            for i, (s, clips) in enumerate(singers.items())]
    # register everyone first, so the counterbalancing index is the participant number
    (out / "study" / "participants.json").write_text(json.dumps({j["code"]: j["index"] for j in jobs}, indent=1))
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        done = pool.map(run_participant, jobs, chunksize=1)
    summary = describe(out, done)
    summary["manual_noise"] = a.manual_noise
    summary["machine"] = machine_info(1)
    summary["wall_minutes"] = round((time.time() - t0) / 60, 1)
    summary["note"] = ("Simulated participants (O4 simulator) judging real MIR-1K singing through the study runner: "
                       "acceptance follows the simulator's assumptions; the objective metrics are measured on the real audio.")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: summary[k] for k in ("C3_minus_C1", "C4_minus_C3", "C3_minus_C2", "naturalness_noninferiority",
                                              "time_to_kept_take_s3_5", "acceptance_by_condition_s3_5", "O6_parts",
                                              "O6_targets_met", "wall_minutes")}, indent=2, default=float))


if __name__ == "__main__":
    main()
