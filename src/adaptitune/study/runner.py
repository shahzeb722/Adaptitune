"""Runs one session of the O6 user study (proposal 5.7) on the participant's own takes.

The app's study window and ``scripts/o6_dry_run.py`` both drive this class, so the simulated
dry run goes through exactly the code that real sessions use. The full plan is in
docs/O6_PROTOCOL.md.

    api = AdaptiveSession("P07", ProfileStore(study_root() / "profiles.sqlite"))
    run = StudySession("P07", next_session("P07"), api)
    for audio in three_takes:
        run.add_take(audio)
    run.start()                                   # the blind, counterbalanced trial list
    for i in range(len(run.trials)):
        res = run.render(i)                       # play res.audio as "Version i+1"
        run.judge(i, keep=1, naturalness=4, seconds=12.0)
    res = run.manual(settings)                    # the hand-tuning task, as often as wanted
    run.finish_manual(settings, seconds=140, adjustments=11, rating=4)
    run.finish()                                  # flags, files, summary

Rules the class enforces, so a session cannot drift from the protocol:

* Trials are judged in order. A C3 version is made only once every earlier trial is judged,
  so the agent has learned from all of them; other versions can be made ahead.
* A repeat plays exactly the audio of the trial it repeats, and only C3 originals teach the
  agent (a repeat would count the same judgment twice).
* C4 is the setting the participant saved by hand in an earlier session, read at the start.
* Everything is written under the study folder on this computer, by participant code only.
  A partial session is kept as ``*.partial.csv`` and left out of the export.
"""

from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..agent.profile import default_path
from ..agent.session import AdaptiveSession, Take
from ..audio import Audio, save
from ..correction.settings import DEFAULT, FIXED_CLASSIC, CorrectionSettings
from ..pipeline import TakeResult
from .protocol import N_SESSIONS, Trial, consistency, flag_inconsistent, schedule

PROTOCOL_VERSION = "O6-1"
TAKES_PER_SESSION = 3  # the participant records three short phrases at the start of a session
MANUAL_LIMIT_S = 300.0  # the hand-tuning task stops after five minutes
CODE_RE = re.compile(r"[A-Za-z]{1,4}[0-9]{1,4}")  # P07, S12: a code, never a name
CONDITION_SETTINGS = {"C1": FIXED_CLASSIC, "C2": DEFAULT}

TRIAL_FIELDS = ["participant", "session", "position", "condition", "stimulus", "repeat_of", "keep", "naturalness",
                "flagged", "seconds", "strength", "retune_ms", "snap", "vibrato", "accuracy", "natural_score",
                "vib_penalty", "before_cents_median", "after_cents_median", "learned", "agent_judgments", "time"]
MANUAL_FIELDS = ["participant", "session", "take", "seconds", "adjustments", "rating", "applied", "strength",
                 "retune_ms", "snap", "vibrato", "time"]


def study_root() -> Path:
    """The study folder, next to the app's own profile (``ADAPTITUNE_HOME`` moves both)."""
    root = default_path().parent / "study"
    root.mkdir(parents=True, exist_ok=True)
    return root


def check_code(code: str) -> str:
    code = code.strip().upper()
    if not CODE_RE.fullmatch(code):
        raise ValueError("Use a participant code such as P07: one to four letters, then digits. Never a name.")
    return code


def participant_index(code: str, root: Path | None = None) -> int:
    """Stable index for counterbalancing: participants are numbered in the order they start."""
    root = Path(root or study_root())
    reg = root / "participants.json"
    data = json.loads(reg.read_text()) if reg.exists() else {}
    if code not in data:
        data[code] = len(data)
        reg.write_text(json.dumps(data, indent=1))
    return int(data[code])


def finished_sessions(code: str, root: Path | None = None) -> list[int]:
    root = Path(root or study_root())
    found = []
    for p in (root / "sessions").glob(f"{code}_s*.json"):
        m = re.fullmatch(rf"{re.escape(code)}_s(\d+)\.json", p.name)
        if m:
            found.append(int(m.group(1)))
    return sorted(found)


def next_session(code: str, root: Path | None = None) -> int | None:
    """The participant's next session number, or None once all five are done."""
    done = finished_sessions(code, root)
    nxt = (max(done) + 1) if done else 1
    return nxt if nxt <= N_SESSIONS else None


def _median_abs(values) -> float | None:
    v = [abs(x) for x in values if x is not None and not np.isnan(x)]
    return round(float(np.median(v)), 1) if v else None


class StudySession:
    def __init__(self, participant: str, session: int, api: AdaptiveSession, root: str | Path | None = None,
                 seed: int = 0, keep_audio: bool = True, build: str | None = None):
        self.participant = check_code(participant)
        if api.user != self.participant:
            raise ValueError("the agent session must belong to the participant")
        if not 1 <= session <= N_SESSIONS:
            raise ValueError(f"sessions run from 1 to {N_SESSIONS}")
        self.root = Path(root or study_root())
        (self.root / "sessions").mkdir(parents=True, exist_ok=True)
        if session in finished_sessions(self.participant, self.root):
            raise ValueError(f"{self.participant} has already done session {session}")
        self.session, self.api, self.seed, self.keep_audio = session, api, seed, keep_audio
        self.index = participant_index(self.participant, self.root)
        self.c4 = api.last_manual() if session >= 3 else None
        self.takes: dict[str, Take] = {}
        self.trials: list[Trial] = []
        self.results: dict[int, TakeResult] = {}
        self.rows: list[dict] = []
        self.manual_row: dict | None = None
        self.manual_applied = 0
        self.started = time.time()
        self.agent_before = api.interactions
        if build is None:
            from ..machine import build_info

            build = build_info().get("git_commit")
        self.build = build
        self.stem = f"{self.participant}_s{self.session}"

    # ------------------------------------------------------------------ takes
    def add_take(self, audio: Audio, key: str | None = None) -> str:
        if self.trials:
            raise RuntimeError("all takes must be added before the session starts")
        if audio.duration < 0.5:
            raise ValueError("a take must be at least half a second long")
        tid = f"t{len(self.takes) + 1}"
        self.takes[tid] = self.api.load_take(audio, key)
        if self.keep_audio:
            save(self.root / "takes" / f"{self.stem}_{tid}.wav", audio, subtype="PCM_16")
        return tid

    def start(self) -> list[Trial]:
        if not self.takes:
            raise RuntimeError("record or import the takes first")
        self.trials = schedule(self.participant, self.session, list(self.takes), self.index, self.seed,
                               with_c4=self.c4 is not None)
        self.started = time.time()
        return self.trials

    # ------------------------------------------------------------------ versions
    @property
    def judged(self) -> int:
        return len(self.rows)

    def settings_for(self, trial: Trial) -> CorrectionSettings | None:
        """The fixed setting of C1, C2 and C4; None for C3, which the agent chooses."""
        if trial.condition == "C4":
            return self.c4
        return CONDITION_SETTINGS.get(trial.condition)

    def can_render(self, i: int) -> bool:
        t = self.trials[i]
        if t.repeat_of:
            return t.repeat_of - 1 < self.judged
        return t.condition != "C3" or i <= self.judged

    def render(self, i: int) -> TakeResult:
        """The version trial ``i`` plays (made once, then reused)."""
        if i in self.results:
            return self.results[i]
        if not self.can_render(i):
            raise RuntimeError("judge the earlier trials first: the agent must learn from them")
        t = self.trials[i]
        if t.repeat_of:
            res = self.results[t.repeat_of - 1]
        elif t.condition == "C3":
            res = self.api.propose(self.takes[t.stimulus])
        else:
            res = self.api.corrector.correct(self.takes[t.stimulus].analysis, self.settings_for(t))
        self.results[i] = res
        return res

    def judge(self, i: int, keep: int, naturalness: int, seconds: float | None = None) -> dict:
        if i != self.judged:
            raise RuntimeError(f"trial {self.judged + 1} is next")
        if keep not in (0, 1) or naturalness not in (1, 2, 3, 4, 5):
            raise ValueError("keep is 0 or 1 and naturalness 1 to 5")
        t, res = self.trials[i], self.render(i)
        t.keep, t.naturalness = int(keep), int(naturalness)
        learned = t.condition == "C3" and not t.repeat_of
        if learned:
            self.api.feedback(self.takes[t.stimulus], res, "keep", t.keep)
        notes = res.note_report()
        s = res.settings
        row = asdict(t) | {
            "seconds": None if seconds is None else round(float(seconds), 2),
            "strength": round(s.strength, 4), "retune_ms": round(s.retune_ms, 2), "snap": s.snap, "vibrato": round(s.vibrato, 4),
            "accuracy": None if np.isnan(res.accuracy) else round(float(res.accuracy), 4),
            "natural_score": round(float(res.natural.score), 4) if res.natural else None,
            "vib_penalty": round(float(res.vib_penalty), 4),
            "before_cents_median": _median_abs(r["before_cents"] for r in notes),
            "after_cents_median": _median_abs(r["after_cents"] for r in notes),
            "learned": learned, "agent_judgments": self.api.interactions,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        self.rows.append(row)
        self._write(self.root / "sessions" / f"{self.stem}_trials.partial.csv", TRIAL_FIELDS, self.rows)
        return row

    # ------------------------------------------------------------------ hand-tuning task (C4)
    @property
    def manual_take(self) -> str:
        return next(iter(self.takes))

    def manual(self, settings: CorrectionSettings) -> TakeResult:
        self.manual_applied += 1
        return self.api.manual(self.takes[self.manual_take], settings)

    def finish_manual(self, settings: CorrectionSettings, seconds: float, adjustments: int, rating: int) -> dict:
        if rating not in (1, 2, 3, 4, 5):
            raise ValueError("rating is 1 to 5")
        seconds = min(float(seconds), MANUAL_LIMIT_S)
        self.api.save_manual(self.takes[self.manual_take], settings, seconds=seconds, adjustments=int(adjustments), rating=rating)
        self.manual_row = {
            "participant": self.participant, "session": self.session, "take": self.manual_take,
            "seconds": round(seconds, 1), "adjustments": int(adjustments), "rating": int(rating),
            "applied": self.manual_applied, "strength": round(settings.strength, 4), "retune_ms": round(settings.retune_ms, 2),
            "snap": settings.snap, "vibrato": round(settings.vibrato, 4), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        return self.manual_row

    # ------------------------------------------------------------------ end
    def finish(self) -> dict:
        if self.judged < len(self.trials) or not self.trials:
            raise RuntimeError("judge every trial first")
        flag_inconsistent(self.trials)
        for row, t in zip(self.rows, self.trials):
            row["flagged"] = t.flagged
        sessions = self.root / "sessions"
        self._write(sessions / f"{self.stem}_trials.csv", TRIAL_FIELDS, self.rows)
        (sessions / f"{self.stem}_trials.partial.csv").unlink(missing_ok=True)
        if self.manual_row:
            self._write(sessions / f"{self.stem}_manual.csv", MANUAL_FIELDS, [self.manual_row])
        summary = {
            "participant": self.participant, "session": self.session, "protocol": PROTOCOL_VERSION, "build": self.build,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(self.started)),
            "minutes": round((time.time() - self.started) / 60, 1),
            "takes": {tid: {"seconds": round(tk.analysis.audio.duration, 1), "key": tk.analysis.key.name} for tid, tk in self.takes.items()},
            "c4_setting": self.c4.as_dict() if self.c4 else None,
            "consistency": consistency(self.trials),
            "flagged_trials": int(sum(t.flagged for t in self.trials)),
            "agent_judgments_before": self.agent_before, "agent_judgments_after": self.api.interactions,
            "manual_task": self.manual_row is not None,
        }
        (sessions / f"{self.stem}.json").write_text(json.dumps(summary, indent=2, default=float))
        return summary

    @staticmethod
    def _write(path: Path, fields: list[str], rows: list[dict]) -> None:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)


def export(root: str | Path | None = None, out: str | Path | None = None) -> dict[str, Path]:
    """Every finished session in one ``trials.csv`` and one ``manual.csv`` (the input of
    ``adaptitune study-analyze``), plus ``sessions.csv``. Participants appear by code only."""
    root = Path(root or study_root())
    out = Path(out or root / "export")
    out.mkdir(parents=True, exist_ok=True)
    sessions = root / "sessions"
    done = sorted(sessions.glob("*_s*.json"))
    trials, manual, summaries = [], [], []
    for js in done:
        summaries.append(json.loads(js.read_text()))
        for kind, rows in (("trials", trials), ("manual", manual)):
            p = sessions / f"{js.stem}_{kind}.csv"
            if p.exists():
                with open(p, newline="") as f:
                    rows.extend(csv.DictReader(f))
    paths = {"trials": out / "trials.csv", "manual": out / "manual.csv", "sessions": out / "sessions.csv"}
    StudySession._write(paths["trials"], TRIAL_FIELDS, trials)
    StudySession._write(paths["manual"], MANUAL_FIELDS, manual)
    sfields = ["participant", "session", "protocol", "build", "started", "minutes", "consistency", "flagged_trials",
               "agent_judgments_before", "agent_judgments_after", "manual_task"]
    StudySession._write(paths["sessions"], sfields, summaries)
    return paths


def withdraw(code: str, store, root: str | Path | None = None) -> int:
    """Delete everything kept about a participant who withdraws: session files, takes and the
    agent's profile. Returns the number of files removed."""
    code = check_code(code)
    root = Path(root or study_root())
    n = 0
    for folder in ("sessions", "takes"):
        for p in (root / folder).glob(f"{code}_s*"):
            p.unlink()
            n += 1
    store.reset(code)
    return n
