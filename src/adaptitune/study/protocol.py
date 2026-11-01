"""User-study protocol (proposal 5.7, objective O6).

Conditions, judged blind on the same recording, tracker and shifter:

* C1 Fixed        - classic auto-tune setting
* C2 Default      - our pipeline with default settings
* C3 Personalized - our pipeline with the agent's choice
* C4 Manual       - in sessions 3-5, the setting the singer saved by hand last session is also
                    hidden among the blind judgments, so it gets a real acceptance rate.

Each session: 12 judged takes (10 unique + 2 repeats, about one in six) and one hand-tuning
task of at most 5 minutes, with a break in the middle. C3 appears 4 times per session, so the
agent learns from about 20 judgments over 5 sessions. The order of conditions is
counterbalanced across participants (rotated Latin square) and shuffled within blocks.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

CONDITIONS = ("C1", "C2", "C3", "C4")
N_SESSIONS = 5
TAKES_PER_SESSION = 12
REPEATS_PER_SESSION = 2
CONSISTENCY_FLAG = 0.70


@dataclass
class Trial:
    participant: str
    session: int
    position: int  # 1..12, order within session (fatigue covariate)
    condition: str
    stimulus: str  # recording id the version is made from
    repeat_of: int | None = None  # position of the earlier identical trial
    keep: int | None = None  # 1 keep / 0 discard (filled in during the session)
    naturalness: int | None = None  # 1..5
    flagged: bool = False


def session_conditions(session: int) -> list[str]:
    if session <= 2:
        return ["C1"] * 3 + ["C2"] * 3 + ["C3"] * 4
    return ["C1"] * 2 + ["C2"] * 2 + ["C3"] * 4 + ["C4"] * 2


def schedule(participant: str, session: int, stimuli: list[str], participant_index: int, seed: int = 0) -> list[Trial]:
    """Blind, counterbalanced trial list for one session."""
    rng = np.random.default_rng(hash((participant, session, seed)) % (2**32))
    conds = session_conditions(session)
    # counterbalancing: rotate the condition cycle by participant and session, then shuffle
    # within two blocks (before / after the break) so every condition appears in both halves
    order = sorted(set(conds), key=lambda c: (CONDITIONS.index(c) + participant_index + session) % 4)
    ranked = sorted(conds, key=lambda c: order.index(c))
    first, second = ranked[0::2], ranked[1::2]
    rng.shuffle(first)
    rng.shuffle(second)
    unique = first + second
    stim = list(rng.choice(stimuli, size=len(unique), replace=len(stimuli) < len(unique)))
    trials: list[Trial] = []
    for c, s in zip(unique, stim):
        trials.append(Trial(participant, session, 0, c, str(s)))
    # insert repeats of earlier trials in the second half
    for _ in range(REPEATS_PER_SESSION):
        src = int(rng.integers(0, len(trials) // 2 + 1))
        pos = int(rng.integers(len(trials) // 2 + 1, len(trials) + 1))
        t = trials[src]
        trials.insert(pos, Trial(participant, session, 0, t.condition, t.stimulus, repeat_of=-1))
        trials[pos].repeat_of = id(t)  # temporary link, fixed below
    for i, t in enumerate(trials, 1):
        t.position = i
    by_id = {id(t): t.position for t in trials}
    for t in trials:
        if t.repeat_of is not None:
            t.repeat_of = by_id.get(t.repeat_of)
    return trials


def consistency(trials: list[Trial]) -> float:
    """Share of repeated trials judged the same way as the original."""
    by_pos = {t.position: t for t in trials}
    pairs = [(by_pos[t.repeat_of], t) for t in trials if t.repeat_of and t.keep is not None]
    if not pairs:
        return float("nan")
    return float(np.mean([a.keep == b.keep for a, b in pairs]))


def flag_inconsistent(trials: list[Trial]) -> list[Trial]:
    """If agreement on repeats drops below 70 %, flag the rest of that session's trials
    (from the first disagreeing repeat on) for separate analysis."""
    by_pos = {t.position: t for t in trials}
    reps = sorted((t for t in trials if t.repeat_of and t.keep is not None), key=lambda t: t.position)
    agree = [by_pos[t.repeat_of].keep == t.keep for t in reps]
    if reps and np.mean(agree) < CONSISTENCY_FLAG:
        first_bad = next(t.position for t, ok in zip(reps, agree) if not ok)
        for t in trials:
            if t.position >= first_bad:
                t.flagged = True
    return trials


def save_trials(trials: list[Trial], path: str | Path) -> None:
    path = Path(path)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(trials[0])))
        if new:
            w.writeheader()
        for t in trials:
            w.writerow(asdict(t))


def write_plan(participants: list[str], stimuli: list[str], out: str | Path) -> Path:
    """Pre-generate every participant's 5-session schedule (for the ethics application and
    so the runner cannot bias the order)."""
    plan = {p: {s: [asdict(t) for t in schedule(p, s, stimuli, i)] for s in range(1, N_SESSIONS + 1)} for i, p in enumerate(participants)}
    out = Path(out)
    out.write_text(json.dumps(plan, indent=1))
    return out
