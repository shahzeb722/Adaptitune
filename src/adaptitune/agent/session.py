"""Glue between the correction pipeline, the personalization agent and the profile store.

This is what the desktop app (and the study runner) call. One ``AdaptiveSession`` per user:

    s = AdaptiveSession(user="ali")
    take = s.load_take("song.wav", key=None)          # FR1, FR2
    res = s.propose(take)                             # agent picks settings, FR3
    s.feedback(take, res, "rating", 4)                # FR4 -> reward -> update -> saved
    a, b = s.propose_ab(take)                         # FR4 A/B
    s.feedback_ab(take, a, b, picked="a")
    man = s.manual(take, CorrectionSettings(...))     # FR5
    s.save_manual(take, man.settings, seconds=95, adjustments=12)
    s.reset()  /  s.export("me.json")                  # FR7
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np

from ..audio import Audio, load
from ..correction.settings import CorrectionSettings, action_grid, action_index
from ..pipeline import Analysis, Corrector, TakeResult
from .features import context_from_analysis
from .policies import POLICIES, PopulationModel
from .profile import ProfileStore
from .reward import Feedback


def load_population(path: str | Path | None = None) -> PopulationModel:
    """The frozen population part: a JSON file shipped with the package, trained in
    simulation by ``scripts/run_agent_sim.py`` (retrain with pilot data when available)."""
    if path:
        return PopulationModel.from_json(Path(path).read_text())
    try:
        txt = resources.files("adaptitune.agent").joinpath("population_model.json").read_text()
        return PopulationModel.from_json(txt)
    except (FileNotFoundError, ModuleNotFoundError):
        return PopulationModel.neutral()


@dataclass
class Take:
    id: str
    analysis: Analysis
    context: np.ndarray
    started: float = field(default_factory=time.time)


class AdaptiveSession:
    def __init__(
        self,
        user: str = "default",
        store: ProfileStore | None = None,
        corrector: Corrector | None = None,
        policy: str = "thompson",
        population: PopulationModel | None = None,
        seed: int | None = None,
    ):
        self.user = user
        self.store = store or ProfileStore()
        self.corrector = corrector or Corrector()
        self.policy = POLICIES[policy](population or load_population(), rng=np.random.default_rng(seed))
        self.policy.set_posterior(self.store.load(user, self.policy.new_posterior))

    # ------------------------------------------------------------------ takes
    def load_take(self, src: str | Path | Audio, key: str | None = None) -> Take:
        audio = src if isinstance(src, Audio) else load(src)
        an = self.corrector.analyze(audio, key=key)
        return Take(uuid.uuid4().hex[:12], an, context_from_analysis(an))

    def set_key(self, take: Take, key: str) -> None:
        from ..music import Key

        take.analysis.key = Key.parse(key)
        take.analysis.key_confidence = 1.0

    # ------------------------------------------------------------------ correction
    def propose(self, take: Take) -> TakeResult:
        s = self.policy.choose(take.context)
        return self.corrector.correct(take.analysis, s)

    def propose_ab(self, take: Take) -> tuple[TakeResult, TakeResult]:
        """Two different versions: the agent's sampled choice and its current best guess
        (or the population's choice if they coincide). The simulation uses the same rule."""
        a = self.policy.select(take.context)
        b = self.policy.rival(take.context, a)
        g = action_grid()
        return self.corrector.correct(take.analysis, g[a]), self.corrector.correct(take.analysis, g[b])

    def manual(self, take: Take, settings: CorrectionSettings) -> TakeResult:
        return self.corrector.correct(take.analysis, settings, score=False)

    # ------------------------------------------------------------------ feedback
    @staticmethod
    def _measured(res: TakeResult) -> tuple[float, float, float]:
        """The pipeline's (a, n, p) for a corrected take."""
        a = res.accuracy if not np.isnan(res.accuracy) else 0.0
        n = res.natural.score if res.natural else 1.0
        return a, n, res.vib_penalty

    def _log(self, take: Take, res: TakeResult, fb: Feedback, r: float) -> None:
        a, n, p = self._measured(res)
        self.store.add_feedback(self.user, take.id, res.settings.as_dict(), take.context, fb.kind, fb.value, a, n, p, r)

    def feedback(self, take: Take, res: TakeResult, kind: str, value: float) -> float:
        """kind: 'keep' (1 keep / 0 discard) or 'rating' (1..5). Returns the reward r."""
        if kind not in ("keep", "rating"):
            raise ValueError("use feedback_ab for A/B picks")
        fb = Feedback(kind, value)
        r = self.policy.observe(take.context, action_index(res.settings.snapped()), kind, fb.scaled(), self._measured(res))
        self._log(take, res, fb, r)
        self.store.save(self.user, self.policy.post)
        return r

    def feedback_ab(self, take: Take, a: TakeResult, b: TakeResult, picked: str) -> tuple[float, float]:
        """One A/B judgment: ``picked`` is "a" or "b". Returns the reward r of each version."""
        ia, ib = action_index(a.settings.snapped()), action_index(b.settings.snapped())
        ra, rb = self.policy.observe_pair(take.context, ia, ib, picked == "a", self._measured(a), self._measured(b))
        self._log(take, a, Feedback("ab", 1.0 if picked == "a" else 0.0), ra)
        self._log(take, b, Feedback("ab", 1.0 if picked == "b" else 0.0), rb)
        self.store.save(self.user, self.policy.post)
        return ra, rb

    def save_manual(self, take: Take, settings: CorrectionSettings, seconds: float | None = None, adjustments: int | None = None, rating: float | None = None) -> None:
        self.store.add_manual(self.user, take.id, settings.as_dict(), seconds, adjustments, rating)

    def last_manual(self) -> CorrectionSettings | None:
        d = self.store.last_manual(self.user)
        return CorrectionSettings(**d) if d else None

    # ------------------------------------------------------------------ profile
    @property
    def interactions(self) -> int:
        """Judgments learned from so far (an A/B pick is one judgment)."""
        return self.policy.post.n

    def current_best(self, take: Take) -> CorrectionSettings:
        return action_grid()[self.policy.greedy(take.context)]

    def reset(self) -> None:
        self.store.reset(self.user)
        self.policy.set_posterior(self.store.load(self.user, self.policy.new_posterior))

    def export(self, path: str | Path) -> Path:
        return self.store.export(self.user, path)
