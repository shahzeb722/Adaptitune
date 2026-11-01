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
from ..correction.settings import CorrectionSettings
from ..pipeline import Analysis, Corrector, TakeResult
from .features import context_from_analysis
from .policies import POLICIES, PopulationModel
from .profile import ProfileStore
from .reward import Feedback, reward


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
        self.policy.post = self.store.load(user, prior_var=self.policy.post.prior_var, noise_var=self.policy.post.noise_var)

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
        (or the population's choice if they coincide)."""
        a = self.policy.select(take.context)
        b = self.policy.greedy(take.context)
        if b == a:
            b = int(np.argmax(self.policy.pop.predict(take.context)))
        if b == a:
            scores = self.policy.mean_scores(take.context)
            scores[a] = -np.inf
            b = int(np.argmax(scores))
        from ..correction.settings import action_grid

        g = action_grid()
        return self.corrector.correct(take.analysis, g[a]), self.corrector.correct(take.analysis, g[b])

    def manual(self, take: Take, settings: CorrectionSettings) -> TakeResult:
        return self.corrector.correct(take.analysis, settings, score=False)

    # ------------------------------------------------------------------ feedback
    def _learn(self, take: Take, res: TakeResult, fb: Feedback) -> float:
        a = res.accuracy if not np.isnan(res.accuracy) else 0.0
        n = res.natural.score if res.natural else 1.0
        r = reward(fb.scaled(), a, n, res.vib_penalty)
        self.policy.learn(take.context, res.settings.snapped(), r)
        self.store.add_feedback(self.user, take.id, res.settings.as_dict(), take.context, fb.kind, fb.value, a, n, res.vib_penalty, r)
        self.store.save(self.user, self.policy.post)
        return r

    def feedback(self, take: Take, res: TakeResult, kind: str, value: float) -> float:
        """kind: 'keep' (1 keep / 0 discard) or 'rating' (1..5)."""
        return self._learn(take, res, Feedback(kind, value))

    def feedback_ab(self, take: Take, a: TakeResult, b: TakeResult, picked: str) -> tuple[float, float]:
        ra = self._learn(take, a, Feedback("ab", 1.0 if picked == "a" else 0.0))
        rb = self._learn(take, b, Feedback("ab", 1.0 if picked == "b" else 0.0))
        return ra, rb

    def save_manual(self, take: Take, settings: CorrectionSettings, seconds: float | None = None, adjustments: int | None = None, rating: float | None = None) -> None:
        self.store.add_manual(self.user, take.id, settings.as_dict(), seconds, adjustments, rating)

    def last_manual(self) -> CorrectionSettings | None:
        d = self.store.last_manual(self.user)
        return CorrectionSettings(**d) if d else None

    # ------------------------------------------------------------------ profile
    @property
    def interactions(self) -> int:
        return self.policy.post.n

    def current_best(self, take: Take) -> CorrectionSettings:
        from ..correction.settings import action_grid

        return action_grid()[self.policy.greedy(take.context)]

    def reset(self) -> None:
        self.store.reset(self.user)
        self.policy.post = self.store.load(self.user, prior_var=self.policy.post.prior_var, noise_var=self.policy.post.noise_var)
        self.policy.anchored = False

    def export(self, path: str | Path) -> Path:
        return self.store.export(self.user, path)
