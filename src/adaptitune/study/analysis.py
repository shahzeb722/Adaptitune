"""User-study analysis (proposal 5.7 and Table 3, O6).

Input: the trials CSV written by :mod:`adaptitune.study.protocol` (one row per judged take)
and the manual-task log (seconds and adjustments per session).

* Acceptance: mixed-effects logistic regression keep ~ condition + position, random
  intercept per singer (statsmodels' Bayesian binomial mixed GLM), with Holm correction
  across the pairwise condition contrasts. Reported as exploratory.
* Primary O6 estimates on sessions 3-5, with singer-clustered bootstrap 95 % CIs:
  acceptance(C3) - acceptance(C1) (target >= 15 points) and
  acceptance(C4) - acceptance(C3) (target <= 10 points).
* Naturalness non-inferiority on sessions 3-5: C3 no more than 0.5 points worse than C1 on
  the 1-5 scale, one-sided Wilcoxon signed-rank on each singer's mean difference shifted by
  the margin, alpha = 0.05.
* Time on sessions 3-5: the singer's time to a take they keep. With the agent that is the
  judging time (first play to answer) per kept C3 version; by hand it is the tuning task's
  time. Target: less time with the agent (the mean paired saving is above zero).
* Power simulation for the design, to run before the study starts.

docs/O6_PROTOCOL.md is the pre-registered plan these functions carry out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def cluster_bootstrap_diff(df: pd.DataFrame, a: str, b: str, col: str = "keep", n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """mean(col | condition a) - mean(col | condition b), resampling singers."""
    rng = np.random.default_rng(seed)
    per = df[df.condition.isin([a, b])].groupby(["participant", "condition"])[col].mean().unstack()
    per = per.dropna()
    if per.empty:
        return float("nan"), float("nan"), float("nan")
    d = (per[a] - per[b]).to_numpy()
    est = float(d.mean())
    boots = [rng.choice(d, size=len(d), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return est, float(lo), float(hi)


def naturalness_noninferiority(df: pd.DataFrame, test: str = "C3", ref: str = "C1", margin: float = 0.5, alpha: float = 0.05) -> dict:
    per = df[df.condition.isin([test, ref])].groupby(["participant", "condition"]).naturalness.mean().unstack().dropna()
    d = (per[test] - per[ref]).to_numpy()
    if len(d) < 3:
        return {"n": len(d), "p": float("nan"), "pass": None}
    # H0: median difference <= -margin ; H1: > -margin
    res = stats.wilcoxon(d + margin, alternative="greater", zero_method="wilcox")
    return {"n": len(d), "mean_diff": float(d.mean()), "p": float(res.pvalue), "pass": bool(res.pvalue < alpha)}


def time_to_kept_take(trials: pd.DataFrame, manual: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """Seconds of the singer's time per take they keep: judging time per kept C3 version
    against the hand-tuning task's time, paired by singer, with a bootstrap over singers.

    A singer who kept none of the agent's versions is counted as having judged them all and
    then tuned by hand, which is what they would have had to do."""
    hand = manual.dropna(subset=["seconds"]).groupby("participant").seconds.mean()
    c3 = trials[(trials.condition == "C3") & trials.seconds.notna()]
    agent = {}
    for p, g in c3.groupby("participant"):
        if p not in hand:
            continue
        kept = float(g.keep.sum())
        agent[p] = float(g.seconds.sum()) / kept if kept else float(g.seconds.sum()) + float(hand[p])
    if not agent:
        return {"n": 0, "agent_seconds": float("nan"), "manual_seconds": float("nan"), "saved": [float("nan")] * 3, "pass": None}
    codes = sorted(agent)
    d = np.array([hand[p] - agent[p] for p in codes])
    rng = np.random.default_rng(seed)
    boots = [rng.choice(d, size=len(d), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": len(codes), "agent_seconds": float(np.mean([agent[p] for p in codes])),
            "manual_seconds": float(hand[codes].mean()), "saved": [float(d.mean()), float(lo), float(hi)],
            "pass": bool(d.mean() > 0)}


def mixed_logit(df: pd.DataFrame) -> pd.DataFrame | None:
    """keep ~ C(condition, Treatment('C1')) + position, random intercept per participant."""
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except Exception:  # pragma: no cover
        return None
    d = df.dropna(subset=["keep"]).copy()
    d["keep"] = d.keep.astype(int)
    model = BinomialBayesMixedGLM.from_formula(
        "keep ~ C(condition, Treatment('C1')) + position", {"participant": "0 + C(participant)"}, d
    )
    fit = model.fit_vb()
    names = model.exog_names
    mean, sd = fit.fe_mean, fit.fe_sd
    z = mean / sd
    p = 2 * stats.norm.sf(np.abs(z))
    tab = pd.DataFrame({"term": names, "log_odds": mean, "sd": sd, "z": z, "p": p})
    cond = tab[tab.term.str.startswith("C(condition")]
    adj = holm(dict(zip(cond.term, cond.p)))
    tab["p_holm"] = tab.term.map(adj)
    return tab


def analyze(trials_csv: str, manual_csv: str | None = None) -> dict:
    df = pd.read_csv(trials_csv)
    clean = df[~df.flagged.astype(bool)] if "flagged" in df else df
    late = clean[clean.session >= 3]
    out = {"participants": int(df.participant.nunique()), "trials": len(df), "flagged_share": float(df.flagged.mean()) if "flagged" in df else 0.0}
    out["acceptance_by_condition_s3_5"] = late.groupby("condition").keep.mean().round(3).to_dict()
    out["C3_minus_C1"] = cluster_bootstrap_diff(late, "C3", "C1")
    out["C4_minus_C3"] = cluster_bootstrap_diff(late, "C4", "C3")
    out["C3_minus_C2"] = cluster_bootstrap_diff(late, "C3", "C2")
    every = df[df.session >= 3]  # flagged judgments are analysed separately (proposal 5.7)
    out["with_flagged"] = {"C3_minus_C1": cluster_bootstrap_diff(every, "C3", "C1"), "C4_minus_C3": cluster_bootstrap_diff(every, "C4", "C3")}
    out["naturalness_noninferiority"] = naturalness_noninferiority(late)
    out["naturalness_noninferiority_all_sessions"] = naturalness_noninferiority(clean)  # exploratory
    tab = mixed_logit(clean)
    out["mixed_logit"] = tab.round(4).to_dict("records") if tab is not None else None
    time_ok = None
    if manual_csv:
        man = pd.read_csv(manual_csv)
        out["manual_seconds_mean"] = float(man.seconds.mean())
        out["manual_adjustments_mean"] = float(man.adjustments.mean())
        if "seconds" in late:
            out["time_to_kept_take_s3_5"] = time_to_kept_take(late, man[man.session >= 3])
            time_ok = out["time_to_kept_take_s3_5"]["pass"]
    est, lo, hi = out["C3_minus_C1"]
    gap, glo, ghi = out["C4_minus_C3"]
    nat = out["naturalness_noninferiority"]["pass"]
    out["O6_parts"] = {"C3_beats_C1_by_15": bool(est >= 0.15), "C3_within_10_of_C4": bool(gap <= 0.10),
                       "less_time": time_ok, "naturalness_noninferior": nat}
    out["O6_targets_met"] = bool(est >= 0.15 and gap <= 0.10 and bool(nat) and time_ok is not False)
    return out


def simulate_study(
    n_singers: int = 12,
    p_c1: float = 0.45,
    effect: float = 0.15,
    singer_sd: float = 0.8,
    n_sims: int = 500,
    seed: int = 0,
) -> dict:
    """Power of the design for the C3-vs-C1 acceptance difference in sessions 3-5.

    Acceptance per singer is logistic with a random intercept (sd on the log-odds scale).
    'Detected' = the 95 % singer-bootstrap CI excludes 0 (fast proxy for the GLMM test).
    """
    rng = np.random.default_rng(seed)
    logit = lambda p: np.log(p / (1 - p))  # noqa: E731
    expit = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
    b1 = logit(p_c1)
    b3 = logit(p_c1 + effect)
    n_trials = 3 * 2  # C1 trials in sessions 3-5 per singer; C3 gets 3 * 4
    hits, ests = 0, []
    for _ in range(n_sims):
        u = rng.normal(0, singer_sd, n_singers)
        a1 = rng.binomial(n_trials, expit(b1 + u)) / n_trials
        a3 = rng.binomial(12, expit(b3 + u)) / 12
        d = a3 - a1
        boots = rng.choice(d, size=(400, n_singers), replace=True).mean(axis=1)
        lo = np.percentile(boots, 2.5)
        hits += lo > 0
        ests.append(d.mean())
    return {"n_singers": n_singers, "effect": effect, "power": hits / n_sims, "mean_estimate": float(np.mean(ests))}
