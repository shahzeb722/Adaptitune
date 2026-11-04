"""Simulation experiments for the personalization agent (objective O4).

1. Train the population part on simulated singers that are *not* used for evaluation.
2. For at least 100 singers per preference model (logistic, ideal point, threshold), with
   drift and fatigue on, run each policy for 20 interactions (about what a real singer gives
   over five sessions) and longer for learning curves.
3. Report mean expected reward of the policy's greedy choice after 20 interactions relative
   to the oracle (best fixed setting for that singer in hindsight), and cumulative regret
   against the same oracle.

O4 passes if Thompson sampling is within 10 % of the oracle after 20 interactions and has
lower cumulative regret than epsilon-greedy and the fixed setting.

Seeds: the population is trained with seed 10000, the agent's settings were chosen on
development singers (seed 1, ``scripts/tune_agent.py``) and the reported results use seed 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..correction.settings import action_matrix
from .features import action_quad
from .policies import LAPSE, POLICIES, PopulationModel, expand_context, fit_logistic
from .reward import reward
from .simulator import MODELS, SimSinger

FEEDBACK_MIX = (("keep", 0.4), ("rating", 0.3), ("ab", 0.3))


# ----------------------------------------------------------------------------- population


def train_population(n_singers: int = 3000, actions_per_singer: int = 24, seed: int = 10_000, ridge: float = 1e-2, lapse: float = LAPSE, model: str | None = None) -> PopulationModel:
    """Fit the population part on simulated singers that are never used for evaluation.

    For each singer, fit their logit of acceptance over all 504 settings (fractional logistic
    regression on the quadratic basis, with acceptance clipped to [lapse, 1 - lapse]) and their
    objective part 0.2 a + 0.2 n - p. h and O are ridge regressions of these per-singer weights
    on the expanded context. The personal prior covariance is the spread of the per-singer logit
    weights around h. The linear reward model of the ablation is fitted as in the first design,
    on ``actions_per_singer`` random settings per singer. ``model`` restricts the training
    singers to one preference model (a robustness check; by default all three are mixed)."""
    rng = np.random.default_rng(seed)
    X, Q = action_matrix(), action_quad()
    ctx, logit_w, obj_w = [], [], []
    lin_ctx, lin_idx, lin_rew = [], [], []
    ridge_w = 1e-3 * np.eye(Q.shape[1])
    for _ in range(n_singers):
        s = SimSinger.sample(rng, model)
        a = rng.integers(0, len(X), actions_per_singer)
        lin_ctx.append(np.repeat(s.context[None], actions_per_singer, 0))
        lin_idx.append(a)
        lin_rew.append(s.expected_reward(X[a]))
        p = np.clip(s.p_accept(X), lapse, 1 - lapse)
        logit_w.append(fit_logistic(Q, p, prior_precision=ridge_w)[0])
        acc, nat, pen = s.objective(X)
        obj_w.append(np.linalg.lstsq(Q, reward(0.0, acc, nat, pen), rcond=None)[0])
        ctx.append(expand_context(s.context))
    C, V, Ob = np.array(ctx), np.array(logit_w), np.array(obj_w)
    A = C.T @ C + ridge * np.eye(C.shape[1])
    logit_coef = np.linalg.solve(A, C.T @ V)
    objective_coef = np.linalg.solve(A, C.T @ Ob)
    resid = V - C @ logit_coef
    prior_cov = np.cov(resid.T) + 1e-3 * np.eye(Q.shape[1])
    return PopulationModel(logit_coef, objective_coef, prior_cov, _fit_linear(np.concatenate(lin_ctx), np.concatenate(lin_idx), np.concatenate(lin_rew), ridge))


def _fit_linear(contexts: np.ndarray, action_idx: np.ndarray, rewards: np.ndarray, ridge: float) -> np.ndarray:
    """Ridge regression of r on joint(expanded context, quadratic action) -> (36, 15)."""
    from .features import joint

    ctx = expand_context(contexts)
    Xj = joint(ctx, action_quad()[action_idx])
    A = Xj.T @ Xj + ridge * np.eye(Xj.shape[1])
    return np.linalg.solve(A, Xj.T @ rewards).reshape(ctx.shape[1], -1)


# ----------------------------------------------------------------------------- one singer


@dataclass
class RunLog:
    chosen_er: np.ndarray  # expected reward of the action actually shown, per step
    greedy_er: np.ndarray  # expected reward of the policy's greedy choice after each update
    er_table: np.ndarray  # (steps, 504) expected reward of every action at each step


def run_singer(policy, singer: SimSinger, horizon: int, session_len: int, rng: np.random.Generator, obj_noise: float = 0.03, feedback_mix=FEEDBACK_MIX) -> RunLog:
    """One simulated singer, judging one take per interaction with the app's own calls.

    Each interaction is a keep/discard, a 1-5 rating or an A/B pick (FEEDBACK_MIX). The A/B
    rival is chosen by ``policy.rival`` as in the app. The pipeline's measurements of a and n
    carry a little noise."""
    X = action_matrix()
    kinds = [k for k, _ in feedback_mix]
    probs = [p for _, p in feedback_mix]
    chosen, greedy, table = [], [], []
    c = singer.context

    def measure(act):
        a, n, p = (v[0] for v in singer.objective(X[act][None]))
        return np.clip(a + rng.normal(0, obj_noise), 0, 1), np.clip(n + rng.normal(0, obj_noise), 0, 1), p

    for t in range(horizon):
        if t > 0 and t % session_len == 0:
            singer.new_session()
        er_all = singer.expected_reward(X)
        table.append(er_all)
        a = policy.select(c)
        chosen.append(er_all[a])
        kind = str(rng.choice(kinds, p=probs))
        if kind == "ab":
            b = policy.rival(c, a)
            fa = singer.judge(X[a][None], "ab", X[b][None])
            ma, mb = measure(a), measure(b)
            policy.observe_pair(c, a, b, fa > 0.5, ma, mb)
        else:
            f = singer.judge(X[a][None], kind)
            policy.observe(c, a, kind, f, measure(a))
        greedy.append(er_all[policy.greedy(c)])
    return RunLog(np.array(chosen), np.array(greedy), np.array(table))


# ----------------------------------------------------------------------------- experiment


def singer_seed(seed: int, model: str, i: int) -> int:
    """Seed of the i-th simulated singer of a preference model. Stable across runs (Python's
    hash() of a string is not), and seeds 0 and 1 never share a singer for i < 1000."""
    return seed * 1_000_003 + MODELS.index(model) * 10_007 + i


def run(
    policies: tuple[str, ...] = ("thompson", "ucb", "epsilon_greedy", "thompson_discounted", "population", "fixed", "thompson_linear"),
    singers_per_model: int = 100,
    horizon: int = 20,
    session_len: int = 4,
    seed: int = 0,
    population: PopulationModel | None = None,
    out_dir: str | Path | None = "results/agent",
    feedback_mix=FEEDBACK_MIX,
) -> tuple[pd.DataFrame, dict]:
    pop = population or train_population()
    rows = []
    for model in MODELS:
        for i in range(singers_per_model):
            sseed = singer_seed(seed, model, i)
            for name in policies:
                rng = np.random.default_rng(sseed)  # same singer and noise stream for every policy
                singer = SimSinger.sample(rng, model)
                pol = POLICIES[name](pop, rng=np.random.default_rng(sseed + 1))
                log = run_singer(pol, singer, horizon, session_len, rng, feedback_mix=feedback_mix)
                # oracle: best fixed setting in hindsight over the whole run
                mean_er = log.er_table.mean(axis=0)
                best = int(np.argmax(mean_er))
                oracle_steps = log.er_table[:, best]
                for t in range(horizon):
                    rows.append(
                        dict(
                            model=model, singer=i, policy=name, step=t + 1,
                            chosen=log.chosen_er[t], greedy=log.greedy_er[t], oracle=oracle_steps[t],
                            oracle_now=log.er_table[t].max(),
                        )
                    )
    df = pd.DataFrame(rows)
    df["regret"] = df.oracle - df.chosen
    df["cum_regret"] = df.groupby(["model", "singer", "policy"]).regret.cumsum()
    summ = summarize(df, horizon)
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "steps.csv.gz", index=False)
        pd.DataFrame(summ["table"]).to_csv(out / "summary.csv", index=False)
        (out / "summary.md").write_text(_markdown(summ))
        plot(df, out)
    return df, summ


def sensitivity(population: PopulationModel | None = None, singers_per_model: int = 100, horizon: int = 20, seed: int = 0,
                policies: tuple[str, ...] = ("thompson", "epsilon_greedy")) -> pd.DataFrame:
    """How much the O4 result depends on the assumptions the agent was built on.

    Each row changes one thing and reruns the main comparison on the report singers: only
    keep/discard feedback, a personal prior that is too narrow or too wide, a population
    trained on 10 times fewer singers, or on one preference model only."""
    from dataclasses import replace

    pop = population or train_population()
    variants = [
        ("as reported", pop, FEEDBACK_MIX),
        ("keep/discard feedback only", pop, (("keep", 1.0),)),
        ("personal prior x0.5 (too narrow)", replace(pop, prior_cov=0.5 * pop.prior_cov), FEEDBACK_MIX),
        ("personal prior x2 (too wide)", replace(pop, prior_cov=2.0 * pop.prior_cov), FEEDBACK_MIX),
        ("population from 300 singers", train_population(300), FEEDBACK_MIX),
    ]
    variants += [(f"population from {m} singers only", train_population(model=m), FEEDBACK_MIX) for m in MODELS]
    rows = []
    for name, p, mix in variants:
        _, summ = run(policies, singers_per_model, horizon, seed=seed, population=p, out_dir=None, feedback_mix=mix)
        t = pd.DataFrame(summ["table"]).set_index("policy")
        row = {"variant": name}
        for pol in policies:
            row[f"{pol}_ratio"] = t.loc[pol, "ratio_to_oracle"]
            row[f"{pol}_regret"] = t.loc[pol, "cum_regret_mean"]
        row["O4_pass"] = summ["O4_pass"] if "fixed" in policies else bool(
            row["thompson_ratio"] >= 0.9 and row["thompson_regret"] < row.get("epsilon_greedy_regret", np.inf))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, horizon: int) -> dict:
    last = df[df.step == horizon]
    table = []
    for (pol), g in last.groupby("policy"):
        ratio = g.greedy.mean() / g.oracle.mean()
        table.append(
            dict(
                policy=pol,
                singers=len(g),
                greedy_reward_at_h=g.greedy.mean(),
                oracle_reward=g.oracle.mean(),
                ratio_to_oracle=ratio,
                cum_regret_mean=g.cum_regret.mean(),
                cum_regret_se=g.cum_regret.std(ddof=1) / np.sqrt(len(g)),
                **{f"ratio_{m}": gg.greedy.mean() / gg.oracle.mean() for m, gg in g.groupby("model")},
            )
        )
    t = pd.DataFrame(table).set_index("policy")
    ts = t.loc["thompson"] if "thompson" in t.index else None
    passed = None
    if ts is not None:
        beats = all(ts.cum_regret_mean < t.loc[p].cum_regret_mean for p in ("epsilon_greedy", "fixed") if p in t.index)
        passed = bool(ts.ratio_to_oracle >= 0.9 and beats)
    return {"horizon": horizon, "table": t.reset_index().to_dict("records"), "O4_pass": passed}


def _markdown(s: dict) -> str:
    t = pd.DataFrame(s["table"]).round(4)
    return (
        f"# Agent simulation (O4)\n\nHorizon: {s['horizon']} interactions. O4 pass: **{s['O4_pass']}**\n\n"
        + t.to_markdown(index=False)
        + "\n"
    )


def plot(df: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for pol, g in df.groupby("policy"):
        m = g.groupby("step").greedy.mean() / g.groupby("step").oracle.mean()
        axes[0].plot(m.index, m.values, label=pol)
        r = g.groupby("step").cum_regret.mean()
        axes[1].plot(r.index, r.values, label=pol)
    axes[0].axhline(0.9, color="grey", ls="--", lw=1)
    axes[0].set(xlabel="interaction", ylabel="greedy reward / oracle", title="Learning curve")
    axes[1].set(xlabel="interaction", ylabel="cumulative regret", title="Regret vs best fixed setting")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=130)
    plt.close(fig)

    models = sorted(df.model.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.5), sharey=True)
    for ax, model in zip(np.atleast_1d(axes), models):
        for pol, g in df[df.model == model].groupby("policy"):
            ax.plot(g.groupby("step").cum_regret.mean(), label=pol)
        ax.set(title=model, xlabel="interaction")
    np.atleast_1d(axes)[0].set_ylabel("cumulative regret")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "regret_by_model.png", dpi=130)
    plt.close(fig)
