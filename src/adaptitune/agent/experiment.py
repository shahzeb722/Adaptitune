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
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..correction.settings import action_matrix
from .policies import POLICIES, PopulationModel
from .reward import reward
from .simulator import MODELS, SimSinger

FEEDBACK_MIX = (("keep", 0.4), ("rating", 0.3), ("ab", 0.3))


# ----------------------------------------------------------------------------- population


def train_population(n_singers: int = 3000, actions_per_singer: int = 24, seed: int = 10_000, ridge: float = 1e-2) -> PopulationModel:
    rng = np.random.default_rng(seed)
    X = action_matrix()
    ctx, idx, rew = [], [], []
    for _ in range(n_singers):
        s = SimSinger.sample(rng)
        a = rng.integers(0, len(X), actions_per_singer)
        er = s.expected_reward(X[a])
        ctx.append(np.repeat(s.context[None], actions_per_singer, 0))
        idx.append(a)
        rew.append(er)
    return PopulationModel.fit(np.concatenate(ctx), np.concatenate(idx), np.concatenate(rew), ridge)


# ----------------------------------------------------------------------------- one singer


@dataclass
class RunLog:
    chosen_er: np.ndarray  # expected reward of the action actually shown, per step
    greedy_er: np.ndarray  # expected reward of the policy's greedy choice after each update
    er_table: np.ndarray  # (steps, 504) expected reward of every action at each step


def run_singer(policy, singer: SimSinger, horizon: int, session_len: int, rng: np.random.Generator, obj_noise: float = 0.03) -> RunLog:
    X = action_matrix()
    kinds = [k for k, _ in FEEDBACK_MIX]
    probs = [p for _, p in FEEDBACK_MIX]
    chosen, greedy, table = [], [], []
    c = singer.context
    for t in range(horizon):
        if t > 0 and t % session_len == 0:
            singer.new_session()
        er_all = singer.expected_reward(X)
        table.append(er_all)
        a = policy.select(c)
        chosen.append(er_all[a])
        kind = str(rng.choice(kinds, p=probs))
        if kind == "ab":
            # the comparison version: the population's choice, or a random neighbour if equal
            b = int(np.argmax(policy.pop.predict(c)))
            if b == a:
                b = int(rng.integers(len(X)))
            fa = singer.judge(X[a][None], "ab", X[b][None])
            for act, f in ((a, fa), (b, 1 - fa)):
                ao, no, po = (v[0] for v in singer.objective(X[act][None]))
                r = reward(f, np.clip(ao + rng.normal(0, obj_noise), 0, 1), np.clip(no + rng.normal(0, obj_noise), 0, 1), po)
                policy.update(c, act, r)
        else:
            f = singer.judge(X[a][None], kind)
            ao, no, po = (v[0] for v in singer.objective(X[a][None]))
            r = reward(f, np.clip(ao + rng.normal(0, obj_noise), 0, 1), np.clip(no + rng.normal(0, obj_noise), 0, 1), po)
            policy.update(c, a, r)
        greedy.append(er_all[policy.greedy(c)])
    return RunLog(np.array(chosen), np.array(greedy), np.array(table))


# ----------------------------------------------------------------------------- experiment


def run(
    policies: tuple[str, ...] = ("thompson", "linucb", "epsilon_greedy", "fixed", "population", "thompson_discounted"),
    singers_per_model: int = 100,
    horizon: int = 20,
    session_len: int = 4,
    seed: int = 0,
    population: PopulationModel | None = None,
    out_dir: str | Path | None = "results/agent",
) -> tuple[pd.DataFrame, dict]:
    pop = population or train_population()
    rows = []
    for model in MODELS:
        for i in range(singers_per_model):
            sseed = seed * 1_000_003 + hash(model) % 997 * 10_007 + i
            for name in policies:
                rng = np.random.default_rng(sseed)  # same singer and noise stream for every policy
                singer = SimSinger.sample(rng, model)
                pol = POLICIES[name](pop, rng=np.random.default_rng(sseed + 1))
                log = run_singer(pol, singer, horizon, session_len, rng)
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
