"""How the agent's settings were chosen (O4), on DEVELOPMENT singers only.

The reported O4 results (scripts/run_agent_sim.py) use seed 0. Every choice here was made on
seed 1, which gives a disjoint set of simulated singers, and the population is trained on a
third stream (seed 10000).

    python scripts/tune_agent.py --singers 100 --out results/agent_tuning.csv
"""

import argparse
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from adaptitune.agent.experiment import run_singer, singer_seed, train_population  # noqa: E402
from adaptitune.agent.policies import POLICIES  # noqa: E402
from adaptitune.agent.simulator import MODELS, SimSinger  # noqa: E402

DEV_SEED = 1


def evaluate(name, pop, singers, horizon=20, session_len=4, **kw):
    rows = []
    for model in MODELS:
        for i in range(singers):
            sseed = singer_seed(DEV_SEED, model, i)
            rng = np.random.default_rng(sseed)
            singer = SimSinger.sample(rng, model)
            pol = POLICIES[name](pop, rng=np.random.default_rng(sseed + 1), **kw)
            log = run_singer(pol, singer, horizon, session_len, rng)
            best = int(np.argmax(log.er_table.mean(axis=0)))
            rows.append(dict(model=model, greedy=log.greedy_er[-1], oracle=log.er_table[-1, best],
                             regret=float((log.er_table[:, best] - log.chosen_er).sum())))
    d = pd.DataFrame(rows)
    out = dict(ratio=d.greedy.mean() / d.oracle.mean(), regret=d.regret.mean(), regret_se=d.regret.std(ddof=1) / np.sqrt(len(d)))
    out.update({f"ratio_{m}": g.greedy.mean() / g.oracle.mean() for m, g in d.groupby("model")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--singers", type=int, default=100, help="per preference model")
    ap.add_argument("--out", default="results/agent_tuning.csv")
    a = ap.parse_args()
    pop = train_population()
    grid = [("epsilon_greedy", {}), ("ucb", {"alpha": 0.5}), ("ucb", {"alpha": 1.0}), ("thompson_linear", {})]
    grid += [("thompson", {"explore": ex, "rating_weight": rw}) for ex in (0.5, 0.75, 1.0) for rw in (1.0, 3.0)]
    rows = []
    for name, kw in grid:
        t0 = time.time()
        r = evaluate(name, pop, a.singers, **kw)
        rows.append(dict(policy=name, **{k: v for k, v in kw.items()}, **r))
        print(f"{name:16s} {kw}  ratio {r['ratio']:.3f}  regret {r['regret']:.2f}  ({time.time() - t0:.0f} s)", flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
