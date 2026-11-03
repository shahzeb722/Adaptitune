"""O4: train the population model and run the agent simulation on the REPORT singers (seed 0).

    python scripts/run_agent_sim.py --singers 100 --horizon 20 --out results/agent
    python scripts/run_agent_sim.py --horizon 60 --out results/agent_h60     # learning curves
    python scripts/run_agent_sim.py --sensitivity --out results/agent        # robustness table

The agent's settings were chosen on development singers (seed 1) with scripts/tune_agent.py.
"""

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from adaptitune.agent.experiment import run, sensitivity, train_population  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--singers", type=int, default=100, help="per preference model")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--session-len", type=int, default=4)
    ap.add_argument("--pop-singers", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/agent")
    ap.add_argument("--sensitivity", action="store_true", help="only the robustness table")
    a = ap.parse_args()
    pop = train_population(a.pop_singers)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "population_model.json").write_text(pop.to_json())
    if a.sensitivity:
        t = sensitivity(pop, a.singers, a.horizon, a.seed)
        t.to_csv(out / "sensitivity.csv", index=False)
        (out / "sensitivity.md").write_text(t.round(3).to_markdown(index=False) + "\n")
        print(t.round(3).to_markdown(index=False))
        return
    _, s = run(singers_per_model=a.singers, horizon=a.horizon, session_len=a.session_len, seed=a.seed, population=pop, out_dir=out)
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
