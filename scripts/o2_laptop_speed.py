"""O2 speed on the reference laptop (docs/O2_PROTOCOL.md, A4 and A9).

The accuracy results were computed on a cloud machine. The speed numbers in the report must come
from the team's reference laptop, so this script times every tracker there, at 4 threads, on the
first of 8 parts of MIR-1K and the first of 24 parts of MDB-stem-synth (the same items as the
cloud's parts 0/8 and 0/24, so the accuracy on them must also match). That is the six trackers
of the protocol plus FCPE's ONNX export, which is how the app runs the chosen tracker (its speed
belongs to O5). Then it times the whole correction path the way the app runs it (FCPE through
ONNX Runtime, the planner, WORLD/PSOLA with the quality check) on one minute of MIR-1K vocals,
for O5. About 50 minutes. If the tracker part has already run, ``--pipeline-only`` does just
the last step (about 2 minutes).

Windows, in an Anaconda Prompt, from the unzipped repository folder:

    conda create -n adaptitune python=3.11 -y
    conda activate adaptitune
    pip install -e .[neural,eval]
    python scripts/export_onnx.py --model pesto
    python scripts/export_onnx.py --model fcpe
    python scripts/o2_laptop_speed.py --mir1k E:\\FYP\\MIR1K --mdb E:\\FYP\\MDB-stem-synth.tar.gz

It writes results/laptop_speed.zip. Send that file back; then
``adaptitune choose-tracker --results results/trackers --speed results/laptop_speed`` makes the
official choice.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


def run(cmd, env=None):
    print(">", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def join_vocals(mir1k_root: str, path: Path, seconds: float = 60.0) -> None:
    """The vocal channel of the first MIR-1K clips, with 0.3 s gaps, until ``seconds`` long
    (the same audio the app's own speed test times)."""
    from adaptitune.bench.rtf import mir1k_minute

    mir1k_minute(mir1k_root, path, seconds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mir1k", required=True, help="folder that contains MIR-1K (any depth)")
    ap.add_argument("--mdb", help="MDB-stem-synth folder, or the .tar.gz (it is extracted next to it)")
    ap.add_argument("--out", default="results/laptop_speed")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--trackers", nargs="+", help="default: every installed tracker")
    ap.add_argument("--limit", type=int, help="only for a quick check of the setup")
    ap.add_argument("--pipeline-only", action="store_true", help="skip the trackers; only time the full correction path")
    a = ap.parse_args()
    if not a.pipeline_only and not a.mdb:
        ap.error("--mdb is needed unless --pipeline-only is given")

    mdb = Path(a.mdb or ".")
    if not a.pipeline_only and mdb.suffixes[-2:] == [".tar", ".gz"]:
        dest = mdb.parent / "MDB-stem-synth_extracted"
        if not dest.exists():
            print(f"extracting {mdb} to {dest} (a few minutes)", flush=True)
            with tarfile.open(mdb) as t:
                t.extractall(dest, filter="data")
        mdb = dest
    for model in ("pesto", "fcpe"):
        if not Path(f"models/{model}.onnx").exists():
            sys.exit(f"models/{model}.onnx is missing: run  python scripts/export_onnx.py --model {model}  first")

    exe = [sys.executable, "-m", "adaptitune.cli"]
    out = Path(a.out)
    if not a.pipeline_only:
        for ds, root, shard in (("mir1k", a.mir1k, "0/8"), ("mdb", str(mdb), "0/24")):
            extra = (["--trackers", *a.trackers] if a.trackers else []) + (["--limit", str(a.limit)] if a.limit else [])
            run(exe + ["bench-trackers", "--dataset", ds, "--root", root, "--shard", shard, "--threads", str(a.threads), "--out", str(out)] + extra)

    # O5: the full correction path on real singing, pinned to the same number of threads
    from adaptitune.machine import thread_env

    vocals = out / "pipeline_input.wav"
    join_vocals(a.mir1k, vocals)
    run(exe + ["rtf", str(vocals), "--tracker", "fcpe-onnx", "--out", str(out / "pipeline_rtf.json")], env=thread_env(a.threads))
    zip_path = shutil.make_archive(str(out), "zip", root_dir=out)
    print(f"\nDone. Send this file back: {os.path.abspath(zip_path)}")


if __name__ == "__main__":
    main()
