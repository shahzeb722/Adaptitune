"""O2 speed on the reference laptop (docs/O2_PROTOCOL.md, A4 and A9).

The accuracy results were computed on a cloud machine. The speed numbers in the report must come
from the team's reference laptop, so this script times every tracker there, at 4 threads, on the
first of 8 parts of MIR-1K and the first of 24 parts of MDB-stem-synth (the same items as the
cloud's parts 0/8 and 0/24, so the accuracy on them must also match). About 45 minutes.

Windows, in an Anaconda Prompt, from the unzipped repository folder:

    conda create -n adaptitune python=3.11 -y
    conda activate adaptitune
    pip install -e .[neural,eval]
    python scripts/export_onnx.py --out models/pesto.onnx
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


def run(cmd):
    print(">", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mir1k", required=True, help="folder that contains MIR-1K (any depth)")
    ap.add_argument("--mdb", required=True, help="MDB-stem-synth folder, or the .tar.gz (it is extracted next to it)")
    ap.add_argument("--out", default="results/laptop_speed")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--trackers", nargs="+", help="default: all six")
    ap.add_argument("--limit", type=int, help="only for a quick check of the setup")
    a = ap.parse_args()

    mdb = Path(a.mdb)
    if mdb.suffixes[-2:] == [".tar", ".gz"]:
        dest = mdb.parent / "MDB-stem-synth_extracted"
        if not dest.exists():
            print(f"extracting {mdb} to {dest} (a few minutes)", flush=True)
            with tarfile.open(mdb) as t:
                t.extractall(dest, filter="data")
        mdb = dest
    if not Path("models/pesto.onnx").exists():
        sys.exit("models/pesto.onnx is missing: run  python scripts/export_onnx.py --out models/pesto.onnx  first")

    exe = [sys.executable, "-m", "adaptitune.cli"]
    out = Path(a.out)
    for ds, root, shard in (("mir1k", a.mir1k, "0/8"), ("mdb", str(mdb), "0/24")):
        extra = (["--trackers", *a.trackers] if a.trackers else []) + (["--limit", str(a.limit)] if a.limit else [])
        run(exe + ["bench-trackers", "--dataset", ds, "--root", root, "--shard", shard, "--threads", str(a.threads), "--out", str(out)] + extra)
    zip_path = shutil.make_archive(str(out), "zip", root_dir=out)
    print(f"\nDone. Send this file back: {os.path.abspath(zip_path)}")


if __name__ == "__main__":
    main()
