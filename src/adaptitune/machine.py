"""Which machine and software produced a number: CPU, cores, memory, versions and the build.

Kept free of numpy and pandas so the packaged app can pin thread counts before they load.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

# every numeric library's thread setting; the speed protocol pins them all to the same count
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "ADAPTITUNE_THREADS")


def thread_env(threads: int, base: dict | None = None) -> dict:
    """Environment that pins every numeric library to ``threads`` threads."""
    env = dict(os.environ if base is None else base)
    for var in THREAD_VARS:
        env[var] = str(threads)
    return env


def cpu_name() -> str:
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        elif sys.platform == "win32":
            import winreg

            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def version(pkg: str) -> str | None:
    from importlib import metadata

    try:
        return metadata.version(pkg)
    except Exception:
        return None


def git_commit() -> str | None:
    try:
        here = Path(__file__).resolve().parent
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def build_info() -> dict:
    """The commit the code came from: ``build_info.json`` in a packaged build (written by
    packaging/adaptitune.spec), else the git checkout."""
    try:
        from importlib import resources

        return json.loads(resources.files("adaptitune").joinpath("build_info.json").read_text())
    except Exception:
        return {"git_commit": git_commit(), "packaged": False}


def machine_info(threads: int, packages: tuple[str, ...] = ()) -> dict:
    """Everything needed to say which machine and software produced a speed number."""
    info = {
        "cpu": cpu_name(),
        "logical_cores": os.cpu_count(),
        "threads": threads,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git_commit": build_info().get("git_commit"),
        "packaged_app": bool(getattr(sys, "frozen", False)),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        import psutil

        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        pass
    info["versions"] = {p: version(p) for p in packages}
    return info
