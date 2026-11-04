# PyInstaller spec for the Windows 10+ desktop build (NFR4). Run from the repository root:
#   pip install -e ".[app]" "pyinstaller==6.22.3" "backports.tarfile>=1.2"   (one call: setuptools<81)
#   python scripts/export_onnx.py --model fcpe        (needs PyTorch and torchfcpe once)
#   pyinstaller packaging/adaptitune.spec --noconfirm
# or packaging/build_windows.bat, which does all three.
# Produces dist/AdaptiTune/AdaptiTune.exe (one-folder build; no installer needed). The same exe
# runs the checks: AdaptiTune.exe --self-test / --speed-test MIR1K_FOLDER (adaptitune.app.checks).
# .github/workflows/windows-app.yml builds it and runs the self-test on every pull request.
import json
import os
import subprocess
import time

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hidden = collect_submodules("adaptitune") + ["onnxruntime", "pyworld", "parselmouth", "soundfile", "sounddevice", "psutil",
                                             # pyworld imports pkg_resources, whose vendored jaraco.context needs
                                             # backports.tarfile on Python 3.11 (pip install backports.tarfile)
                                             "backports.tarfile"]

datas = collect_data_files("adaptitune") + collect_data_files("librosa")
# the trackers exported by scripts/export_onnx.py: FCPE (the O2 choice, what the app uses) and
# PESTO (its fallback); without them the app falls back to pYIN, which is too slow for NFR2.
# PyInstaller 6 puts them in _internal/models (find_onnx_model looks there).
for model in ("fcpe", "pesto"):
    if os.path.exists(f"models/{model}.onnx"):
        datas.append((f"../models/{model}.onnx", "models"))

# which commit this build is (Help > About, and every check report)
try:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
except Exception:
    commit = None
os.makedirs("build", exist_ok=True)
with open("build/build_info.json", "w") as f:
    json.dump({"git_commit": os.environ.get("GITHUB_SHA") or commit or None, "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "packaged": True}, f)
datas.append(("../build/build_info.json", "adaptitune"))

a = Analysis(
    ["../src/adaptitune/app/launch.py"],
    pathex=["../src"],
    hiddenimports=hidden,
    datas=datas,
    # the app runs the trackers with ONNX Runtime, and the study and plotting code is not part of it
    excludes=["torch", "torchaudio", "tensorflow", "pesto", "torchfcpe", "torchcrepe", "matplotlib", "statsmodels",
              "sklearn", "tkinter"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="AdaptiTune", console=False)
coll = COLLECT(exe, a.binaries, a.datas, name="AdaptiTune")
