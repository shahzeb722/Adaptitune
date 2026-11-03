# PyInstaller spec for the Windows 10+ desktop build (NFR4).
#   pip install pyinstaller "adaptitune[app]"
#   pyinstaller packaging/adaptitune.spec --noconfirm
# Produces dist/AdaptiTune/AdaptiTune.exe (one-folder build; no installer needed).
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hidden = collect_submodules("adaptitune") + ["onnxruntime", "pyworld", "parselmouth", "soundfile", "sounddevice", "scipy.special._cdflib"]
import os

datas = collect_data_files("adaptitune") + collect_data_files("librosa") + collect_data_files("parselmouth")
# the trackers exported by scripts/export_onnx.py: FCPE (the O2 choice, what the app uses) and
# PESTO (its fallback); without them the app falls back to pYIN
for model in ("fcpe", "pesto"):
    if os.path.exists(f"models/{model}.onnx"):
        datas.append((f"../models/{model}.onnx", "models"))

a = Analysis(
    ["../src/adaptitune/app/main.py"],
    pathex=["../src"],
    hiddenimports=hidden,
    datas=datas,
    excludes=["torch", "torchaudio", "tensorflow", "pesto", "torchfcpe", "torchcrepe"],  # the app runs the trackers with ONNX Runtime
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="AdaptiTune", console=False)
coll = COLLECT(exe, a.binaries, a.datas, name="AdaptiTune")
