@echo off
REM Build the offline Windows app. Run from the repository root in a Python 3.11 venv.
REM .github/workflows/windows-app.yml does the same on every pull request and keeps the result
REM as the "AdaptiTune-windows" artifact, so most people never need to run this.
python -m pip install --upgrade pip
REM One pip call, so setuptools stays below 81 (pyworld imports pkg_resources).
python -m pip install -e ".[app]" "pyinstaller==6.22.3" "backports.tarfile>=1.2"
REM The app runs FCPE, the tracker O2 chose, with ONNX Runtime. Export it once (this needs PyTorch,
REM which the built app does not). Without models\fcpe.onnx the app falls back to PESTO, then pYIN.
if not exist models\fcpe.onnx (
    python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    python -m pip install "torchfcpe>=0.0.4" "onnx>=1.15" "setuptools<81"
    python scripts\export_onnx.py --model fcpe
)
pyinstaller packaging\adaptitune.spec --noconfirm --distpath dist --workpath build
echo.
echo Built dist\AdaptiTune\AdaptiTune.exe
echo Check it with: dist\AdaptiTune\AdaptiTune.exe --self-test --out self_test.json
