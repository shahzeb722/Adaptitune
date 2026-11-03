@echo off
REM Build the offline Windows app. Run from the repository root in a Python 3.11 venv.
python -m pip install --upgrade pip
pip install -e ".[app]" pyinstaller
REM Export the trackers to ONNX first (needs the [neural] extra once, on any machine). The app uses
REM FCPE, the tracker the O2 benchmark chose; without models\fcpe.onnx it falls back to PESTO, then pYIN.
REM python scripts\export_onnx.py --model fcpe
REM python scripts\export_onnx.py --model pesto
pyinstaller packaging\adaptitune.spec --noconfirm --distpath dist --workpath build
echo.
echo Built dist\AdaptiTune\AdaptiTune.exe
