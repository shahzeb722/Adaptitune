@echo off
REM Build the offline Windows app. Run from the repository root in a Python 3.11 venv.
python -m pip install --upgrade pip
pip install -e ".[app]" pyinstaller
REM Optional: export the tracker to ONNX first (needs the [neural] extra once, on any machine)
REM python scripts\export_onnx.py --out models\pesto.onnx
pyinstaller packaging\adaptitune.spec --noconfirm --distpath dist --workpath build
echo.
echo Built dist\AdaptiTune\AdaptiTune.exe
