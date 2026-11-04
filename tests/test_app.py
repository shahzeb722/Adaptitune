import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")


def test_app_drives_all_features(tmp_path, detuned, monkeypatch):
    monkeypatch.setenv("ADAPTITUNE_HOME", str(tmp_path))
    from PySide6.QtWidgets import QApplication

    from adaptitune.agent.profile import ProfileStore
    from adaptitune.agent.session import AdaptiveSession
    from adaptitune.app.main import MainWindow
    from adaptitune.audio import Audio
    from adaptitune.pipeline import Corrector

    app = QApplication.instance() or QApplication([])
    store = ProfileStore(tmp_path / "p.sqlite")
    w = MainWindow(session_factory=lambda u: AdaptiveSession(u, store, Corrector("yin"), seed=0))
    w.player = None  # no audio device in CI
    take = w.session.load_take(Audio(detuned.samples, detuned.sr), key="C major")
    w._loaded(take)
    assert w.btn_correct.isEnabled()
    w._auto_done(w.session.propose(take))
    assert w.table.rowCount() == 5
    w._feedback("rating", 5)
    w._ab_done(w.session.propose_ab(take))
    w._pick("a")
    w._manual_changed()
    w._manual_done(w.session.manual(take, w._manual_settings()))
    w._manual_save()
    assert w.session.interactions == 2  # a rating and one A/B pick
    assert "learned from 2" in w.status.currentMessage()
    w._switch_user("second")
    assert w.session.interactions == 0
    w.grab()
    w.close()
    app.processEvents()


def _wait(app, cond, timeout=120):
    import time

    t0 = time.time()
    while not cond():
        app.processEvents()
        time.sleep(0.01)
        assert time.time() - t0 < timeout, "the job's result never reached the window"


def test_buttons_deliver_results_from_the_worker_thread(tmp_path, detuned, monkeypatch):
    """Import, correct and apply run on a worker thread; their results must come back to the
    window (they never did while nothing held the job, so every button hung on 'Analysing…')."""
    monkeypatch.setenv("ADAPTITUNE_HOME", str(tmp_path))
    from PySide6.QtWidgets import QApplication

    from adaptitune.agent.profile import ProfileStore
    from adaptitune.agent.session import AdaptiveSession
    from adaptitune.app.main import MainWindow
    from adaptitune.audio import Audio, save
    from adaptitune.pipeline import Corrector

    app = QApplication.instance() or QApplication([])
    store = ProfileStore(tmp_path / "p.sqlite")
    w = MainWindow(session_factory=lambda u: AdaptiveSession(u, store, Corrector("yin"), seed=0), show_errors=False)
    w.player = None
    save(tmp_path / "take.wav", Audio(detuned.samples, detuned.sr))
    w._load(str(tmp_path / "take.wav"))
    _wait(app, lambda: w.take is not None)
    w._auto_correct()
    _wait(app, lambda: w.result is not None)
    assert w.table.rowCount() == 5 and not w._jobs and not w.errors
    w._run(lambda: 1 / 0, lambda r: None, "failing on purpose")
    _wait(app, lambda: bool(w.errors))
    assert "ZeroDivisionError" in w.errors[-1]  # failures are reported, not swallowed
    w.close()


def test_record_without_a_microphone_resets_the_button(tmp_path, monkeypatch):
    """With no microphone, Record must say why and come back up, not stay pressed; a recording
    too short to analyse is refused the same way."""
    monkeypatch.setenv("ADAPTITUNE_HOME", str(tmp_path))
    from PySide6.QtWidgets import QApplication

    from adaptitune.agent.profile import ProfileStore
    from adaptitune.agent.session import AdaptiveSession
    from adaptitune.app.main import MainWindow
    from adaptitune.audio import Audio
    from adaptitune.pipeline import Corrector

    class NoMicrophone:
        def start(self):
            raise OSError("Error querying device -1")

    class Click:
        def start(self):
            pass

        def stop(self):
            return Audio(np.zeros(2400), 24_000)

    app = QApplication.instance() or QApplication([])
    store = ProfileStore(tmp_path / "p.sqlite")
    w = MainWindow(session_factory=lambda u: AdaptiveSession(u, store, Corrector("yin"), seed=0), show_errors=False)
    w.player = None
    w.btn_record.setEnabled(True)
    w.recorder = NoMicrophone()
    w.btn_record.click()
    assert not w.btn_record.isChecked() and w.btn_record.text() == "● Record"
    assert "Recording could not start" in w.errors[-1]
    w.recorder = Click()
    w.btn_record.click()
    assert w.btn_record.isChecked()
    w.btn_record.click()
    assert "shorter than half a second" in w.errors[-1] and w.take is None and not w._jobs
    w.close()
    app.processEvents()


def test_self_test_passes_every_requirement(monkeypatch, tmp_path):
    monkeypatch.setenv("ADAPTITUNE_HOME", str(tmp_path / "real_profile"))
    from adaptitune.app.checks import self_test

    report = self_test(tracker="yin")
    assert report["pass"], report["checks"]
    assert {"FR1 import a vocal", "FR2 key by hand or detected", "FR3 correct a take", "FR4 feedback: keep, rating, A/B",
            "FR5 manual settings saved", "FR6 per-note pitch error", "FR7 reset or export the profile",
            "NFR1 no network", "NFR3 data stays in the profile folder"} <= set(report["checks"])
    assert not (tmp_path / "real_profile").exists()  # the test used a throwaway folder


def test_self_test_notices_network_access():
    import socket

    from adaptitune.app.checks import NoNetwork

    with NoNetwork() as net, pytest.raises(OSError):
        socket.create_connection(("example.org", 80), timeout=1)
    assert net.attempts and socket.create_connection.__name__ == "create_connection"  # restored


def test_packaged_app_finds_its_bundled_model(tmp_path, monkeypatch):
    """PyInstaller 6 puts data files in _internal/ (sys._MEIPASS), not next to the exe."""
    import sys

    from adaptitune.pitch.trackers import find_onnx_model

    internal = tmp_path / "AdaptiTune" / "_internal"
    (internal / "models").mkdir(parents=True)
    (internal / "models" / "fcpe.onnx").write_bytes(b"model")
    for var in ("ADAPTITUNE_ONNX", "ADAPTITUNE_FCPE_ONNX"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ADAPTITUNE_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(internal), raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "AdaptiTune" / "AdaptiTune.exe"))
    assert find_onnx_model("fcpe") == str(internal / "models" / "fcpe.onnx")


def test_speed_test_reports_the_machine(tmp_path, detuned):
    import json

    from adaptitune.app.checks import cli
    from adaptitune.audio import Audio, save

    save(tmp_path / "take.wav", Audio(detuned.samples, detuned.sr))
    cli(["--speed-test", str(tmp_path / "take.wav"), "--out", str(tmp_path / "speed.json"), "--repeats", "1", "--tracker", "yin"])
    r = json.loads((tmp_path / "speed.json").read_text())
    assert r["rtf"] > 0 and r["target"] == 0.5 and r["tracker"] == "yin" and r["input"] == "the file take.wav"
    assert r["machine"]["cpu"] and r["machine"]["logical_cores"] and "platform" in r


def test_speed_test_input_is_the_first_mir1k_clips(tmp_path):
    import soundfile as sf

    from adaptitune.bench.rtf import mir1k_minute

    sr = 16_000
    (tmp_path / "MIR-1K" / "Wavfile").mkdir(parents=True)
    (tmp_path / "MIR-1K" / "PitchLabel").mkdir()
    for i in range(4):
        voc = 0.1 * np.sin(2 * np.pi * (200 + 20 * i) * np.arange(sr) / sr)
        sf.write(tmp_path / "MIR-1K" / "Wavfile" / f"amy_1_0{i}.wav", np.stack([np.zeros(sr), voc], axis=1), sr)
        np.savetxt(tmp_path / "MIR-1K" / "PitchLabel" / f"amy_1_0{i}.pv", np.full(49, 55.0))
    x, file_sr = sf.read(mir1k_minute(tmp_path, tmp_path / "minute.wav", seconds=3.0))
    assert file_sr == sr and len(x) == 3 * (sr + int(0.3 * sr))  # 1 s clips plus 0.3 s gaps: three reach 3 s


def test_qt_binding_keeps_none_alive():
    """PySide6 6.12.0 under Python 3.11 drops a reference to None on every call that returns
    nothing (setEnabled, lineTo, ...), so the app crashed after a few thousand UI calls, for
    example when drawing a minute-long pitch curve. pyproject.toml pins PySide6 below 6.12."""
    import sys

    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QPainterPath
    from PySide6.QtWidgets import QApplication, QPushButton

    QApplication.instance() or QApplication([])
    path, button, pt = QPainterPath(), QPushButton(), QPointF(1.0, 2.0)
    before = sys.getrefcount(None)
    for _ in range(500):
        path.lineTo(pt)
        button.setEnabled(True)
    assert sys.getrefcount(None) - before > -100
