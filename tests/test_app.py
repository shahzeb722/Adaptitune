import os

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
    assert w.session.interactions == 3
    assert "learned from 3" in w.status.currentMessage()
    w._switch_user("second")
    assert w.session.interactions == 0
    w.grab()
    w.close()
    app.processEvents()
