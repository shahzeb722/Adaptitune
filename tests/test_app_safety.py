"""The app's look and its safeguards: the stylesheet parses in both appearances, nothing can be
started while the window is busy, late results are dropped, judgments count once, unusable
input gets a plain message, and unexpected errors are logged instead of closing the app."""

import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")


def _wait(app, cond, timeout=120):
    t0 = time.time()
    while not cond():
        app.processEvents()
        time.sleep(0.01)
        assert time.time() - t0 < timeout, "the window never got there"


@pytest.fixture
def window(tmp_path, monkeypatch):
    monkeypatch.setenv("ADAPTITUNE_HOME", str(tmp_path))
    from PySide6.QtWidgets import QApplication

    from adaptitune.agent.profile import ProfileStore
    from adaptitune.agent.session import AdaptiveSession
    from adaptitune.app.main import MainWindow
    from adaptitune.pipeline import Corrector

    app = QApplication.instance() or QApplication([])
    store = ProfileStore(tmp_path / "p.sqlite")
    w = MainWindow(session_factory=lambda u: AdaptiveSession(u, store, Corrector("yin"), seed=0), show_errors=False)
    w.player = None
    yield app, w
    w.close()
    app.processEvents()


def test_the_stylesheet_parses_in_light_and_dark(window):
    from PySide6.QtCore import qInstallMessageHandler

    from adaptitune.app import theme

    app, w = window
    messages = []
    old = qInstallMessageHandler(lambda mode, ctx, msg: messages.append(msg))
    try:
        for mode in ("light", "dark", "auto"):
            dark = theme.apply(app, mode)
            assert dark == (mode == "dark" or (mode == "auto" and theme.system_is_dark()))
            assert theme.colors() is (theme.DARK if dark else theme.LIGHT)
            w.show()
            w.grab()  # paints every widget, so a bad rule would be reported here
            app.processEvents()
    finally:
        qInstallMessageHandler(old)
        theme.apply(app, "light")
    assert not [m for m in messages if "style" in m.lower() or "parse" in m.lower()], messages
    with pytest.raises(ValueError):
        theme.apply(app, "sepia")
    assert not theme.app_icon().isNull()


def test_nothing_starts_while_the_window_is_busy(window, detuned):
    from adaptitune.audio import Audio

    app, w = window
    take = w.session.load_take(Audio(detuned.samples, detuned.sr), key="C major")
    w._loaded(take)
    assert w.btn_correct.isEnabled() and w.btn_import.isEnabled() and w.user_box.isEnabled()
    w._run(lambda: time.sleep(0.5), lambda r: None, "working")
    busy = (w.btn_import, w.btn_correct, w.btn_make_ab, w.btn_apply, w.user_box, w.key_box, w.btn_new_user, w.btn_reset,
            w.act_import, w.act_study)
    assert not any(x.isEnabled() for x in busy)
    w._auto_correct()  # a second job is refused while one runs
    assert len(w._jobs) == 1
    _wait(app, lambda: not w._jobs)
    assert all(x.isEnabled() for x in busy)


def test_a_result_for_an_earlier_take_is_dropped(window, detuned):
    from adaptitune.audio import Audio

    app, w = window
    take = w.session.load_take(Audio(detuned.samples, detuned.sr), key="C major")
    w._loaded(take)
    res = w.session.propose(take)

    def slow():
        time.sleep(0.3)
        return res

    w._run(slow, w._auto_done, "Correcting…")
    w._switch_user("someone else")  # the singer changes before the version is back
    _wait(app, lambda: not w._jobs)
    assert w.result is None and w.auto_result is None and not w.errors


def test_each_version_is_judged_once(window, detuned):
    from adaptitune.audio import Audio

    app, w = window
    take = w.session.load_take(Audio(detuned.samples, detuned.sr), key="C major")
    w._loaded(take)
    w._auto_done(w.session.propose(take))
    w._feedback("keep", 1)
    w._feedback("rating", 5)  # the buttons are off by now, and a direct call does nothing either
    assert w.session.interactions == 1 and not w.btn_keep.isEnabled() and not w.rating.isEnabled()
    w._ab_done(w.session.propose_ab(take))
    w._pick("a")
    w._pick("b")
    assert w.session.interactions == 2 and not w.btn_pick_b.isEnabled()


def test_a_manual_setting_is_saved_only_once_heard(window, detuned):
    from adaptitune.audio import Audio

    app, w = window
    take = w.session.load_take(Audio(detuned.samples, detuned.sr), key="C major")
    w._loaded(take)
    assert not w.btn_save_man.isEnabled()
    w.s_strength.setValue(40)
    w._manual_done(w.session.manual(take, w._manual_settings()))
    assert w.btn_save_man.isEnabled()
    w.s_strength.setValue(90)  # moved again but not applied: not heard yet
    assert not w.btn_save_man.isEnabled()


def test_unusable_input_gets_a_plain_message(window, tmp_path, monkeypatch):
    from PySide6.QtCore import QMimeData, QUrl
    from PySide6.QtWidgets import QInputDialog

    from adaptitune.audio import Audio

    app, w = window
    w._load(Audio(np.zeros(601 * 8000, dtype=np.float32), 8000))  # over ten minutes
    _wait(app, lambda: bool(w.errors))
    assert "up to 10 minutes" in w.errors[-1] and "Traceback" not in w.errors[-1] and w.take is None

    for typed in ("x" * 41, "bell\x07"):  # too long, and a character that cannot be shown
        monkeypatch.setattr(QInputDialog, "getText", lambda *a, typed=typed, **k: (typed, True))
        w._new_user()
        assert "shorter name" in w.errors[-1] and w.session.user == "default"
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("  Ali   Khan ", True))
    w._new_user()
    assert w.session.user == "Ali Khan" and w.user_box.currentText() == "Ali Khan"

    class Drop:
        def __init__(self, path):
            self.md = QMimeData()
            self.md.setUrls([QUrl.fromLocalFile(str(path))])

        def mimeData(self):  # noqa: N802
            return self.md

    assert w._dropped_file(Drop(tmp_path / "song.WAV")) == str(tmp_path / "song.WAV")
    assert w._dropped_file(Drop(tmp_path / "notes.txt")) is None


def test_unexpected_errors_are_logged_not_fatal(tmp_path, monkeypatch):
    import sys
    import threading

    from adaptitune.app import safety

    monkeypatch.setenv("ADAPTITUNE_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)  # restored after the test
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    monkeypatch.setitem(safety._state, "installed", False)
    shown = []
    monkeypatch.setattr(safety, "_show", shown.append)
    path = safety.install()
    try:
        try:
            raise ValueError("boom in a slot")
        except ValueError:
            sys.excepthook(*sys.exc_info())
        assert shown and "boom in a slot" in shown[0]
        for h in safety.log.handlers:
            h.flush()
        assert "boom in a slot" in path.read_text(encoding="utf-8") and path.parent.name == "logs"
    finally:
        from PySide6.QtCore import qInstallMessageHandler

        qInstallMessageHandler(None)  # Qt's own handler again, for the tests that follow
        for h in list(safety.log.handlers):
            if not isinstance(h, __import__("logging").NullHandler):
                safety.log.removeHandler(h)
                h.close()
