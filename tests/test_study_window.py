"""The study window driven the way a researcher and a participant use it, offscreen."""

import json
import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")


def _wait(app, cond, timeout=180):
    t0 = time.time()
    while not cond():
        app.processEvents()
        time.sleep(0.01)
        assert time.time() - t0 < timeout, "the window never got there"


def test_a_whole_session_through_the_window(tmp_path, monkeypatch):
    monkeypatch.setenv("ADAPTITUNE_HOME", str(tmp_path / "home"))
    from PySide6.QtWidgets import QApplication
    from test_study_runner import _takes

    from adaptitune.app.study import StudyWindow
    from adaptitune.pipeline import Corrector

    app = QApplication.instance() or QApplication([])
    w = StudyWindow(Corrector("yin"), player=None, root=tmp_path / "study", show_errors=False)
    w.show()
    assert not w.btn_start.isEnabled()
    w.code.setText("Ali Khan")
    assert "letters, then digits" in w.session_label.text() and not w.btn_start.isEnabled()
    w.code.setText("p05")
    assert w.session_label.text() == "Session 1 of 5"
    for a in _takes():
        w._add_take(a)
    assert not w.btn_start.isEnabled()  # no consent yet
    w.consent.setChecked(True)
    assert w.btn_start.isEnabled() and not w.btn_import_take.isEnabled()  # three takes is the limit
    w.btn_start.click()
    _wait(app, lambda: w.pages.currentIndex() == w.PAGE_TRIAL and w.btn_play.isEnabled())
    n = len(w.run.trials)
    breaks = 0
    while w.pages.currentIndex() != w.PAGE_MANUAL:
        if w.pages.currentIndex() == w.PAGE_BREAK:
            breaks += 1
            w.btn_continue.click()
        _wait(app, lambda: w.btn_play.isEnabled())
        assert not w.btn_next.isEnabled()  # nothing heard or answered yet
        w.btn_play.click()
        w.keep_row.buttons()[w.i % 2].click()
        w.nat_row.buttons()[3].click()
        assert w.btn_next.isEnabled()
        w.btn_next.click()
        app.processEvents()
    assert breaks == 1 and w.run.judged == n
    assert not w.btn_happy.isEnabled()  # hear your own setting before saying you are happy
    w.manual_panel.strength.setValue(40)
    w.btn_apply.click()
    _wait(app, lambda: w.btn_happy.isEnabled())
    w.btn_happy.click()
    assert not w.btn_finish.isEnabled()
    w.rate_row.buttons()[4].click()
    w.btn_finish.click()
    assert w.pages.currentIndex() == w.PAGE_DONE and not w.errors
    summary = json.loads((tmp_path / "study" / "sessions" / "P05_s1.json").read_text())
    assert summary["manual_task"] and summary["agent_judgments_after"] == 4
    assert w.run.manual_row["strength"] == 0.4 and w.run.manual_row["adjustments"] >= 1
    w.close()
    # the participant's next visit starts at session 2
    w2 = StudyWindow(Corrector("yin"), player=None, root=tmp_path / "study", show_errors=False)
    w2.code.setText("P05")
    assert w2.session_label.text() == "Session 2 of 5"
    w2.close()
    app.processEvents()
