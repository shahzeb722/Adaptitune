"""AdaptiTune desktop application (objective O5, FR1-FR7).

Runs fully offline: nothing in the app or the pipeline opens a network connection, and the
profile lives in a local SQLite file (``~/.adaptitune/profile.sqlite``, override with the
``ADAPTITUNE_HOME`` environment variable). Help has a speed test and a self-test
(``adaptitune.app.checks``); Study runs a session of the O6 user study (``adaptitune.app.study``).

The window follows one rule to stay out of trouble: while something runs in the background,
everything that would start other work or change the take is disabled (``_refresh``), and a
result that arrives after the take or the singer changed is dropped. The look is in
``adaptitune.app.theme``.

    adaptitune-app            # or: python -m adaptitune.app.launch
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, QSize, QStandardPaths, Qt, QThreadPool, QTimer, QUrl
from PySide6.QtGui import QActionGroup, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .. import metrics as M
from ..audio import Audio, load, save
from ..correction.settings import CorrectionSettings
from ..music import Key
from . import safety, theme  # noqa: F401 (safety: the log has a handler before anything logs)
from .audio_io import Player, Recorder  # Recorder: checks.py and the tests import it from here
from .jobs import Job, Jobs, UserMessage
from .pitchview import PitchView
from .widgets import Card, SettingsPanel, StarRating, button, hline, label

APP_NAME = "AdaptiTune"
AUDIO_SUFFIXES = (".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif")
AUDIO_FILTER = "Audio (" + " ".join(f"*{s}" for s in AUDIO_SUFFIXES) + ")"
MAX_TAKE_S = 600.0  # longer files are whole songs or mistakes, and would keep the app busy for minutes
MAX_NAME = 40
log = logging.getLogger("adaptitune.app")

__all__ = ["APP_NAME", "Job", "Jobs", "MainWindow", "Player", "Recorder", "UserMessage", "check_command", "main"]


def _home() -> Path:
    from ..agent.profile import default_path

    return default_path().parent


# ----------------------------------------------------------------------------- main window


class _TabPage(QWidget):
    """A tab page that asks for no room while another tab is on show (QTabWidget sizes itself
    to its tallest page otherwise, which leaves a gap under the short ones)."""

    active = True

    def sizeHint(self):  # noqa: N802 (Qt naming)
        return super().sizeHint() if self.active else QSize(0, 0)

    def minimumSizeHint(self):  # noqa: N802
        return super().minimumSizeHint() if self.active else QSize(0, 0)

    def heightForWidth(self, width):  # noqa: N802 (QStackedLayout asks every page for this)
        return super().heightForWidth(width) if self.active else 0


class MainWindow(QMainWindow):
    def __init__(self, session_factory=None, show_errors: bool = True):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(theme.app_icon())
        self.setAcceptDrops(True)
        self.pool = QThreadPool.globalInstance()
        self._jobs = Jobs(self.pool)  # holds each running job until its result is back
        self.errors = []  # messages and tracebacks shown to the user (the self-test reads them)
        self.show_errors = show_errors
        self._session_factory = session_factory or self._default_session
        self._store = self._corrector = None  # shared by every singer's session (default factory)
        self.session = None
        self.take = None
        self.take_name = ""
        self._pending_name = ""
        self.result = None  # the version on show in the pitch view and the table
        self.auto_result, self.auto_judged = None, False  # the agent's version, judged by Keep / Discard / stars
        self.ab, self.ab_judged = None, False
        self.manual_result, self.manual_dirty = None, False  # dirty: the sliders moved since the last Apply
        self.manual_started = None
        self.manual_adjustments = 0
        self._busy_n = 0
        self._gen = 0  # bumped when the take or the singer changes; older jobs' results are dropped
        self.study_window = None
        self.recorder = Recorder() if Recorder.available() else None
        try:
            self.player = Player()
        except Exception:  # pragma: no cover - no audio backend
            self.player = None
        self.settings = QSettings(str(_home() / "settings.ini"), QSettings.IniFormat)
        self._rec_started = 0.0
        self._rec_timer = QTimer(self)
        self._rec_timer.setInterval(250)
        self._rec_timer.timeout.connect(self._rec_tick)
        self._build()
        geometry = self.settings.value("geometry")
        if geometry is None or not self.restoreGeometry(geometry):
            room = self.screen().availableGeometry() if self.screen() else None
            self.resize(min(1100, room.width() - 40) if room else 1100, min(860, room.height() - 60) if room else 860)
        self._switch_user(str(self.settings.value("singer", "default") or "default"))

    # ------------------------------------------------------------------ setup
    def _default_session(self, user):
        from ..agent.profile import ProfileStore
        from ..agent.session import AdaptiveSession
        from ..pipeline import Corrector

        if self._store is None:
            self._store = ProfileStore()
        if self._corrector is None:
            self._corrector = Corrector()
        return AdaptiveSession(user=user, store=self._store, corrector=self._corrector)

    def _build(self):
        root = QWidget()
        root.setObjectName("page")
        outer = self._root_layout = QVBoxLayout(root)
        outer.setContentsMargins(20, 14, 20, 10)
        outer.setSpacing(12)

        # header: the app and the singer whose profile is in use (FR7)
        head = QHBoxLayout()
        head.setSpacing(10)
        icon = QLabel()
        icon.setPixmap(theme.app_icon().pixmap(40, 40))
        head.addWidget(icon)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        titles.addWidget(label(APP_NAME, "title", wrap=False))
        titles.addWidget(label("Pitch correction that learns how you like your voice", "caption", wrap=False))
        head.addLayout(titles)
        head.addStretch(1)
        head.addWidget(label("Singer", "caption", wrap=False))
        self.user_box = QComboBox()
        self.user_box.setMinimumWidth(170)
        self.user_box.setToolTip("Each singer has their own profile")
        self.user_box.activated.connect(lambda _: self._switch_user(self.user_box.currentText()))
        head.addWidget(self.user_box)
        self.btn_new_user = button("New singer…", "secondary", "Add a singer with a fresh profile")
        self.btn_new_user.clicked.connect(self._new_user)
        head.addWidget(self.btn_new_user)
        self.btn_export = button("Export…", "plain", "Save this singer's profile as a JSON file")
        self.btn_export.clicked.connect(self._export_profile)
        self.btn_reset = button("Reset", "destructive-plain", "Forget everything AdaptiTune learned about this singer")
        self.btn_reset.clicked.connect(self._reset_profile)
        head.addWidget(self.btn_export)
        head.addWidget(self.btn_reset)
        outer.addLayout(head)

        # the take (FR1, FR2)
        take = Card()
        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn_import = button("Import vocal…", "primary", "A dry solo vocal in WAV, FLAC, OGG, MP3 or AIFF, not a full mix (Ctrl+O)")
        self.btn_import.clicked.connect(self._import)
        self.btn_record = button("● Record", "record", "Record from the microphone; click again to stop (Ctrl+R)")
        self.btn_record.setCheckable(True)
        self.btn_record.toggled.connect(self._record)
        if self.recorder is None:
            self.btn_record.setToolTip("Recording needs a microphone and the sounddevice package. Importing a file still works.")
        row.addWidget(self.btn_import)
        row.addWidget(self.btn_record)
        row.addSpacing(14)
        row.addWidget(label("Key", "caption", wrap=False))
        self.key_box = QComboBox()
        self.key_box.addItem("Auto-detect")
        for k in Key.all():
            self.key_box.addItem(k.name)
        self.key_box.setMaxVisibleItems(16)
        self.key_box.setToolTip("Leave on Auto-detect, or choose the song's key yourself")
        self.key_box.activated.connect(self._key_changed)
        row.addWidget(self.key_box)
        self.key_label = label("", "badge", wrap=False)
        self.key_label.setVisible(False)
        row.addWidget(self.key_label)
        row.addStretch(1)
        self.btn_play_orig = button("▶  Original", "secondary", "Play the take as sung")
        self.btn_play_orig.clicked.connect(lambda: self._play(self.take.analysis.audio if self.take else None))
        self.btn_stop = button("■  Stop", "secondary", "Stop playback")
        self.btn_stop.clicked.connect(self._stop)
        row.addWidget(self.btn_play_orig)
        row.addWidget(self.btn_stop)
        take.box.addLayout(row)
        self.take_info = label("No take yet. Import a vocal, record one, or drop an audio file on this window.", "caption")
        take.box.addWidget(self.take_info)
        outer.addWidget(take)

        # the three ways to correct it
        self.tabs = QTabWidget()
        self.tabs.addTab(self._auto_tab(), "Personalized")
        self.tabs.addTab(self._ab_tab(), "Compare A / B")
        self.tabs.addTab(self._manual_tab(), "Manual")
        self.tabs.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.tabs.currentChanged.connect(self._fit_tabs)
        self._fit_tabs(0)
        outer.addWidget(self.tabs)

        # FR6: pitch view + per-note table
        pitch = Card()
        hdr = QHBoxLayout()
        hdr.addWidget(label("Pitch, note by note", "section", wrap=False))
        self.showing = label("", "caption", wrap=False)
        hdr.addWidget(self.showing)
        hdr.addStretch(1)
        self.btn_save = button("Save WAV…", "secondary", "Save the version on show as a WAV file (Ctrl+S)")
        self.btn_save.clicked.connect(self._save_wav)
        hdr.addWidget(self.btn_save)
        pitch.box.addLayout(hdr)
        self.view = PitchView()
        self.view.setMinimumHeight(140)
        pitch.box.addWidget(self.view, 3)
        pitch.box.addWidget(hline())
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Time (s)", "Target", "Before (cents)", "After (cents)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setMinimumHeight(96)
        self.table.setMaximumHeight(220)
        pitch.box.addWidget(self.table, 2)
        outer.addWidget(pitch, 1)

        # on a small screen (a laptop at 150% scaling has about 1280 x 680 to spare) the page scrolls
        scroll = QScrollArea()
        scroll.setObjectName("pageScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(root)
        self.setCentralWidget(scroll)
        self.status = self.statusBar()
        self.busy_bar = QProgressBar()
        self.busy_bar.setRange(0, 0)  # moving stripe: working, length unknown
        self.busy_bar.setFixedWidth(120)
        self.busy_bar.setVisible(False)
        self.status.addPermanentWidget(self.busy_bar)
        self._build_menu()
        self._refresh()

    def _build_menu(self):
        mb = self.menuBar()
        f = mb.addMenu("&File")
        self.act_import = f.addAction("Import vocal…")
        self.act_import.setShortcut(QKeySequence.Open)
        self.act_import.triggered.connect(self._import)
        self.act_record = f.addAction("Record / stop recording")
        self.act_record.setShortcut(QKeySequence("Ctrl+R"))
        self.act_record.triggered.connect(lambda: self.btn_record.isEnabled() and self.btn_record.click())
        self.act_save = f.addAction("Save WAV…")
        self.act_save.setShortcut(QKeySequence.Save)
        self.act_save.triggered.connect(self._save_wav)
        f.addSeparator()
        self.act_export = f.addAction("Export profile…")
        self.act_export.triggered.connect(self._export_profile)
        f.addSeparator()
        quit_ = f.addAction("Quit")
        quit_.setShortcut(QKeySequence.Quit)
        quit_.triggered.connect(self.close)

        v = mb.addMenu("&View")
        ap = v.addMenu("Appearance")
        group = QActionGroup(self)
        current = str(self.settings.value("appearance", "auto"))
        for key, text in (("auto", "Match the system"), ("light", "Light"), ("dark", "Dark")):
            a = ap.addAction(text)
            a.setCheckable(True)
            a.setChecked(key == current)
            group.addAction(a)
            a.triggered.connect(lambda _=False, k=key: self._set_appearance(k))

        st = mb.addMenu("&Study")
        self.act_study = st.addAction("Run a study session…")
        self.act_study.triggered.connect(self._study)
        m = mb.addMenu("&Help")
        self.act_speed = m.addAction("Speed test…")
        self.act_speed.triggered.connect(self._speed_test)
        self.act_check = m.addAction("Check this installation")
        self.act_check.triggered.connect(self._self_test)
        m.addSeparator()
        m.addAction("Where my data is stored").triggered.connect(self._data_folder)
        m.addAction(f"About {APP_NAME}").triggered.connect(self._about)

    def _auto_tab(self):
        page, card = self._tab_page()
        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn_correct = button("Correct with my profile", "primary", "AdaptiTune picks the settings it thinks you will like")
        self.btn_correct.clicked.connect(self._auto_correct)
        self.btn_play_corr = button("▶  Corrected", "secondary", "Play the corrected version again")
        self.btn_play_corr.clicked.connect(lambda: self._play(self.auto_result.audio if self.auto_result else None))
        row.addWidget(self.btn_correct)
        row.addWidget(self.btn_play_corr)
        row.addStretch(1)
        card.box.addLayout(row)
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(label("Would you keep it?", "body", wrap=False))
        self.btn_keep = button("👍  Keep", "secondary", "Keep this version: AdaptiTune learns that you like it")
        self.btn_keep.clicked.connect(lambda: self._feedback("keep", 1))
        self.btn_discard = button("👎  Discard", "secondary", "Discard it: AdaptiTune learns that you do not")
        self.btn_discard.clicked.connect(lambda: self._feedback("keep", 0))
        row.addWidget(self.btn_keep)
        row.addWidget(self.btn_discard)
        row.addSpacing(18)
        row.addWidget(label("or rate it", "body", wrap=False))
        self.rating = StarRating()
        self.rating.setToolTip("Rate this version from 1 to 5 stars")
        self.rating.rated.connect(lambda i: self._feedback("rating", i))
        row.addWidget(self.rating)
        row.addStretch(1)
        card.box.addLayout(row)
        self.auto_note = label("Each judgment teaches AdaptiTune a little more about your taste.", "caption")
        card.box.addWidget(self.auto_note)
        return page

    def _ab_tab(self):
        page, card = self._tab_page()
        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn_make_ab = button("Make two versions", "primary", "Two versions with different settings, to compare by ear")
        self.btn_make_ab.clicked.connect(self._make_ab)
        self.btn_play_a = button("▶  A", "secondary")
        self.btn_play_a.clicked.connect(lambda: self._play(self.ab[0].audio if self.ab else None))
        self.btn_play_b = button("▶  B", "secondary")
        self.btn_play_b.clicked.connect(lambda: self._play(self.ab[1].audio if self.ab else None))
        self.btn_pick_a = button("I prefer A", "secondary")
        self.btn_pick_a.clicked.connect(lambda: self._pick("a"))
        self.btn_pick_b = button("I prefer B", "secondary")
        self.btn_pick_b.clicked.connect(lambda: self._pick("b"))
        for x in (self.btn_make_ab, self.btn_play_a, self.btn_play_b):
            row.addWidget(x)
        row.addSpacing(18)
        row.addWidget(self.btn_pick_a)
        row.addWidget(self.btn_pick_b)
        row.addStretch(1)
        card.box.addLayout(row)
        card.box.addWidget(label("Listen to both, then pick the one you like better. AdaptiTune learns from your choice.", "caption"))
        return page

    def _manual_tab(self):
        page, card = self._tab_page()
        row = QHBoxLayout()
        row.setSpacing(18)
        self.manual_panel = SettingsPanel()
        self.manual_panel.changed.connect(self._manual_changed)
        # the names the self-test and older code use
        self.s_strength, self.s_retune = self.manual_panel.strength, self.manual_panel.retune
        self.c_inkey, self.s_vibrato = self.manual_panel.in_key, self.manual_panel.vibrato
        row.addWidget(self.manual_panel, 3)
        col = QVBoxLayout()
        col.setSpacing(8)
        self.btn_apply = button("Apply", "primary", "Correct the take with these settings and play it")
        self.btn_apply.clicked.connect(self._manual_apply)
        self.btn_play_man = button("▶  Play", "secondary", "Play your version again")
        self.btn_play_man.clicked.connect(lambda: self._play(self.manual_result.audio if self.manual_result else None))
        self.btn_save_man = button("Save as my setting", "secondary",
                                   "Keep these settings as yours. Apply them first, so you have heard them.")
        self.btn_save_man.clicked.connect(self._manual_save)
        for x in (self.btn_apply, self.btn_play_man, self.btn_save_man):
            col.addWidget(x)
        col.addStretch(1)
        row.addLayout(col, 1)
        card.box.addLayout(row)
        return page

    def _fit_tabs(self, current: int):
        """Let the tabs take the height of the page on show, not of the tallest page."""
        for k in range(self.tabs.count()):
            page = self.tabs.widget(k)
            page.active = k == current
            page.updateGeometry()
        self.tabs.updateGeometry()
        if getattr(self, "_root_layout", None) is not None:
            self._root_layout.invalidate()  # the box layout keeps the old heights otherwise

    @staticmethod
    def _tab_page():
        page = _TabPage()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 8, 0, 0)
        card = Card()
        v.addWidget(card)
        return page, card

    # ------------------------------------------------------------------ state
    def _refresh(self):
        """Enable exactly what makes sense now. Called after every change of state."""
        busy = self._busy_n > 0
        rec = self.btn_record.isChecked()
        free = not busy and not rec
        has_take = self.take is not None
        for w in (self.user_box, self.btn_new_user, self.btn_export, self.btn_reset, self.btn_import, self.key_box,
                  self.act_import, self.act_export, self.act_study, self.act_speed, self.act_check):
            w.setEnabled(free)
        self.btn_record.setEnabled(self.recorder is not None and (rec or not busy))
        self.act_record.setEnabled(self.btn_record.isEnabled())
        self.btn_play_orig.setEnabled(has_take and not rec)
        for w in (self.btn_correct, self.btn_make_ab, self.btn_apply):
            w.setEnabled(free and has_take)
        self.btn_play_corr.setEnabled(self.auto_result is not None and not rec)
        for w in (self.btn_keep, self.btn_discard, self.rating):
            w.setEnabled(self.auto_result is not None and not self.auto_judged and not busy)
        for w in (self.btn_play_a, self.btn_play_b):
            w.setEnabled(self.ab is not None and not rec)
        for w in (self.btn_pick_a, self.btn_pick_b):
            w.setEnabled(self.ab is not None and not self.ab_judged and not busy)
        self.btn_play_man.setEnabled(self.manual_result is not None and not rec)
        self.btn_save_man.setEnabled(self.manual_result is not None and not self.manual_dirty and not busy)
        self.btn_save.setEnabled(self.result is not None and not busy)
        self.act_save.setEnabled(self.btn_save.isEnabled())

    def _set_enabled(self, on: bool):  # kept for older callers
        self._refresh()

    def _forget_versions(self):
        self.result = self.auto_result = self.manual_result = self.ab = None
        self.auto_judged = self.ab_judged = self.manual_dirty = False
        self.rating.clear()
        self.table.setRowCount(0)
        self.showing.setText("")

    def _busy(self, msg):
        self._busy_n += 1
        if self._busy_n == 1:
            QApplication.setOverrideCursor(Qt.BusyCursor)
        self.busy_bar.setVisible(True)
        self.status.showMessage(msg)
        self._refresh()

    def _idle(self, msg=""):
        self._busy_n = max(0, self._busy_n - 1)
        if self._busy_n == 0:
            QApplication.restoreOverrideCursor()
            self.busy_bar.setVisible(False)
        self._refresh()
        self._update_status(msg)

    def _update_status(self, extra=""):
        n = self.session.interactions if self.session else 0
        self.status.showMessage(f"Singer: {self.session.user if self.session else '-'} · learned from {n} judgments"
                                + (f" · {extra}" if extra else ""))

    def _run(self, fn, done, msg, *a, **kw):
        """Run ``fn`` on a worker thread, then ``done(result)`` back on the UI thread, unless
        the take or the singer changed in the meantime."""
        gen = self._gen
        self._busy(msg)

        def ok(result):
            self._idle()
            if gen != self._gen:
                log.info("dropped a result made for an earlier take or singer")
                return
            done(result)

        def failed(text):
            self._idle()
            self._error(text)

        self._jobs.start(fn, ok, failed, *a, **kw)

    def _error(self, text):
        self.errors.append(text)
        if text.startswith("Traceback"):
            log.error("job failed\n%s", text)
        else:
            log.info("told the user: %s", text)
        if not self.show_errors:
            return
        box = QMessageBox(QMessageBox.Warning, APP_NAME, text, QMessageBox.Ok, self)
        if text.startswith("Traceback"):  # a bug, not something the user did: keep the details for a report
            from .safety import log_path

            box.setText("Something went wrong, but AdaptiTune is still running.")
            box.setInformativeText(f"The details are saved in {log_path()}. Please send that file to the AdaptiTune team.")
            box.setDetailedText(text)
        box.exec()

    def _play(self, audio):
        if audio is None or not self.player:
            return
        try:
            self.player.play(audio)
        except Exception as e:  # a full disk or a lost audio device must not take the window down
            self._error(f"AdaptiTune could not play this: {e}")

    def _stop(self):
        if self.player:
            self.player.stop()

    def _manual_settings(self) -> CorrectionSettings:
        return self.manual_panel.settings()

    def _show_result(self, res, label_text=""):
        self.result = res
        an = self.take.analysis
        orig = M.contour_midi(an.track.f0)
        corr = res.plan.corrected_midi  # what the corrector applied; the table shows the re-tracked result
        rows = res.note_report()
        self.view.set_data(an.track.hop_s, orig, corr, [(r["start"], r["end"], r["target"]) for r in rows])
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            vals = [f"{r['start']:.2f}–{r['end']:.2f}", f"{Key(r['target'] % 12, 'major').name.split()[0]}{r['target'] // 12 - 1}",
                    f"{r['before_cents']:+.0f}", "–" if r["after_cents"] is None else f"{r['after_cents']:+.0f}"]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, j, item)
        s = res.settings
        self.showing.setText(f"· showing {label_text.rstrip(':')}" if label_text else "")
        self._refresh()
        self._update_status(f"{label_text} strength {s.strength:.2f}, retune {s.retune_ms:.0f} ms, {s.snap.replace('_', '-')}, "
                            f"vibrato {s.vibrato:.2f} · {res.rtf:.2f}× real time")

    # ------------------------------------------------------------------ users / profile (FR7)
    @staticmethod
    def _clean_name(name: str) -> str:
        return " ".join(str(name).split())

    def _switch_user(self, user):
        user = self._clean_name(user) or "default"
        self.session = self._session_factory(user)
        self._gen += 1
        # a version made for the previous singer must not collect this singer's feedback
        self._forget_versions()
        self.user_box.blockSignals(True)
        self.user_box.clear()
        self.user_box.addItems(sorted(set(self.session.store.users()) | {user}, key=str.lower))
        self.user_box.setCurrentText(user)
        self.user_box.blockSignals(False)
        last = self.session.last_manual()
        if last:
            self.manual_panel.set_settings(last)
        self.manual_started, self.manual_adjustments = None, 0
        if self.take:  # the take stays; only the versions made from it go
            an = self.take.analysis
            self.view.set_data(an.track.hop_s, M.contour_midi(an.track.f0))
        self.settings.setValue("singer", user)
        self._refresh()
        self._update_status()

    def _new_user(self):
        name, ok = QInputDialog.getText(self, "New singer", "Name of the singer (stays on this computer):")
        if not ok:
            return
        name = self._clean_name(name)
        if not name:
            return
        if len(name) > MAX_NAME or not name.isprintable():
            self._error(f"Please use a shorter name, of at most {MAX_NAME} letters, numbers and spaces.")
            return
        self._switch_user(name)

    def _reset_profile(self):
        if self.show_errors and QMessageBox.question(
                self, APP_NAME, f"Forget everything AdaptiTune learned about {self.session.user}? This cannot be undone.") != QMessageBox.Yes:
            return
        self.session.reset()
        self._update_status("profile reset")

    def _export_profile(self):
        start = Path(str(self.settings.value("last_dir", ""))) / f"{self.session.user}_profile.json"
        path, _ = QFileDialog.getSaveFileName(self, "Export profile", str(start), "JSON (*.json)")
        if not path:
            return
        try:
            self.session.export(path)
        except OSError as e:
            self._error(f"AdaptiTune could not save the profile to {path}: {e.strerror or e}")
            return
        self._remember_dir(path)
        self._update_status(f"exported to {path}")

    def _remember_dir(self, path):
        self.settings.setValue("last_dir", str(Path(path).parent))

    # ------------------------------------------------------------------ input (FR1, FR2)
    def _import(self):
        if self._busy_n:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Import vocal", str(self.settings.value("last_dir", "")),
                                              f"{AUDIO_FILTER};;All files (*)")
        if path:
            self._remember_dir(path)
            self._load(path)

    def _record(self, on):
        if on:
            try:
                self.recorder.start()
            except Exception as e:  # no microphone, or Windows does not let desktop apps use it
                self._record_off()
                self._error(f"Recording could not start: {e}\n\nCheck that a microphone is connected and that "
                            "Windows lets desktop apps use it (Settings > Privacy & security > Microphone). "
                            "Importing a file still works.")
                return
            self._rec_started = time.time()
            self._rec_tick()
            self._rec_timer.start()
            self._refresh()
        else:
            self._rec_timer.stop()
            self.btn_record.setText("● Record")
            try:
                audio = self.recorder.stop()
            except Exception as e:  # the microphone went away mid-take
                self._refresh()
                self._error(f"The recording stopped with an error: {e}")
                return
            self._refresh()
            if audio.duration < 0.5:
                self._error("The recording is shorter than half a second, so there is nothing to correct.")
                return
            self._load(audio)

    def _record_off(self):
        self._rec_timer.stop()
        self.btn_record.blockSignals(True)
        self.btn_record.setChecked(False)
        self.btn_record.blockSignals(False)
        self.btn_record.setText("● Record")
        self._refresh()

    def _rec_tick(self):
        s = int(time.time() - self._rec_started)
        self.btn_record.setText(f"■  Stop  {s // 60}:{s % 60:02d}")

    def _load(self, src):
        if self._busy_n:  # one thing at a time
            return
        key = None if self.key_box.currentIndex() == 0 else self.key_box.currentText()
        self._pending_name = "Recording" if isinstance(src, Audio) else Path(src).name
        self._run(self._read_and_analyse, self._loaded, "Analysing…", src, key)

    def _read_and_analyse(self, src, key):
        """Runs on the worker thread: read the file, check it is usable, then analyse it."""
        if not isinstance(src, Audio):
            try:
                src = load(src)
            except Exception as e:
                raise UserMessage(f"AdaptiTune could not read {Path(src).name}. It reads WAV, FLAC, OGG, MP3 and AIFF "
                                  "files; convert anything else to WAV first.") from e
        if src.duration < 0.5:
            raise UserMessage("This take is shorter than half a second, so there is nothing to correct.")
        if src.duration > MAX_TAKE_S:
            raise UserMessage(f"This file is {src.duration / 60:.0f} minutes long. AdaptiTune corrects takes of up to "
                              f"{MAX_TAKE_S / 60:.0f} minutes; cut out the part you want first.")
        return self.session.load_take(src, key)

    def _loaded(self, take):
        self.take = take
        self._gen += 1
        self.take_name, self._pending_name = self._pending_name or "Take", ""
        self._forget_versions()
        an = take.analysis
        self._show_key(an)
        contour = M.contour_midi(an.track.f0)
        self.view.set_data(an.track.hop_s, contour)
        voiced = float(np.mean(np.isfinite(contour))) if len(contour) else 0.0
        info = f"{self.take_name} · {an.audio.duration:.1f} s"
        if voiced < 0.05:
            info += " · hardly any singing was found in it, so there is little to correct"
        self.take_info.setText(info)
        self._refresh()
        self._update_status(f"loaded {an.audio.duration:.1f} s")

    def _show_key(self, an):
        auto = self.key_box.currentIndex() == 0
        self.key_label.setText(f"Detected {an.key.name} · {an.key_confidence:.0%}" if auto else "")
        self.key_label.setVisible(auto)

    def _key_changed(self, i):
        if self.take is None:
            self.key_label.setVisible(False)
            return
        if i > 0:
            self.session.set_key(self.take, self.key_box.currentText())
        else:
            from ..pitch.key import estimate_key

            k, c = estimate_key(self.take.analysis.track.f0, self.take.analysis.track.voicing)
            self.take.analysis.key, self.take.analysis.key_confidence = k, c
        self._show_key(self.take.analysis)

    # ------------------------------------------------------------------ personalized (FR3, FR4)
    def _auto_correct(self):
        if self.take is None or self._busy_n:
            return
        self._run(self.session.propose, self._auto_done, "Correcting…", self.take)

    def _auto_done(self, res):
        self.auto_result, self.auto_judged = res, False
        self.rating.clear()
        self.auto_note.setText("Keep it, discard it, or rate it: each judgment teaches AdaptiTune a little more.")
        self._show_result(res, "the agent's version:")
        self._play(res.audio)

    def _feedback(self, kind, value):
        if self.auto_result is None or self.auto_judged or self.take is None:
            return
        r = self.session.feedback(self.take, self.auto_result, kind, value)
        self.auto_judged = True
        if kind == "rating":
            self.rating.set_value(int(value))
        self.auto_note.setText("Thanks. AdaptiTune has learned from that; correct again to hear what it chooses now.")
        self._refresh()
        self._update_status(f"thanks (reward {r:.2f})")

    def _save_wav(self):
        if self.result is None:
            return
        stem = Path(self.take_name).stem if self.take_name else "take"
        start = Path(str(self.settings.value("last_dir", ""))) / f"{stem}_corrected.wav"
        path, _ = QFileDialog.getSaveFileName(self, "Save corrected vocal", str(start), "WAV (*.wav)")
        if not path:
            return
        if not path.lower().endswith(".wav"):
            path += ".wav"
        try:
            save(path, self.result.audio)
        except Exception as e:
            self._error(f"AdaptiTune could not save {Path(path).name}: {e}")
            return
        self._remember_dir(path)
        self._update_status(f"saved {Path(path).name}")

    # ------------------------------------------------------------------ A/B (FR3, FR4)
    def _make_ab(self):
        if self.take is None or self._busy_n:
            return
        self._run(self.session.propose_ab, self._ab_done, "Making two versions…", self.take)

    def _ab_done(self, pair):
        self.ab, self.ab_judged = pair, False
        self._show_result(pair[0], "A:")

    def _pick(self, which):
        if not self.ab or self.ab_judged or self.take is None:
            return
        self.session.feedback_ab(self.take, self.ab[0], self.ab[1], which)
        self.ab_judged = True
        self._show_result(self.ab[0] if which == "a" else self.ab[1], f"you picked {which.upper()}:")

    # ------------------------------------------------------------------ manual (FR5)
    def _load_manual(self, s: CorrectionSettings):
        self.manual_panel.set_settings(s)

    def _manual_changed(self, *_):
        if self.manual_started is None and self.take is not None:
            self.manual_started = time.time()
        self.manual_adjustments += 1
        self.manual_dirty = True
        self._refresh()

    def _manual_apply(self):
        if self.take is None or self._busy_n:
            return
        if self.manual_started is None:
            self.manual_started = time.time()
        self._run(self.session.manual, self._manual_done, "Applying…", self.take, self._manual_settings())

    def _manual_done(self, res):
        self.manual_result, self.manual_dirty = res, False
        self._show_result(res, "your setting:")
        self._play(res.audio)

    def _manual_save(self):
        if self.take is None:
            return
        secs = time.time() - self.manual_started if self.manual_started else None
        self.session.save_manual(self.take, self._manual_settings(), seconds=secs, adjustments=self.manual_adjustments)
        self.manual_started, self.manual_adjustments = None, 0
        self._update_status("saved as your manual setting")

    # ------------------------------------------------------------------ view
    def _set_appearance(self, key: str):
        self.settings.setValue("appearance", key)
        theme.apply(QApplication.instance(), key)
        self.view.update()

    # ------------------------------------------------------------------ drag and drop
    @staticmethod
    def _dropped_file(event) -> str | None:
        md = event.mimeData()
        if not md.hasUrls():
            return None
        for url in md.urls():
            if url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in AUDIO_SUFFIXES:
                return url.toLocalFile()
        return None

    def dragEnterEvent(self, e):  # noqa: N802 (Qt naming)
        if self._dropped_file(e) and not self._busy_n and not self.btn_record.isChecked():
            e.acceptProposedAction()
        else:
            e.ignore()

    def dropEvent(self, e):  # noqa: N802
        path = self._dropped_file(e)
        if path and not self._busy_n and not self.btn_record.isChecked():
            e.acceptProposedAction()
            self._remember_dir(path)
            self._load(path)

    # ------------------------------------------------------------------ study (O6)
    def _study(self):
        from .study import StudyWindow

        if self.study_window is not None and self.study_window.isVisible():
            self.study_window.raise_()
            return
        self._stop()
        self.study_window = StudyWindow(self.session.corrector, self.player, self, show_errors=self.show_errors)
        self.study_window.setWindowModality(Qt.ApplicationModal)  # nothing else happens during a session
        self.study_window.show()

    # ------------------------------------------------------------------ help: checks (O5)
    def _speed_test(self):
        box = QMessageBox(self)
        box.setWindowTitle("Speed test")
        box.setText("Times the whole correction path on this computer: pitch tracking, key, note planning and "
                    "WORLD/PSOLA. The target is 0.5× real time or less on the reference laptop.")
        box.setInformativeText("For the report, choose the MIR-1K folder: the test then times the same minute of singing "
                               "as the cloud run. It takes one to three minutes; close other programs first.")
        b_mir = box.addButton("MIR-1K folder…", QMessageBox.AcceptRole)
        b_file = box.addButton("Audio file…", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Cancel)
        box.exec()
        if box.clickedButton() is b_mir:
            src = QFileDialog.getExistingDirectory(self, "Folder that contains MIR-1K")
        elif box.clickedButton() is b_file:
            src, _ = QFileDialog.getOpenFileName(self, "Audio file to time", "", AUDIO_FILTER)
        else:
            return
        if src:
            self._run_check(["--speed-test", src], "speed", "Speed test running (1–3 minutes)…")

    def _self_test(self):
        self._run_check(["--self-test"], "self", "Checking every feature with the network blocked…")

    def _run_check(self, args, kind, msg):
        """Run a check in a separate process of this app, so its thread settings and throwaway
        profile cannot touch this window's session."""
        out = Path(tempfile.mkdtemp(prefix="adaptitune_check_")) / f"{kind}_test.json"
        cmd = check_command(args + ["--out", str(out)])

        def work():
            env = dict(os.environ)
            env.pop("QT_QPA_PLATFORM", None)  # the check sets its own (offscreen)
            p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if not out.exists():
                raise RuntimeError(f"the check did not finish (exit code {p.returncode})\n{p.stderr[-2000:]}")
            return json.loads(out.read_text())

        self._run(work, lambda r: self._check_done(r, kind), msg)

    def _check_done(self, report, kind):
        from .checks import summary

        keep = Path(self.session.store.path).parent / "checks" / f"{kind}_test_{time.strftime('%Y%m%d_%H%M%S')}.json"
        keep.parent.mkdir(parents=True, exist_ok=True)
        keep.write_text(json.dumps(report, indent=2, default=str))
        box = QMessageBox(self)
        box.setWindowTitle("Speed test" if kind == "speed" else "Installation check")
        box.setText(report.get("error") or summary(report, kind))
        save_btn = box.addButton("Save report…", QMessageBox.AcceptRole)
        box.addButton(QMessageBox.Close)
        box.exec()
        if box.clickedButton() is save_btn:
            docs = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
            path, _ = QFileDialog.getSaveFileName(self, "Save report", str(Path(docs) / keep.name), "JSON (*.json)")
            if path:
                try:
                    shutil.copyfile(keep, path)
                except OSError as e:
                    self._error(f"AdaptiTune could not save the report to {path}: {e.strerror or e}")
        self._update_status(f"{'speed test' if kind == 'speed' else 'check'} report kept in {keep.parent}")

    def _data_folder(self):
        folder = Path(self.session.store.path).parent
        box = QMessageBox(self)
        box.setWindowTitle("Your data")
        box.setText(f"Your profile and everything AdaptiTune learned about each singer are in\n{folder}")
        box.setInformativeText("Nothing is sent anywhere: the app works with no internet connection. Recordings are "
                               "saved only where you save them. The logs folder there helps the team if something goes wrong.")
        opened = box.addButton("Open folder", QMessageBox.AcceptRole)
        box.addButton(QMessageBox.Close)
        box.exec()
        if box.clickedButton() is opened:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _about(self):
        from ..machine import build_info

        b = build_info()
        commit = (b.get("git_commit") or "unknown")[:7]
        tracker = self.session.corrector.tracker.name if self.session else "?"
        QMessageBox.about(self, f"About {APP_NAME}",
                          f"{APP_NAME} {__import__('adaptitune').__version__} (build {commit})\n"
                          f"Pitch tracker: {tracker}\n\nOffline vocal pitch correction that learns each singer's "
                          "preferred settings. COMSATS University Islamabad, Lahore Campus, final year project.")

    def closeEvent(self, e):  # noqa: N802
        if self.study_window is not None and self.study_window.isVisible() and not self.study_window.close():
            e.ignore()  # the study window asked, and the researcher kept the session going
            return
        if len(self._jobs) and self.show_errors:
            ask = "AdaptiTune is still working. Quit anyway? What it is doing now will be lost."
            if QMessageBox.question(self, APP_NAME, ask) != QMessageBox.Yes:
                e.ignore()
                return
        if self.btn_record.isChecked() and self.recorder is not None:
            self._rec_timer.stop()
            try:
                self.recorder.stop()
            except Exception:  # pragma: no cover - the device is gone already
                pass
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.sync()
        if self.player:
            self.player.close()
        super().closeEvent(e)


def check_command(args: list[str]) -> list[str]:
    """How to start this app with the given arguments: the packaged executable, or Python."""
    if getattr(sys, "frozen", False):
        return [sys.executable, *args]
    return [sys.executable, "-m", "adaptitune.app.launch", *args]


def main(argv=None):
    argv = argv or sys.argv
    if sys.platform == "win32":  # the taskbar shows AdaptiTune's icon, not Python's
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("COMSATS.AdaptiTune")
        except Exception:  # pragma: no cover
            pass
    app = QApplication(argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_NAME)
    app.setWindowIcon(theme.app_icon())
    from .safety import install, log_path

    install()
    settings = QSettings(str(_home() / "settings.ini"), QSettings.IniFormat)
    appearance = str(settings.value("appearance", "auto"))
    theme.apply(app, appearance if appearance in theme.APPEARANCES else "auto")
    w = MainWindow()
    w.show()
    from ..machine import build_info

    log.info("started %s %s, build %s, tracker %s, log %s", APP_NAME, __import__("adaptitune").__version__,
             (build_info().get("git_commit") or "unknown")[:7], w.session.corrector.tracker.name, log_path())
    if "--smoke-test" in argv:  # used by the packaging check
        QTimer.singleShot(500, app.quit)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
