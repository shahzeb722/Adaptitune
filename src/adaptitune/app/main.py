"""AdaptiTune desktop application (objective O5, FR1-FR7).

Runs fully offline: nothing in the app or the pipeline opens a network connection, and the
profile lives in a local SQLite file (``~/.adaptitune/profile.sqlite``, override with the
``ADAPTITUNE_HOME`` environment variable). Help has a speed test and a self-test
(``adaptitune.app.checks``).

    adaptitune-app            # or: python -m adaptitune.app.launch
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import (
    QObject,
    QRunnable,
    QStandardPaths,
    Qt,
    QThreadPool,
    QTimer,
    QUrl,
    Signal,
    Slot,
)
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .. import metrics as M
from ..audio import Audio, preprocess, save
from ..correction.settings import RETUNE_MS_LEVELS, CorrectionSettings
from ..music import Key
from .pitchview import PitchView

APP_NAME = "AdaptiTune"


# ----------------------------------------------------------------------------- workers


class _Signals(QObject):
    done = Signal(object)
    failed = Signal(str)


class Job(QRunnable):
    """Run slow pipeline work off the UI thread."""

    def __init__(self, fn, *a, **kw):
        super().__init__()
        self.fn, self.a, self.kw = fn, a, kw
        self.signals = _Signals()

    @Slot()
    def run(self):
        try:
            self.signals.done.emit(self.fn(*self.a, **self.kw))
        except Exception:  # pragma: no cover - surfaced in a dialog
            self.signals.failed.emit(traceback.format_exc())


class Player:
    """Plays an Audio buffer. Uses QtMultimedia (no extra dependency) with a temp WAV."""

    def __init__(self):
        from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

        self.out = QAudioOutput()
        self.player = QMediaPlayer()
        self.player.setAudioOutput(self.out)
        self.tmp = Path(tempfile.mkdtemp(prefix="adaptitune_"))
        self.n = 0

    def play(self, audio: Audio):
        self.stop()
        self.n += 1
        f = self.tmp / f"play_{self.n}.wav"
        save(f, audio)
        self.player.setSource(QUrl.fromLocalFile(str(f)))
        self.player.play()

    def stop(self):
        self.player.stop()

    def close(self):
        self.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)


class Recorder:
    """Microphone capture through sounddevice (PortAudio). Optional: import still works
    without it."""

    def __init__(self, sr: int = 44_100):
        self.sr = sr
        self.chunks = []
        self.stream = None

    @staticmethod
    def available() -> bool:
        try:
            import sounddevice  # noqa: F401

            return True
        except Exception:
            return False

    def start(self):
        import sounddevice as sd

        self.chunks = []
        self.stream = sd.InputStream(samplerate=self.sr, channels=1, callback=lambda d, *_: self.chunks.append(d.copy()))
        self.stream.start()

    def stop(self) -> Audio:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        x = np.concatenate(self.chunks)[:, 0] if self.chunks else np.zeros(1)
        return preprocess(x, self.sr)


# ----------------------------------------------------------------------------- main window


class MainWindow(QMainWindow):
    def __init__(self, session_factory=None, show_errors: bool = True):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1000, 720)
        self.pool = QThreadPool.globalInstance()
        self._jobs = set()  # running jobs; PySide drops a job's result if nothing holds the job
        self.errors = []  # tracebacks of failed jobs (the self-test reads them)
        self.show_errors = show_errors
        self._session_factory = session_factory or self._default_session
        self.session = None
        self.take = None
        self.result = None  # current corrected version
        self.ab = None
        self.manual_started = None
        self.manual_adjustments = 0
        self.recorder = Recorder() if Recorder.available() else None
        try:
            self.player = Player()
        except Exception:  # pragma: no cover - no audio backend
            self.player = None
        self._build()
        self._switch_user("default")

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _default_session(user):
        from ..agent.session import AdaptiveSession

        return AdaptiveSession(user=user)

    def _build(self):
        root = QWidget()
        lay = QVBoxLayout(root)

        # top: user + profile actions
        top = QHBoxLayout()
        top.addWidget(QLabel("Singer:"))
        self.user_box = QComboBox()
        self.user_box.setEditable(False)
        self.user_box.activated.connect(lambda _: self._switch_user(self.user_box.currentText()))
        top.addWidget(self.user_box)
        b = QPushButton("New singer…")
        b.clicked.connect(self._new_user)
        top.addWidget(b)
        top.addStretch(1)
        self.btn_reset = QPushButton("Reset profile")
        self.btn_reset.clicked.connect(self._reset_profile)
        self.btn_export = QPushButton("Export profile…")
        self.btn_export.clicked.connect(self._export_profile)
        top.addWidget(self.btn_reset)
        top.addWidget(self.btn_export)
        lay.addLayout(top)

        # input row: FR1 + FR2
        inp = QHBoxLayout()
        self.btn_import = QPushButton("Import vocal…")
        self.btn_import.clicked.connect(self._import)
        self.btn_record = QPushButton("● Record")
        self.btn_record.setCheckable(True)
        self.btn_record.toggled.connect(self._record)
        self.btn_record.setEnabled(self.recorder is not None)
        if self.recorder is None:
            self.btn_record.setToolTip("Recording needs the sounddevice package and a microphone")
        inp.addWidget(self.btn_import)
        inp.addWidget(self.btn_record)
        inp.addSpacing(20)
        inp.addWidget(QLabel("Key:"))
        self.key_box = QComboBox()
        self.key_box.addItem("Auto-detect")
        for k in Key.all():
            self.key_box.addItem(k.name)
        self.key_box.activated.connect(self._key_changed)
        inp.addWidget(self.key_box)
        self.key_label = QLabel("")
        inp.addWidget(self.key_label)
        inp.addStretch(1)
        self.btn_play_orig = QPushButton("▶ Original")
        self.btn_play_orig.clicked.connect(lambda: self._play(self.take.analysis.audio if self.take else None))
        self.btn_stop = QPushButton("■ Stop")
        self.btn_stop.clicked.connect(lambda: self.player and self.player.stop())
        inp.addWidget(self.btn_play_orig)
        inp.addWidget(self.btn_stop)
        lay.addLayout(inp)

        # modes
        self.tabs = QTabWidget()
        self.tabs.addTab(self._auto_tab(), "Personalized")
        self.tabs.addTab(self._ab_tab(), "Compare A / B")
        self.tabs.addTab(self._manual_tab(), "Manual")
        self.tabs.setMaximumHeight(170)
        lay.addWidget(self.tabs)

        # FR6: pitch view + per-note table
        self.view = PitchView()
        lay.addWidget(self.view, 3)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Time (s)", "Target", "Before (cents)", "After (cents)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setMaximumHeight(150)
        lay.addWidget(self.table, 1)

        self.setCentralWidget(root)
        self.status = self.statusBar()
        self._set_enabled(False)
        self._build_menu()

    def _build_menu(self):
        m = self.menuBar().addMenu("&Help")
        m.addAction("Speed test…").triggered.connect(self._speed_test)
        m.addAction("Check this installation").triggered.connect(self._self_test)
        m.addSeparator()
        m.addAction("Where my data is stored").triggered.connect(self._data_folder)
        m.addAction(f"About {APP_NAME}").triggered.connect(self._about)

    def _auto_tab(self):
        w = QWidget()
        h = QHBoxLayout(w)
        self.btn_correct = QPushButton("Correct with my profile")
        self.btn_correct.clicked.connect(self._auto_correct)
        self.btn_play_corr = QPushButton("▶ Corrected")
        self.btn_play_corr.clicked.connect(lambda: self._play(self.result.audio if self.result else None))
        self.btn_keep = QPushButton("👍 Keep")
        self.btn_keep.clicked.connect(lambda: self._feedback("keep", 1))
        self.btn_discard = QPushButton("👎 Discard")
        self.btn_discard.clicked.connect(lambda: self._feedback("keep", 0))
        self.rating = QComboBox()
        self.rating.addItems(["Rate…", "1 ★", "2 ★★", "3 ★★★", "4 ★★★★", "5 ★★★★★"])
        self.rating.activated.connect(lambda i: i > 0 and self._feedback("rating", i))
        self.btn_save = QPushButton("Save WAV…")
        self.btn_save.clicked.connect(self._save_wav)
        for x in (self.btn_correct, self.btn_play_corr, self.btn_keep, self.btn_discard, self.rating, self.btn_save):
            h.addWidget(x)
        h.addStretch(1)
        return w

    def _ab_tab(self):
        w = QWidget()
        h = QHBoxLayout(w)
        self.btn_make_ab = QPushButton("Make two versions")
        self.btn_make_ab.clicked.connect(self._make_ab)
        self.btn_play_a = QPushButton("▶ A")
        self.btn_play_a.clicked.connect(lambda: self._play(self.ab[0].audio if self.ab else None))
        self.btn_play_b = QPushButton("▶ B")
        self.btn_play_b.clicked.connect(lambda: self._play(self.ab[1].audio if self.ab else None))
        self.btn_pick_a = QPushButton("I prefer A")
        self.btn_pick_a.clicked.connect(lambda: self._pick("a"))
        self.btn_pick_b = QPushButton("I prefer B")
        self.btn_pick_b.clicked.connect(lambda: self._pick("b"))
        for x in (self.btn_make_ab, self.btn_play_a, self.btn_play_b, self.btn_pick_a, self.btn_pick_b):
            h.addWidget(x)
        h.addStretch(1)
        return w

    def _manual_tab(self):
        w = QWidget()
        outer = QHBoxLayout(w)
        box = QGroupBox("Settings")
        f = QFormLayout(box)
        self.s_strength = QSlider(Qt.Horizontal)
        self.s_strength.setRange(0, 100)
        self.s_strength.setValue(75)
        self.s_retune = QSlider(Qt.Horizontal)
        self.s_retune.setRange(0, len(RETUNE_MS_LEVELS) - 1)
        self.s_retune.setValue(3)
        self.c_inkey = QCheckBox("Snap to key (off = nearest semitone)")
        self.c_inkey.setChecked(True)
        self.s_vibrato = QSlider(Qt.Horizontal)
        self.s_vibrato.setRange(0, 100)
        self.s_vibrato.setValue(67)
        self.l_settings = QLabel()
        for s in (self.s_strength, self.s_retune, self.s_vibrato):
            s.valueChanged.connect(self._manual_changed)
        self.c_inkey.toggled.connect(self._manual_changed)
        f.addRow("Correction strength", self.s_strength)
        f.addRow("Retune speed (fast → slow)", self.s_retune)
        f.addRow("", self.c_inkey)
        f.addRow("Keep vibrato", self.s_vibrato)
        f.addRow(self.l_settings)
        outer.addWidget(box, 2)
        col = QVBoxLayout()
        self.btn_apply = QPushButton("Apply")
        self.btn_apply.clicked.connect(self._manual_apply)
        self.btn_play_man = QPushButton("▶ Play")
        self.btn_play_man.clicked.connect(lambda: self._play(self.result.audio if self.result else None))
        self.btn_save_man = QPushButton("Save as my setting")
        self.btn_save_man.clicked.connect(self._manual_save)
        for x in (self.btn_apply, self.btn_play_man, self.btn_save_man):
            col.addWidget(x)
        col.addStretch(1)
        outer.addLayout(col, 1)
        self._manual_changed()
        return w

    # ------------------------------------------------------------------ helpers
    def _set_enabled(self, on: bool):
        for b in (self.btn_correct, self.btn_make_ab, self.btn_apply, self.btn_play_orig):
            b.setEnabled(on)
        for b in (self.btn_play_corr, self.btn_keep, self.btn_discard, self.rating, self.btn_save,
                  self.btn_play_a, self.btn_play_b, self.btn_pick_a, self.btn_pick_b, self.btn_play_man, self.btn_save_man):
            b.setEnabled(False)

    def _busy(self, msg):
        self.status.showMessage(msg)
        QApplication.setOverrideCursor(Qt.WaitCursor)

    def _idle(self, msg=""):
        QApplication.restoreOverrideCursor()
        self._update_status(msg)

    def _update_status(self, extra=""):
        n = self.session.interactions if self.session else 0
        self.status.showMessage(f"Singer: {self.session.user if self.session else '-'} · learned from {n} judgments" + (f" · {extra}" if extra else ""))

    def _run(self, fn, done, msg, *a, **kw):
        """Run ``fn`` on a worker thread, then ``done(result)`` back on the UI thread."""
        self._busy(msg)
        job = Job(fn, *a, **kw)
        self._jobs.add(job)  # without this reference PySide discards the finished signal

        def finished():
            self._jobs.discard(job)
            self._idle()

        job.signals.done.connect(lambda r: (finished(), done(r)))
        job.signals.failed.connect(lambda e: (finished(), self._error(e)))
        self.pool.start(job)

    def _error(self, text):
        self.errors.append(text)
        if self.show_errors:
            QMessageBox.critical(self, APP_NAME, text)

    def _play(self, audio):
        if audio is not None and self.player:
            self.player.play(audio)

    def _manual_settings(self) -> CorrectionSettings:
        return CorrectionSettings(
            strength=self.s_strength.value() / 100,
            retune_ms=float(RETUNE_MS_LEVELS[self.s_retune.value()]),
            snap="in_key" if self.c_inkey.isChecked() else "chromatic",
            vibrato=self.s_vibrato.value() / 100,
        )

    def _show_result(self, res, label=""):
        self.result = res
        an = self.take.analysis
        orig = M.contour_midi(an.track.f0)
        corr = res.plan.corrected_midi  # what the corrector applied; the table shows the re-tracked result
        rows = res.note_report()
        self.view.set_data(an.track.hop_s, orig, corr, [(r["start"], r["end"], r["target"]) for r in rows])
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            vals = [f"{r['start']:.2f}–{r['end']:.2f}", f"{Key(r['target'] % 12, 'major').name.split()[0]}{r['target'] // 12 - 1}",
                    f"{r['before_cents']:+.0f}", "" if r["after_cents"] is None else f"{r['after_cents']:+.0f}"]
            for j, v in enumerate(vals):
                self.table.setItem(i, j, QTableWidgetItem(v))
        s = res.settings
        self._update_status(f"{label} strength {s.strength:.2f}, retune {s.retune_ms:.0f} ms, {s.snap.replace('_', '-')}, vibrato {s.vibrato:.2f} · {res.rtf:.2f}× real time")

    # ------------------------------------------------------------------ users / profile (FR7)
    def _switch_user(self, user):
        self.session = self._session_factory(user)
        self.user_box.clear()
        self.user_box.addItems(sorted(set(self.session.store.users()) | {user}))
        self.user_box.setCurrentText(user)
        last = self.session.last_manual()
        if last:
            self._load_manual(last)
        if self.take:
            # re-analyse under the new session's corrector so caches are consistent
            self.take = self.session.load_take(self.take.analysis.audio)
        self._update_status()

    def _new_user(self):
        name, ok = QInputDialog.getText(self, APP_NAME, "Singer name:")
        if ok and name.strip():
            self._switch_user(name.strip())

    def _reset_profile(self):
        if QMessageBox.question(self, APP_NAME, f"Forget everything AdaptiTune learned about {self.session.user}?") == QMessageBox.Yes:
            self.session.reset()
            self._update_status("profile reset")

    def _export_profile(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export profile", f"{self.session.user}_profile.json", "JSON (*.json)")
        if path:
            self.session.export(path)
            self._update_status(f"exported to {path}")

    # ------------------------------------------------------------------ input (FR1, FR2)
    def _import(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import vocal", "", "Audio (*.wav *.flac *.ogg *.mp3 *.aiff);;All files (*)")
        if path:
            self._load(path)

    def _record(self, on):
        if on:
            try:
                self.recorder.start()
            except Exception as e:  # no microphone, or Windows does not let desktop apps use it
                self.btn_record.blockSignals(True)
                self.btn_record.setChecked(False)
                self.btn_record.blockSignals(False)
                self._error(f"Recording could not start: {e}\n\nCheck that a microphone is connected and that "
                            "Windows lets desktop apps use it (Settings > Privacy & security > Microphone). "
                            "Importing a file still works.")
                return
            self.btn_record.setText("■ Stop recording")
        else:
            self.btn_record.setText("● Record")
            audio = self.recorder.stop()
            if audio.duration < 0.5:
                self._error("The recording is shorter than half a second, so there is nothing to correct.")
                return
            self._load(audio)

    def _load(self, src):
        key = None if self.key_box.currentIndex() == 0 else self.key_box.currentText()
        self._run(self.session.load_take, self._loaded, "Analysing…", src, key)

    def _loaded(self, take):
        self.take = take
        self.result = None
        an = take.analysis
        self.key_label.setText(f"detected: {an.key.name} ({an.key_confidence:.0%})" if self.key_box.currentIndex() == 0 else "")
        self.view.set_data(an.track.hop_s, M.contour_midi(an.track.f0))
        self.table.setRowCount(0)
        self._set_enabled(True)
        self._update_status(f"loaded {an.audio.duration:.1f} s")

    def _key_changed(self, i):
        if self.take and i > 0:
            self.session.set_key(self.take, self.key_box.currentText())
            self.key_label.setText("")
        elif self.take and i == 0:
            from ..pitch.key import estimate_key

            k, c = estimate_key(self.take.analysis.track.f0, self.take.analysis.track.voicing)
            self.take.analysis.key, self.take.analysis.key_confidence = k, c
            self.key_label.setText(f"detected: {k.name} ({c:.0%})")

    # ------------------------------------------------------------------ personalized (FR3, FR4)
    def _auto_correct(self):
        self._run(self.session.propose, self._auto_done, "Correcting…", self.take)

    def _auto_done(self, res):
        self._show_result(res, "agent chose")
        for b in (self.btn_play_corr, self.btn_keep, self.btn_discard, self.rating, self.btn_save):
            b.setEnabled(True)
        self.rating.setCurrentIndex(0)
        self._play(res.audio)

    def _feedback(self, kind, value):
        if not self.result:
            return
        r = self.session.feedback(self.take, self.result, kind, value)
        for b in (self.btn_keep, self.btn_discard, self.rating):
            b.setEnabled(False)
        self._update_status(f"thanks (reward {r:.2f})")

    def _save_wav(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save corrected vocal", "corrected.wav", "WAV (*.wav)")
        if path and self.result:
            save(path, self.result.audio)

    # ------------------------------------------------------------------ A/B (FR3, FR4)
    def _make_ab(self):
        self._run(self.session.propose_ab, self._ab_done, "Making two versions…", self.take)

    def _ab_done(self, pair):
        self.ab = pair
        self._show_result(pair[0], "A:")
        for b in (self.btn_play_a, self.btn_play_b, self.btn_pick_a, self.btn_pick_b):
            b.setEnabled(True)

    def _pick(self, which):
        if not self.ab:
            return
        self.session.feedback_ab(self.take, self.ab[0], self.ab[1], which)
        self._show_result(self.ab[0] if which == "a" else self.ab[1], f"you picked {which.upper()}:")
        for b in (self.btn_pick_a, self.btn_pick_b):
            b.setEnabled(False)

    # ------------------------------------------------------------------ manual (FR5)
    def _load_manual(self, s: CorrectionSettings):
        self.s_strength.setValue(int(round(s.strength * 100)))
        self.s_retune.setValue(int(np.argmin([abs(np.log(r / s.retune_ms)) for r in RETUNE_MS_LEVELS])))
        self.c_inkey.setChecked(s.snap == "in_key")
        self.s_vibrato.setValue(int(round(s.vibrato * 100)))

    def _manual_changed(self, *_):
        s = self._manual_settings()
        self.l_settings.setText(f"{s.strength:.2f} · {s.retune_ms:.0f} ms · {s.snap.replace('_', '-')} · vibrato {s.vibrato:.2f}")
        if self.manual_started is None and self.take is not None:
            self.manual_started = time.time()
        self.manual_adjustments += 1

    def _manual_apply(self):
        if self.manual_started is None:
            self.manual_started = time.time()
        self._run(self.session.manual, self._manual_done, "Applying…", self.take, self._manual_settings())

    def _manual_done(self, res):
        self._show_result(res, "manual:")
        self.btn_play_man.setEnabled(True)
        self.btn_save_man.setEnabled(True)
        self._play(res.audio)

    def _manual_save(self):
        secs = time.time() - self.manual_started if self.manual_started else None
        self.session.save_manual(self.take, self._manual_settings(), seconds=secs, adjustments=self.manual_adjustments)
        self.manual_started, self.manual_adjustments = None, 0
        self._update_status("saved as your manual setting")

    # ------------------------------------------------------------------ help: checks (O5)
    def _speed_test(self):
        box = QMessageBox(self)
        box.setWindowTitle("Speed test")
        box.setText("Times the whole correction path on this computer: pitch tracking, key, note planning and "
                    "WORLD/PSOLA. The target is 0.5× real time or less on the reference laptop.\n\n"
                    "For the report, choose the MIR-1K folder: the test then times the same minute of singing "
                    "as the cloud run. It takes one to three minutes; close other programs first.")
        b_mir = box.addButton("MIR-1K folder…", QMessageBox.AcceptRole)
        b_file = box.addButton("Audio file…", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Cancel)
        box.exec()
        if box.clickedButton() is b_mir:
            src = QFileDialog.getExistingDirectory(self, "Folder that contains MIR-1K")
        elif box.clickedButton() is b_file:
            src, _ = QFileDialog.getOpenFileName(self, "Audio file to time", "", "Audio (*.wav *.flac *.ogg *.mp3 *.aiff)")
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
        save = box.addButton("Save report…", QMessageBox.AcceptRole)
        box.addButton(QMessageBox.Close)
        box.exec()
        if box.clickedButton() is save:
            docs = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
            path, _ = QFileDialog.getSaveFileName(self, "Save report", str(Path(docs) / keep.name), "JSON (*.json)")
            if path:
                shutil.copyfile(keep, path)
        self._update_status(f"{'speed test' if kind == 'speed' else 'check'} report kept in {keep.parent}")

    def _data_folder(self):
        folder = Path(self.session.store.path).parent
        box = QMessageBox(self)
        box.setWindowTitle("Your data")
        box.setText(f"Your profile and everything AdaptiTune learned about each singer are in\n{folder}\n\n"
                    "Nothing is sent anywhere: the app works with no internet connection. Recordings are "
                    "saved only where you save them.")
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

    def closeEvent(self, e):
        if self.player:
            self.player.close()
        super().closeEvent(e)


def check_command(args: list[str]) -> list[str]:
    """How to start this app with the given arguments: the packaged executable, or Python."""
    if getattr(sys, "frozen", False):
        return [sys.executable, *args]
    return [sys.executable, "-m", "adaptitune.app.launch", *args]


def main(argv=None):
    app = QApplication(argv or sys.argv)
    app.setApplicationName(APP_NAME)
    w = MainWindow()
    w.show()
    if "--smoke-test" in (argv or sys.argv):  # used by the packaging check
        QTimer.singleShot(500, app.quit)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
