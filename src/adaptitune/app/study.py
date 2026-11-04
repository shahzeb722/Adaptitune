"""The O6 study window (Study > Run a study session): one session of the user study.

The researcher enters the participant's code, confirms consent and records the three takes;
then the participant judges twelve versions blind (keep or discard, and how natural each
sounds), takes a break halfway, and finally sets the correction by hand for up to five
minutes. :class:`adaptitune.study.runner.StudySession` enforces the protocol and writes the
data; this window only presents it. docs/O6_PROTOCOL.md is the full plan.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

from PySide6.QtCore import Qt, QThreadPool, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..agent.profile import ProfileStore
from ..agent.session import AdaptiveSession
from ..audio import Audio, load
from ..music import Key
from ..study.protocol import N_SESSIONS
from ..study.runner import (
    MANUAL_LIMIT_S,
    TAKES_PER_SESSION,
    StudySession,
    check_code,
    export,
    next_session,
    study_root,
    withdraw,
)
from .audio_io import Recorder
from .jobs import Jobs, UserMessage
from .widgets import Card, ChoiceRow, SettingsPanel, button, label

NATURALNESS = [("1", 1), ("2", 2), ("3", 3), ("4", 4), ("5", 5)]
MAX_TAKE_S = 45.0  # the protocol asks for phrases of 10 to 20 seconds


class StudyWindow(QDialog):
    PAGE_SETUP, PAGE_TRIAL, PAGE_BREAK, PAGE_MANUAL, PAGE_DONE = range(5)

    def __init__(self, corrector, player=None, parent=None, root: str | Path | None = None, show_errors: bool = True):
        super().__init__(parent)
        self.setWindowTitle("AdaptiTune study session")
        self.setObjectName("StudyWindow")
        self.resize(760, 600)
        self.root = Path(root or study_root())
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = ProfileStore(self.root / "profiles.sqlite")
        self.corrector, self.player = corrector, player
        self.recorder = Recorder() if Recorder.available() else None
        pool = QThreadPool(self)
        pool.setMaxThreadCount(1)  # one version at a time, in order
        self.jobs = Jobs(pool)
        self.show_errors, self.errors = show_errors, []
        self.run: StudySession | None = None
        self.takes: list[Audio] = []
        self.i = 0
        self.first_play: float | None = None
        self.played = False
        self.finished = False
        self.manual_started: float | None = None
        self.manual_seconds: float | None = None
        self.manual_adjustments = 0
        self.manual_applied = False
        self.manual_done = False
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._tick)

        self.pages = QStackedWidget()
        for build in (self._setup_page, self._trial_page, self._break_page, self._manual_page, self._done_page):
            self.pages.addWidget(build())
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.addWidget(self.pages)
        self._code_changed()

    # ------------------------------------------------------------------ pages
    def _setup_page(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setSpacing(14)
        v.addWidget(label("Study session", "title"))
        v.addWidget(label("For the researcher. Fill this in, then hand the computer to the participant. "
                          "Everything stays on this computer, filed under the participant's code.", "subtitle"))
        who = Card("Participant")
        row = QHBoxLayout()
        self.code = QLineEdit()
        self.code.setPlaceholderText("Code, such as P07")
        self.code.setMaxLength(8)
        self.code.textChanged.connect(self._code_changed)
        row.addWidget(self.code, 1)
        self.session_label = label("", "caption", wrap=False)
        row.addWidget(self.session_label, 2)
        who.box.addLayout(row)
        self.consent = QCheckBox("The participant has read the information sheet and signed the consent form")
        self.consent.toggled.connect(self._refresh_setup)
        who.box.addWidget(self.consent)
        v.addWidget(who)

        takes = Card(f"The participant's {TAKES_PER_SESSION} takes")
        takes.box.addWidget(label("Short phrases of 10 to 20 seconds each, sung without accompaniment.", "caption"))
        self.take_list = QListWidget()
        self.take_list.setMaximumHeight(96)
        takes.box.addWidget(self.take_list)
        row = QHBoxLayout()
        self.btn_rec_take = button("● Record a take", "record")
        self.btn_rec_take.setCheckable(True)
        self.btn_rec_take.toggled.connect(self._record_take)
        self.btn_rec_take.setEnabled(self.recorder is not None)
        if self.recorder is None:
            self.btn_rec_take.setToolTip("Recording needs a microphone")
        self.btn_import_take = button("Import a take…", "secondary")
        self.btn_import_take.clicked.connect(self._import_take)
        self.btn_remove_take = button("Remove last", "secondary")
        self.btn_remove_take.clicked.connect(self._remove_take)
        self.key_box = QComboBox()
        self.key_box.addItem("Detect the key")
        for k in Key.all():
            self.key_box.addItem(k.name)
        for w in (self.btn_rec_take, self.btn_import_take, self.btn_remove_take):
            row.addWidget(w)
        row.addStretch(1)
        row.addWidget(label("Key:", "caption", wrap=False))
        row.addWidget(self.key_box)
        takes.box.addLayout(row)
        v.addWidget(takes)

        v.addStretch(1)
        row = QHBoxLayout()
        self.btn_export = button("Export study data…", "plain", "Every finished session as CSV files for adaptitune study-analyze")
        self.btn_export.clicked.connect(self._export)
        self.btn_withdraw = button("Withdraw a participant…", "destructive-plain", "Delete everything kept about one participant")
        self.btn_withdraw.clicked.connect(self._withdraw)
        self.btn_start = button("Start session", "primary")
        self.btn_start.clicked.connect(self._start)
        row.addWidget(self.btn_export)
        row.addWidget(self.btn_withdraw)
        row.addStretch(1)
        row.addWidget(self.btn_start)
        v.addLayout(row)
        return page

    def _trial_page(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setSpacing(14)
        self.trial_title = label("", "title")
        v.addWidget(self.trial_title)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(6)
        v.addWidget(self.progress)
        listen = Card("Listen")
        row = QHBoxLayout()
        self.btn_play = button("▶  Play this version", "primary")
        self.btn_play.clicked.connect(self._play_version)
        self.btn_original = button("▶  My original take", "secondary")
        self.btn_original.clicked.connect(self._play_original)
        self.btn_stop = button("■  Stop", "secondary")
        self.btn_stop.clicked.connect(lambda: self.player and self.player.stop())
        for w in (self.btn_play, self.btn_original, self.btn_stop):
            row.addWidget(w)
        row.addStretch(1)
        listen.box.addLayout(row)
        self.preparing = label("Preparing this version…", "caption")
        listen.box.addWidget(self.preparing)
        self.btn_retry = button("Try again", "secondary", "Make this version again")
        self.btn_retry.clicked.connect(lambda: self._show_trial(self.i))
        self.btn_retry.setVisible(False)
        listen.box.addWidget(self.btn_retry)
        v.addWidget(listen)
        ask = Card("Your judgment")
        ask.box.addWidget(label("Would you keep this version of your take?", "body"))
        self.keep_row = ChoiceRow([("Keep it", 1), ("Discard it", 0)])
        self.keep_row.chosen.connect(lambda _: self._refresh_trial())
        ask.box.addWidget(self.keep_row)
        ask.box.addWidget(label("How natural does it sound? 1 is very unnatural, 5 is completely natural.", "body"))
        self.nat_row = ChoiceRow(NATURALNESS)
        self.nat_row.chosen.connect(lambda _: self._refresh_trial())
        ask.box.addWidget(self.nat_row)
        v.addWidget(ask)
        v.addStretch(1)
        row = QHBoxLayout()
        row.addStretch(1)
        self.btn_next = button("Next", "primary")
        self.btn_next.clicked.connect(self._next)
        row.addWidget(self.btn_next)
        v.addLayout(row)
        return page

    def _break_page(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.addStretch(1)
        v.addWidget(label("Halfway there", "title"))
        v.addWidget(label("Take a short break: stretch, have some water. Press Continue when you are ready.", "subtitle"))
        row = QHBoxLayout()
        row.addStretch(1)
        self.btn_continue = button("Continue", "primary")
        self.btn_continue.clicked.connect(lambda: self._show_trial(self.i))
        row.addWidget(self.btn_continue)
        v.addLayout(row)
        v.addStretch(2)
        return page

    def _manual_page(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setSpacing(14)
        v.addWidget(label("Now set it yourself", "title"))
        v.addWidget(label("Move the sliders until the correction sounds right to you, and press Apply to hear it. "
                          "You have up to five minutes.", "subtitle"))
        card = Card()
        self.manual_panel = SettingsPanel()
        self.manual_panel.changed.connect(self._manual_changed)
        card.box.addWidget(self.manual_panel)
        row = QHBoxLayout()
        self.btn_apply = button("Apply and play", "primary")
        self.btn_apply.clicked.connect(self._manual_apply)
        self.btn_manual_stop = button("■  Stop", "secondary")
        self.btn_manual_stop.clicked.connect(lambda: self.player and self.player.stop())
        self.time_left = label("", "caption", wrap=False)
        row.addWidget(self.btn_apply)
        row.addWidget(self.btn_manual_stop)
        row.addStretch(1)
        row.addWidget(self.time_left)
        card.box.addLayout(row)
        v.addWidget(card)
        rate = Card("When you are happy with it")
        self.btn_happy = button("I'm happy with this setting", "secondary")
        self.btn_happy.clicked.connect(self._manual_happy)
        rate.box.addWidget(self.btn_happy)
        self.rate_label = label("How good is your own version? 1 is poor, 5 is excellent.", "body")
        self.rate_row = ChoiceRow(NATURALNESS)
        self.rate_row.chosen.connect(lambda _: self.btn_finish.setEnabled(True))
        rate.box.addWidget(self.rate_label)
        rate.box.addWidget(self.rate_row)
        v.addWidget(rate)
        v.addStretch(1)
        row = QHBoxLayout()
        row.addStretch(1)
        self.btn_finish = button("Finish session", "primary")
        self.btn_finish.clicked.connect(self._finish)
        row.addWidget(self.btn_finish)
        v.addLayout(row)
        return page

    def _done_page(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setSpacing(12)
        v.addStretch(1)
        mark = label("✓", "donecheck", wrap=False)
        mark.setAlignment(Qt.AlignCenter)
        v.addWidget(mark)
        v.addWidget(label("Thank you!", "title"))
        self.done_text = label("", "subtitle")
        v.addWidget(self.done_text)
        details = Card("For the researcher")
        self.done_details = label("", "caption")
        details.box.addWidget(self.done_details)
        v.addWidget(details)
        row = QHBoxLayout()
        row.addStretch(1)
        b = button("Close", "primary")
        b.clicked.connect(self.close)
        row.addWidget(b)
        v.addLayout(row)
        v.addStretch(2)
        return page

    # ------------------------------------------------------------------ setup
    def _code(self) -> str | None:
        try:
            return check_code(self.code.text())
        except ValueError:
            return None

    def _code_changed(self, *_):
        code = self._code()
        if not self.code.text().strip():
            self.session_label.setText("Use a code, never the participant's name.")
        elif code is None:
            self.session_label.setText("A code is one to four letters, then digits, such as P07.")
        else:
            n = next_session(code, self.root)
            self.session_label.setText(f"All {N_SESSIONS} sessions are done." if n is None else f"Session {n} of {N_SESSIONS}")
        self._refresh_setup()

    def _refresh_setup(self, *_):
        code = self._code()
        ready = (code is not None and next_session(code, self.root) is not None and self.consent.isChecked()
                 and len(self.takes) == TAKES_PER_SESSION and not len(self.jobs))
        self.btn_start.setEnabled(ready)
        full = len(self.takes) >= TAKES_PER_SESSION
        self.btn_import_take.setEnabled(not full)
        self.btn_rec_take.setEnabled(self.recorder is not None and (not full or self.btn_rec_take.isChecked()))
        self.btn_remove_take.setEnabled(bool(self.takes))

    def _add_take(self, audio: Audio, name: str = "") -> None:
        if audio.duration < 0.5:
            self._error("That take is shorter than half a second. Please record it again.")
            return
        if audio.duration > MAX_TAKE_S:
            self._error(f"That take is {audio.duration:.0f} seconds long. The study uses short phrases of 10 to 20 "
                        "seconds; please record or choose a shorter one.")
            return
        self.takes.append(audio)
        self.take_list.addItem(f"Take {len(self.takes)} · {audio.duration:.1f} s" + (f" · {name}" if name else ""))
        self._refresh_setup()

    def _remove_take(self):
        if self.takes:
            self.takes.pop()
            self.take_list.takeItem(self.take_list.count() - 1)
        self._refresh_setup()

    def _import_take(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import a take", "", "Audio (*.wav *.flac *.ogg *.mp3 *.aiff)")
        if not path:
            return
        try:
            self._add_take(load(path), Path(path).name)
        except Exception:
            self._error(f"AdaptiTune could not read {Path(path).name}. Use a WAV, FLAC, OGG, MP3 or AIFF file.")

    def _record_take(self, on: bool):
        if on:
            try:
                self.recorder.start()
            except Exception as e:
                self.btn_rec_take.blockSignals(True)
                self.btn_rec_take.setChecked(False)
                self.btn_rec_take.blockSignals(False)
                self._error(f"Recording could not start: {e}")
                return
            self.btn_rec_take.setText("■ Stop recording")
        else:
            self.btn_rec_take.setText("● Record a take")
            try:
                audio = self.recorder.stop()
            except Exception as e:  # the microphone went away mid-take
                self._error(f"The recording stopped with an error: {e}")
            else:
                self._add_take(audio)
        self._refresh_setup()

    def _start(self):
        code = self._code()
        n = next_session(code, self.root) if code else None
        if code is None or n is None or not self.consent.isChecked() or len(self.takes) != TAKES_PER_SESSION or len(self.jobs):
            return
        if self.player is None and self.show_errors:
            self._error("AdaptiTune found no audio output, so the participant could not hear the versions. "
                        "Connect the study headphones and open this window again.")
            return
        api = AdaptiveSession(code, self.store, self.corrector)
        try:
            self.run = StudySession(code, n, api, root=self.root)
        except ValueError as e:
            self._error(str(e))
            return
        key = None if self.key_box.currentIndex() == 0 else self.key_box.currentText()
        run, takes = self.run, list(self.takes)

        def prepare():
            for a in takes:
                run.add_take(a, key)
            run.start()
            return run.render(0)

        self._lock_setup(True)
        self.session_label.setText("Analysing the takes…")
        self.jobs.start(prepare, lambda _: self._show_trial(0), self._setup_failed)

    def _lock_setup(self, locked: bool) -> None:
        for w in (self.code, self.consent, self.key_box, self.btn_export, self.btn_withdraw):
            w.setEnabled(not locked)
        if locked:
            for w in (self.btn_start, self.btn_import_take, self.btn_rec_take, self.btn_remove_take):
                w.setEnabled(False)
        else:
            self._refresh_setup()

    def _setup_failed(self, text: str):
        """The takes could not be analysed: nothing was judged, so start again from the setup."""
        self.run = None
        self._lock_setup(False)
        self._code_changed()
        self._error(text)

    # ------------------------------------------------------------------ trials
    def _show_trial(self, i: int):
        self.i = i
        n = len(self.run.trials)
        self.pages.setCurrentIndex(self.PAGE_TRIAL)
        self.trial_title.setText(f"Version {i + 1} of {n}")
        self.progress.setRange(0, n)
        self.progress.setValue(i)
        self.keep_row.clear()
        self.nat_row.clear()
        self.played, self.first_play = False, None
        self.btn_retry.setVisible(False)
        self.preparing.setText("Preparing this version…")
        self._wait_ready(i)

    def _wait_ready(self, i: int):
        if self.pages.currentIndex() != self.PAGE_TRIAL or self.i != i:
            return
        if i in self.run.results:
            self._ready()
            return
        self._refresh_trial()
        if len(self.jobs):  # it is being made already, while the last one was judged
            QTimer.singleShot(200, lambda: self._wait_ready(i))
        else:
            self.jobs.start(self.run.render, lambda _: self._wait_ready(i), self._render_failed, i)

    def _render_failed(self, text: str):
        self.preparing.setText("This version could not be made.")
        self.btn_retry.setVisible(True)
        self._error(text)

    def _ready(self):
        self._refresh_trial()
        nxt = self.i + 1
        # make the next version while this one is judged, unless the agent must hear this judgment first
        if nxt < len(self.run.trials) and nxt not in self.run.results and self.run.can_render(nxt) and not len(self.jobs):
            self.jobs.start(self.run.render, lambda _: None, lambda text: None, nxt)  # a failure shows when it is needed

    def _refresh_trial(self):
        ready = self.run is not None and self.i in self.run.results
        self.preparing.setVisible(not ready)
        self.btn_play.setEnabled(ready)
        for row in (self.keep_row, self.nat_row):  # answer after listening, not before
            row.setEnabled(ready and self.played)
        self.btn_next.setEnabled(ready and self.played and self.keep_row.value() is not None and self.nat_row.value() is not None)

    def _play_version(self):
        if self.i not in self.run.results:
            return
        if self.first_play is None:
            self.first_play = time.time()
        self.played = True
        if self.player:
            self.player.play(self.run.results[self.i].audio)
        self._refresh_trial()

    def _play_original(self):
        t = self.run.trials[self.i]
        if self.player:
            self.player.play(self.run.takes[t.stimulus].analysis.audio)

    def _next(self):
        if not self.btn_next.isEnabled():
            return
        if self.player:
            self.player.stop()
        secs = time.time() - self.first_play if self.first_play else None
        try:
            self.run.judge(self.i, self.keep_row.value(), self.nat_row.value(), secs)
        except Exception as e:  # pragma: no cover - the buttons only allow valid answers
            self._error(str(e))
            return
        nxt = self.i + 1
        if nxt == len(self.run.trials):
            self._start_manual()
        elif nxt == len(self.run.trials) // 2:
            self.i = nxt
            self.pages.setCurrentIndex(self.PAGE_BREAK)
        else:
            self._show_trial(nxt)

    # ------------------------------------------------------------------ hand-tuning task
    def _start_manual(self):
        self.pages.setCurrentIndex(self.PAGE_MANUAL)
        self.manual_started = time.time()
        self.manual_adjustments, self.manual_applied, self.manual_done = 0, False, False
        self.rate_row.clear()
        for w in (self.rate_label, self.rate_row):
            w.setVisible(False)
        self.btn_finish.setEnabled(False)
        self.btn_happy.setEnabled(False)
        self._tick()
        self.timer.start()

    def _left(self) -> float:
        return MANUAL_LIMIT_S - (time.time() - self.manual_started)

    def _tick(self):
        left = max(0.0, self._left())
        self.time_left.setText(f"{int(left) // 60}:{int(left) % 60:02d} left")
        if left <= 0:
            self.timer.stop()
            self.time_left.setText("Time is up")
            self._manual_happy()

    def _manual_changed(self):
        self.manual_adjustments += 1

    def _manual_apply(self):
        settings = self.manual_panel.settings()
        self.btn_apply.setEnabled(False)

        def done(res):
            self.manual_applied = True
            if self.manual_done:  # time ran out while it was being made: the setting is taken already
                return
            self.btn_apply.setEnabled(self._left() > 0)
            self.btn_happy.setEnabled(True)
            if self.player:
                self.player.play(res.audio)

        def failed(text):
            if not self.manual_done:
                self.btn_apply.setEnabled(self._left() > 0)
            self._error(text)

        self.jobs.start(self.run.manual, done, failed, settings)

    def _manual_happy(self):
        if self.manual_done:
            return
        self.manual_done = True
        self.timer.stop()
        self.manual_seconds = min(time.time() - self.manual_started, MANUAL_LIMIT_S)
        for w in (self.manual_panel, self.btn_apply, self.btn_happy):
            w.setEnabled(False)
        for w in (self.rate_label, self.rate_row):
            w.setVisible(True)

    def _finish(self):
        rating = self.rate_row.value()
        if rating is None:
            return
        self.run.finish_manual(self.manual_panel.settings(), self.manual_seconds, self.manual_adjustments, rating)
        summary = self.run.finish()
        self.finished = True
        if self.player:
            self.player.stop()
        n = summary["session"]
        self.done_text.setText(f"Session {n} of {N_SESSIONS} is complete." + ("" if n == N_SESSIONS else " See you next time."))
        cons = summary["consistency"]
        self.done_details.setText(
            f"{summary['flagged_trials']} judgments flagged; agreement on repeats "
            f"{'n/a' if math.isnan(cons) else f'{cons:.0%}'}; {summary['minutes']} minutes. Saved in {self.root / 'sessions'}. "
            "Copy this line into the study log.")
        self.pages.setCurrentIndex(self.PAGE_DONE)

    # ------------------------------------------------------------------ researcher tools
    def _export(self):
        folder = QFileDialog.getExistingDirectory(self, "Export the study data to")
        if folder:
            try:
                paths = export(self.root, Path(folder) / "adaptitune_study_export")
            except OSError as e:
                self._error(f"The study data could not be written to {folder}: {e.strerror or e}")
                return
            QMessageBox.information(self, "Study data exported", "Wrote " + ", ".join(p.name for p in paths.values())
                                    + f" to\n{paths['trials'].parent}\n\nAnalyse with: adaptitune study-analyze trials.csv --manual manual.csv")

    def _withdraw(self):
        code, ok = QInputDialog.getText(self, "Withdraw a participant", "Participant code:")
        if not ok or not code.strip():
            return
        try:
            code = check_code(code)
        except ValueError as e:
            self._error(str(e))
            return
        if QMessageBox.question(self, "Withdraw a participant", f"Delete every session, take and learned setting of {code}? "
                                "This cannot be undone.") != QMessageBox.Yes:
            return
        try:
            n = withdraw(code, self.store, self.root)
        except OSError as e:
            self._error(f"Not everything of {code} could be deleted: {e}. Close any program that has the files open "
                        "and withdraw again.")
            return
        QMessageBox.information(self, "Participant withdrawn", f"Deleted {n} files and the learned profile of {code}.")
        self._code_changed()

    # ------------------------------------------------------------------ errors and closing
    def _failed(self, text: str):
        self._error(text)

    def _error(self, text: str):
        self.errors.append(text)
        if not self.show_errors:
            return
        box = QMessageBox(QMessageBox.Critical, "AdaptiTune study", text, QMessageBox.Ok, self)
        if text.startswith("Traceback"):
            box.setText("Something went wrong. Please save the details below and send them to the AdaptiTune team.")
            box.setDetailedText(text)
        box.exec()

    def keyPressEvent(self, e):  # noqa: N802 (Qt naming)
        if e.key() == Qt.Key_Escape:  # a dialog closes on Esc without asking; a session must ask first
            self.close()
            return
        super().keyPressEvent(e)

    def in_progress(self) -> bool:
        return self.run is not None and bool(self.run.trials) and not self.finished

    def closeEvent(self, e):
        if self.in_progress() and self.show_errors:
            ans = QMessageBox.question(self, "Stop this session?", "The judgments so far are kept as an unfinished session "
                                       "and left out of the analysis. Stop anyway?")
            if ans != QMessageBox.Yes:
                e.ignore()
                return
        self.timer.stop()
        if self.player:
            self.player.stop()
        if self.recorder is not None and self.recorder.stream is not None:
            self.recorder.stop()
        super().closeEvent(e)


__all__ = ["StudyWindow", "UserMessage"]
