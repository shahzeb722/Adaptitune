"""Widgets shared by the main window and the study window. Each sets a ``role`` property that
the stylesheet in :mod:`adaptitune.app.theme` keys on, so the look lives in one place."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..correction.settings import RETUNE_MS_LEVELS, CorrectionSettings


def styled(widget, role: str):
    """Tag a widget with a style role (title, subtitle, caption, card, primary, ...)."""
    widget.setProperty("role", role)
    return widget


def label(text: str, role: str = "body", wrap: bool = True) -> QLabel:
    lab = styled(QLabel(text), role)
    lab.setWordWrap(wrap)
    return lab


def button(text: str, role: str = "secondary", tip: str = "") -> QPushButton:
    b = styled(QPushButton(text), role)
    b.setCursor(Qt.PointingHandCursor)
    if tip:
        b.setToolTip(tip)
    return b


class Card(QFrame):
    """A rounded panel that groups related controls."""

    def __init__(self, title: str | None = None, parent=None):
        super().__init__(parent)
        styled(self, "card")
        self.box = QVBoxLayout(self)
        self.box.setContentsMargins(18, 14, 18, 16)
        self.box.setSpacing(10)
        if title:
            self.box.addWidget(label(title, "section", wrap=False))


class ChoiceRow(QWidget):
    """Exclusive choice shown as a row of joined buttons (a segmented control)."""

    chosen = Signal(object)

    def __init__(self, options: list[tuple[str, object]], parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.values = []
        for i, (text, value) in enumerate(options):
            b = styled(QPushButton(text), "segment")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setProperty("position", "first" if i == 0 else "last" if i == len(options) - 1 else "middle")
            self.group.addButton(b, i)
            self.values.append(value)
            lay.addWidget(b)
        self.group.idClicked.connect(lambda i: self.chosen.emit(self.values[i]))

    def value(self):
        i = self.group.checkedId()
        return None if i < 0 else self.values[i]

    def set_value(self, value) -> None:
        if value is None:
            self.clear()
            return
        self.group.button(self.values.index(value)).setChecked(True)

    def clear(self) -> None:
        self.group.setExclusive(False)
        for b in self.group.buttons():
            b.setChecked(False)
        self.group.setExclusive(True)

    def buttons(self) -> list[QPushButton]:
        return self.group.buttons()


class SettingsPanel(QWidget):
    """The four correction settings as sliders (FR5), with a plain-language readout."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(10)
        self.strength = QSlider(Qt.Horizontal)
        self.strength.setRange(0, 100)
        self.strength.setValue(75)
        self.strength.setToolTip("How much of each note's pitch error is removed")
        self.retune = QSlider(Qt.Horizontal)
        self.retune.setRange(0, len(RETUNE_MS_LEVELS) - 1)
        self.retune.setValue(3)
        self.retune.setPageStep(1)
        self.retune.setToolTip("How quickly the voice glides to the right pitch: fast sounds robotic, slow sounds natural")
        self.in_key = QCheckBox("Snap to the key")
        self.in_key.setChecked(True)
        self.in_key.setToolTip("On: notes move to the nearest note of the key. Off: to the nearest semitone")
        self.vibrato = QSlider(Qt.Horizontal)
        self.vibrato.setRange(0, 100)
        self.vibrato.setValue(67)
        self.vibrato.setToolTip("How much of your natural vibrato is kept")
        self.readout = label("", "caption")
        form.addRow("Correction strength", self.strength)
        form.addRow("Retune speed (fast → slow)", self.retune)
        form.addRow("", self.in_key)
        form.addRow("Keep vibrato", self.vibrato)
        form.addRow("", self.readout)
        for s in (self.strength, self.retune, self.vibrato):
            s.valueChanged.connect(self._changed)
        self.in_key.toggled.connect(self._changed)
        self._update_readout()

    def settings(self) -> CorrectionSettings:
        return CorrectionSettings(
            strength=self.strength.value() / 100,
            retune_ms=float(RETUNE_MS_LEVELS[self.retune.value()]),
            snap="in_key" if self.in_key.isChecked() else "chromatic",
            vibrato=self.vibrato.value() / 100,
        )

    def set_settings(self, s: CorrectionSettings) -> None:
        import numpy as np

        for w in (self.strength, self.retune, self.in_key, self.vibrato):
            w.blockSignals(True)
        self.strength.setValue(int(round(s.strength * 100)))
        self.retune.setValue(int(np.argmin([abs(np.log(r / s.retune_ms)) for r in RETUNE_MS_LEVELS])))
        self.in_key.setChecked(s.snap == "in_key")
        self.vibrato.setValue(int(round(s.vibrato * 100)))
        for w in (self.strength, self.retune, self.in_key, self.vibrato):
            w.blockSignals(False)
        self._update_readout()

    def _changed(self, *_):
        self._update_readout()
        self.changed.emit()

    def _update_readout(self):
        s = self.settings()
        self.readout.setText(f"{s.strength:.0%} strength · {s.retune_ms:.0f} ms glide · "
                             f"{'in key' if s.snap == 'in_key' else 'nearest semitone'} · {s.vibrato:.0%} vibrato kept")


def hline() -> QFrame:
    line = styled(QFrame(), "separator")
    line.setFrameShape(QFrame.HLine)
    return line
