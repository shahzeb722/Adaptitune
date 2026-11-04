"""The app's look, after Apple's Human Interface Guidelines: system colours, rounded cards and
controls, segmented choices, and light and dark appearances.

One stylesheet serves both windows. It keys on the ``role`` property that
:mod:`adaptitune.app.widgets` sets (title, subtitle, caption, section, card, primary,
secondary, plain, destructive-plain, record, segment, separator), so the look lives here and the
windows only say what each widget is. The appearance follows the system's light or dark setting
unless View → Appearance fixes one. Apple's SF Pro font is not bundled (its licence does not
allow it); the font stack uses it where it is installed and falls back to Segoe UI on Windows.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from string import Template

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QGuiApplication,
    QIcon,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QApplication, QStyleFactory

FONT_FAMILIES = ["SF Pro Text", ".AppleSystemUIFont", "Segoe UI Variable Text", "Segoe UI", "Helvetica Neue", "Arial"]
APPEARANCES = ("auto", "light", "dark")

# Apple's system colours (iOS / macOS), light and dark
LIGHT = {
    "window": "#F2F2F7", "card": "#FFFFFF", "card_border": "#E5E5EA", "raised": "#FFFFFF", "selected_segment": "#FFFFFF",
    "label": "#1D1D1F", "secondary": "#6E6E73", "tertiary": "#AEAEB2",
    "separator": "#D8D8DC", "fill": "#E9E9EE", "fill_hover": "#DFDFE4", "fill_pressed": "#D2D2D7", "fill_disabled": "#F2F2F5",
    "input": "#FFFFFF", "input_border": "#D2D2D7", "alt_row": "#F7F7FA", "scroll": "#C7C7CC",
    "accent": "#007AFF", "accent_hover": "#1A88FF", "accent_pressed": "#0062CC", "accent_text": "#FFFFFF",
    "accent_soft": "#DCEBFF", "accent_disabled": "#A6CDFF", "focus": "#7FB5FF",
    "red": "#FF3B30", "green": "#34C759", "orange": "#FF9500",
    "track": "#E5E5EA", "handle": "#FFFFFF", "handle_border": "#C7C7CC",
}
DARK = {
    "window": "#1C1C1E", "card": "#2C2C2E", "card_border": "#3A3A3C", "raised": "#3A3A3C", "selected_segment": "#636366",
    "label": "#F5F5F7", "secondary": "#98989D", "tertiary": "#636366",
    "separator": "#3A3A3C", "fill": "#3A3A3C", "fill_hover": "#48484A", "fill_pressed": "#545458", "fill_disabled": "#353538",
    "input": "#1C1C1E", "input_border": "#48484A", "alt_row": "#323234", "scroll": "#545458",
    "accent": "#0A84FF", "accent_hover": "#409CFF", "accent_pressed": "#0071E3", "accent_text": "#FFFFFF",
    "accent_soft": "#173A63", "accent_disabled": "#1E4A7A", "focus": "#0A5CB8",
    "red": "#FF453A", "green": "#30D158", "orange": "#FF9F0A",
    "track": "#48484A", "handle": "#F5F5F7", "handle_border": "#636366",
}

QSS = Template("""
QMainWindow, QDialog, QWidget#page { background-color: $window; }
QScrollArea#pageScroll { background-color: $window; border: none; }
QWidget { color: $label; }
QFrame[role="card"] { background-color: $card; border: 1px solid $card_border; border-radius: 12px; }
QFrame[role="separator"] { background-color: $separator; border: none; max-height: 1px; min-height: 1px; }

QLabel { background: transparent; }
QLabel[role="largetitle"] { font-size: 22pt; font-weight: 700; }
QLabel[role="title"] { font-size: 17pt; font-weight: 700; }
QLabel[role="subtitle"] { font-size: 11pt; color: $secondary; }
QLabel[role="section"] { font-size: 11pt; font-weight: 600; }
QLabel[role="caption"] { font-size: 9pt; color: $secondary; }
QLabel[role="badge"] { background-color: $accent_soft; color: $accent; border-radius: 9px; padding: 2px 9px; font-size: 9pt; font-weight: 600; }
QLabel[role="hint"] { color: $tertiary; }
QLabel[role="donecheck"] { color: #FFFFFF; background-color: $green; border-radius: 28px; font-size: 24pt; font-weight: 700;
                           min-width: 56px; max-width: 56px; min-height: 56px; max-height: 56px; }

QPushButton { background-color: $fill; color: $label; border: 2px solid transparent; border-radius: 8px;
              padding: 4px 12px; min-height: 22px; }
QPushButton:hover { background-color: $fill_hover; }
QPushButton:pressed { background-color: $fill_pressed; }
QPushButton:focus { border-color: $focus; }
QPushButton:disabled { background-color: $fill_disabled; color: $tertiary; }
QPushButton[role="primary"], QMessageBox QPushButton:default, QDialogButtonBox QPushButton:default {
    background-color: $accent; color: $accent_text; font-weight: 600; }
QPushButton[role="primary"]:hover, QMessageBox QPushButton:default:hover, QDialogButtonBox QPushButton:default:hover {
    background-color: $accent_hover; }
QPushButton[role="primary"]:pressed, QMessageBox QPushButton:default:pressed { background-color: $accent_pressed; }
QPushButton[role="primary"]:disabled { background-color: $accent_disabled; color: $accent_text; }
QPushButton[role="plain"], QPushButton[role="destructive-plain"] { background: transparent; color: $accent; padding: 4px 6px; }
QPushButton[role="destructive-plain"] { color: $red; }
QPushButton[role="plain"]:hover, QPushButton[role="destructive-plain"]:hover { background-color: $fill; }
QPushButton[role="plain"]:disabled, QPushButton[role="destructive-plain"]:disabled { background: transparent; color: $tertiary; }
QPushButton[role="record"] { color: $red; font-weight: 600; }
QPushButton[role="record"]:checked { background-color: $red; color: #FFFFFF; }
QPushButton[role="record"]:disabled { color: $tertiary; }
QPushButton[role="star"] { background: transparent; border: none; color: $orange; font-size: 17pt; padding: 0px 3px; min-height: 26px; }
QPushButton[role="star"]:hover { color: $accent; }
QPushButton[role="star"]:disabled { color: $tertiary; background: transparent; }
QPushButton[role="segment"] { border-radius: 0px; padding: 4px 16px; min-width: 28px; }
QPushButton[role="segment"][position="first"] { border-top-left-radius: 8px; border-bottom-left-radius: 8px; }
QPushButton[role="segment"][position="last"] { border-top-right-radius: 8px; border-bottom-right-radius: 8px; }
QPushButton[role="segment"]:checked { background-color: $accent; color: $accent_text; font-weight: 600; }
QPushButton[role="segment"]:checked:disabled { background-color: $accent_disabled; }

QLineEdit, QComboBox { background-color: $input; border: 1px solid $input_border; border-radius: 7px; padding: 4px 8px;
                       min-height: 22px; selection-background-color: $accent; selection-color: #FFFFFF; }
QLineEdit:focus, QComboBox:focus { border: 1px solid $accent; }
QLineEdit:disabled, QComboBox:disabled { background-color: $fill_disabled; color: $tertiary; }
QComboBox { padding-right: 26px; }
QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: center right; width: 24px; border: none; }
QComboBox::down-arrow { image: url("$chevron"); width: 10px; height: 10px; }
QComboBox::down-arrow:disabled { image: url("$chevron_disabled"); }
QComboBox QAbstractItemView { background-color: $raised; color: $label; border: 1px solid $card_border; padding: 4px;
                              selection-background-color: $accent; selection-color: #FFFFFF; outline: none; }

QCheckBox { spacing: 8px; background: transparent; }
QCheckBox::indicator { width: 18px; height: 18px; border-radius: 5px; border: 1px solid $input_border; background-color: $input; }
QCheckBox::indicator:hover { border-color: $accent; }
QCheckBox::indicator:checked { background-color: $accent; border-color: $accent; image: url("$check"); }
QCheckBox::indicator:disabled { background-color: $fill_disabled; border-color: $card_border; }
QCheckBox:disabled { color: $tertiary; }

QSlider { min-height: 26px; background: transparent; }
QSlider::groove:horizontal { height: 4px; background: $track; border-radius: 2px; }
QSlider::sub-page:horizontal { background: $accent; border-radius: 2px; }
QSlider::add-page:horizontal { background: $track; border-radius: 2px; }
QSlider::handle:horizontal { background: $handle; border: 1px solid $handle_border; width: 20px; height: 20px;
                             margin: -9px 0; border-radius: 11px; }
QSlider::handle:horizontal:hover { border-color: $accent; }
QSlider::sub-page:horizontal:disabled { background: $tertiary; }

QProgressBar { background-color: $track; border: none; border-radius: 3px; max-height: 6px; min-height: 6px; text-align: center; }
QProgressBar::chunk { background-color: $accent; border-radius: 3px; }

QTabWidget::pane { border: none; background: transparent; top: 4px; }
QTabWidget::tab-bar { alignment: center; }
QTabBar { background: transparent; }
QTabBar::tab { background-color: $fill; color: $label; border: 2px solid $fill; padding: 4px 22px; min-width: 96px; }
QTabBar::tab:first { border-top-left-radius: 9px; border-bottom-left-radius: 9px; }
QTabBar::tab:last { border-top-right-radius: 9px; border-bottom-right-radius: 9px; }
QTabBar::tab:hover:!selected { background-color: $fill_hover; border-color: $fill_hover; }
QTabBar::tab:selected { background-color: $selected_segment; border-radius: 7px; font-weight: 600; }
QTabBar::tab:disabled { color: $tertiary; }

QTableWidget, QTableView { background-color: $card; alternate-background-color: $alt_row; border: none;
                           gridline-color: transparent; selection-background-color: $accent_soft; selection-color: $label; }
QHeaderView { background: transparent; }
QHeaderView::section { background-color: $card; color: $secondary; border: none; border-bottom: 1px solid $separator;
                       padding: 6px 8px; font-weight: 600; }
QTableCornerButton::section { background-color: $card; border: none; }

QListWidget { background-color: $input; border: 1px solid $input_border; border-radius: 8px; padding: 4px; }
QListWidget::item { padding: 4px 6px; border-radius: 6px; }
QListWidget::item:selected { background-color: $accent_soft; color: $label; }

QMenuBar { background-color: $window; border-bottom: 1px solid $separator; padding: 2px 4px; }
QMenuBar::item { background: transparent; padding: 4px 10px; border-radius: 6px; }
QMenuBar::item:selected { background-color: $fill; }
QMenu { background-color: $raised; border: 1px solid $card_border; padding: 6px; }
QMenu::item { padding: 5px 26px 5px 14px; border-radius: 6px; background: transparent; }
QMenu::item:selected { background-color: $accent; color: #FFFFFF; }
QMenu::item:disabled { color: $tertiary; }
QMenu::separator { height: 1px; background: $separator; margin: 4px 8px; }
QMenu::indicator { width: 14px; height: 14px; left: 4px; }

QStatusBar { background-color: $window; color: $secondary; border-top: 1px solid $separator; }
QStatusBar QLabel { color: $secondary; }
QStatusBar::item { border: none; }
QToolTip { background-color: $raised; color: $label; border: 1px solid $card_border; padding: 5px 8px; }

QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 2px; }
QScrollBar::handle { background: $scroll; border-radius: 3px; }
QScrollBar::handle:vertical { min-height: 30px; }
QScrollBar::handle:horizontal { min-width: 30px; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0px; height: 0px; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

QMessageBox, QInputDialog, QFileDialog { background-color: $window; }
""")

_state = {"appearance": "auto", "dark": False, "watching": False}


def colors() -> dict:
    """The colours of the appearance in use, for widgets that paint themselves (the pitch view)."""
    return DARK if _state["dark"] else LIGHT


def is_dark() -> bool:
    return _state["dark"]


def system_is_dark() -> bool:
    hints = QGuiApplication.styleHints()
    return hints is not None and hints.colorScheme() == Qt.ColorScheme.Dark


def _icons(tokens: dict, dark: bool) -> dict:
    """The few images the stylesheet needs (a chevron and a check mark), drawn at 2x for sharp
    screens. Stylesheets can only load images from files, so they are written once per run."""
    folder = Path(tempfile.gettempdir()) / "adaptitune_theme"
    folder.mkdir(exist_ok=True)
    out = {}
    for name, color, kind in (("chevron", tokens["secondary"], "chevron"), ("chevron_disabled", tokens["tertiary"], "chevron"),
                              ("check", "#FFFFFF", "check")):
        path = folder / f"{name}_{'dark' if dark else 'light'}.png"
        img = QImage(40, 40, QImage.Format_ARGB32)
        img.fill(Qt.transparent)
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(color))
        pen.setWidthF(5.0 if kind == "check" else 4.5)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        path_ = QPainterPath()
        if kind == "chevron":
            path_.moveTo(8, 14)
            path_.lineTo(20, 27)
            path_.lineTo(32, 14)
        else:
            path_.moveTo(9, 21)
            path_.lineTo(17, 29)
            path_.lineTo(31, 11)
        p.drawPath(path_)
        p.end()
        img.save(str(path))
        out[name] = path.as_posix()
    return out


def palette(tokens: dict) -> QPalette:
    c = {k: QColor(v) for k, v in tokens.items()}
    pal = QPalette()
    roles = {
        QPalette.Window: c["window"], QPalette.WindowText: c["label"], QPalette.Base: c["input"],
        QPalette.AlternateBase: c["alt_row"], QPalette.Text: c["label"], QPalette.Button: c["fill"],
        QPalette.ButtonText: c["label"], QPalette.BrightText: c["red"], QPalette.Highlight: c["accent"],
        QPalette.HighlightedText: QColor("#FFFFFF"), QPalette.ToolTipBase: c["raised"], QPalette.ToolTipText: c["label"],
        QPalette.PlaceholderText: c["tertiary"], QPalette.Link: c["accent"], QPalette.Light: c["card"],
        QPalette.Midlight: c["fill"], QPalette.Mid: c["separator"], QPalette.Dark: c["scroll"], QPalette.Shadow: c["scroll"],
    }
    for role, color in roles.items():
        pal.setColor(role, color)
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, c["tertiary"])
    return pal


def apply(app: QApplication | None = None, appearance: str | None = None) -> bool:
    """Style the whole application. ``appearance`` is auto (follow the system), light or dark;
    leave it out to keep the current choice. Returns whether the dark appearance is in use."""
    app = app or QApplication.instance()
    if appearance is not None:
        if appearance not in APPEARANCES:
            raise ValueError(f"appearance is one of {APPEARANCES}")
        _state["appearance"] = appearance
    mode = _state["appearance"]
    dark = system_is_dark() if mode == "auto" else mode == "dark"
    _state["dark"] = dark
    tokens = DARK if dark else LIGHT
    if app.style().name().lower() != "fusion":
        app.setStyle(QStyleFactory.create("Fusion"))
    font = QFont(app.font())
    font.setFamilies(FONT_FAMILIES)
    font.setPointSizeF(10)
    app.setFont(font)
    app.setPalette(palette(tokens))
    app.setStyleSheet(QSS.substitute(tokens, **_icons(tokens, dark)))
    if not _state["watching"] and QGuiApplication.styleHints() is not None:
        # follow the system when it switches between light and dark while the app is open
        QGuiApplication.styleHints().colorSchemeChanged.connect(lambda *_: _state["appearance"] == "auto" and apply(app))
        _state["watching"] = True
    for w in app.topLevelWidgets():
        w.update()
    return dark


def app_icon() -> QIcon:
    """AdaptiTune's icon: a wavering sung line that settles on its note, on a blue squircle."""
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        pm = QPixmap(size, size)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        s = float(size)
        grad = QLinearGradient(QPointF(0, 0), QPointF(0, s))
        grad.setColorAt(0, QColor("#5E5CE6"))
        grad.setColorAt(1, QColor("#0A84FF"))
        p.setPen(Qt.NoPen)
        p.setBrush(grad)
        p.drawRoundedRect(QRectF(0, 0, s, s), s * 0.225, s * 0.225)
        pen = QPen(QColor("#FFFFFF"))
        pen.setWidthF(max(1.2, s * 0.075))
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        line = QPainterPath()
        mid = s * 0.52
        line.moveTo(s * 0.18, mid)
        line.cubicTo(s * 0.26, mid - s * 0.2, s * 0.34, mid + s * 0.2, s * 0.43, mid)
        line.cubicTo(s * 0.48, mid - s * 0.09, s * 0.53, mid + s * 0.05, s * 0.58, mid)
        line.lineTo(s * 0.82, mid)
        p.drawPath(line)
        dot = max(1.5, s * 0.06)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#FFFFFF"))
        p.drawEllipse(QPointF(s * 0.82, mid), dot, dot)
        p.end()
        icon.addPixmap(pm)
    return icon
