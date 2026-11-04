"""Per-note pitch view (FR6): original contour, corrected contour and note targets, drawn in the
theme's colours (``adaptitune.app.theme``) so it reads the same in light and dark."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from ..music import NOTE_NAMES
from . import theme


class PitchView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(170)
        self.hop = 0.005
        self.orig = None
        self.corr = None
        self.notes = []  # (t0, t1, target_midi, ...)

    def set_data(self, hop_s, orig_midi, corr_midi=None, notes=None):
        self.hop = hop_s
        self.orig = orig_midi
        self.corr = corr_midi
        self.notes = notes or []
        self.update()

    def clear(self):
        self.orig = self.corr = None
        self.notes = []
        self.update()

    def paintEvent(self, ev):  # noqa: N802 (Qt naming)
        c = theme.colors()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(c["card"]))
        small = QFont(self.font())
        small.setPointSizeF(max(7.5, small.pointSizeF() - 1.5))
        p.setFont(small)
        if self.orig is None or len(self.orig) == 0 or np.all(np.isnan(self.orig)):
            p.setPen(QColor(c["secondary"]))
            p.drawText(self.rect(), Qt.AlignCenter, "Import or record a take to see its pitch here")
            return
        m = self.orig[~np.isnan(self.orig)]
        # the range of the singing itself: a tracker's stray octave jump must not squash the plot
        lo, hi = np.percentile(m, [1, 99]) if len(m) > 20 else (m.min(), m.max())
        lo, hi = np.floor(lo) - 1, np.ceil(hi) + 1
        n = len(self.orig)
        left, right, top, bottom = 44, 10, 24, 10
        w = max(1, self.width() - left - right)
        h = max(1, self.height() - top - bottom)

        def x_at(i):
            return left + w * i / max(n - 1, 1)

        def y_at(v):
            return top + h * (hi - v) / (hi - lo)

        # semitone grid with note names; thin out the labels when the range is tall
        step = max(1, int(np.ceil((hi - lo) * 13 / h)))
        grid = QPen(QColor(c["separator"]))
        grid.setWidthF(0.6)
        for s in range(int(lo), int(hi) + 1):
            p.setPen(grid)
            p.drawLine(QPointF(left, y_at(s)), QPointF(left + w, y_at(s)))
            if (s - int(lo)) % step == 0:
                p.setPen(QColor(c["secondary"]))
                p.drawText(QRectF(0, y_at(s) - 7, left - 6, 14), Qt.AlignRight | Qt.AlignVCenter, f"{NOTE_NAMES[s % 12]}{s // 12 - 1}")

        # note targets: soft green bars
        green = QColor(c["green"])
        green.setAlpha(90)
        tgt = QPen(green)
        tgt.setWidthF(6)
        tgt.setCapStyle(Qt.RoundCap)
        p.setPen(tgt)
        for t0, t1, target, *_ in self.notes:
            p.drawLine(QPointF(x_at(t0 / self.hop), y_at(target)), QPointF(x_at(t1 / self.hop), y_at(target)))

        def curve(arr, color, width):
            pen = QPen(QColor(color))
            pen.setWidthF(width)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            p.setPen(pen)
            path = QPainterPath()
            pen_down = False
            for i, v in enumerate(arr):
                if np.isnan(v) or v < lo or v > hi:
                    pen_down = False
                    continue
                pt = QPointF(x_at(i), y_at(v))
                if pen_down:
                    path.lineTo(pt)
                else:
                    path.moveTo(pt)
                    pen_down = True
            p.drawPath(path)

        curve(self.orig, c["orange"], 1.5)
        if self.corr is not None:
            curve(self.corr, c["accent"], 2.0)

        # legend, top right: a dot or bar in each colour, then its name
        items = [(c["orange"], "As sung")]
        if self.corr is not None:
            items.append((c["accent"], "Corrected"))
        if self.notes:
            items.append((c["green"], "Target note"))
        fm = p.fontMetrics()
        xr = left + w
        for color, text in reversed(items):
            tw = fm.horizontalAdvance(text)
            p.setPen(QColor(c["secondary"]))
            p.drawText(QRectF(xr - tw, 2, tw, 18), Qt.AlignVCenter | Qt.AlignLeft, text)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(color))
            p.drawRoundedRect(QRectF(xr - tw - 16, 8, 10, 6), 3, 3)
            xr -= tw + 30
