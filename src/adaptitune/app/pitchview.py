"""Per-note pitch view (FR6): original contour, corrected contour and note targets."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from ..music import NOTE_NAMES


class PitchView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(220)
        self.hop = 0.005
        self.orig = None
        self.corr = None
        self.notes = []  # (t0, t1, target_midi, before_cents, after_cents)

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
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        pal = self.palette()
        p.fillRect(self.rect(), pal.base())
        if self.orig is None or np.all(np.isnan(self.orig)):
            p.setPen(pal.text().color())
            p.drawText(self.rect(), Qt.AlignCenter, "Import or record a take to see its pitch")
            return
        m = self.orig[~np.isnan(self.orig)]
        lo, hi = np.floor(m.min()) - 1, np.ceil(m.max()) + 1
        n = len(self.orig)
        left, right, top, bottom = 44, 8, 8, 20
        W = max(1, self.width() - left - right)
        H = max(1, self.height() - top - bottom)

        def X(i):
            return left + W * i / max(n - 1, 1)

        def Y(v):
            return top + H * (hi - v) / (hi - lo)

        # semitone grid with note names
        grid = QPen(pal.mid().color())
        grid.setWidthF(0.5)
        p.setFont(QFont(p.font().family(), 8))
        for s in range(int(lo), int(hi) + 1):
            p.setPen(grid)
            p.drawLine(QPointF(left, Y(s)), QPointF(left + W, Y(s)))
            p.setPen(pal.text().color())
            p.drawText(QRectF(0, Y(s) - 7, left - 4, 14), Qt.AlignRight | Qt.AlignVCenter, f"{NOTE_NAMES[s % 12]}{s // 12 - 1}")

        # note targets
        tgt = QPen(QColor(46, 160, 67, 160))
        tgt.setWidthF(4)
        for t0, t1, target, *_ in self.notes:
            p.setPen(tgt)
            p.drawLine(QPointF(X(t0 / self.hop), Y(target)), QPointF(X(t1 / self.hop), Y(target)))

        def curve(arr, color, width):
            pen = QPen(color)
            pen.setWidthF(width)
            p.setPen(pen)
            path = QPainterPath()
            pen_down = False
            for i, v in enumerate(arr):
                if np.isnan(v) or v < lo or v > hi:
                    pen_down = False
                    continue
                pt = QPointF(X(i), Y(v))
                if pen_down:
                    path.lineTo(pt)
                else:
                    path.moveTo(pt)
                    pen_down = True
            p.drawPath(path)

        curve(self.orig, QColor(200, 80, 60), 1.4)
        if self.corr is not None:
            curve(self.corr, QColor(40, 110, 220), 1.8)

        # legend
        p.setPen(pal.text().color())
        p.drawText(QRectF(left, self.height() - bottom + 2, W, bottom - 2), Qt.AlignLeft,
                   "red: as sung    blue: corrected    green: target note")
