"""Keeps the app standing when something unexpected happens.

Every error, and every warning from Qt, goes to a log file in the profile folder
(``<profile folder>/logs/adaptitune.log``, three old files kept, 1 MB each), and the person
sees a plain message with the details folded away, instead of the app vanishing. The log holds
no recordings and no profile data, only what went wrong and where.
"""

from __future__ import annotations

import logging
import sys
import threading
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path

log = logging.getLogger("adaptitune")
log.addHandler(logging.NullHandler())  # nothing goes to stderr unless install() is called
_state = {"installed": False, "showing": False}


def log_path() -> Path:
    from ..agent.profile import default_path

    folder = default_path().parent / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / "adaptitune.log"


def _qt_messages(mode, context, message):
    from PySide6.QtCore import QtMsgType

    level = {QtMsgType.QtDebugMsg: logging.DEBUG, QtMsgType.QtInfoMsg: logging.INFO,
             QtMsgType.QtWarningMsg: logging.WARNING}.get(mode, logging.ERROR)
    logging.getLogger("adaptitune.qt").log(level, "%s", message)


def _show(text: str) -> None:
    from PySide6.QtWidgets import QApplication, QMessageBox

    if QApplication.instance() is None:
        return
    box = QMessageBox(QMessageBox.Warning, "AdaptiTune", "Something went wrong, but AdaptiTune is still running.")
    box.setInformativeText(f"If it happens again, please send the AdaptiTune team the log file:\n{log_path()}")
    box.setDetailedText(text)
    box.exec()


def excepthook(etype, value, tb) -> None:
    text = "".join(traceback.format_exception(etype, value, tb))
    log.error("unhandled error\n%s", text)
    if issubclass(etype, KeyboardInterrupt):
        sys.__excepthook__(etype, value, tb)
        return
    # one dialog at a time, and only from the window's thread
    if _state["showing"] or threading.current_thread() is not threading.main_thread():
        return
    _state["showing"] = True
    try:
        _show(text)
    except Exception:  # pragma: no cover - never let the error handler raise
        pass
    finally:
        _state["showing"] = False


def install() -> Path:
    """Log to the profile folder and catch every unhandled error. Called once by the app's main()."""
    path = log_path()
    if not _state["installed"]:
        handler = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        sys.excepthook = excepthook
        threading.excepthook = lambda a: excepthook(a.exc_type, a.exc_value, a.exc_traceback)
        try:
            from PySide6.QtCore import qInstallMessageHandler

            qInstallMessageHandler(_qt_messages)
        except Exception:  # pragma: no cover
            pass
        _state["installed"] = True
    return path
