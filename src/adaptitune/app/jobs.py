"""Background work for the windows: run slow pipeline calls off the UI thread and bring the
result back to it."""

from __future__ import annotations

import traceback

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot


class UserMessage(Exception):
    """A failure the user can act on: the window shows its message, not a traceback."""


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
        except UserMessage as e:
            self.signals.failed.emit(str(e))
        except Exception:  # surfaced in a dialog, with the traceback under "Show Details"
            self.signals.failed.emit(traceback.format_exc())


class Jobs:
    """Starts jobs and keeps each one referenced until its result is delivered: PySide drops
    the finished signal of a job that nothing holds, which left every button on "Analysing…"."""

    def __init__(self, pool: QThreadPool | None = None):
        self.pool = pool or QThreadPool.globalInstance()
        self.running: set[Job] = set()

    def start(self, fn, done, failed, *a, **kw) -> Job:
        job = Job(fn, *a, **kw)
        self.running.add(job)

        def finish(handler, value):
            self.running.discard(job)
            handler(value)

        job.signals.done.connect(lambda r: finish(done, r))
        job.signals.failed.connect(lambda e: finish(failed, e))
        self.pool.start(job)
        return job

    def __len__(self) -> int:
        return len(self.running)
