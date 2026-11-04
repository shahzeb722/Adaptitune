"""Playback and recording for the windows."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
from PySide6.QtCore import QUrl

from ..audio import Audio, preprocess, save


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
