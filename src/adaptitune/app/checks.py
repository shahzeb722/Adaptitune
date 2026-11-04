"""Checks that the installed app works on the machine it runs on (O5).

* ``self_test`` drives every feature (FR1-FR7) through the main window, the way the buttons do,
  on a short detuned synthetic take, with the network blocked (NFR1) and a throwaway profile
  folder, and reports each requirement. It also loads the light and dark looks and paints the
  window with each. The Windows build runs it in CI on the packaged app.
* ``speed_test`` times the whole correction path the way the app runs it (NFR2): the tracker,
  the key, the planner and WORLD/PSOLA with the quality check. Given the MIR-1K folder it times
  the same minute of vocals as the cloud run (docs/RESULTS.md, O5); given a file, that file.

    AdaptiTune.exe --self-test  --out self_test.json
    AdaptiTune.exe --speed-test E:\\FYP\\MIR1K --out speed_test.json

Both are also under Help in the app. Each writes JSON and exits with 0 on a pass and 1 on a
fail. Use ``python -m adaptitune.app.launch`` in place of ``AdaptiTune.exe`` in a checkout.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import sys
import tempfile
import time
import traceback
from pathlib import Path

# C major, each note sung off pitch by a known amount: what the self-test corrects
DEMO_NOTES = [(60, 0.6, 35, 40), (62, 0.5, -30, 0), (64, 0.7, 20, 50), (65, 0.5, -40, 0), (67, 0.9, 25, 45),
              (65, 0.5, 30, 0), (64, 0.6, -35, 40), (62, 0.5, 25, 0), (60, 1.0, -20, 45)]
DEMO_KEY = "C major"


def demo_take():
    from ..audio import Audio
    from ..data.synth import SungNote, sing

    s = sing([SungNote(m, d, offset_cents=o, vibrato_cents=v) for m, d, o, v in DEMO_NOTES], seed=1)
    return Audio(s.samples, s.sr)


class NoNetwork:
    """Blocks every way of opening a connection and records each attempt (NFR1)."""

    def __init__(self):
        self.attempts: list[str] = []

    def _block(self, name):
        def blocked(*a, **k):
            self.attempts.append(f"{name}{a[1:2] if name == 'socket.connect' else a[:1]}")
            raise OSError("network access is blocked by the AdaptiTune self-test")

        return blocked

    def __enter__(self):
        self.saved = [(socket.socket, "connect", socket.socket.connect), (socket.socket, "connect_ex", socket.socket.connect_ex),
                      (socket, "create_connection", socket.create_connection), (socket, "getaddrinfo", socket.getaddrinfo)]
        socket.socket.connect = self._block("socket.connect")
        socket.socket.connect_ex = self._block("socket.connect_ex")
        socket.create_connection = self._block("create_connection")
        socket.getaddrinfo = self._block("getaddrinfo")
        return self

    def __exit__(self, *exc):
        for obj, name, fn in self.saved:
            setattr(obj, name, fn)


def _wait(app, cond, errors, timeout: float = 600.0) -> float:
    t0 = time.time()
    while not cond():
        if errors:
            raise RuntimeError(errors[-1])
        if time.time() - t0 > timeout:
            raise TimeoutError(f"no result after {timeout:.0f} s")
        app.processEvents()
        time.sleep(0.01)
    return time.time() - t0


def self_test(tracker: str = "auto", timeout: float = 600.0) -> dict:
    """Run FR1-FR7 through the main window, offline, in a throwaway profile folder."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from ..agent.session import AdaptiveSession
    from ..audio import save
    from ..pipeline import Corrector
    from .main import MainWindow, Recorder

    report = {"checks": {}, "info": {}}
    checks, info = report["checks"], report["info"]

    def check(name, ok, detail=""):
        checks[name] = {"pass": bool(ok), "detail": detail}

    home = Path(tempfile.mkdtemp(prefix="adaptitune_selftest_"))
    old_home = os.environ.get("ADAPTITUNE_HOME")
    os.environ["ADAPTITUNE_HOME"] = str(home)
    app = QApplication.instance() or QApplication([])
    w = None
    try:
        with NoNetwork() as net:
            factory = None if tracker == "auto" else (lambda u: AdaptiveSession(u, corrector=Corrector(tracker)))
            w = MainWindow(session_factory=factory, show_errors=False)
            info["tracker"] = w.session.corrector.tracker.name
            info["playback"] = "Qt Multimedia loaded" if w.player is not None else "no audio output"
            w.player = None  # the self-test never makes a sound
            errors = w.errors

            # FR1: import a file, as the Import button does (recording needs a microphone)
            audio = demo_take()
            wav = home / "demo_take.wav"
            save(wav, audio)
            w._load(str(wav))
            secs = _wait(app, lambda: w.take is not None, errors, timeout)
            check("FR1 import a vocal", abs(w.take.analysis.audio.duration - audio.duration) < 0.05,
                  f"{audio.duration:.1f} s take analysed in {secs:.1f} s")
            rec = Recorder.available()
            info["recording"] = "available" if rec else "sounddevice/PortAudio missing"
            if rec:
                try:
                    import sounddevice as sd

                    info["input_devices"] = sum(1 for d in sd.query_devices() if d["max_input_channels"] > 0)
                except Exception as e:  # pragma: no cover - device query
                    info["input_devices"] = f"query failed: {e}"

            # FR2: detected key, then a key set by hand
            detected = w.take.analysis.key.name
            w.key_box.setCurrentText("G major")
            w._key_changed(w.key_box.currentIndex())
            by_hand = w.take.analysis.key.name
            w.key_box.setCurrentText(DEMO_KEY)
            w._key_changed(w.key_box.currentIndex())
            check("FR2 key by hand or detected", detected == DEMO_KEY and by_hand == "G major" and w.take.analysis.key.name == DEMO_KEY,
                  f"detected {detected}; set by hand: {by_hand}")

            # FR3 + FR6: correct with the profile, then the per-note table
            w._auto_correct()
            secs = _wait(app, lambda: w.result is not None, errors, timeout)
            res = w.result
            # the speed test measures real-time factors; res.rtf would also count the analysis above
            check("FR3 correct a take", res.audio.duration > 0.95 * audio.duration, f"corrected in {secs:.1f} s")
            rows = res.note_report()
            before = sorted(abs(r["before_cents"]) for r in rows)
            after = sorted(abs(r["after_cents"]) for r in rows if r["after_cents"] is not None)
            med = lambda v: v[len(v) // 2] if v else float("nan")  # noqa: E731
            check("FR6 per-note pitch error", w.table.rowCount() == len(rows) and abs(len(rows) - len(DEMO_NOTES)) <= 2 and med(after) < med(before),
                  f"{len(rows)} notes; median error {med(before):.0f} cents before, {med(after):.0f} after")

            # FR4: keep, rate, and pick between two versions
            w._feedback("keep", 1)
            first = w.result
            w._auto_correct()
            _wait(app, lambda: w.result is not first, errors, timeout)
            w._feedback("rating", 4)
            w._make_ab()
            _wait(app, lambda: w.ab is not None, errors, timeout)
            w._pick("a")
            check("FR4 feedback: keep, rating, A/B", w.session.interactions == 3, f"{w.session.interactions} judgments stored")

            # FR5: manual settings, saved, and still there after a restart
            prev = w.result
            w.tabs.setCurrentIndex(2)
            w.s_strength.setValue(50)
            w._manual_apply()
            _wait(app, lambda: w.result is not prev, errors, timeout)
            w._manual_save()
            w._switch_user(w.session.user)  # a fresh session reads the profile from disk
            saved = w.session.last_manual()
            check("FR5 manual settings saved", saved is not None and abs(saved.strength - 0.5) < 1e-9 and w.session.interactions == 3,
                  f"saved strength {saved.strength if saved else None}; {w.session.interactions} judgments after reload")

            # FR7: export, then reset
            out = w.session.export(home / "profile_export.json")
            exported = out.exists() and "posterior" in out.read_text()
            w.session.reset()
            check("FR7 reset or export the profile", exported and w.session.interactions == 0, f"exported {out.stat().st_size} bytes; reset to 0 judgments")

            # the look: the stylesheet loads in both appearances and the window paints with it
            from PySide6.QtCore import qInstallMessageHandler

            from . import theme

            qt_messages = []
            previous = qInstallMessageHandler(lambda mode, context, message: qt_messages.append(message))
            try:
                for mode in ("dark", "light"):
                    theme.apply(app, mode)
                    w.grab()
                    app.processEvents()
            finally:
                qInstallMessageHandler(previous)
            bad = [m for m in qt_messages if "style" in m.lower() or "parse" in m.lower()]
            check("Look in light and dark", not bad and not theme.app_icon().isNull(),
                  "stylesheet and icon load, window painted in both" if not bad else "; ".join(bad[:3]))

            check("NFR1 no network", not net.attempts, "no connection attempted" if not net.attempts else f"attempts: {net.attempts[:5]}")
        files = sorted(str(p.relative_to(home)) for p in home.rglob("*") if p.is_file())
        check("NFR3 data stays in the profile folder", "profile.sqlite" in files, f"files written: {', '.join(files)}")
        if getattr(sys, "frozen", False):
            check("NFR4 packaged app runs the bundled pitch model", info["tracker"] == "fcpe-onnx", f"tracker {info['tracker']}")
    except Exception:
        check("error", False, traceback.format_exc(limit=6))
    finally:
        if w is not None:
            w.close()
            app.processEvents()
        if old_home is None:
            os.environ.pop("ADAPTITUNE_HOME", None)
        else:
            os.environ["ADAPTITUNE_HOME"] = old_home
    info.update(_about())
    report["pass"] = all(c["pass"] for c in checks.values())
    return report


def speed_test(src: str | Path | None = None, repeats: int = 3, tracker: str = "fcpe-onnx") -> dict:
    """Time the full correction path (NFR2) on the MIR-1K minute, a file, or synthetic singing."""
    from ..bench.rtf import measure, mir1k_minute

    if src and Path(src).is_dir():
        tmp = Path(tempfile.mkdtemp(prefix="adaptitune_speed_"))
        path, what = mir1k_minute(src, tmp / "mir1k_minute.wav"), "the first minute of MIR-1K vocals (the input of the cloud O5 run)"
    elif src:
        path, what = Path(src), f"the file {Path(src).name}"
    else:
        path, what = None, "20 s of synthetic singing"
    res = measure(path, tracker, repeats)
    res["input"] = what
    res.update({k: v for k, v in _about().items() if k not in res})
    return res


def _about() -> dict:
    from ..machine import build_info, version

    return {"platform": platform.platform(), "packaged_app": bool(getattr(sys, "frozen", False)), "build": build_info(),
            "pyside6": version("PySide6"), "onnxruntime": version("onnxruntime")}


def summary(report: dict, kind: str) -> str:
    """A few lines for a dialog or a console."""
    if kind == "speed":
        r = report
        return (f"Real-time factor {r['rtf']:.3f} (target {r['target']}): {'PASS' if r['pass'] else 'FAIL'}\n"
                f"{r['audio_seconds']:.1f} s of audio, best of {r.get('repeats', 3)} runs; tracker {r['tracker']}, "
                f"{r['threads']} threads\n{r['cpu']}, {r['machine'].get('ram_gb', '?')} GB RAM\nInput: {r['input']}")
    lines = [f"{'PASS' if c['pass'] else 'FAIL'}  {name}: {c['detail']}" for name, c in report["checks"].items()]
    return ("All checks passed." if report["pass"] else "Some checks failed.") + "\n\n" + "\n".join(lines)


def cli(argv: list[str]) -> int:
    """``--self-test`` or ``--speed-test [PATH]``, each with ``--out FILE``."""
    import argparse

    ap = argparse.ArgumentParser(prog="AdaptiTune")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--self-test", action="store_true")
    g.add_argument("--speed-test", nargs="?", const="", metavar="MIR1K_FOLDER_OR_FILE")
    ap.add_argument("--out", help="write the report here (JSON)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tracker", help="default: the bundled FCPE (speed test) or the app's own choice (self-test)")
    a = ap.parse_args(argv)
    try:
        if a.self_test:
            report, kind = self_test(a.tracker or "auto"), "self"
        else:
            report, kind = speed_test(a.speed_test or None, a.repeats, a.tracker or "fcpe-onnx"), "speed"
        ok = bool(report["pass"])
    except Exception:
        report, kind, ok = {"pass": False, "error": traceback.format_exc()}, "error", False
    out = Path(a.out or ("self_test.json" if a.self_test else "speed_test.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    if sys.stdout is not None:  # a windowed build has no console
        print(report.get("error") or summary(report, kind))
        print(f"\nReport: {out.resolve()}")
    return 0 if ok else 1
