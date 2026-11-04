"""Entry point of the packaged app (packaging/adaptitune.spec), also ``python -m adaptitune.app.launch``.

``--self-test`` and ``--speed-test`` run the checks in ``adaptitune.app.checks`` instead of
opening the window. The speed test pins every numeric library to 4 threads before numpy loads,
as the speed protocol does (docs/O2_PROTOCOL.md, A4); ``--threads N`` changes that.
"""

import os
import sys


def _report_crash(argv: list[str]) -> None:
    import json
    import traceback

    out = argv[argv.index("--out") + 1] if "--out" in argv[:-1] else "check_error.json"
    try:
        with open(out, "w") as f:
            json.dump({"pass": False, "error": traceback.format_exc()}, f, indent=2)
    except OSError:
        pass


def run(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--self-test" in argv or "--speed-test" in argv:
        from adaptitune.machine import THREAD_VARS  # no numpy yet

        if "--threads" in argv:
            i = argv.index("--threads")
            threads = argv[i + 1]
            del argv[i : i + 2]
        else:
            threads = "4"
        if "--speed-test" in argv:
            for var in THREAD_VARS:
                os.environ[var] = threads
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # no window for the checks
        try:
            from adaptitune.app.checks import cli

            return cli(argv)
        except BaseException:  # a windowed build would otherwise stop on an error dialog
            _report_crash(argv)
            return 1
    from adaptitune.app.main import main

    return main()


if __name__ == "__main__":
    sys.exit(run())
