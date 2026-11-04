# O5 app checks

Reports written by the packaged app's own checks (`adaptitune/app/checks.py`): `--self-test` (Help → Check this installation) and `--speed-test` (Help → Speed test). All are from commit 12e292c. The Windows reports name 93e6d37, GitHub's merge of 12e292c into `main`, which has the same tree (276e5f3).

| File | Machine | What |
|---|---|---|
| `self_test_windows_ci.json` | GitHub's Windows runner: Windows Server 2025 (build 26100), AMD EPYC 9V74, 4 vCPUs, 16 GB, no microphone or audio device | Self-test of the packaged exe, first launch after the build, PyTorch removed from the machine ([run 41799036872](https://github.com/shahzeb722/Adaptitune/actions/runs/41799036872)) |
| `speed_test_windows_ci.json` | the same runner | Speed test on 20 s of synthetic singing, 4 threads, best of 3. Synthetic voices send half the audio through PSOLA, so this is a smoke test, not the NFR2 number |
| `self_test_packaged_linux_first_run.json` | Cloud VM: Linux, Intel Xeon @ 2.10 GHz, 4 cores, 15.7 GB | Self-test of the packaged app (PyInstaller, same spec), first launch: the analysis includes compiling librosa's numba code once |
| `self_test_packaged_linux.json` | the same VM | The second launch, with that code cached |
| `speed_test_packaged_linux.json` | the same VM | Speed test on the MIR-1K folder (the 64 s the cloud O5 run used), packaged app |
| `speed_test_source_linux.json` | the same VM | The same, run from the source checkout (`python -m adaptitune.app.launch`) |

The Windows reports were copied from the job log, where the workflow prints them; the run's `windows-checks` artifact holds the same files. The packaged reports show `null` for most library versions because these builds did not bundle the packages' metadata; later builds do, and `speed_test_source_linux.json` lists them.
