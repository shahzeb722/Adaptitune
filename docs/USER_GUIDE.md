# AdaptiTune user guide (Windows)

AdaptiTune corrects the pitch of a recorded vocal and learns, from your feedback, how much correction you like. It runs entirely on your computer: it needs no internet connection and sends nothing anywhere.

## Get the app

1. Open the [windows-app runs](https://github.com/shahzeb722/Adaptitune/actions/workflows/windows-app.yml) on GitHub (you need to be signed in) and click the newest run with a green tick.
2. Under **Artifacts**, download **AdaptiTune-windows**. It is a zip of a few hundred MB, kept for 30 days after each build.
3. Unzip it anywhere, for example into `Documents\AdaptiTune`, and run `AdaptiTune.exe`. There is no installer and it needs no administrator rights.
4. The exe is not signed, so Windows may say "Windows protected your PC". Click **More info**, then **Run anyway**.

The first take you import after unzipping takes about 20 seconds longer than later ones, because the app compiles some of its signal-processing code for your processor once and keeps the result. Later imports, and later launches, skip this.

To build it yourself instead, run `packaging\build_windows.bat` from the repository root in a Python 3.11 virtual environment. The result is `dist\AdaptiTune\AdaptiTune.exe`.

## Correct a take

| Step | What to do | Requirement |
|---|---|---|
| Choose the singer | Pick a name under **Singer**, or **New singer…**. Each singer has their own profile. | |
| Bring in a vocal | **Import vocal…** (WAV, FLAC, OGG, MP3 or AIFF), drag the file onto the window, or **● Record** and click it again to stop. Recording needs a microphone. Use a dry solo vocal of up to 10 minutes, not a full mix. | FR1 |
| Set the key | Leave **Key** on *Auto-detect*, or choose the key yourself. The detected key is shown next to the box. | FR2 |
| Correct | **Personalized → Correct with my profile**. Then **▶ Original** and **▶ Corrected** to compare, and **■ Stop** to stop. **Save WAV…** above the pitch view saves the version on show. | FR3 |
| Tell it what you think | **👍 Keep** or **👎 Discard**, or a star rating. Or **Compare A / B → Make two versions**, listen to both and click **I prefer A** or **I prefer B**. Every judgment updates your profile, and each version can be judged once. | FR4 |
| Set it by hand | **Manual**: correction strength, retune speed, snap to key, and how much vibrato to keep. **Apply** to hear it, then **Save as my setting** to keep it. | FR5 |
| See how far each note was off | The pitch view and the table below it show every note's error in cents before and after correction. | FR6 |
| Start again or take your profile with you | **Reset** (it asks first) or **Export…** (a JSON file), at the top right. | FR7 |

While the app is working on something, the buttons that would start something else are greyed out until it finishes.

## Light and dark, and shortcuts

The app follows Windows' light or dark mode. **View → Appearance** sets it to **Light** or **Dark** instead. The app remembers that choice, the last singer, the last folder you used, and the window's size and position.

| Shortcut | Does |
|---|---|
| Ctrl+O | Import a vocal |
| Ctrl+R | Start or stop recording |
| Ctrl+S | Save the version on show as a WAV file |

## Where your data is

Everything the app learns is in one file, `C:\Users\<you>\.adaptitune\profile.sqlite`. **Help → Where my data is stored** opens that folder. Corrected recordings are saved only where you save them. The same folder holds `settings.ini` (the choices above) and `logs\adaptitune.log`, which records anything that went wrong. Set the `ADAPTITUNE_HOME` environment variable to keep all of this somewhere else.

## Run a study session (O6)

**Study → Run a study session…** opens the window for one session of the user study. Use it only once the study has ethics approval, and follow [study/session_script.md](study/session_script.md). The plan is [O6_PROTOCOL.md](O6_PROTOCOL.md).

1. The researcher types the participant's code (such as P07, never a name). The window shows which session is next, "Session 3 of 5". Tick the consent box, then record or import the participant's three takes. **Start session** becomes available once all three are in.
2. The participant plays each of twelve versions, answers *keep or discard* and *how natural* (1–5), and presses **Next**. There is a break after the sixth.
3. The participant sets the correction by hand for up to five minutes, presses **I'm happy with this setting**, and rates it.
4. The last page gives the researcher a one-line summary for the study log.

The study keeps its own files, apart from your own profile, in `C:\Users\<you>\.adaptitune\study\`. **Export study data…** writes the CSV files for the analysis (`adaptitune study-analyze trials.csv --manual manual.csv` in a checkout). **Withdraw a participant…** deletes everything kept about one code. Closing the window in the middle of a session asks first. An unfinished session is kept apart and left out of the analysis.

## Check the app on your laptop

These checks are how O5 is measured on the reference laptop. Each one shows PASS or FAIL and keeps its report in the `checks` folder next to your profile; **Save report…** puts a copy wherever you like, for example to send it to the team.

1. **Check this installation** (Help menu). It runs every feature (FR1–FR7) on a short built-in take with all network access blocked, in a separate throwaway profile, so your own profile is not touched, and checks that the light and dark looks load. It takes about a minute.
2. **Speed test** (Help menu, NFR2). Close other programs, plug in the charger and set Windows to its best-performance power mode, because a laptop on battery runs slower. Click **MIR-1K folder…** and choose the folder that holds MIR-1K, such as `E:\FYP\MIR1K`. The test times the whole correction path three times on the first minute of MIR-1K vocals, the same audio as the cloud run, and reports the best. It takes one to three minutes. The target is a real-time factor of 0.5 or less.
3. **Offline check** (NFR1). Turn Wi-Fi off and unplug any network cable. Open the app, then import a take, correct it, keep or rate it, make an A/B pair, apply and save a manual setting, and export your profile. Everything should work. Run **Check this installation** once more while you are offline.

The same checks run from a command prompt in the app's folder:

```
AdaptiTune.exe --self-test --out self_test.json
AdaptiTune.exe --speed-test E:\FYP\MIR1K --out speed_test.json
```

The window does not open for these. Each writes its report to the `--out` file and finishes with exit code 0 on a pass and 1 on a fail. The speed test uses 4 threads, as the speed protocol does; add `--threads N` to change that.

## If something goes wrong

* **"Something went wrong, but AdaptiTune is still running."** The app caught an unexpected error and wrote the details to `C:\Users\<you>\.adaptitune\logs\adaptitune.log`. Please send that file to the team. It holds no recordings and nothing from your profile, only what went wrong and where, with a file's path if a file was involved.
* **A take is refused as too long.** The app corrects takes of up to 10 minutes, and a study take may be up to 45 seconds. Cut out the part you want first.
* **The Record button is greyed out.** No microphone was found. Importing a file still works.
* **No sound.** Check the Windows output device; the app plays through the default one.
* **A file will not import.** Convert it to WAV first. M4A and other AAC files are not supported.
* **Help → About** shows the build (the commit it came from) and the pitch tracker in use. It should say `fcpe-onnx`; anything else means the bundled model was not found and the app is using a slower fallback.
