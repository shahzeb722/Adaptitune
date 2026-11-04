# Session script for the researcher

What to do and say in each session of the O6 study. The plan behind it is [../O6_PROTOCOL.md](../O6_PROTOCOL.md). Read the script aloud in your own words, but keep the instructions for the two questions as written, so every participant hears the same thing.

## Before the study starts

* Ethics approval is in place. Nobody is recorded before it.
* The study laptop runs the frozen study build. **Help → Check this installation** and **Help → Speed test** pass on it.
* The headphones, the microphone and the room are the ones the protocol names. Write the volume setting in the study log.
* Do a practice session yourself under code **T1**, then delete it with **Study → Withdraw a participant…**.

## Session 1 only: consent

1. Give the participant the information sheet and time to read it. Answer their questions.
2. If they agree, both of you sign two copies of the consent form. They keep one.
3. Give them the next free code (P01, P02, …). Write their name and code in the list of codes, which is kept apart from the laptop. **Never type a name into the app.**
4. Set the headphone volume together to a level they find comfortable. Write it in the study log, and use the same level in later sessions.

## Every session

**1. Check in (about 1 minute).**
> "Thanks for coming back. Are you still happy to take part today? You can stop at any time."

Open **Study → Run a study session**. Type the code, and check that the app shows the session you expect, such as "Session 3 of 5". Tick the consent box.

**2. Record three takes (about 3 minutes).**
> "Please sing three short phrases of about 10 to 20 seconds each, without music. Any song you like. You can use the same song each time or different ones."

Use **● Record a take** for each one. If a take is cut short, clipped or noisy, use **Remove last** and record it again. Leave the key on **Detect the key**, unless the participant tells you the key. Press **Start session**. The app needs a few seconds to analyse the takes.

**3. Hand over and explain the judgments (about 1 minute).**
> "You'll now hear twelve versions of your phrases, each corrected in a different way. They come in a random order, and I don't know which is which either. Play each one at least once. You can play it again, and you can listen to your own original take whenever you like. Then answer two questions:
> *Would you keep this version?* Imagine it is your own recording and you are deciding whether to keep it.
> *How natural does it sound?* 1 means very unnatural, 5 means completely natural.
> There are no right answers. We want your own taste. Press Next when you have answered both."

Sit where you cannot see the screen. Do not comment on any version.

**4. Break.** After version 6 the app shows a break page. Offer water. The participant presses **Continue** when ready.

**5. Hand-tuning task (up to 5 minutes).**
> "Now set the correction yourself. Move the sliders, press Apply and play to hear the result, and keep going until it sounds right to you. When you're happy, press 'I'm happy with this setting' and rate your version. You have up to five minutes. The clock is on the screen."

Do not suggest settings. If they ask what a slider does, the tooltip text is the only explanation to give.

**6. End (about 1 minute).** Press **Finish session**. Read the researcher line on the last page, which shows flagged judgments, agreement on repeats and minutes. Copy it into the study log with the date and code. Agree on the next session's date: at least one day later, with all five within about four weeks.
> "Thank you, that's session N of 5 done."

## If something goes wrong

* **The participant wants to stop.** Stop at once, close the window and confirm. The unfinished session is kept apart and is not analysed. Ask, without pressing them, whether they want to continue another day, or to withdraw.
* **The app shows an error.** Note the time, press "Show Details" and save the text. Close the window. The unfinished session is not analysed, and the session is run again on another day with new takes. Tell the team.
* **The participant withdraws.** Use **Study → Withdraw a participant…** with their code. It deletes their sessions, takes and learned settings. Delete any exported copies too, and strike their line in the list of codes. Note the date in the study log, without a reason.
* **Distress or discomfort.** Stop the session. The participant's wellbeing comes before the data.

## After each day

* Check that the day's sessions appear under `study/sessions/` on the laptop.
* Make the encrypted backup described in [data_management.md](data_management.md). Never copy study data to a personal device or to cloud storage.
