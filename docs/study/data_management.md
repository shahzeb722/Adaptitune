# Data management plan for the O6 study

Draft, version 1, 4 November 2026, for the ethics application. It covers the user study ([../O6_PROTOCOL.md](../O6_PROTOCOL.md)). The other activities with people (the real recordings, the pilot, the usability test and the listening test) follow the same rules. Responsible: Muhammad Shahzeb Khan (ethics and data), with Afzaal Afzal, who runs the sessions.

## What is collected

| Data | Personal? | Where it lives |
|---|---|---|
| Name, contact detail, signed consent form | Yes | Paper in a locked drawer, or one encrypted file. Never on the study laptop with the data |
| List of names and codes | Yes, it is the key | Same as above |
| Sung takes (three per session, 10–20 s each, 16-bit WAV) | A voice can identify someone, so yes | `study/takes/` on the study laptop |
| Judgments, ratings, timings, chosen settings | Coded | `study/sessions/` on the study laptop |
| What the agent learned for each code | Coded | `study/profiles.sqlite` on the study laptop |
| Code → number for counterbalancing | Coded, holds no personal data | `study/participants.json` |

No video, no other audio, no location, and no device or account data. AdaptiTune makes no network connections (O5, NFR1), and the study data never leave the project machines.

## Storage and security

* The study laptop's disk is encrypted (BitLocker), and the laptop has a password that only the team knows.
* The study folder is `~/.adaptitune/study/` on that laptop.
* **Backup (decided here).** After each study day, the study folder is copied to one encrypted USB drive kept in the same locked drawer as the consent forms. No cloud storage, personal device or email is used for study data.
* Exports for analysis (`trials.csv`, `manual.csv`, `sessions.csv`) contain codes and numbers only, with no audio. They stay on project machines.

## Access

The three student researchers and the supervisor. Nobody else sees takes or answers.

## Retention and deletion

* Takes and session files are deleted from the laptop and the backup when the final report has been assessed, by [date, about March 2028]. The exception is participants who agreed on the consent form (item 9) that their takes may be kept for the project's later evaluation. Theirs are deleted by [date].
* The list of codes and the consent forms are kept until the takes are deleted, then shredded or deleted. After that the coded numbers can no longer be linked to anyone.
* The coded result tables (no audio) may be kept with the report.

## Withdrawal

**Study → Withdraw a participant…** in the app deletes the participant's session files, takes and learned settings. The profile database is rewritten, and SQLite's secure delete is on, so nothing stays behind in the file's free space (`tests/test_study_runner.py` checks this). The code stays in `participants.json` so later participants keep their counterbalancing order. The researcher also deletes any exported copies and the backup's copy, strikes the name from the list, and notes the date in the study log without a reason. Withdrawal from the analysis is possible until [date, about December 2027], when the analysis is written up.

## Publication

The report and any paper give group results, or results by code. They never give names, and never publish recordings.

## If something goes wrong

If the laptop or the backup drive is lost or stolen, the disk encryption protects the data. Tell the supervisor the same day. The supervisor decides with the department and the ethics committee whether participants must be told.
