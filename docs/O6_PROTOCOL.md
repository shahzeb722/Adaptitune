# O6 protocol: the multi-session user study

**Objective O6.** Compare personalized, fixed and manually tuned correction through simulation, objective metrics and a multi-session user study.

**Success criterion (proposal, Table 3).** Across sessions three to five:

1. the acceptance rate of personalized correction (C3) is at least 15 percentage points above the fixed baseline (C1);
2. it is within 10 points of the singer's own saved manual setting judged blind (C4);
3. it takes less of the singer's time;
4. naturalness passes the non-inferiority test at a 0.5-point margin (α = 0.05).

O6 is judged on the observed differences with their 95% confidence intervals. The significance tests are exploratory, except the naturalness test, which is part of the criterion. If the targets are missed, that is reported and analysed as a valid result, not a failure.

This plan is fixed on **4 November 2026, before anyone has taken part**. Nobody is recorded or tested before COMSATS University's ethics committee approves the study through the supervisor and the department. The application is due in February 2027, and the sessions are planned for October and November 2027. Any change after today goes in the change log at the end, with its reason and date. The code that carries out the plan is `src/adaptitune/study/` (protocol, runner, analysis) and the study window in the app (`src/adaptitune/app/study.py`). Its tests are `tests/test_study_runner.py`, `tests/test_study_window.py` and `tests/test_session_study.py`.

Points the proposal leaves open are marked **(decided here)**. The team can still change them before the ethics application, and should then record the change below.

---

## 1. Participants

* **Number.** At least 10 singers, and 15 if we can get them. Recruitment goes through student societies (proposal, Table 2). The texts are in [study/recruitment.md](study/recruitment.md).
* **Who can take part (decided here).** Adults aged 18 or over who sing at any level, from beginners to trained singers. They must be able to come to five sessions on different days, and must not know of any hearing loss. Members of the AdaptiTune team are not eligible, nor is anyone who helped choose the app's settings.
* **Codes.** Each participant gets a code such as P07: one to four letters, then digits. The study files hold only the code. The list that links names to codes is kept on paper or in one encrypted file, apart from the study data ([study/data_management.md](study/data_management.md)). The app's code field only takes codes of that form, and it reminds the researcher never to type a name.
* **Consent.** Written informed consent before the first session ([study/information_sheet.md](study/information_sheet.md), [study/consent_form.md](study/consent_form.md)). Before every session the researcher asks again whether the participant is happy to continue, and ticks the consent box in the app.

## 2. Setting and equipment

* The same laptop for every session, with the frozen study build of AdaptiTune. Each session file records the build's commit.
* **(Decided here.)** The same closed-back headphones and the same microphone for every session and every participant. The volume is set to a comfortable level in session 1 and written down, then kept the same in later sessions.
* A quiet room. Sessions are at least one day apart (proposal: "five sessions on different days"). **(Decided here.)** All five should fall within about four weeks.

## 3. A session, step by step

A session takes at most 20 minutes (proposal 5.7). The researcher follows [study/session_script.md](study/session_script.md). In the app, under **Study → Run a study session**:

1. **Setup (researcher).** The researcher types the participant's code, and the app shows which session is next ("Session 3 of 5"). The researcher ticks the consent box and records or imports **three takes**: short phrases of 10 to 20 seconds, sung without accompaniment. The key is detected, or set when the participant knows it. **Start session** stays disabled until there is a valid code, consent and exactly three takes.
2. **Twelve blind versions (participant).** For each version the participant presses **Play this version** at least once. They can play it again, and can hear their own original take as often as they like. Then they answer two questions: *"Would you keep this version of your take?"* (Keep it / Discard it) and *"How natural does it sound?"* (1 = very unnatural, 5 = completely natural). **Next** stays disabled until the version has been played and both questions are answered. Nothing on screen says which condition a version comes from.
3. **Break.** After the sixth version, a break page asks the participant to rest. They press **Continue** when ready.
4. **Hand-tuning task (participant, up to 5 minutes).** On the first take of the session, the participant moves the four sliders (strength, retune speed, snap to key, vibrato kept). **Apply and play** lets them hear the result. When they are happy they press **I'm happy with this setting**, which becomes available after the first Apply, and rate their own version from 1 to 5. The clock stops at "I'm happy", or at 5:00, when the current setting is taken.
5. **End.** The app writes the session's files and shows the researcher a short summary: flagged judgments, agreement on repeats, and minutes taken.

## 4. Conditions

All versions are made from the same takes with the same tracker (FCPE) and the same shifter. Only the way the settings are chosen changes.

| Condition | Settings | Where they come from |
|---|---|---|
| **C1 Fixed** | strength 1.0, retune 5 ms, snap to key, vibrato kept 0 | `FIXED_CLASSIC`: classic hard auto-tune |
| **C2 Default** | strength 0.75, retune 44.7 ms, snap to key, vibrato kept 2/3 | `DEFAULT`: our pipeline's default |
| **C3 Personalized** | chosen by the agent (Thompson sampling) for this take, from the 504-setting grid | The agent, which starts from the population part and learns from this singer's C3 judgments (section 6) |
| **C4 Manual** | the setting this singer saved in the hand-tuning task of the **previous** session | From session 3 on, once a setting has been saved (proposal 5.7) |

## 5. Trials in a session

| Sessions | Unique versions | Repeats | Judged |
|---|---|---|---|
| 1–2 | C1 ×3, C2 ×3, C3 ×4 | 2 | 12 |
| 3–5 | C1 ×2, C2 ×2, C3 ×4, C4 ×2 | 2 | 12 |

* **Takes.** Each condition cycles through the session's three takes in its own random order, so no condition hears one take twice while another of its takes is never used.
* **Order.** Counterbalanced across participants: the order of the conditions is rotated by the participant's number and the session (a rotated Latin square). The conditions are then shuffled within the two halves of the session, so every condition appears both before and after the break. The participant's number is the order of first registration in the app.
* **Repeats.** Two versions from the first half are played again in the second half. The audio is identical, not a new rendering (about one judgment in six, proposal 5.7).
* **Seeds.** Each schedule comes from a fixed hash of the code and session, so it can be rebuilt exactly.
* **C4 without a saved setting.** The app does not let a session finish without the hand-tuning task, so every participant has a saved setting by session 3. If one were missing, the runner would give that session the mix of sessions 1–2.

## 6. What the agent learns from

* **Only the keep or discard answers on C3 versions teach the agent, and only on their first hearing.** That gives 4 judgments per session and 20 over the study, the "roughly 20 real interactions" that O4 tests in simulation. In O4's sensitivity table, an agent that gets keep or discard only, with no ratings or A/B picks, still reaches 90.4% of the oracle.
* **Repeats do not teach it,** so the consistency check does not count one judgment twice.
* **The naturalness ratings do not teach it (decided here).** They are the study's outcome for the non-inferiority test, and they ask how natural a version sounds, not whether the singer likes it.
* **The hand-tuned setting does not teach it (decided here).** C3 against C4 compares what the agent learns from judgments alone with what the singer reaches by hand. If the agent copied the saved setting, C3 would borrow the singer's own tuning time, and part 3 of the criterion would mean nothing.
* **C1, C2 and C4 judgments do not teach it,** because they are not the agent's choices (proposal: "the agent learns from every judgment that involves its output").
* **Prior.** The agent starts from the population part shipped with the app. If the pilot (task S5) gives data first, the population part is retrained on it before the study, and the study build records which model it uses.
* **Waiting.** The agent chooses each C3 version only after it has learned from every earlier C3 judgment in the session. The app prepares the next version while the current one is judged, unless that next version has to wait for the agent.

## 7. Consistency

If a participant's agreement on the two repeats in a session drops below 70%, the rest of that session's judgments are flagged, from the first disagreeing repeat on, and analysed separately (proposal 5.7). With two repeats, agreement can only be 0%, 50% or 100%, so a single disagreement flags the rest of the session. Flagged judgments stay in the data with `flagged = True`. The primary analysis leaves them out, and a second analysis keeps them in.

## 8. What is recorded

Everything is written to the study folder on the laptop: `~/.adaptitune/study/`, or `<ADAPTITUNE_HOME>/../study/` if `ADAPTITUNE_HOME` is set. Files are named by code, never by name.

| File | Contents |
|---|---|
| `sessions/P07_s3_trials.csv` | One row per judged version: condition, take, position, whether it is a repeat, keep, naturalness, flagged, seconds from first play to answer, the four settings, and the version's in-key accuracy, naturalness score, vibrato penalty, and median pitch error before and after correction. Also whether the judgment taught the agent, and how many judgments the agent had learned from |
| `sessions/P07_s3_manual.csv` | The hand-tuning task: seconds (capped at 300), slider adjustments, Apply presses, the rating, and the saved setting |
| `sessions/P07_s3.json` | Session summary: protocol version, build, start time, length, takes and keys, the C4 setting, agreement on repeats, flagged judgments, agent judgments before and after |
| `takes/P07_s3_t1.wav` … | The three takes (16-bit WAV) |
| `profiles.sqlite` | The agent's state for each code, and the saved manual settings |
| `participants.json` | Code → number, for the counterbalancing |

A session written as it runs is `…_trials.partial.csv`. Only finished sessions are exported and analysed. **Study → Export study data…** writes `trials.csv`, `manual.csv` and `sessions.csv` for `adaptitune study-analyze`. **Study → Withdraw a participant…** deletes that code's session files, takes and agent state (section 11).

## 9. Analysis

Run `adaptitune study-analyze trials.csv --manual manual.csv` (`adaptitune.study.analysis.analyze`). Everything below is fixed now.

**Primary (the O6 criterion): sessions 3–5, unflagged judgments.**

| Part | Measure | Met when |
|---|---|---|
| 1 | Acceptance C3 − C1: the mean over singers of each singer's difference in keep rate, with a 95% CI from 2,000 bootstrap resamples of singers | estimate ≥ 0.15 |
| 2 | Acceptance C4 − C3, computed the same way | estimate ≤ 0.10 |
| 3 | Time to a kept take. With the agent: the seconds from first play to answer on C3 versions, divided by the number kept. A singer who kept none counts as having judged them all and then tuned by hand. By hand: the tuning task's seconds. Paired by singer, with a bootstrap CI | mean saving > 0 |
| 4 | Naturalness C3 against C1: each singer's mean rating difference, shifted by the 0.5 margin, in a one-sided Wilcoxon signed-rank test | p < 0.05 |

O6 is met when all four parts hold. Each estimate is reported with its 95% CI.

**Exploratory.**

* A mixed-effects logistic model of keep with condition and position in the session as fixed effects and a random intercept per singer (variational Bayes, statsmodels), with Holm correction across the three contrasts against C1.
* C3 − C2.
* Acceptance by session for each condition, as the agent's learning curve.
* The distance from the agent's C3 settings to the singer's saved setting, by session (proposal 5.7: "a personal target").
* The naturalness test on all five sessions.
* Parts 1 and 2 with flagged judgments included.
* The objective metrics of every version by condition: pitch error before and after, in-key accuracy, naturalness score and vibrato penalty.

**Missing data.** A singer who misses sessions is analysed on the sessions they finished. A singer counts towards a difference when they have judgments in both of its conditions in sessions 3–5. A session stopped before the end is not analysed. The agent keeps what it learned from the judgments made before the stop, because those were real judgments. That session is run again on another day with new takes, and the gap is noted in the report.

**Power.** The power simulation (`adaptitune.study.analysis.simulate_study`, in [RESULTS.md](RESULTS.md#o6-user-study-dry-run-and-power)) gives 0.50 power at 10 singers and 0.62 at 15 for a 15-point difference. That is why O6 is judged on estimates with confidence intervals.

## 10. Dry run

`scripts/o6_dry_run.py` runs the whole study through the same runner the app uses, with 15 simulated participants. Each is one MIR-1K singer, whose real takes are corrected by the real pipeline. Keep or discard comes from the O4 simulator, with drift between sessions and fatigue within each. The ratings follow the measured naturalness of each version. The hand-tuned setting lands near the simulated singer's ideal. The results are in [RESULTS.md](RESULTS.md#o6-user-study-dry-run-and-power) and `docs/results/o6/dry_run/`. They test the code and the analysis, and say nothing about people.

## 11. Ethics and data protection

* One ethics application covers every activity that involves people (proposal 5.7, task S3, February 2027): the real recordings, the pilot, the usability test with three users, this study and the listening test.
* Written informed consent. Anyone can stop a session or withdraw at any time without giving a reason.
* Recordings and answers are kept under codes, on project machines only, and never uploaded. The app makes no network connections (O5, NFR1). Details: [study/data_management.md](study/data_management.md).
* **Withdrawal.** **Study → Withdraw a participant…** deletes the participant's session files, takes and agent state. The profile database is rewritten, so nothing stays behind in its free space. The code stays in `participants.json` so the counterbalancing of later participants does not change. The code holds no personal data. Copies already exported for analysis are deleted as well, up to the date in the information sheet after which results can no longer be withdrawn.
* **Risks** are small: listening fatigue, which the 20-minute limit, the break and the 12-version limit address, and singing in front of a researcher. Sessions stop if the participant wishes.

## 12. Before the first session

1. Ethics approval (S3).
2. The usability test with three users (A5, September 2027), and fixes.
3. Freeze the study build. Record its commit and the population model in the study log.
4. Run Help → Check this installation and Help → Speed test on the study laptop.
5. Run a practice session with a team member under a test code such as T1, then withdraw T1.

## Change log

| Date | Change | Why |
|---|---|---|
| 4 Nov 2026 | First version, written before anyone takes part | — |
