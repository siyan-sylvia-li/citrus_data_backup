# Intervention iteration — working notes

State of the Othello arm of CITRUS as of 2026-08-21. Covers what the phase-2 and
3-arm data say, the intervention forms built so far, and the infrastructure and
measurement traps that produced wrong answers at least once.

Scope note: everything here is about the **non-adopter** population unless a row says
otherwise, and every number is conditional on the assistant being **gpt-5.5** — coach
model changes assistant-side act behaviour a lot.

---

## 1. The headline: no intervention lifts transfer

Every form raises the *count* of the taught behaviour and none of them moves the
unassisted outcome. The manipulation check passes; the endpoint does not.

Analysis set throughout: `~carryover & reached_demographics`. `solo margin` is the
**summed** final margin over the two unassisted rounds (R2+R3), participants who
completed both only.

| condition | population | n | R1 optimal % | R1 margin | R1 solved | n solo | solo margin |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline_adopters | adopters | 100 | 43.2 | −21.0 | 0/100 | 100 | −15.3 |
| vanilla_adopters | adopters | 111 | 69.6 | −8.4 | 21/111 | 111 | −13.4 |
| baseline_non_adopters | non-adopters | 50 | 35.4 | −22.2 | 0/50 | 50 | −18.4 |
| vanilla_non_adopters | non-adopters | 46 | 69.4 | −8.0 | 8/46 | 46 | −14.0 |
| intervention_v2 (v2_think_pair) | non-adopters | 16 | 72.2 | −7.5 | 3/16 | 16 | −18.6 |
| intervention_v9 (v9_two_rows) | non-adopters | 15 | 71.1 | −4.5 | 6/15 | 15 | −14.5 |
| intervention_johnny | non-adopters | 34 | 75.5 | −4.4 | 11/34 | 13 | −14.3 |
| intervention_modelled_ta | non-adopters | 15 | 61.3 | −12.4 | 2/15 | 0 | — |
| intervention_esr | non-adopters | 14 | 67.1 | −13.8 | 1/14 | 0 | — |

Reading it:

- **The assistant is the only thing that moves R1.** Baseline → vanilla is +26 points
  in adopters and +34 in non-adopters. Every intervention sits within a few points of
  its own vanilla.
- **The assistant erases the ability gap.** Unaided, non-adopters are only borderline
  weaker than adopters (35.4 vs 43.2, p ≈ .051). With the assistant, 69.4 vs 69.6 —
  gone.
- **Transfer margins are flat across arms.** In the concurrent 3-arm batch, johnny beat
  vanilla on R1 (80.4 vs 70.2) and the transfer margins were identical (−14.3 vs −14.1).
  This is the cleanest within-window comparison available, because both arms were
  recruited in the same 70 minutes.
- johnny's earlier R1 advantage shrank as n grew (75.4 → 71.2 → 72.5 over batches);
  treat single-batch R1 gaps as noise until replicated.

## 2. Coached vs spontaneous Think Aloud — the dissociation

Replicates in three samples. Spontaneous Think Aloud predicts transfer; *taught*
Think Aloud does not, even though the teaching demonstrably raises its frequency.

| | sample 1 | sample 2 | sample 3 |
|---|---:|---:|---:|
| spontaneous (vanilla arms) | +0.356 | +0.210 | +0.167 |
| coached (scaffolded arms) | −0.084 | +0.024 | — |

The reading the data supports: the intervention teaches the **announcement** and not
the **reasoning**. That is a production-vs-mediation story — you can install a
behaviour whose underlying cognition the participant cannot supply — not a compliance
failure. Participants do comply.

Two things this is *not*:

- It is **not** about reason clauses. Bare think-aloud utterances predict too, so
  narrowing the act to require a justification is falsified — do not re-open it.
- It is **not** visible in the assistant's acts. Phase-1 tutor-move acts predict
  nothing; they mirror participant help-seeking (r = .58 SR ↔ ProvideCorrectAnswer).

Reason-clause **rate** does differ by form, and the two contrasting-cases variants are
the worst: v2 8%, v9 11%, against ≥17% for modelled_TA, esr and plain vanilla.

## 3. The transfer outcome has very low reliability

This caps everything above and is the most important methodological fact in the file.

- R2–R3 correlation: r = .076 – .149 across samples.
- Composite (Spearman-Brown) reliability: 0.07 – 0.26.
- Attenuation ceiling is √reliability, so a true effect of any size is mostly
  unmeasurable at these n's.

Concrete power figure, pooled non-adopter vanilla (n = 46): Think Aloud vs solo margin
is rho = +0.198, p = .186. Smallest detectable |rho| at that n is 0.291; **n ≈ 195**
for 80% power. A null transfer result at n = 15 per arm carries almost no information.

The margin distribution is bimodal — participants who finish the position land at the
ceiling (+18 in R1, +16 across both solo rounds) and the rest cluster near −20 — so a
condition mean is close to a re-encoding of its solve rate. Do not median-split it.

## 4. Where the arms *do* differ

Behavioural profiles separate; outcomes do not. Checked and found null or negligible:

- **AI-assistant helpfulness Likert items** — no arm differences.
- **Optimal-move rate** — scaffolded marginally higher, not significant.
- **Per-decision latency** — flat. An earlier `s1_sec_per_move` result (p = .044) was
  a measurement artefact: it counted inter-move gaps only and included forced moves.
  The related "slower predicts better margin" claim was also wrong in sign and is
  adopter-specific anyway (−0.218, not +0.29).
- **Ability covariates** (gate correctness, skill rating) — scaffolded participants are
  not detectably weaker; the gate control does not rescue any arm difference.

## 5. Dialogue acts and elicitation

Scheme: AutoTutor-derived, 6 participant acts and 8 fine assistant acts. Annotation is
a 3-seat LLM panel (gpt / llama / sonnet) with a majority re-vote.

- Reportable set = fine acts clearing κ ≥ .6, and it differs by source: **Hint** fails
  on the simulator only; **Board Report** is weak in Connect Four.
- Elicitation ORs are cluster-robust logistic regressions with a
  `SEPARATION_LOGIT = 5.0` guard; separated cells print `sep` rather than a number.
- ORs near a ceiling are **non-collapsible** — a pooled OR is not the average of the
  stratum ORs, which is why Think Aloud → Board Report can land on opposite sides of 1
  in pooled vs split tables. Report the split.
- Moderation claims need an explicit interaction test, not two subgroup ORs.

Style and length, adopters vs non-adopters: compared with StyleDistance
content-independent embeddings, within-group vs cross-group pairwise similarity, tested
by participant-level permutation with same-participant pairs excluded
(`cross_condition_scores/style_similarity.py`).

## 6. Intervention forms

All live in `game_user_study_intervention_iter/interventions/forms/`, one module per
form (`_skeleton.py` is the contract). Selected at deploy time by `INTERVENTION_FORM`
(pin one) or `LIVE_FORMS` (randomise among several).

| form | what the participant does |
|---|---|
| `contrasting_cases` | forced choice between two messages; version selects the table (`INTERVENTION_VERSION`) |
| `johnny_connect_four_boost` | animated Connect Four narrative; watch two ways of asking play out |
| `johnny_write` | same frame, but the participant writes the message (authored examples, not participant text) |
| `modelled_think_aloud` | worked model of thinking aloud |
| `principles_first` | participant states the abstract principles that would improve a weak message |
| `elicit_select_reply` | elicit → pick from candidates → see the assistant's reply; set in **chess** so the game reads as hard |
| `rewrite_to_pass` | rewrite a weak message until it passes |

Standings on the manipulation check are in memory (`intervention-form-standings`) —
v9 leads; don't re-derive. On reason rate specifically the ranking inverts (§2).

Design commitment, settled: this is a **boost, not a nudge**. It runs pre-task only,
with nothing changed during interaction, so free adoption is what R1 measures. Do not
propose extra rounds or decision-time architecture — participant time and budget are
limited.

## 7. Infrastructure

`game_user_study_intervention_iter/` is a copy of `game_user_study_phase_2` with the
pluggable form registry. Flask on Cloud Run, gunicorn, participant data on a
gcsfuse-mounted GCS bucket, one bucket per `SERVICE`.

Arm assignment (`app.py`): `ARMS = (["vanilla"] if INCLUDE_VANILLA else []) + LIVE_FORMS`,
uniform random per session, recorded in `intervention.json` (`form`, `variant`).
Vanilla goes straight to the game — the `julie_advice` filler is gone and arm durations
are deliberately unmatched.

Deploy knobs that matter: `SERVICE`, `INTERVENTION_FORM`, `INTERVENTION_VERSION`,
`LIVE_FORMS`, `INCLUDE_VANILLA`, `OTH_ASSISTED_ONLY`, `OTH_TIME_LIMIT_SECONDS` (480),
`OTH_TIME_LIMIT_TRANSFER` (90), `OPENAI_MODEL` (gpt-5.5).

Four deployment bugs, all of which produced silently wrong runs:

1. **Comma in a `--set-env-vars` value.** gcloud parses the value as a dict. A `^##^`
   alternate delimiter set *zero* env vars and the worker died on a missing
   `OPENAI_API_KEY` (503). Fix: `LIVE_FORMS` is semicolon-separated and the app parser
   accepts either.
2. **`${VAR:-default}` fires on empty**, so `INTERVENTION_FORM=` could not clear the
   pin — a 3-arm run silently became 2 arms (17 vanilla / 13 johnny / **0
   v9_reasons**). Fix: `${VAR-default}` (single dash) plus a `SystemExit` guard in
   `app.py` when a pin and a multi-form `LIVE_FORMS` are both set.
3. **`OTH_ASSISTED_ONLY=0` did nothing** because the two transfer rounds had been
   deleted from the `OTH_ROUNDS` literal. Restored verbatim from phase_2.
4. **Route defaults vs form context** collided (`render_template() got multiple values
   for 'min_chars'`). Fixed by applying route defaults first, then `form.context()`.

## 8. Data layout

Per-study recordings live under `game_user_study_*/recordings-*`. `cross_condition_scores/`
turns them into one long DataFrame, one row per participant-round:

- `build_scores.py` — `build(write=True)` writes `scores_<condition>.csv` per condition
  plus `scores_all.csv`. Reuses `game_user_study_phase_2/score_from_logs.py` for the
  scoring, and overrides the `ai` flag from the condition definition (the baselines
  played R1 with no assistant, which every copy of the scorer hardcodes as assisted).
- No exclusions are baked in. `carryover`, `reached_demographics`, `prompt_*`,
  `effort_median`, `consented_at`, `pretask_at`, `form`, `variant` are carried so each
  analysis applies its own.
- **`POOL_INTO`** pools separately-collected batches into one analysis condition —
  currently `3arm_vanilla → vanilla_non_adopters` and `3arm_johnny → intervention_johnny`.
  `condition` is the analysis key; the new **`batch`** column keeps the original, so any
  pooling can be split back apart. Pooled vanilla = 46 participants (29 + 17); pooled
  johnny = 34 R1 rounds (21 + 13) but only 13 transfer blocks, because the earlier
  johnny batch ran assisted-only.
- `elicitation.py` / `elicitation_plot.py` — turn-level acts, prevalence, OR tables,
  forest plots. `style_similarity.py` — StyleDistance embeddings and the permutation test.
- Notebooks: `scores_adopter_non_adopter.ipynb` (outcomes),
  `elicitation_odds_ratios.ipynb` (acts).
- Assistant-side fine acts are a separate pass:
  `bad_user_sim/phase2_assistant_fine.py --root <recordings dir>`.

## 9. Traps that have already cost time

- **A stale kernel module clobbers hand-edited CSVs.** `build()` writes one CSV per
  condition it knows about, so a notebook holding a pre-pooling copy of `build_scores`
  rewrites them from the old condition list. Both notebooks now `importlib.reload`
  before calling in.
- **String-compared timestamps are open-ended.** `consented_at >= "2026-08-20T18"`
  also matches all of 2026-08-21, so the late-johnny exclusion would have swallowed the
  whole 3-arm batch once it was pooled in. That mask is now scoped on `batch`.
- **`DROP_LATE_JOHNNY` is correlated with the outcome** and is applied asymmetrically
  (every other condition also sits in one narrow recruitment window and none are
  dropped). Keep `False` as primary; `True` is a stated sensitivity check only.
- **Restricting a sample by turn count biases a performance measure.** An over-restriction
  to ≥4 turns produced a spurious rho of −0.675 where the honest figure on all 15
  participants is −0.101.
- Johnny's disc-drop animation originally implied the reasoning option had no effect;
  the board animation was removed and the motion moved to `flyToBackpack`.

## 10. Open

- **v9 solo deploy.** `INTERVENTION_FORM=contrasting_cases INTERVENTION_VERSION=v9_reasons
  OTH_ASSISTED_ONLY=0 ./deploy.sh`; add `INCLUDE_VANILLA=0` for no concurrent control.
  Recommendation is to keep vanilla in — every previous v9 estimate lacked a
  same-window control.
- `3arm_v9_reasons` has **0 participants**. `v9_reasons` is the same tables as
  `v9_two_rows` with a reworded correction sentence; it is a separate key so the wording
  change is recorded, and it is deliberately *not* pooled into `intervention_v9`.
- Reusing `SERVICE=game-study-3arm` keeps arms accumulating in one bucket; windows stay
  separable via `consented_at`. A fresh `SERVICE` gives a clean bucket instead.
- Not done: a third forest plot restricted to `condition == "vanilla"` (39 non-adopter
  participants, no pre-task step) for a like-for-like population comparison.
- The paper's headline is the planned 3-arm baseline/vanilla/scaffolded study in the
  phase-2 format. The Othello baseline notebook is preliminary — don't try to rescue its
  significance.
