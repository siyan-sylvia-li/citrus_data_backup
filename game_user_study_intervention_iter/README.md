# Intervention iteration

A copy of `game_user_study_phase_2` set up to iterate on the intervention **form**, not
just on the material inside one form. Two differences from phase 2:

1. **Assisted round only, by default.** The two unassisted transfer puzzles are dropped, so
   a session is ~3 minutes and one survey shorter. There is therefore **no transfer
   outcome** — the readable endpoint is the manipulation check (do the dialogue acts move?).
   Set `OTH_ASSISTED_ONLY=0` for the full three-round protocol.
2. **Pluggable intervention forms.** `/intervention` is form-agnostic. A form declares its
   own page, fields and record; the route renders, validates, effort-gates and writes. See
   `interventions/__init__.py` for the contract.
3. **No filler.** The vanilla arm has no pre-task step: it goes from the prompt task
   straight to the primer. So vanilla writes no pre-task record, is not effort-screened,
   and clears one filter where scaffolded clears two.

Everything else — recruitment, consent, eligibility, the prompt task, randomisation, the
primer, the coach, the surveys — is phase 2 unchanged.

## Running

Forms are **randomised per participant** across `LIVE_FORMS`, assigned once on first
visit and stored in the session.

```bash
./run_local.sh                                       # randomise over all forms but skeleton
INTERVENTION_FORM=johnny_connect_four_boost ./run_local.sh          # pin ONE form
LIVE_FORMS=contrasting_cases,johnny_connect_four_boost ./run_local.sh  # randomise over a subset
INTERVENTION_VERSION=v9_two_rows ./run_local.sh      # a different TABLE, same form
OTH_ASSISTED_ONLY=0 ./run_local.sh                   # include the transfer puzzles
```

`INTERVENTION_FORM` pins one form and wins over `LIVE_FORMS`. An unknown name fails at
import, not at the participant's first page load, and the resolved set is printed at
startup as `[study] intervention forms in play: [...]` — check that line if you are not
seeing the form you expect.

`skeleton` is excluded from the default set: it is a copy-me template with placeholder
copy and must never reach a real participant. Pin it explicitly to test the plumbing.

## Forms shipped

| id | shape | correction step | effort-gated |
|---|---|---|---|
| `contrasting_cases` | phase 2's form, ported unchanged: A/B tables → observation → forced choice | yes | yes |
| `johnny_connect_four_boost` | interactive: choose how Johnny talks to the AI, then see what he could reuse | built in | no — gated on the activity log |
| `skeleton` | placeholder: stated procedure → practice attempt, no forced choice | no | yes |

`skeleton` exists so the plumbing can be exercised before any real copy is written. Its
content is placeholder text and it is registered only for that purpose.

## Adding a form

1. Copy `interventions/forms/_skeleton.py`.
2. Write `templates/forms/<your_form>.html`. Post to `url_for('intervention')` — the route
   name is stable across forms, so the template does not change when the form is swapped.
3. Register the id in `REGISTRY` in `interventions/__init__.py`.

The module needs `ID`, `TEMPLATE`, `FEEDBACK_TEMPLATE`, `TEXT_FIELD`, `JUDGE_TASK`,
`context()` and `validate(form, min_chars)`, plus `feedback_context(record)` if it has a
correction step. `validate` owns every rule the form needs and returns the fields to
record; the route adds `form`, `effort` and `submitted_at`.

Never reuse an `ID`, and keep retired forms in `REGISTRY` — a form that ran on real
participants has to stay resolvable so its data can be read back.

## What the arms differ by

Scaffolded: prompt task → intervention form (+ effort screen) → primer → game.
Vanilla: prompt task → primer → game.

So the arms differ in time-on-task and in number of screens passed, by design. That is a
deliberate choice for an iteration build whose endpoint is the manipulation check; it is
not the phase-2 design, where a length-matched filler kept the screening symmetric.

## Analysis

The phase-2 notebooks are deliberately not copied here. `phase2_analysis.ipynb` defines
`analysed` as requiring both solo rounds, which under `OTH_ASSISTED_ONLY=1` is empty by
construction — an iteration run needs a filter on `n_turns > 0` plus the demographics and
carry-over conditions, without the margin requirement.

Arm is read from the presence of `intervention.json`, so vanilla participants are now
identified by its ABSENCE rather than by a `filler.json` — any loader copied from phase 2
needs that change, or it will label every vanilla participant `None`.

`annotate_phase2.py` works unchanged and is what produces the participant-side acts the
manipulation check reads. `../bad_user_sim/phase2_assistant_fine.py` does the
assistant side; point its `REC` at this directory's recordings.

`score_from_logs.py` still runs; its `transfer_*` columns come out empty.
