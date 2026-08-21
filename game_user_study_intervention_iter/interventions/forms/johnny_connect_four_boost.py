"""Interactive boost: the participant chooses how Johnny talks to the AI.

A structurally different FORM from contrasting_cases -- the participant makes choices
inside a simulated interaction and then sees whether Johnny could reuse what he got,
rather than reading two transcripts and writing an observation. The activity is
self-contained in the template; this module only defines what is recorded and what counts
as having done it.

The activity's own instrumentation is the seam: `logEvent` accumulates to
`window.johnnyBoostLog`, and the completion button dispatches `johnnyBoostComplete` with
`{log, knowledge, transferSuccess}`. The template's integration glue posts that JSON here.

TWO THINGS TO DECIDE BEFORE THIS RUNS ON PARTICIPANTS

1. **The gate is a different KIND.** There is no free text, so the LLM effort judge cannot
   apply. `validate` gates on the interaction log instead: the activity must be completed
   and every assisted round must have a recorded choice. That is objective and needs no
   judge, but it is not the same instrument the filler arm passes through -- so either pair
   this with a filler gated the same way, or accept that the arms are screened differently
   and say so. See the symmetry note in interventions/__init__.py.

2. **Length matching.** `julie_advice` was matched to contrasting_cases, not to this. Time
   this activity before pairing it with a filler.
"""
from __future__ import annotations

import json

ID = "johnny_connect_four_boost"
TEMPLATE = "forms/johnny_connect_four_boost.html"
FEEDBACK_TEMPLATE = None          # the activity delivers its own outcome screen
TEXT_FIELD = None                 # no free text -> NOT effort-gated, see above
JUDGE_TASK = None

# Assisted rounds in the activity. Kept here so `validate` can check the participant
# actually made a choice in each one rather than clicking through.
N_ROUNDS = 2


def context():
    return {}


def validate(form, min_chars):
    """Accept the activity's event log, and require evidence it was worked through.

    The log is the record: it carries which option was chosen in each round, the order the
    options were presented in (they are shuffled), what was added to the backpack, and
    whether the independent-phase choice matched. That is the behavioural measure this
    form produces, so it is stored whole rather than summarised.
    """
    raw = form.get("payload") or ""
    try:
        detail = json.loads(raw)
    except (ValueError, TypeError):
        return {}, ("Something went wrong recording the activity. Please reload the page "
                    "and work through it again.")

    log = detail.get("log") or []
    events = [row.get("event") for row in log]
    if "completed" not in events:
        return {}, "Please finish the activity before continuing."

    choices = [row for row in log if row.get("event") == "johnny_choice"]
    rounds_answered = {row.get("round") for row in choices}
    if len(rounds_answered) < N_ROUNDS:
        return {}, ("Please make a choice in each of Johnny's turns before continuing.")

    independent = next((row for row in log if row.get("event") == "independent_choice"), {})
    return {"log": log,
            "knowledge": detail.get("knowledge") or [],
            "transfer_success": detail.get("transferSuccess"),
            "n_restarts": events.count("restarted"),
            "choice_types": [row.get("choice_type") for row in choices],
            "independent_choice": independent.get("selected"),
            "independent_target": independent.get("target")}, None
