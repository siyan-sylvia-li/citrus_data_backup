"""Pluggable intervention FORMS.

Phase 2 hard-wired one form -- contrasting cases, free-text observation, forced choice,
correction -- into the /intervention route. Iterating on the form itself (a worked example,
a stated procedure, a video, a quiz) meant editing the route, which made the arms hard to
keep symmetric and made "which form did this participant see" a question about git history
rather than about the data.

Here a form is a module that declares its own page, its own fields and its own record. The
route is form-agnostic: it renders what the form names, validates with what the form
provides, applies the SHARED effort gate, and writes whatever the form returns.

    INTERVENTION_FORM=contrasting_cases python app.py      # pin one form
    LIVE_FORMS=contrasting_cases,johnny_connect_four_boost python app.py   # randomise

Only the scaffolded arm has a pre-task step here. Vanilla goes from the prompt task
straight to the primer, so it writes no pre-task record and is not effort-screened.

**Adding a form.** Copy `forms/_skeleton.py`, write the template, register it below. The
module needs:

    ID                  str, recorded in the participant's json -- never reuse one
    TEMPLATE            path under templates/
    FEEDBACK_TEMPLATE   path, or None for a form with no correction step
    TEXT_FIELD          name of the free-text input, or None if the form has none
    JUDGE_TASK          rubric passed to the effort judge (None = its default rubric)
    context()           -> dict passed to the template
    validate(form, min_chars) -> (record: dict, error: str|None)
    feedback_context(r) -> dict passed to FEEDBACK_TEMPLATE   (only if FEEDBACK_TEMPLATE)

`validate` receives the raw POST and owns every rule the form needs. It returns the fields
to record, so a form that collects three things records three things without the route
knowing what they are.

TEXT_FIELD is what the route prefills on a validation error and what the effort judge
scores. A form with TEXT_FIELD = None takes no free text and so is not effort-gated; gate
it in `validate` instead if the form has something objective to check.
"""
from __future__ import annotations

import importlib
import os

# id -> module path. Keep retired forms listed; a form that ran on real participants
# must stay resolvable so its data can be replayed.
REGISTRY = {
    "contrasting_cases": "interventions.forms.contrasting_cases",
    "johnny_connect_four_boost": "interventions.forms.johnny_connect_four_boost",
    "elicit_select_reply": "interventions.forms.elicit_select_reply",
    "johnny_write": "interventions.forms.johnny_write",
    "modelled_think_aloud": "interventions.forms.modelled_think_aloud",
    "principles_first": "interventions.forms.principles_first",
    "rewrite_to_pass": "interventions.forms.rewrite_to_pass",
    "skeleton": "interventions.forms._skeleton",
}



def load(form_id: str):
    if form_id not in REGISTRY:
        raise SystemExit(f"unknown form {form_id!r}; expected one of {sorted(REGISTRY)}")
    mod = importlib.import_module(REGISTRY[form_id])
    for attr in ("ID", "TEMPLATE", "FEEDBACK_TEMPLATE", "TEXT_FIELD", "JUDGE_TASK",
                 "context", "validate"):
        if not hasattr(mod, attr):
            raise SystemExit(f"form {form_id!r} is missing {attr}")
    if mod.ID != form_id:
        raise SystemExit(f"form {form_id!r} declares ID {mod.ID!r}; they must match")
    if mod.FEEDBACK_TEMPLATE and not hasattr(mod, "feedback_context"):
        raise SystemExit(f"form {form_id!r} has a FEEDBACK_TEMPLATE but no "
                         f"feedback_context()")
    return mod


# NOTE: form selection lives in app.py (INTERVENTION_FORM / LIVE_FORMS). There is
# deliberately no accessor here -- an unused one silently shadowed the real selection once.
