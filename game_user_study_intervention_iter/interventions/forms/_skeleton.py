"""Template for a new intervention FORM. Copy, rename, register in interventions/__init__.

The CONTENT below is placeholder. Every string a participant reads is the measurement
instrument, so the wording is the study owner's to write -- this file only shows where it
goes and which hooks the route calls.

This skeleton is a worked example of the shape that phase 2's form could NOT express: a
stated procedure plus a practice attempt, with no forced choice and no correction step.
It is registered so `INTERVENTION_FORM=skeleton python app.py` runs end to end and the
plumbing can be checked before any real copy exists.
"""
from __future__ import annotations

ID = "skeleton"
TEMPLATE = "forms/skeleton.html"
FEEDBACK_TEMPLATE = None          # set a template path to add a correction step
TEXT_FIELD = "attempt"            # prefilled on error, and what the effort judge scores
JUDGE_TASK = None                 # None = the judge's default observation rubric

# PLACEHOLDER CONTENT -- replace. Kept as data rather than baked into the template so a
# variant is a change here, not a change to markup.
STEPS = [
    "PLACEHOLDER: first step of the procedure the participant should follow.",
    "PLACEHOLDER: second step.",
    "PLACEHOLDER: third step.",
]
PRACTICE_PROMPT = ("PLACEHOLDER: the situation the participant writes a response to, so "
                   "they execute the procedure once before the game.")


def context():
    """Everything the template needs. Called on GET and on a validation error."""
    return {"steps": STEPS, "practice_prompt": PRACTICE_PROMPT}


def validate(form, min_chars):
    """Parse and check the POST. Return (record_to_write, error_message_or_None).

    Return the fields you want in the participant's json. The route adds `form`, `effort`
    and `submitted_at`, so do not set those here.
    """
    attempt = (form.get(TEXT_FIELD) or "").strip()
    if len(attempt) < min_chars:
        return {TEXT_FIELD: attempt}, f"Please write at least {min_chars} characters."
    return {TEXT_FIELD: attempt, "n_steps_shown": len(STEPS)}, None
