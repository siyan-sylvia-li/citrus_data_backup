"""Phase 2's form, ported unchanged: contrasting cases -> observation -> forced choice.

Behaviour is deliberately identical to phase 2 so this form is the baseline any new form
is compared against. The variant tables live in `intervention_content.py`; changing which
table is shown is an INTERVENTION_VERSION change, not a form change.
"""
from __future__ import annotations

import os

from intervention_content import INTERVENTION_VARIANTS

ID = "contrasting_cases"
TEMPLATE = "forms/contrasting_cases.html"
FEEDBACK_TEMPLATE = "forms/contrasting_cases_feedback.html"
TEXT_FIELD = "observation"
JUDGE_TASK = None                      # None = the judge's default observation rubric

VERSION = os.environ.get("INTERVENTION_VERSION", "v0_live")
if VERSION not in INTERVENTION_VARIANTS:
    raise SystemExit(f"unknown INTERVENTION_VERSION {VERSION!r}; "
                     f"expected one of {sorted(INTERVENTION_VARIANTS)}")
PAIRS = INTERVENTION_VARIANTS[VERSION]

# Which group the correction names. A property of the material, so it lives with it.
CORRECT_CHOICE = "B"


def context():
    return {"pairs": PAIRS, "variant": VERSION}


def validate(form, min_chars):
    """Free text first, then the forced choice -- the order the page asks in.

    The participant writes what they noticed BEFORE being asked which group did better,
    so the observation is their own reading rather than a rationalisation of the answer
    they just picked. Validating in the same order keeps the error messages consistent
    with that.
    """
    observation = (form.get("observation") or "").strip()
    choice = (form.get("choice") or "").strip().upper()
    if len(observation) < min_chars:
        return {"observation": observation}, (
            f"Please write at least {min_chars} characters about how the two groups differ.")
    if choice not in ("A", "B"):
        return {"observation": observation}, (
            "Please choose which group you think plays better.")
    return {"variant": VERSION,
            "observation": observation,
            "choice": choice,
            "choice_correct": choice == CORRECT_CHOICE}, None


def feedback_context(record):
    return {"correct": record["choice_correct"]}
