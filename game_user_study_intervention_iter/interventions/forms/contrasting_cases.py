"""Phase 2's form, ported unchanged: contrasting cases -> observation -> forced choice.

Behaviour is deliberately identical to phase 2 so this form is the baseline any new form
is compared against. The variant tables live in `intervention_content.py`; changing which
table is shown is an INTERVENTION_VERSION change, not a form change.
"""
from __future__ import annotations

import os
import re

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


# How many words of each assistant reply a participant sees. The replies are stored in
# full in intervention_content, so this is purely a display setting: raise it to show more
# of what the assistant said, or set it to 0 to show the reply untruncated. It is a knob
# rather than a constant because how much assistant text to show is exactly the kind of
# thing worth varying -- too little and Group A's outsourcing has nothing to outsource TO,
# too much and the participants' own messages stop being what the eye lands on.
REPLY_WORDS = int(os.environ.get("INTERVENTION_REPLY_WORDS", "20"))

_MD = ((re.compile(r"\*\*(.+?)\*\*"), r"\1"),     # bold
       (re.compile(r"\*(.+?)\*"), r"\1"),           # italic
       (re.compile(r"^[\s>]*[-*]\s+", re.M), ""),   # list bullets
       (re.compile(r"`+"), ""))                     # code ticks


def shorten(reply, words=None):
    """Flatten one assistant reply to plain text and cut it to `words` words.

    Returns {"text": ..., "more": n} where `more` is how many words were cut, or 0 when
    the whole reply is shown. The count is rendered as "[n more words]" so the LENGTH of
    what each style pulled back is visible even though the body is cut: Group A's
    "Reply with only the column number" gets a one-word answer and no suffix at all,
    while a reasoned question pulls back paragraphs. That difference is part of the
    contrast, and truncating to a fixed width without it would hide the difference.

    The bubbles render as plain text, so stored markdown is stripped rather than shown
    literally. `words=0` disables the cut.
    """
    if not reply:
        return None
    text = reply
    for pattern, sub in _MD:
        text = pattern.sub(sub, text)
    parts = " ".join(text.split()).split()
    n = REPLY_WORDS if words is None else words
    if n <= 0 or len(parts) <= n:
        return {"text": " ".join(parts), "more": 0}
    return {"text": " ".join(parts[:n]) + " ...", "more": len(parts) - n}


def _turns(block):
    """One shape for the template: a list of (participant message, assistant reply).

    Variants differ in whether their material carries the assistant side. v9_reasons pairs
    each message with the reply it actually received; the earlier variants are message-only
    and render an ellipsis instead. Normalising here means the template does not branch and
    an older variant renders exactly as it did before.
    """
    out = []
    for t in block:
        message, reply = t if isinstance(t, (tuple, list)) else (t, None)
        out.append((message, shorten(reply)))
    return out


def context():
    pairs = [(_turns(a), _turns(b)) for a, b in PAIRS]
    return {"pairs": pairs, "variant": VERSION, "reply_words": REPLY_WORDS}


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
