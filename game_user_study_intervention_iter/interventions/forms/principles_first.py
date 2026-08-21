"""Elicit the principle, then reveal it. No board, no simulated dialogue.

Why this shape. The concrete forms (rewrite_to_pass, johnny_write) put the participant in a
position and asked them to compose about it, which brings two costs: they have to read a
board, and the assistant's reply has to be canned even though they can write anything, so a
mismatch is visible. This form drops both -- there is no position and no reply, only one real
message and a question about it.

Why elicit BEFORE revealing. Phase 2's effect came from the modelling, not the free-text
step: `choice_correct` predicted neither Think Aloud nor transfer, so comprehension of the
material was not what moved anything. But reveal-only is also not enough --
modelled_think_aloud showed three examples and no question, and produced Think Aloud 18.4%,
the highest knowledge-deficit rate of any form (29.5%) and the worst round-1 play. This form
is that same reveal with an attempt in front of it, so running the two against each other
isolates whether trying-before-being-told is what pure display lacked. Attempting first
helps even when the attempt is wrong.

The free text restores the EFFORT JUDGE (see TASK_PRINCIPLES in intervention_filter.py), so
this form is screened by the same instrument phase 2 used, rather than by a pattern check.

BY-PRODUCT: `principles` is what the participant already believed about using an assistant,
written before they were told anything -- a pre-task measure on the target construct.
"""
from __future__ import annotations

import os

from intervention_content import _B_REASON_MID
from intervention_filter import TASK_PRINCIPLES

ID = "principles_first"
TEMPLATE = "forms/principles_first.html"
FEEDBACK_TEMPLATE = "forms/principles_first_reveal.html"
TEXT_FIELD = "principles"
EFFORT_GATE = True                 # abstract text cannot be pattern-checked; judge it
JUDGE_TASK = TASK_PRINCIPLES

MIN_CHARS = int(os.environ.get("PRINCIPLES_MIN_CHARS", "50"))

# The message they are asked about. Verbatim from a phase-1 Connect Four participant.
START_MESSAGE = "What do you think would be the best move right now?"

# --- DRAFT COPY for the study owner to edit ------------------------------------
FRAMING = ("In a moment you will play a game with an AI assistant. What you get back from "
           "it depends on what you give it: a message that only asks for the answer gets "
           "you an answer, and nothing you can weigh up yourself.")

QUESTION = ("What would you change about how this person is using the assistant, so that "
            "they come away able to make this kind of decision themselves?")

# The reveal. Deliberately framed around what the message DOES TO THE ASSISTANT rather than
# around the people who sent it: modelled_think_aloud told participants they were looking at
# players who had succeeded, and that framing coincided with a jump in "what don't I know"
# questions rather than in narrating their own thinking.
REVEAL = ("Notice what that message leaves out: what the person was actually considering. "
          "When a message says what you are thinking of doing and what you think it would "
          "achieve, the assistant has something of your reasoning to respond to — it can "
          "confirm it, correct it, or point at what you missed. When it does not, there is "
          "nothing to respond to except the question, so all you get back is a move.")

# One verbatim example, as an existence proof that the principle is something people
# actually do. One rather than several: a list of examples makes this a display form, which
# is the thing modelled_think_aloud already tested.
EXAMPLE = _B_REASON_MID[0]
EXAMPLE_LABEL = "One participant put it like this:"


def context():
    return {"start_message": START_MESSAGE, "framing": FRAMING, "question": QUESTION,
            "min_chars": MIN_CHARS}


def validate(form, min_chars):
    text = (form.get(TEXT_FIELD) or "").strip()
    if len(text) < MIN_CHARS:
        return {TEXT_FIELD: text}, (f"Please write at least {MIN_CHARS} characters about "
                                    f"what you would change.")
    return {"principles": text, "start_message": START_MESSAGE,
            "n_chars": len(text)}, None


def feedback_context(record):
    return {"reveal": REVEAL, "example": EXAMPLE, "example_label": EXAMPLE_LABEL,
            "start_message": START_MESSAGE, "answer": record["principles"]}
