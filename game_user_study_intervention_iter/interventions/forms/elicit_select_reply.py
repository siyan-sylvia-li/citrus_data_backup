"""Elicit -> select -> see the assistant's actual reply.

This resolves the defect in the compose-it-yourself forms. There the assistant's reply had
to be canned while the participant could write anything, so a mismatch was visible exactly
when they had most invested in their own wording. Here the reply is keyed to a SELECTION, so
it always answers what was sent -- and the participant still generates first, so the
elicitation step is not lost.

Three phases on one page:
  1. ELICIT   they write what they would change about a bare request. Effort-judged.
  2. SELECT   three real messages; they choose which one to send.
  3. REPLY    the assistant's response to what they sent, then the responses every OTHER
              option would have got -- so the full contingency is visible and everyone sees
              the same set regardless of what they chose.

Phase 3 is why the selection is not gated on being "right": whichever they pick, seeing the
bare request earn nothing but a move is the demonstration, not a failure. Their choice is
still recorded, and is a one-bit pre-task measure of what they would reach for unprompted.

PROVENANCE: these messages and replies are AUTHORED, not verbatim participant text. Johnny's
were too -- none of its seven messages match any block in intervention_content.py. Only
contrasting_cases and modelled_think_aloud use verbatim phase-1 messages.

WHY CHESS. Connect Four is easy enough that participants may not see why anyone would need
an assistant at all, which makes the situation hard to take seriously. Chess needs no rules
explanation and nobody has to be persuaded it is difficult. The messages deliberately turn on
PIECE-SPECIFIC content (a knight, a bishop) rather than on positional maxims like "control
the centre", because a maxim of that kind has a loose Othello analogue and the pre-task step
must not carry task-relevant strategy content.

BY-PRODUCTS: `idea` is what they believed before being shown anything, and `selected_type`
is which kind of message they chose when it was their call -- both pre-task measures on the
target construct.
"""
from __future__ import annotations

import json
import os

import intervention_filter  # noqa: F401  (shared rubric template lives there)

ID = "elicit_select_reply"
TEMPLATE = "forms/elicit_select_reply.html"
FEEDBACK_TEMPLATE = None          # the reply phase happens in-page, before the single POST
TEXT_FIELD = "idea"
EFFORT_GATE = True
JUDGE_TASK = None            # set at the bottom, once GAME/QUESTION exist

MIN_CHARS = int(os.environ.get("ESR_MIN_CHARS", "50"))

START_MESSAGE = "What do you think would be the best move right now?"

# id/type are recorded; `text` and `reply` are verbatim. `carries_forward` marks whether the
# reply leaves the sender with something reusable -- used to pick the contrast in phase 3,
# never shown as a label.
CANDIDATES = [
    {"id": "ask_answer", "type": "solution_request",
     "text": "Which piece should I move? I am feeling quite confused.",
     "reply": "Move your knight to f5.",
     "carries_forward": False},
    {"id": "say_thinking", "type": "think_aloud",
     "text": ("I'm thinking of moving my knight to f5 because it would attack their queen "
              "and their bishop at the same time. Is that the best move?"),
     "reply": ("Good thinking. A move that creates two threats at once is hard to answer, "
               "because defending one of them can leave the other open."),
     "carries_forward": True},
    {"id": "look_ahead", "type": "knowledge_question",
     "text": "Could you help me look one move ahead at their strongest reply?",
     "reply": ("Yes. Checking their best answer first often shows whether a tempting move "
               "is actually safe."),
     "carries_forward": True},
]
_BY_ID = {c["id"]: c for c in CANDIDATES}

# --- DRAFT COPY for the study owner to edit ------------------------------------
# GAME names the game the EXAMPLE messages come from. It is not the game the participant
# plays afterwards -- that is Othello -- and it has to stay a different game, so the
# pre-task step cannot teach the task.
GAME = "chess"

FRAMING = ("In a moment you will play a game with an AI assistant. This task prepares you for interacting with the assistant by teaching you to make your interactions more successful.")
QUESTION = (f"What would you change about how this person is using the assistant, so that "
            f"they could play {GAME} better without AI assistance?")
SELECT_PROMPT = ("Now pick the message you would send instead. You will see how the "
                 "assistant replies.")
CONTRAST_LABEL = "And here is what the other messages would have got:"

# Caption over the opening message. NOTE: it does not say "real" -- unlike contrasting_cases
# and modelled_think_aloud, whose messages are verbatim phase-1 participant text, the chess
# messages here are authored. Keeping "novice" (your wording) because it explains why the
# sender is asking at all.
MESSAGE_LABEL = f"A message a complete {GAME} novice sent to an AI assistant"


def context():
    return {"start_message": START_MESSAGE, "framing": FRAMING, "question": QUESTION,
            "select_prompt": SELECT_PROMPT, "contrast_label": CONTRAST_LABEL,
            "candidates": CANDIDATES, "min_chars": MIN_CHARS, "game": GAME,
            "message_label": MESSAGE_LABEL}


def validate(form, min_chars):
    text = (form.get(TEXT_FIELD) or "").strip()
    if len(text) < MIN_CHARS:
        return {TEXT_FIELD: text}, (f"Please write at least {MIN_CHARS} characters about "
                                    f"what you would change.")
    try:
        payload = json.loads(form.get("payload") or "{}")
    except (ValueError, TypeError):
        payload = {}
    sel = payload.get("selected")
    if sel not in _BY_ID:
        return {TEXT_FIELD: text}, "Please choose a message to send before continuing."
    chosen = _BY_ID[sel]
    return {"idea": text,
            "n_chars": len(text),
            "start_message": START_MESSAGE,
            "selected": sel,
            "selected_type": chosen["type"],
            "selected_carries_forward": chosen["carries_forward"],
            "others_shown": payload.get("others_shown"),
            "option_order": payload.get("order"),
            "seconds_on_page": payload.get("seconds")}, None


# The effort judge is told exactly what THIS form showed, instead of reusing the
# Connect-Four-worded TASK_PRINCIPLES: a rubric describing the wrong stimulus scores the
# wrong thing. Only the task description differs from the other arms' rubrics -- the scoring
# bar is the shared one in intervention_filter._RUBRIC_TEMPLATE, so no arm is filtered harder.
JUDGE_TASK = (
    f"A participant was shown one chat message that a complete {GAME} novice sent to an AI "
    f"assistant while playing: '{START_MESSAGE}'. They were asked: '{QUESTION}'\n"
    "A real answer names something the person could do DIFFERENTLY -- a thing to say, ask, "
    "include or avoid. Restating the goal without naming a change ('ask better questions', "
    "'be more specific', 'engage more') is not an answer."
)
