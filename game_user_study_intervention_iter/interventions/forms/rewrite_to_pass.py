"""Rewrite-to-pass: the participant must PRODUCE the target form of message once.

Why this shape. Every earlier form asked for recognition -- read two tables, pick a group,
click a pre-written option -- while the behaviour being taught is production. Johnny showed
the gap: message length rose to the highest of any form (participants copied what they saw)
but Think Aloud stayed level with the static tables, because nobody ever composed anything.
Here the only action is composing, and the page does not advance until the composition has
both parts.

Why TIC-TAC-TOE. The task is incoherent without a position to reason about -- "column 4
because it blocks their diagonal" is invented if there is no board. Tic-tac-toe is the
smallest situation that fixes that: no rules primer needed, anyone can form a real candidate
and a real consequence in seconds, and its strategy is worth nothing for Othello, so the
pre-task step still carries no task-relevant content. That firewall is why the material has
always been another game: if the intervention taught Othello strategy, a downstream effect
could be content rather than behaviour.

The gate is a PATTERN CHECK on the FORM of the message, never on whether the move is right.
Same principle as the effort judge it replaces: scoring correctness would confound low
effort with the low ability that defines the sample. Whether they picked the move that
actually blocks is RECORDED (`named_correct_cell`) but never gated.

BY-PRODUCT WORTH HAVING: `attempts` records every draft in order with which checks it
passed, and `first_attempt` is what they wrote before any feedback -- a pre-task baseline of
the exact behaviour the study is trying to move, on the same construct as the outcome.
"""
from __future__ import annotations

import json
import os
import re

ID = "rewrite_to_pass"
TEMPLATE = "forms/rewrite_to_pass.html"
FEEDBACK_TEMPLATE = None
TEXT_FIELD = "rewrite"
EFFORT_GATE = False          # gated by the pattern check below, not the LLM judge
JUDGE_TASK = None

# --- the position -------------------------------------------------------------
# The participant is X. O has the top row's left and centre, so top-right is the one
# urgent move. Deliberately unambiguous: the point is that they CAN state a real
# consequence, not that the position is hard.
#
#   O | O | .
#   . | X | .
#   X | . | .
BOARD = [["O", "O", ""],
         ["", "X", ""],
         ["X", "", ""]]
CORRECT_CELL = "top right"

# The message they are asked to improve. Verbatim from a phase-1 Connect Four participant
# (the opening of _A_TERSE) -- the outsourcing pattern in its purest form, and worded
# without any game-specific vocabulary, so it reads naturally over this board too.
START_MESSAGE = "What do you think would be the best move right now?"

# --- requirement 1: names a cell ----------------------------------------------
# Per-cell patterns so the record can say WHICH cell they named, not just that they named
# one. Bare "middle"/"centre" means the centre square; the row words only bind a cell when
# paired with a column word.
_V = "(?:top|upper)", "(?:middle|centre|center|mid)", "(?:bottom|lower)"
_H = "(?:left|l)", "(?:middle|centre|center|mid)", "(?:right|r)"
CELL_PATTERNS = {
    "top left":      r"\b(?:top|upper)[\s\-]*(?:left|l)\b",
    "top centre":    r"\b(?:top|upper)[\s\-]*(?:middle|centre|center|mid)\b",
    "top right":     r"\b(?:top|upper)[\s\-]*(?:right|r)\b",
    "middle left":   r"\b(?:middle|centre|center|mid)[\s\-]*(?:left|l)\b|\bleft[\s\-]*(?:middle|centre|center)\b",
    "centre":        r"\b(?:the\s+)?(?:middle|centre|center)\b(?![\s\-]*(?:left|right|row|l\b|r\b))",
    "middle right":  r"\b(?:middle|centre|center|mid)[\s\-]*(?:right|r)\b|\bright[\s\-]*(?:middle|centre|center)\b",
    "bottom left":   r"\b(?:bottom|lower)[\s\-]*(?:left|l)\b",
    "bottom centre": r"\b(?:bottom|lower)[\s\-]*(?:middle|centre|center|mid)\b",
    "bottom right":  r"\b(?:bottom|lower)[\s\-]*(?:right|r)\b",
}
# One combined pattern for the live client-side hint.
CANDIDATE_RE = "|".join(f"(?:{p})" for p in CELL_PATTERNS.values())

# --- requirement 2: says what the move would do -------------------------------
CONSEQUENCE_RE = (r"(because|since|so that|so i|so they|that way|otherwise"
                  r"|to (?:block|stop|prevent|win|deny|avoid|take|get|keep|set)"
                  r"|block|blocks|blocking|stop|stops|stopping|prevent|prevents"
                  r"|win|wins|winning|three in a row|3 in a row|row|column|diagonal"
                  r"|fork|threat|they(?:'| a)?d|they would|gives me|lets me|opens|sets up)")

_CELLS = {name: re.compile(p, re.I) for name, p in CELL_PATTERNS.items()}
_CONS = re.compile(CONSEQUENCE_RE, re.I)

MIN_CHARS = int(os.environ.get("REWRITE_MIN_CHARS", "20"))

# DRAFT for the study owner to edit. This is the only prose the participant reads, so the
# wording matters more than anything else on the page.
#
# Three things it deliberately avoids:
#  * "people who were successful" -- modelled_think_aloud used that framing and produced
#    the highest knowledge-deficit-question rate of any form (29.5%, above vanilla) and the
#    worst round-1 play. Telling participants they are looking at winners invites "what
#    don't I know" rather than "narrate what I'm thinking".
#  * any claim that the tic-tac-toe move matters -- the gate scores the form of the
#    message, so the framing should not read as a skill test.
#  * Othello vocabulary -- the pre-task step must carry no task-relevant strategy content.
#
# What it does assert is measured, not invented: asking for the answer elicits a bare
# verdict (Solution Request -> Move Verdict, OR 9.9) while saying what you are considering
# elicits prompts and justifications (Think Aloud -> Prompt 2.5, Hint 1.7).
FRAMING = ("In a moment you will play a game with an AI assistant. What you get back from "
           "it depends on what you give it: a message that only asks for the answer gets "
           "you an answer, and nothing you can weigh up yourself.")

# The instruction proper. Kept separate from FRAMING because the two do different jobs --
# FRAMING says why, TASK says what to do -- and because the earlier version buried the
# instruction in a form label below the board, where it read as a caption.
TASK = ("This exercise helps you think about what you should ask the AI assistant during "
        "the Othello game. Below is a real message someone sent an assistant while playing. "
        "Can you rewrite it so that the sender of the message learns more about tic-tac-toe "
        "strategy, and is therefore more likely to succeed without AI? Your message needs to "
        "name the square you are thinking of playing and say what that move would do. The "
        "Continue button unlocks once it does both.")


def named_cells(text: str):
    return [n for n, rx in _CELLS.items() if rx.search(text or "")]


def check(text: str) -> dict:
    """Which requirements the text meets. Same logic the page runs client-side."""
    t = (text or "").strip()
    cells = named_cells(t)
    return {"long_enough": len(t) >= MIN_CHARS,
            "names_candidate": bool(cells),
            "says_consequence": bool(_CONS.search(t))}


def context():
    return {"board": BOARD, "start_message": START_MESSAGE, "min_chars": MIN_CHARS,
            "framing": FRAMING, "task": TASK, "candidate_re": CANDIDATE_RE,
            "consequence_re": CONSEQUENCE_RE}


def validate(form, min_chars):
    """Re-check server-side. The client gate only stops the button being clickable."""
    text = (form.get(TEXT_FIELD) or "").strip()
    c = check(text)
    if not c["long_enough"]:
        return {TEXT_FIELD: text}, f"Please write at least {MIN_CHARS} characters."
    if not c["names_candidate"]:
        return {TEXT_FIELD: text}, ("Name the square you are thinking of playing — "
                                    "for example “top right”.")
    if not c["says_consequence"]:
        return {TEXT_FIELD: text}, ("Say what that move would do — what it blocks, "
                                    "wins or sets up.")
    try:
        attempts = json.loads(form.get("attempts") or "[]") or []
    except (ValueError, TypeError):
        attempts = []
    first = attempts[0]["text"] if attempts and attempts[0].get("text") else text
    cells = named_cells(text)
    return {"rewrite": text,
            "start_message": START_MESSAGE,
            "named_cells": cells,
            # Recorded, never gated: whether they picked the move that actually blocks.
            "named_correct_cell": CORRECT_CELL in cells,
            "attempts": attempts,
            "n_attempts": max(len(attempts), 1),
            "first_attempt": first,
            "first_attempt_checks": check(first),
            "first_attempt_cells": named_cells(first)}, None
