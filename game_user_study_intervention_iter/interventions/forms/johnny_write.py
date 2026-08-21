"""Johnny, reconstructed as a PRODUCTION task.

The original johnny_connect_four_boost asked for recognition -- pick one of two pre-written
messages -- and the results showed the gap: message length rose to the highest of any form
(participants copied what they saw) while Think Aloud stayed level with the static tables,
because nobody composed anything. Here the participant WRITES the message Johnny sends.

Two things are handed to them so the task is about the message, not about the game:
  * the column Johnny is considering is stated outright, and
  * the page marks the square the disc would land on,
so no board-reading or visualisation is required. What is left to produce is exactly the
target behaviour: say what the move would do.

Boards, scenarios and assistant replies are reused from the original Johnny activity. Note
that those were AUTHORED for that activity, not verbatim participant text -- none of Johnny's
seven messages match any block in intervention_content.py. Connect Four rather than Othello
for the usual reason: the pre-task step must not carry task-relevant strategy content.

The gate is a pattern check on the FORM of the message, never on whether the move is good.
`attempts` keeps every draft per round, so the FIRST draft of round 1 is a pre-task baseline
of the behaviour the study is trying to move.
"""
from __future__ import annotations

import json
import os
import re

ID = "johnny_write"
TEMPLATE = "forms/johnny_write.html"
FEEDBACK_TEMPLATE = None
TEXT_FIELD = None            # the page posts a JSON payload of per-round messages
EFFORT_GATE = False
JUDGE_TASK = None

# Boards are 7 wide x 6 tall, listed top row first. "Y" = Johnny, "R" = opponent.
# candidate_column is 1-indexed, as the participants' own messages phrase it.
ROUNDS = [
    {"board": ["", "", "", "", "", "", "",
               "", "", "", "", "", "", "",
               "", "", "", "", "", "", "",
               "", "", "", "R", "", "", "",
               "", "", "Y", "Y", "R", "", "",
               "R", "Y", "R", "Y", "R", "", ""],
     "candidate_column": 4,
     "assistant_reply": ("Good strategy. Creating more than one future threat can make it "
                         "harder for Red to block everything."),
     "prompt": "Johnny is thinking about playing column 4."},
    {"board": ["", "", "", "", "", "", "",
               "", "", "", "", "", "", "",
               "", "", "", "", "", "", "",
               "", "", "", "Y", "", "", "",
               "", "R", "Y", "R", "Y", "", "",
               "R", "Y", "R", "Y", "R", "", ""],
     "candidate_column": 6,
     "assistant_reply": ("Yes. Looking at Red's strongest reply can reveal whether an "
                         "attractive move is actually safe."),
     "prompt": "Johnny is thinking about playing column 6."},
]

# --- the two requirements, shared with the page so they cannot drift ------------
CANDIDATE_RE = r"(column\s*[1-7]|col\.?\s*[1-7]|\b[1-7]\b)"
CONSEQUENCE_RE = (r"(because|since|so that|so i|so they|that way|otherwise"
                  r"|to (?:block|stop|prevent|win|deny|avoid|take|get|keep|set|connect)"
                  r"|block|blocks|blocking|stop|stops|prevent|prevents"
                  r"|win|wins|winning|four|connect|diagonal|row|threat|threats"
                  r"|they(?:'| a)?d|they would|gives me|lets me|opens|sets up|safe)")
_CAND = re.compile(CANDIDATE_RE, re.I)
_CONS = re.compile(CONSEQUENCE_RE, re.I)

MIN_CHARS = int(os.environ.get("JW_MIN_CHARS", "20"))

# DRAFT wording for the study owner to edit.
FRAMING = ("In a moment you will play a game with an AI assistant. What you get back from "
           "it depends on what you give it: a message that only asks for the answer gets "
           "you an answer, and nothing you can weigh up yourself.")
TASK = ("Johnny is playing Connect Four with an AI assistant. You will be told which column "
        "he is thinking about playing — the board marks where the disc would land. Write the "
        "message Johnny should send so that he learns something he could use again later, "
        "not just whether the move is right.")
REQUIREMENTS = ["Mentions the column Johnny is considering",
                "Says what that move would do — what it blocks, wins or sets up"]


def check(text: str) -> dict:
    t = (text or "").strip()
    return {"long_enough": len(t) >= MIN_CHARS,
            "names_candidate": bool(_CAND.search(t)),
            "says_consequence": bool(_CONS.search(t))}


def context():
    return {"rounds": ROUNDS, "framing": FRAMING, "task": TASK,
            "requirements": REQUIREMENTS, "min_chars": MIN_CHARS,
            "candidate_re": CANDIDATE_RE, "consequence_re": CONSEQUENCE_RE}


def validate(form, min_chars):
    """Every round must carry a message that meets the requirements."""
    try:
        payload = json.loads(form.get("payload") or "{}")
    except (ValueError, TypeError):
        return {}, ("Something went wrong recording your messages. Please reload the page "
                    "and try again.")
    rounds = payload.get("rounds") or []
    if len(rounds) < len(ROUNDS):
        return {}, "Please write a message for each of Johnny's turns before continuing."
    for i, r in enumerate(rounds, 1):
        c = check(r.get("final") or "")
        if not all(c.values()):
            return {}, (f"The message for turn {i} still needs to mention the column and "
                        f"say what the move would do.")
    first = (rounds[0].get("attempts") or [{}])[0].get("text") or rounds[0].get("final")
    return {"messages": [r.get("final") for r in rounds],
            "rounds": rounds,
            "n_attempts": [len(r.get("attempts") or []) for r in rounds],
            "first_attempt": first,
            "first_attempt_checks": check(first),
            "candidate_columns": [r["candidate_column"] for r in ROUNDS]}, None
