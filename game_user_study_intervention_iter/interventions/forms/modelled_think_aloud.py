"""Direct modelling: show the target behaviour, ask nothing.

Rationale for dropping the questions. In phase 2 the forced choice discriminated nothing
downstream -- participants who read the contrast correctly averaged Think Aloud 33% against
32% for those who did not, and solo optimal rate 39% against 40% -- and every participant
is told the answer in the correction regardless. Meanwhile the modelling half is what
demonstrably moved the acts (+17.7 points, d 0.67), and the mechanism appeared to be that
participants copied the most REPEATED surface feature. Displaying only the target
behaviour makes that feature something you choose rather than something they induce.

Content provenance is unchanged: every message is verbatim from a phase-1 Connect Four
participant, drawn from the same blocks contrasting_cases uses. Connect Four rather than
Othello is deliberate -- the material must not carry task-relevant strategy content, or a
transfer effect could be content rather than behaviour.

NO EFFORT JUDGE. There is no free text, so `validate` gates on DWELL TIME instead: the
page must have been open for MIN_DWELL_SECONDS. Both a client-side elapsed reading and a
server-side one (from the session) are recorded; the server value is the one enforced,
because the client value is trivially editable.
"""
from __future__ import annotations

import os

from intervention_content import _B_REASON_MID, _B_REASON_SHORT, _B_THINK1

ID = "modelled_think_aloud"
TEMPLATE = "forms/modelled_think_aloud.html"
FEEDBACK_TEMPLATE = None
TEXT_FIELD = None                 # no free text -> gated on dwell, not on the judge
JUDGE_TASK = None

# Long enough to have read three short exchanges, short enough not to feel like a wall.
MIN_DWELL_SECONDS = int(os.environ.get("MIN_DWELL_SECONDS", "30"))

# Three real exchanges, ordered shortest-reasoning to fullest. Each message pairs a
# CANDIDATE with what it would DO -- that pairing is the repeated feature, which is the
# whole point of the form.
EXAMPLES = [
    {"label": "Proposing a candidate move after thinking about it", "messages": _B_THINK1},
    {"label": "Reasoning about consequences of the move", "messages": _B_REASON_SHORT},
    {"label": "Discussing the rationale", "messages": _B_REASON_MID},
]

# PLACEHOLDER -- the framing sentence is the part that does the teaching, so the wording
# belongs to the study owner. Replace before this runs on participants.
FRAMING = ("You are looking at participants who were successful in a previous version of this study (Connect Four instead of Othello). Observe these utterances and think about how this would affect your interactions with the AI assistant during the study.")


def context():
    return {"examples": EXAMPLES, "framing": FRAMING,
            "min_dwell": MIN_DWELL_SECONDS}


def validate(form, min_chars):
    """Gate on time on page. `server_dwell` is injected by the route from the session."""
    try:
        client = float(form.get("elapsed_ms") or 0) / 1000.0
    except (TypeError, ValueError):
        client = 0.0
    server = float(form.get("server_dwell") or 0)
    if server < MIN_DWELL_SECONDS:
        return {}, (f"Please take a moment to read the examples "
                    f"({MIN_DWELL_SECONDS}s minimum) before continuing.")
    return {"dwell_seconds_server": round(server, 1),
            "dwell_seconds_client": round(client, 1),
            "n_examples_shown": len(EXAMPLES),
            "min_dwell_seconds": MIN_DWELL_SECONDS}, None
