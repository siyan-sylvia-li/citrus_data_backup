"""Pretest intervention variants by simulating the reader, before spending participants.

The scaffolded arm works only if the contrast makes the RIGHT dimension salient. Two live
participants both named "Group A just asks for the move" and neither named thinking aloud
-- and Think Aloud is the target with the strongest transfer link. So the question is
whether the material foregrounds it, and that is testable without recruiting anyone.

Simulated readers are a CEILING, not a model of participants. They read more carefully
than a Prolific worker paid for ten minutes. If a variant fails to make a dimension
salient to them, it will not become salient to a human; if it succeeds, that is
necessary but not sufficient. Calibration check: the harness reports how often readers
reproduce the two real human observations' pattern (request-type yes, think-aloud no).

    python intervention_pretest.py --list
    python intervention_pretest.py --readers 6
    python intervention_pretest.py --variant v2_think_pair --readers 8
"""
from __future__ import annotations

import argparse
import concurrent.futures
import re
import json
import sys
from collections import Counter
from pathlib import Path

import dotenv

BASE = Path(__file__).resolve().parent
dotenv.load_dotenv(BASE / ".env")

# ---------------------------------------------------------------------------
# Variants. Every message below is VERBATIM from a phase-1 Connect Four
# participant -- variants differ only in which real conversations are shown and
# in what order, never in invented copy.
# ---------------------------------------------------------------------------
A1 = ["Which move would be the best one and why", "What about next move?", "what about 6",
      "so which one would be the best move?", "What about now?", "Give me the best move"]
A2 = ["What do you think would be the best move right now?", "And now?", "Now?"]
A3 = ["Analyze the current Connect Four board carefully. I am Red and need to choose the "
      "best first three columns to keep Red on track to force a win. Please identify the "
      "strongest move now, explain the key threat it creates, and give the best follow-up "
      "columns after the opponent responds. Use 1-indexed column numbers."]
A4 = ["what is the best move for red now?", "and now?",
      "what are the best starting moves in this game as well to be able to win?",
      "so always start at column 4 with red is best?"]

B1 = ["I am about to move into column 3. will this work", "what would you suggest",
      "so just block up column 4 here. what's best to win with 2 moves left",
      "ok what next", "any risks to column 6"]
B2 = ["Where is the biggest threat from Red?",
      "But if I let Red place in that location, I can put a disc on top of it in 5, then "
      "2 and 3 gets me a win."]
B3 = ["Was my previous move effective?",
      "That worked well to block the immediate threat - it felt like the only option, am I right?"]
# Heavier think-aloud examples, also verbatim phase-1 Connect Four participants.
B4 = ["I was thinking of putting one in 3-3",
      "it would prevent yellow and open up one opportuhity for me",
      "what is the best next move"]
B5 = ["I was considering column 3, is that a good option?", "so column 4 first?",
      "is it better to focus on blocking or building my own win?"]

# Reasoning-heavy Group B material. The live v2 table makes CANDIDATE-NAMING the
# repeated feature and shows an actual justification only once, and that is what
# participants copied: scaffolded think-aloud turns are ~3x more frequent than vanilla's
# but carry a stated reason less often (~14% vs ~25%). These conversations attach a
# consequence to the move -- "column 3? this blocks their diagonal" rather than
# "I was considering column 3, is that a good option?" -- so justification becomes the
# feature that repeats.
# Long Group A material, needed to balance the reasoning-heavy Group B rows. These are
# verbose SOLUTION REQUESTS -- elaborate, well-formed, and still handing the decision
# over -- which is the point: it keeps Group A at least as long as Group B, so "writes
# more" stays unavailable as a reading of the contrast.
A_LONG2 = [            # 66ca1c321a, 70 words
    'Analyze the current board and tell me the best sequence of my next three moves '
    'for Red. Please list only the column numbers in order and briefly explain why '
    'they are optimal.',
    "Given Yellow's last move, what is now the best second move for Red? Reply with "
    "only the column number.",
    "Given Yellow's latest move, what is now the best second move for Red? Reply with "
    "only the column number.",
]

B_REASON_LONG = [      # 546e8bc6fd, 82 words, reason in 4 of 6 turns
    'I am thinking row 3, column 2. This would give me three in a row, as well as '
    'blocking the opponents diagonal streak. Thoughts?',
    'Do you recommend a different move?',
    "Oh I've just seen I NEED to go column 4, or they will have a diagonal four.",
    'Column 3? This blocks their diagonal move, gives me three diagonal, and also '
    'gives me two horizontally',
    'If I go there, they can block my existing diagonal three, so I will avoid that.',
    'Column 5?',
]
B_REASON_MID = [       # 6447bd2acb, 64 words
    'I\u2019m considering row 3 because I think that gives me a chance to win '
    'horizontally or diagonally. Is that the best move?',
    'Thanks! Should I do column 3 now? It gives me a chance to win diagonally going '
    'from the bottom right up too',
    'Thanks! I didn\u2019t see that threat. Column 3 might set me up for a win, should '
    'I do it now?',
]
B_REASON_SHORT = [     # 60147f790b, 32 words
    'If I add to column 2, will that reduce my possibilities rather than improve my '
    'chances?',
    'How about column 5, then?',
    "how about 4, blocking yellow's progress and adding to the diagonal?",
]

def first(block, k):
    """First k messages of a real conversation.

    A prefix, never a hand-picked subset: truncating keeps what the participant actually
    said FIRST, so the excerpt is still a conversation rather than a curated highlight
    reel. It is the lever for balancing Group A and B length without dropping the
    reasoning-heavy material -- explaining a move simply takes more words than naming
    one, so the columns will not balance by block choice alone.
    """
    return block[:k]


VARIANTS = {
    "v0_live":        [(A1, B1), (A2, B2), (A3, B3)],           # what is deployed now
    "v1_think_first": [(A2, B4), (A1, B1), (A3, B3)],           # narration in row 1
    "v2_think_pair":  [(A2, B4), (A1, B5), (A3, B2)],           # two narration rows
    "v3_short":       [(A2, B4), (A3, B3)],                     # fewer rows, sharper
    "v4_all_think":   [(A2, B4), (A4, B5), (A3, B2), (A1, B1)], # narration in three rows
    # --- externalised reasoning: every Group B row attaches a REASON to the move ---
    "v5_reasoning":   [(A2, B_REASON_SHORT), (A1, B_REASON_MID),
                       (A3, B_REASON_LONG), (A_LONG2, B4)],
    # same idea, Group B kept terse so length cannot be the perceived difference
    "v6_reason_lean": [(A2, B_REASON_SHORT), (A4, B_REASON_MID), (A3, B2)],
    # --- truncated to balance length: A and B within ~5%, B ~70% reason-bearing ---
    "v7_balanced":    [(A2,                  first(B_REASON_SHORT, 3)),
                       (A1,                  first(B_REASON_MID, 2)),
                       (A3,                  first(B_REASON_LONG, 4)),
                       (first(A_LONG2, 2),   first(B4, 2))],
    # three rows instead of four, at the cost of a wider length gap
    "v8_balanced3":   [(A2,                  first(B_REASON_SHORT, 3)),
                       (first(A4, 4),        first(B_REASON_MID, 2)),
                       (A3,                  first(B_REASON_LONG, 3))],
    # --- two rows only ---------------------------------------------------------
    # Uses A_LONG2 rather than A3 for the long Group A material. Participant
    # observations on the live table show A3 backfiring: three of four WRONG choices
    # cited Group A's elaborate prompt as a virtue ("quite in-depth instructions and
    # very detailed", "a full single prompt to get the AI to lay out a plan"). A_LONG2
    # is equally long but ends "Reply with only the column number" twice, so its
    # answer-seeking is unmistakable rather than reading as sophistication.
    "v9_two_rows":    [(first(A_LONG2, 2),   first(B_REASON_MID, 2)),
                       (A1,                  first(B_REASON_SHORT, 3))],
    "v10_two_rows_b": [(first(A_LONG2, 2),   first(B_REASON_LONG, 3)),
                       (A1,                  first(B_REASON_SHORT, 3))],
}

READER_PROMPT = """You are taking part in a paid online study. You are not an expert at
writing prompts for AI and you are moving fairly quickly.

Below are messages that earlier participants sent to an AI assistant while solving a
Connect Four puzzle, split into two groups.

{table}

Answer in EXACTLY this format, two lines, nothing else:

OBSERVATION: <one or two sentences, in your own words, on how the two groups behave
differently -- written the way a normal person types in a study text box>
CHOICE: <A or B, which group you think plays better once the AI is taken away>"""

CODER_PROMPT = """A study participant was shown two groups of messages that earlier
participants sent to an AI assistant, and wrote this observation:

  "{obs}"

Decide which of these three differences the observation actually NAMES. Be strict: the
observation must point at the feature, not merely be compatible with it.

  request_type    -- that one group asks the AI which move to play / asks for the
                     answer, while the other proposes a specific move of its own
  states_candidate -- that one group says WHICH move it is considering
  explains_why     -- that one group gives a REASON for its move: what it would achieve,
                     block, set up, or risk. Naming a move without saying what it does
                     is states_candidate only, NOT explains_why.
  says_longer      -- that the difference is LENGTH or verbosity: one group writes more,
                     longer messages, more detail. Mark this whenever length is offered
                     as the difference, even alongside something else.

The last two are deliberately separate. A participant who copies only the announcing
half produces "I think c8" and no reasoning, which is the failure this pretest exists to
detect -- so do not mark explains_why unless a reason is actually named.

Answer in EXACTLY this format, four lines, nothing else:

REQUEST_TYPE: <yes or no>
STATES_CANDIDATE: <yes or no>
EXPLAINS_WHY: <yes or no>
SAYS_LONGER: <yes or no>"""


# ---------------------------------------------------------------------------
# Adoption pretest. The observation-coding above measures whether a reader NOTICES a
# feature; what predicts the study outcome is whether they COPY it. Those diverge
# sharply: v0_live scores 93% on explains_why because readers spot its single
# justification, yet real participants on v0 produced reason-bearing utterances in only
# ~14% of think-aloud turns -- they copied candidate-naming, the feature that repeats.
#
# So this asks the reader to WRITE the messages they would send, and codes those.
# Still a ceiling -- a model reads more carefully than a paid participant -- but it
# measures the right thing, and the ranking is what is being used.
# ---------------------------------------------------------------------------
PLAY_PROMPT = """You are taking part in a paid online study. You are not an expert at
writing prompts for AI and you are moving fairly quickly.

You were just shown these messages from earlier participants, split into two groups:

{table}

Group B went on to play better than Group A once the AI assistant was taken away.

Now YOU are about to play a puzzle game (Othello) with an AI assistant available. You
must send the assistant 5 messages while you play.

Write the 5 messages you would send. Write them the way a normal person types in a chat
box -- short, informal, imperfect. One per line, numbered 1-5, nothing else."""

PLAY_CODER = """Here is one message a study participant sent to an AI assistant while
playing Othello:

  "{msg}"

Answer in EXACTLY this format, two lines, nothing else:

NAMES_MOVE: <yes or no -- does it name a specific square or move the participant is
considering or has played?>
GIVES_REASON: <yes or no -- does it say WHY: what the move would achieve, block, set up,
or risk? Naming a move without saying what it does is no.>"""


def render(variant):
    rows = []
    for i, (a, b) in enumerate(VARIANTS[variant], 1):
        rows.append(f"Group A, Participant {i}:\n" + "\n".join(f"  - {m}" for m in a))
        rows.append(f"Group B, Participant {i}:\n" + "\n".join(f"  - {m}" for m in b))
    return "\n\n".join(rows)


def _fields(text):
    """Line-based parsing: weaker readers emit malformed JSON but follow KEY: value."""
    out = {}
    for line in text.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            k = k.strip().strip("*- ").lower()
            if k in ("observation", "choice", "request_type", "states_candidate",
                     "explains_why", "says_longer", "names_move", "gives_reason"):
                out[k] = v.strip()
    return out


def run_variant(variant, readers, reader_models, coder_model):
    import dspy
    table = render(variant)
    coder = dspy.LM(coder_model, cache=False, max_tokens=2048)

    def one(i):
        lm = dspy.LM(reader_models[i % len(reader_models)], cache=False,
                     temperature=1.0, max_tokens=2048)
        try:
            out = _fields(lm(READER_PROMPT.format(table=table))[0])
            obs = out.get("observation", "").strip()
            if len(obs) < 15:
                raise ValueError(f"no observation parsed from: {out}")
            choice = out.get("choice", "").upper().strip()[:1]
            codes = _fields(coder(CODER_PROMPT.format(obs=obs))[0])
            yes = lambda k: codes.get(k, "").lower().startswith("y")
            return {"observation": obs, "choice": choice,
                    **{k: yes(k) for k in ("request_type", "states_candidate",
                                           "explains_why", "says_longer")}}
        except Exception as e:
            print(f"    reader {i} failed: {type(e).__name__}: {e}", file=sys.stderr)
            return None

    with concurrent.futures.ThreadPoolExecutor(min(8, readers)) as ex:
        rows = [r for r in ex.map(one, range(readers)) if r]
    return rows


def run_adoption(variant, readers, reader_models, coder_model):
    """Ask readers to produce the messages they would send, then code those."""
    import dspy
    table = render(variant)
    coder = dspy.LM(coder_model, cache=False, max_tokens=1024)

    def one(i):
        lm = dspy.LM(reader_models[i % len(reader_models)], cache=False,
                     temperature=1.0, max_tokens=2048)
        try:
            raw = lm(PLAY_PROMPT.format(table=table))[0]
            msgs = []
            for line in raw.splitlines():
                line = line.strip()
                m = re.match(r"^\s*\d+[\.\)]\s*(.+)$", line)
                if m and len(m.group(1).split()) >= 2:
                    msgs.append(m.group(1).strip().strip('"'))
            msgs = msgs[:5]
            if len(msgs) < 3:
                raise ValueError(f"only parsed {len(msgs)} messages")
            out = []
            for msg in msgs:
                f = _fields(coder(PLAY_CODER.format(msg=msg))[0])
                out.append((msg,
                            f.get("names_move", "").lower().startswith("y"),
                            f.get("gives_reason", "").lower().startswith("y")))
            return out
        except Exception as e:
            print(f"    reader {i} failed: {type(e).__name__}: {e}", file=sys.stderr)
            return None

    with concurrent.futures.ThreadPoolExecutor(min(8, readers)) as ex:
        return [r for r in ex.map(one, range(readers)) if r]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=None)
    ap.add_argument("--readers", type=int, default=6)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--show", action="store_true", help="print each observation")
    ap.add_argument("--adopt", action="store_true",
                    help="adoption pretest: readers WRITE the messages they would send, "
                         "and those are coded. Measures copying, not noticing.")
    ap.add_argument("--coder", default="openai/gpt-5.4-mini",
                    help="coder model; use one OUTSIDE the reader pool to check that "
                         "the ranking is not an artifact of same-family legibility")
    a = ap.parse_args()

    if a.list:
        for k, v in VARIANTS.items():
            print(f"{k:<16} {len(v)} rows")
        return 0

    # Sonnet comes from Bedrock; the other two from their own vendors. Region for the
    # Bedrock reader is taken from AWS_REGION_NAME, since these are bare model strings.
    reader_models = ["openai/gpt-5.4-mini", llm_backends.anthropic_model("claude-sonnet-5"),
                     "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"]
    coder_model = a.coder
    names = [a.variant] if a.variant else list(VARIANTS)

    if a.adopt:
        print(f"{'variant':<16}{'readers':>8}{'msgs':>6}{'names move':>12}"
              f"{'GIVES REASON':>14}{'reason+move':>13}")
        for name in names:
            res = run_adoption(name, a.readers, reader_models, coder_model)
            if not res:
                print(f"{name:<16}  (all readers failed)"); continue
            flat = [m for r in res for m in r]
            nm = sum(m[1] for m in flat) / len(flat)
            gr = sum(m[2] for m in flat) / len(flat)
            both = sum(m[1] and m[2] for m in flat) / len(flat)
            print(f"{name:<16}{len(res):>8}{len(flat):>6}{nm:>11.0%}{gr:>13.0%}"
                  f"{both:>12.0%}")
            if a.show:
                for msg, n_, g_ in flat[:8]:
                    print(f"    [{'M' if n_ else ' '}{'R' if g_ else ' '}] {msg[:82]}")
        return 0

    print(f"{'variant':<16}{'n':>4}{'req_type':>10}{'states_cand':>13}"
          f"{'explains_why':>14}{'says_LONGER':>13}{'chose B':>9}")
    for name in names:
        rows = run_variant(name, a.readers, reader_models, coder_model)
        if not rows:
            print(f"{name:<16}  (all readers failed)"); continue
        n = len(rows)
        f = lambda k: sum(r[k] for r in rows) / n
        print(f"{name:<16}{n:>4}{f('request_type'):>9.0%}"
              f"{f('states_candidate'):>13.0%}{f('explains_why'):>14.0%}"
              f"{f('says_longer'):>13.0%}"
              f"{sum(r['choice'] == 'B' for r in rows) / n:>9.0%}")
        if a.show:
            for r in rows:
                tags = "".join(c for c, k in
                               (("R", "request_type"), ("C", "states_candidate"),
                                ("W", "explains_why"), ("L", "says_longer"))
                               if r[k]) or "-"
                print(f"    [{r['choice']}|{tags:<3}] {r['observation'][:96]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
