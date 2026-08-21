"""Refined tutor-move scheme: splits the codes that saturate, keeps the old ones recoverable.

WHY
---
Measured on 538 simulated Connect Four assistant turns, the coarse scheme has two
codes that fire on almost every turn and two that never fire:

    Direct Instruction      99.1%   kappa 0.51
    Provide Correct Answer  84.2%   kappa 0.84
    Hint                    21.0%   Positive 18.0%   Prompt 13.4%
    Paraphrase               4.5%   kappa 0.31
    Pose Simplified Problem  0.5%   kappa 0.30
    Comprehension Gauging    0.0%   never used

A code present on 99% of turns carries almost no between-run variance, so it cannot
explain differences in transfer no matter how much data is collected. That is a
structural reason the assistant-acts-vs-outcome analysis came back null, distinct
from statistical power.

THE SPLIT
---------
For a TRANSFER outcome the distinction that should matter is whether the assistant
said anything portable to a different position. `Direct Instruction` currently
swallows that whole, so it is split along transfer scope:

    Board Report        restates what is visible          -> no portable content
    Local Justification why THIS move in THIS position    -> tied to the board
    Worked Line         a concrete consequent sequence    -> shows the derivation
    General Principle   a rule stated portably            -> transferable

`Provide Correct Answer` is renamed `Move Verdict` at the fine level (same meaning,
clearer contrast with Local Justification: the verdict alone versus the verdict plus
its reason).

PRE-REGISTERED PREDICTION, recorded before the re-annotation is run
-------------------------------------------------------------------
    General Principle and Worked Line predict unassisted transfer.
    Board Report and Move Verdict do not.

Written down first on purpose. Refining a taxonomy after seeing a null, in the
direction that might produce an effect, is a garden-of-forking-paths risk; a
directional prediction that can fail is the defence.

HIERARCHY
---------
Every fine code rolls up to a coarse one via ROLLUP, so annotations under either
scheme remain comparable and nothing already coded is stranded — which matters
because the human assistant-side coding cannot be redone here.

One deliberate lossy merge: `Pose Simplified Problem` (0.5%, kappa 0.30) folds into
`Prompt`, so a fine annotation cannot reproduce that coarse label. It was too rare
and too unreliable to support a category of its own.
`Comprehension Gauging Question` is kept despite never firing, so the roll-up stays
total for the codes the human data may contain.
"""

import concurrent.futures
import logging
from collections import Counter
from typing import Literal

import dspy

from dialogue_act_annotation import _complete_cases  # noqa: F401  (shared helpers)

_log = logging.getLogger("dialogue_act_annotation_assistant_fine")

# fine code -> coarse code in ASSISTANT_SCHEME. Order is the canonical output order.
ROLLUP = {
    # --- telling: the assistant supplies content -------------------------------
    "Move Verdict": "Provide Correct Answer",
    "Board Report": "Direct Instruction",
    "Local Justification": "Direct Instruction",
    "Worked Line": "Direct Instruction",
    "General Principle": "Direct Instruction",
    # --- scaffolding: the assistant makes the participant do the work ----------
    "Hint": "Hint",
    "Prompt": "Prompt",                       # absorbs Pose Simplified Problem
    "Comprehension Gauging Question": "Comprehension Gauging Question",
    # --- reacting to what the participant said ---------------------------------
    "Paraphrase": "Paraphrase",
    "Positive Feedback": "Positive Feedback",
    "Negative Feedback": "Negative Feedback",
    "Neutral Feedback": "Neutral Feedback",
}

# Which fine codes the pre-registered prediction says should carry transfer.
TRANSFER_BEARING = ("General Principle", "Worked Line")
TRANSFER_INERT = ("Board Report", "Move Verdict")

EXAMPLES = {
    "Move Verdict": '"Play **column 4**." "I\'d suggest column 4 or 5." (the move alone, no reason)',
    "Board Report": '"Yellow has discs at c7r1, c6r2 and c5r3." "Columns 3 and 6 are full."',
    "Local Justification": '"Column 4 is right because it blocks Yellow\'s diagonal at row 3."',
    "Worked Line": '"If you play 4, Yellow must answer 5, and then 2 gives you a double threat."',
    "General Principle": '"Odd threats matter more when you are second to move." '
                         '"A threat you can\'t reach until the column fills is worth little."',
    "Hint": '"The key area to inspect is column 4." "Look at Yellow\'s cluster on rows 1-2."',
    "Prompt": '"After you play there, what does Yellow threaten?" '
              '"Can Yellow win immediately if you do nothing?"',
    "Comprehension Gauging Question": '"Does that make sense?" "Are you with me?"',
    "Paraphrase": '"Your column 2 idea would land at row 3 and make a vertical three."',
    "Positive Feedback": '"Nice block." "That\'s right."',
    "Negative Feedback": '"That column is full, so it isn\'t playable."',
    "Neutral Feedback": '"That does help build, but it leaves the diagonal open."',
}

TAXONOMY = "".join(f"DIALOGUE ACT: {k}, EXAMPLES: {EXAMPLES[k]}\n" for k in ROLLUP)

FineAssistantAct = Literal[
    "Move Verdict", "Board Report", "Local Justification", "Worked Line",
    "General Principle", "Hint", "Prompt", "Comprehension Gauging Question",
    "Paraphrase", "Positive Feedback", "Negative Feedback", "Neutral Feedback",
]


def to_coarse(fine_acts):
    """Fine label set -> the coarse label set it implies, deduped, canonical order."""
    from dialogue_act_annotation_assistant import ASSISTANT_SCHEME
    mapped = {ROLLUP[a] for a in fine_acts if a in ROLLUP}
    return [k for k in ASSISTANT_SCHEME if k in mapped]


class FineAssistantActSignature(dspy.Signature):
    """Given a reply from an AI assistant helping a user solve a Connect Four puzzle, first split it into clauses, then classify each clause into the tutor dialogue acts that apply. Output every act that applies. Assistant replies are long and usually carry several acts; do not stop at the first one.

The four content acts are the ones that matter most and are the easiest to confuse. Decide them by asking: WOULD THIS SENTENCE STILL BE TRUE AND USEFUL ON A DIFFERENT BOARD?
- "Board Report": states what is on this board — where pieces are, which columns are legal or full, whose turn it is. Pure description. Not true of any other board.
- "Move Verdict": names the move to play, with NO reason attached ("play column 4"). If a reason is attached, also code the reason separately.
- "Local Justification": gives the reason a move is right HERE, naming specific squares, columns, rows or threats on this board ("column 4 blocks their diagonal at row 3"). Would be false on a different board.
- "Worked Line": traces a concrete sequence of consequent moves ("if you play 4, they must take 5, then you have a double threat"). Two or more plies. If only one move is named with no continuation, it is Local Justification.
- "General Principle": states a rule, heuristic or concept that is NOT tied to this position and would hold on other boards ("odd threats favour the second player", "control of the centre gives more winning lines"). A principle stated and THEN applied to this board is BOTH General Principle and Local Justification.

The scaffolding acts:
- "Hint": points at where to look without resolving it ("the key area is column 4"). If the text states the conclusion, it is not a Hint.
- "Prompt": a question the user is expected to answer or work out ("what does Yellow threaten now?"), including an easier sub-question handed back to them.
- "Comprehension Gauging Question": checks the user followed ("does that make sense?"). Not about puzzle content.

Reacting to the user:
- "Paraphrase": restates the USER's own proposal or reasoning back to them. Adding new board facts is Board Report or Local Justification, not Paraphrase.
- "Positive Feedback" / "Negative Feedback" / "Neutral Feedback": endorses / rejects / hedges the user's idea or move.
"""
    utterance = dspy.InputField(desc="The assistant reply to be classified.")
    taxonomy = dspy.InputField(desc="Taxonomy of tutor dialogue acts, with example utterances.")
    dialogue_acts: list[FineAssistantAct] = dspy.OutputField(
        desc="Every fine-grained tutor dialogue act applicable to this reply.")


class FineAssistantActSuite(dspy.Module):
    """Same 3-model panel + per-act majority vote as the coarse suites."""

    def __init__(self, arbiter=None, callbacks=None, lm_timeout=120, wall_timeout=150):
        super().__init__(callbacks)
        self.wall_timeout = wall_timeout
        specs = {
            "llama": dspy.LM("together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
                             temperature=0.0, max_tokens=2048, timeout=lm_timeout),
            "gpt": dspy.LM("openai/gpt-5.4-mini", max_tokens=2048, timeout=lm_timeout),
            "sonnet": dspy.LM("anthropic/claude-sonnet-5", max_tokens=2048, timeout=lm_timeout),
        }
        self.annotators = {}
        for name, lm in specs.items():
            a = dspy.ChainOfThought(FineAssistantActSignature)
            a.set_lm(lm)
            self.annotators[name] = a
        self.arbiter = None
        if arbiter is not None:
            self.arbiter = dspy.ChainOfThought(FineAssistantActSignature)
            self.arbiter.set_lm(arbiter)

    def _annotate(self, name, annotator, utterance):
        try:
            acts = annotator(utterance=utterance, taxonomy=TAXONOMY).dialogue_acts
            valid = set(acts)
            return name, [k for k in ROLLUP if k in valid]
        except Exception as e:
            _log.warning("fine assistant annotator %r failed: %s: %s", name, type(e).__name__, e)
            return name, None

    def forward(self, utterance, min_votes=None):
        per_model = {}
        ex = concurrent.futures.ThreadPoolExecutor(max_workers=len(self.annotators))
        fut_to_name = {ex.submit(self._annotate, n, a, utterance): n
                       for n, a in self.annotators.items()}
        try:
            for fut in concurrent.futures.as_completed(fut_to_name, timeout=self.wall_timeout):
                name, acts = fut.result()
                per_model[name] = acts
        except concurrent.futures.TimeoutError:
            pass
        finally:
            for fut, name in fut_to_name.items():
                if name not in per_model:
                    fut.cancel()
                    _log.warning("fine assistant annotator %r exceeded wall timeout", name)
                    per_model[name] = None
            ex.shutdown(wait=False)

        valid = [a for a in per_model.values() if a is not None]
        n_valid = len(valid)
        votes = Counter(a for acts in valid for a in acts)
        if min_votes is None:
            min_votes = n_valid // 2 + 1 if n_valid else 1
        consensus = [k for k in ROLLUP if votes[k] >= min_votes]
        confidence = {a: votes[a] / n_valid for a in votes} if n_valid else {}
        no_majority = not consensus and bool(votes)
        final, arbiter_used = consensus, False
        if no_majority:
            if self.arbiter is not None:
                _, arb = self._annotate("arbiter", self.arbiter, utterance)
                if arb is not None:
                    final, arbiter_used = arb, True
                else:
                    final = [k for k in ROLLUP if votes[k] >= 1]
            else:
                final = [k for k in ROLLUP if votes[k] >= 1]
        return {"final": final, "consensus": consensus, "votes": dict(votes),
                "confidence": confidence, "per_model": per_model, "n_valid": n_valid,
                "min_votes": min_votes, "no_majority": no_majority,
                "needs_review": no_majority or n_valid < 2, "arbiter_used": arbiter_used,
                "coarse": to_coarse(final)}


def fleiss_kappa(records, raters=None):
    """Multi-label Fleiss' kappa over the FINE codes (same estimator as the coarse
    modules, re-pointed at ROLLUP's key order)."""
    raters = raters or sorted({r for rec in records for r in rec})
    items = _complete_cases(records, raters)
    n = len(raters)
    if len(items) < 2 or n < 2:
        return float("nan"), {}
    by_act = {}
    for act in ROLLUP:
        counts = [sum(act in items[i][r] for r in raters) for i in range(len(items))]
        if all(c == 0 for c in counts) or all(c == n for c in counts):
            continue
        P = [(c * c + (n - c) ** 2 - n) / (n * (n - 1)) for c in counts]
        Pbar = sum(P) / len(P)
        p_present = sum(counts) / (len(counts) * n)
        Pe = p_present ** 2 + (1 - p_present) ** 2
        by_act[act] = 1.0 if Pe == 1 else (Pbar - Pe) / (1 - Pe)
    macro = sum(by_act.values()) / len(by_act) if by_act else float("nan")
    return macro, by_act
