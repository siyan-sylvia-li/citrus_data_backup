"""Othello wording for the assistant (tutor-move) schemes, coarse and fine.

The act inventories are game-agnostic — `ASSISTANT_SCHEME` and `ROLLUP` name tutor
moves, not board features. Only the EXAMPLES and the disambiguation prose are
Connect Four–specific ("play column 4", "Yellow has discs at c7r1"). Running those
on Othello would code the wrong game, exactly as the Connect Four student annotator
would.

So this module re-instructs the EXISTING signatures rather than duplicating them:
`Signature.with_instructions()` returns a new class and leaves the original alone, so
the two games share one act inventory and one voting implementation, and a change to
the taxonomy cannot drift between them. That is deliberately not the pattern used on
the student side, where the Connect Four variant is a whole copied file whose name
differs by a space — which is why the notebook has to load it by path.

Usage:
    from assistant_acts_othello import OthelloAssistantSuite, OthelloFineAssistantSuite
    suite = OthelloFineAssistantSuite()
    suite(utterance="Take the corner at h8 — corners can never be flipped.")
"""
from __future__ import annotations

import dspy

import dialogue_act_annotation_assistant as coarse
import dialogue_act_annotation_assistant_fine as fine

# --- coarse scheme, Othello wording ------------------------------------------
COARSE_EXAMPLES = {
    "Provide Correct Answer": '"Play **h8**." "I\'d take g2 or h1."',
    "Direct Instruction": '"Black has stable discs along the a-file." '
                          '"In Othello, a corner can never be flipped."',
    "Hint": '"The key area to inspect is the h-file." "Look at what happens to the c8 group."',
    "Prompt": '"After you play there, which discs can White flip back?"',
    "Pose Simplified Problem": '"If you do nothing, can White reach a corner next move?"',
    "Comprehension Gauging Question": '"Does that make sense?" "Are you with me?"',
    "Paraphrase": '"Your g2 idea would flip the whole second row."',
    "Positive Feedback": '"Nice — that keeps the edge." "That\'s right."',
    "Negative Feedback": '"g7 isn\'t legal here — it flips nothing."',
    "Neutral Feedback": '"That does win discs now, but it opens the corner."',
}

COARSE_INSTRUCTIONS = """Given a reply from an AI assistant that is helping a user solve an Othello endgame puzzle, first split it into clauses, then classify each clause into the tutor dialogue acts that apply per the taxonomy provided. Output all applicable acts. Assistant replies are long and usually carry several acts; do not stop at the first one.

Disambiguating the acts that are easily confused — read carefully:
- "Provide Correct Answer": names a concrete square for the user to play ("play h8", "I'd suggest g2"). Only for the MOVE ITSELF, not for other factual content.
- "Direct Instruction": asserts content the user did not supply — a rule of the game, a strategic principle, or a reading of the board ("White has stable discs on the h-file"). It RESOLVES the question rather than handing it back.
- "Hint": points the user toward where the answer lies without resolving it ("the key area is the bottom edge", "look at the c8 group"). If the text states the conclusion, it is Direct Instruction, not a Hint.
- "Prompt": a question that cues one specific next step the user should take or check ("after you play there, what can White flip back?"). Rhetorical questions the assistant then answers itself are Prompts only if the user is expected to work it out.
- "Pose Simplified Problem": hands back an easier, self-contained sub-question the user is meant to solve ("can White reach a corner if you pass?"). Prompt cues the next step in the CURRENT line; Pose Simplified Problem substitutes a smaller problem.
- "Comprehension Gauging Question": checks whether the user followed ("does that make sense?"). Not about puzzle content.
- "Paraphrase": restates the user's own proposal, move, or reasoning back to them ("your g2 idea would flip the second row"). Restating the USER; adding new board facts is Direct Instruction.
- "Positive Feedback": affirms the user's idea or move as correct/good.
- "Negative Feedback": marks the user's idea or move as wrong ("that square is not legal, it flips nothing").
- "Neutral Feedback": neither endorses nor rejects — partial credit or a hedge ("that wins discs now, but it opens the corner").
"""

# --- fine scheme, Othello wording --------------------------------------------
# The transfer distinction translates more naturally here than in Connect Four:
# corner permanence, edge stability and parity are stated as portable rules far more
# often than Connect Four threat geometry, so General Principle should be less rare.
FINE_EXAMPLES = {
    "Move Verdict": '"Play **h8**." "I\'d take g2." (the square alone, no reason)',
    "Board Report": '"You have 12 discs to White\'s 8." "Your legal moves are g2, b6 and c8."',
    "Local Justification": '"h8 is right because it takes the corner and locks the h-file."',
    "Worked Line": '"If you play g2, White must answer h1, and then b6 gives you the edge."',
    "General Principle": '"Corners can never be flipped, so they are worth more than the discs they cost." '
                         '"Having fewer moves late is often an advantage."',
    "Hint": '"The key area to inspect is the h-file." "Look at what happens to the c8 group."',
    "Prompt": '"After you play there, which discs can White flip back?"',
    "Comprehension Gauging Question": '"Does that make sense?" "Are you with me?"',
    "Paraphrase": '"Your g2 idea would flip the whole second row."',
    "Positive Feedback": '"Nice — that keeps the edge."',
    "Negative Feedback": '"g7 isn\'t legal here — it flips nothing."',
    "Neutral Feedback": '"That does win discs now, but it opens the corner."',
}

FINE_INSTRUCTIONS = """Given a reply from an AI assistant helping a user solve an Othello endgame puzzle, first split it into clauses, then classify each clause into the tutor dialogue acts that apply. Output every act that applies. Assistant replies are long and usually carry several acts; do not stop at the first one.

The four content acts matter most and are easiest to confuse. Decide them by asking: WOULD THIS SENTENCE STILL BE TRUE AND USEFUL ON A DIFFERENT BOARD?
- "Board Report": states what is on this board — which squares hold which colour, whose move it is, which moves are legal, the disc counts. Pure description, not true of any other board.
- "Move Verdict": names the square to play with NO reason attached ("play h8"). If a reason is attached, code the reason separately too.
- "Local Justification": the reason a move is right HERE, naming specific squares, files, rows or groups on this board ("h8 takes the corner and locks the h-file"). Would be false on a different board.
- "Worked Line": traces a concrete sequence of consequent moves ("if you play g2, White must take h1, then b6 gives you the edge"). Two or more plies. One move named with no continuation is Local Justification.
- "General Principle": a rule, heuristic or concept NOT tied to this position, which would hold on other boards ("corners can never be flipped", "mobility matters more than disc count in the midgame", "parity decides who takes the last move"). A principle stated and THEN applied to this board is BOTH General Principle and Local Justification.

The scaffolding acts:
- "Hint": points at where to look without resolving it. If the text states the conclusion, it is not a Hint.
- "Prompt": a question the user is expected to answer or work out, including an easier sub-question handed back to them.
- "Comprehension Gauging Question": checks the user followed. Not about puzzle content.

Reacting to the user:
- "Paraphrase": restates the USER's own proposal or reasoning. Adding new board facts is Board Report or Local Justification.
- "Positive Feedback" / "Negative Feedback" / "Neutral Feedback": endorses / rejects / hedges the user's idea or move.
"""

COARSE_TAXONOMY = "".join(
    f"DIALOGUE ACT: {k}, EXAMPLES: {COARSE_EXAMPLES[k]}\n" for k in coarse.ASSISTANT_SCHEME)
FINE_TAXONOMY = "".join(
    f"DIALOGUE ACT: {k}, EXAMPLES: {FINE_EXAMPLES[k]}\n" for k in fine.ROLLUP)

OthelloCoarseSignature = coarse.AssistantDialogueActClassifierSignature.with_instructions(
    COARSE_INSTRUCTIONS)
OthelloFineSignature = fine.FineAssistantActSignature.with_instructions(FINE_INSTRUCTIONS)


def _retarget(suite, signature, taxonomy):
    """Point an existing suite's annotators at the Othello signature and taxonomy.

    Reuses the panel, timeouts, fail-open vote and per-act majority rule already
    implemented; only the prompt changes. Keeping one voting implementation is the
    whole point — a second copy would drift.
    """
    for name, ann in suite.annotators.items():
        lm = ann.get_lm()
        new = dspy.ChainOfThought(signature)
        new.set_lm(lm)
        suite.annotators[name] = new
    if suite.arbiter is not None:
        lm = suite.arbiter.get_lm()
        suite.arbiter = dspy.ChainOfThought(signature)
        suite.arbiter.set_lm(lm)
    suite._othello_taxonomy = taxonomy
    return suite


class OthelloAssistantSuite(coarse.AssistantDialogueActSuite):
    """Coarse tutor-move panel, Othello wording."""

    def __init__(self, **kw):
        super().__init__(**kw)
        _retarget(self, OthelloCoarseSignature, COARSE_TAXONOMY)

    def _annotate(self, name, annotator, utterance):
        try:
            acts = annotator(utterance=utterance,
                             taxonomy=self._othello_taxonomy).dialogue_acts
            valid = set(acts)
            return name, [k for k in coarse.ASSISTANT_SCHEME if k in valid]
        except Exception as e:
            coarse._log.warning("othello coarse annotator %r failed: %s: %s",
                                name, type(e).__name__, e)
            return name, None


class OthelloFineAssistantSuite(fine.FineAssistantActSuite):
    """Fine tutor-move panel, Othello wording. Same ROLLUP, so results collapse to
    the coarse codes exactly as on Connect Four."""

    def __init__(self, **kw):
        super().__init__(**kw)
        _retarget(self, OthelloFineSignature, FINE_TAXONOMY)

    def _annotate(self, name, annotator, utterance):
        try:
            acts = annotator(utterance=utterance,
                             taxonomy=self._othello_taxonomy).dialogue_acts
            valid = set(acts)
            return name, [k for k in fine.ROLLUP if k in valid]
        except Exception as e:
            fine._log.warning("othello fine annotator %r failed: %s: %s",
                              name, type(e).__name__, e)
            return name, None
