"""Assistant-side dialogue-act annotation (TUTOR moves from Table I).

Mirror of `dialogue_act_annotation.py`, which codes the participant's turns with the
STUDENT moves. Here we code the AI assistant's reply in the same exchange with the
TUTOR moves, so a turn of `conversation_p15.jsonl` ends up with a student-act label
set and a tutor-act label set that can be cross-tabulated.

Design notes on the subset of Table I kept (the full 26 tutor moves are far more than
this setting supports; the participant side likewise kept ~10 of 17 student moves):

  * KEPT because they carry the pedagogical contrast this study is about --
    does the assistant hand over the answer, or make the participant work?
      Provide Correct Answer, Direct Instruction, Hint, Prompt,
      Pose Simplified Problem, Comprehension Gauging Question, Paraphrase,
      Positive / Negative / Neutral Feedback.

  * The "Elaborated" feedback variants are COLLAPSED into their base act. In the
    source (human tutor speech) "Positive Feedback" is a bare "Right." and the
    elaborated form adds content; an LLM assistant essentially always elaborates,
    so the distinction is near-degenerate here and would only split panel votes.

  * DROPPED as not occurring / not separable in this setting: Attribution
    Acknowledgment, Conversational Ok, Counter Example, Example, Forced Choice,
    Motivational Statement (praise lands in Positive Feedback), Humor, Off topic,
    Pose New Problem (the puzzle is fixed), Preview, Pump (merges with Prompt),
    Repetition (merges with Paraphrase), Solidarity Statement, Summary.

Edit ASSISTANT_SCHEME / the signature docstring below to change the coding scheme;
everything downstream (the DialogueAct Literal, the taxonomy block, the vote) is
derived from it.
"""

import concurrent.futures
import logging
from collections import Counter
from typing import Literal

import dspy

from dialogue_act_annotation import EXAMPLES, _complete_cases

_log = logging.getLogger("dialogue_act_annotation_assistant")

# Tutor moves kept for the AI assistant. All are "T" in Table I's scheme.
ASSISTANT_SCHEME = {
    # --- telling: the assistant supplies content ---
    "Provide Correct Answer": "T",     # names the move to play
    "Direct Instruction": "T",         # states a fact / rule / board reading
    # --- scaffolding: the assistant makes the participant do the work ---
    "Hint": "T",                       # points at where to look without resolving it
    "Prompt": "T",                     # question cueing a specific next step
    "Pose Simplified Problem": "T",    # hands back an easier sub-question
    "Comprehension Gauging Question": "T",
    # --- reacting to what the participant said ---
    "Paraphrase": "T",                 # restates the participant's move/reasoning
    "Positive Feedback": "T",
    "Negative Feedback": "T",
    "Neutral Feedback": "T",
}

# Extra examples for acts whose Table I example is from the biology-tutoring corpus
# and reads oddly against a Connect Four assistant. Table I's wording is kept and
# this setting's wording is appended, so the coder sees both.
SETTING_EXAMPLES = {
    "Provide Correct Answer": '"Play **column 4**." "I\'d suggest column 4 or column 5."',
    "Direct Instruction": '"Yellow has discs at c7r1, c6r2 and c5r3." "In Connect Four, defense vs offense depends on immediacy."',
    "Hint": '"The key area to inspect is column 4." "Look especially at Yellow\'s cluster around columns 3-6."',
    "Prompt": '"After Red plays 3,3, does Red create an immediate threat Yellow must answer?"',
    "Pose Simplified Problem": '"If Yellow gets the next move, is there any square that would make four in a row?"',
    "Paraphrase": '"Your column 2 idea would land at row 3, column 2 and make a vertical three."',
    "Positive Feedback": '"Nice block." "Good idea to look for a can\'t-block-both setup."',
    "Negative Feedback": '"That column is already full, so it isn\'t playable."',
    "Neutral Feedback": '"Your idea at 3,3 does help build, but ask yourself..."',
}

TAXONOMY = ""
for k in ASSISTANT_SCHEME:
    ex = EXAMPLES[k]
    extra = SETTING_EXAMPLES.get(k)
    if extra:
        ex = f"{ex} | in this setting: {extra}"
    TAXONOMY += "DIALOGUE ACT: " + k + ", EXAMPLES: " + ex + "\n"

# Closed label set (Pydantic-validated by dspy, and surfaced to the LM).
AssistantDialogueAct = Literal[
    "Provide Correct Answer", "Direct Instruction", "Hint", "Prompt",
    "Pose Simplified Problem", "Comprehension Gauging Question", "Paraphrase",
    "Positive Feedback", "Negative Feedback", "Neutral Feedback",
]


class AssistantDialogueActClassifierSignature(dspy.Signature):
    """Given a reply from an AI assistant that is helping a user solve a Connect Four puzzle, first split it into clauses, then classify each clause into the tutor dialogue acts that apply per the taxonomy provided. Output all applicable acts. Assistant replies are long and usually carry several acts; do not stop at the first one.

Disambiguating the acts that are easily confused — read carefully:
- "Provide Correct Answer": names a concrete move for the user to play ("play column 4", "I'd suggest column 4"). Only for the MOVE ITSELF, not for other factual content.
- "Direct Instruction": asserts content the user did not supply — a rule of the game, a strategic principle, or a reading of the board ("Yellow threatens a diagonal at c4r4"). It RESOLVES the question rather than handing it back.
- "Hint": points the user toward where the answer lies without resolving it ("the key area to inspect is column 4", "look at Yellow's cluster on rows 1-2"). If the text states the conclusion, it is Direct Instruction, not a Hint.
- "Prompt": a question that cues one specific next step the user should take or check ("after you play there, what does Yellow threaten?"). Rhetorical questions the assistant then answers itself are still Prompts only if the user is expected to work it out.
- "Pose Simplified Problem": hands back an easier, self-contained sub-question that the user is meant to solve ("can Yellow win immediately if you don't respond?"). Prompt cues the next step in the CURRENT line of reasoning; Pose Simplified Problem substitutes a smaller problem for it.
- "Comprehension Gauging Question": checks whether the user followed ("does that make sense?", "are you with me?"). Not about the puzzle content.
- "Paraphrase": restates the user's own proposal, move, or reasoning back to them ("your column 2 idea would land at row 3 and make a vertical three"). Restating the USER; adding new board facts is Direct Instruction.
- "Positive Feedback": affirms the user's idea or move as correct/good ("nice block", "that's right", "good idea to check that").
- "Negative Feedback": marks the user's idea or move as wrong ("that column is full", "no, that wouldn't stop the diagonal").
- "Neutral Feedback": neither endorses nor rejects — partial credit or a hedge ("that does help build, but...", "it's tempting, though consider...").
"""
    utterance = dspy.InputField(desc="The assistant reply to be classified.")
    taxonomy = dspy.InputField(desc="Taxonomy of tutor dialogue acts, with example utterances.")
    dialogue_acts: list[AssistantDialogueAct] = dspy.OutputField(desc="The list of dialogue acts applicable to this assistant reply.")


class AssistantDialogueActSuite(dspy.Module):
    """Same 3-model panel + per-act majority vote as `DialogueActSuite`; see that
    class for the rationale on model choice, timeouts and the fail-open vote."""

    def __init__(self, arbiter=None, callbacks=None, lm_timeout=120, wall_timeout=150):
        super().__init__(callbacks)
        self.wall_timeout = wall_timeout
        specs = {
            "llama": dspy.LM("together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo", temperature=0.0, max_tokens=2048, timeout=lm_timeout),
            "gpt": dspy.LM("openai/gpt-5.4-mini", max_tokens=2048, timeout=lm_timeout),
            "sonnet": dspy.LM("anthropic/claude-sonnet-5", max_tokens=2048, timeout=lm_timeout),
        }
        self.annotators = {}
        for name, lm in specs.items():
            a = dspy.ChainOfThought(AssistantDialogueActClassifierSignature)
            a.set_lm(lm)
            self.annotators[name] = a

        self.arbiter = None
        if arbiter is not None:
            self.arbiter = dspy.ChainOfThought(AssistantDialogueActClassifierSignature)
            self.arbiter.set_lm(arbiter)

    def _annotate(self, name, annotator, utterance):
        try:
            acts = annotator(utterance=utterance, taxonomy=TAXONOMY).dialogue_acts
            valid = set(acts)
            return name, [k for k in ASSISTANT_SCHEME if k in valid]
        except Exception as e:
            _log.warning("assistant dialogue-act annotator %r failed: %s: %s", name, type(e).__name__, e)
            return name, None

    def forward(self, utterance, min_votes=None):
        """Annotate `utterance` with every model in parallel, then majority-vote.
        Returns the same dict shape as `DialogueActSuite.forward`."""
        per_model = {}
        ex = concurrent.futures.ThreadPoolExecutor(max_workers=len(self.annotators))
        fut_to_name = {ex.submit(self._annotate, name, a, utterance): name
                       for name, a in self.annotators.items()}
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
                    _log.warning("assistant dialogue-act annotator %r exceeded wall timeout %ss; "
                                 "dropping from vote", name, self.wall_timeout)
                    per_model[name] = None
            ex.shutdown(wait=False)

        valid = [acts for acts in per_model.values() if acts is not None]
        n_valid = len(valid)
        votes = Counter(act for acts in valid for act in acts)
        if min_votes is None:
            min_votes = n_valid // 2 + 1 if n_valid else 1
        consensus = [k for k in ASSISTANT_SCHEME if votes[k] >= min_votes]
        confidence = {act: votes[act] / n_valid for act in votes} if n_valid else {}

        no_majority = not consensus and bool(votes)
        needs_review = no_majority or n_valid < 2

        final, arbiter_used = consensus, False
        if no_majority:
            if self.arbiter is not None:
                _, arb_acts = self._annotate("arbiter", self.arbiter, utterance)
                if arb_acts is not None:
                    final, arbiter_used = arb_acts, True
                else:
                    final = [k for k in ASSISTANT_SCHEME if votes[k] >= 1]
            else:
                final = [k for k in ASSISTANT_SCHEME if votes[k] >= 1]

        return {
            "final": final,
            "consensus": consensus,
            "votes": dict(votes),
            "confidence": confidence,
            "per_model": per_model,
            "n_valid": n_valid,
            "min_votes": min_votes,
            "no_majority": no_majority,
            "needs_review": needs_review,
            "arbiter_used": arbiter_used,
        }


def fleiss_kappa(records, raters=None):
    """Multi-label Fleiss' κ over ASSISTANT_SCHEME (see the participant module's
    docstring; identical maths, different act list)."""
    raters = raters or sorted({r for rec in records for r in rec})
    items = _complete_cases(records, raters)
    n = len(raters)
    if len(items) < 2 or n < 2:
        return float("nan"), {}

    kappa_by_act = {}
    for act in ASSISTANT_SCHEME:
        counts = [sum(act in items[i][r] for r in raters) for i in range(len(items))]
        if all(c == 0 for c in counts) or all(c == n for c in counts):
            continue
        P = [(c * c + (n - c) ** 2 - n) / (n * (n - 1)) for c in counts]
        Pbar = sum(P) / len(P)
        p_present = sum(counts) / (len(counts) * n)
        Pe = p_present ** 2 + (1 - p_present) ** 2
        kappa_by_act[act] = 1.0 if Pe == 1 else (Pbar - Pe) / (1 - Pe)

    macro = sum(kappa_by_act.values()) / len(kappa_by_act) if kappa_by_act else float("nan")
    return macro, kappa_by_act


if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()

    suite = AssistantDialogueActSuite()
    samples = [
        "Play **column 4**.\n\nWhy: Yellow has strong center threats, and column 4 both contests the center and keeps your winning chances alive.",
        "Your idea at **3,3** does help build, but ask yourself: after Red plays 3,3, does Red create an immediate threat Yellow must answer?",
    ]
    for u in samples:
        r = suite(utterance=u)
        print(f"\n{u[:60]!r}{'  <needs review>' if r['needs_review'] else ''}")
        print("  final:     ", r["final"])
        print("  per model: ", r["per_model"])
