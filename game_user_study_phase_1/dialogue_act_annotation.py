import concurrent.futures
import logging
from collections import Counter
from typing import Literal

import dspy

_log = logging.getLogger("dialogue_act_annotation")

# Scheme membership for the acts actually used (T = Tutor move, S = Student move)
SCHEME = {
    # Student moves
    "Think Aloud": "S", "Conversational Acknowledgment": "S",
    "Knowledge Deficit Question": "S", "Misconception": "S",
    "Common Ground Question": "S", "Vague Answer": "S", "Partial Answer": "S",
    "Social Coordination Action": "S", "Metacomment": "S", "Read Aloud": "S",
    # "Correct Answer" dropped: in the source scheme it means a student answering a
    # tutor's posed question — there's no tutor quizzing the participant here, so it
    # was misfiring on terse move statements (e.g. "Column 4").
    # Added for the human<->AI help-seeking setting: an OPEN request for the
    # assistant to supply the move/answer (no candidate proposed). The published
    # tutoring scheme has no clean student act for this, so such requests were
    # scattering into Knowledge Deficit Question / Prompt / Forced Choice.
    "Solution Request": "S",
    # Tutor moves (used where the user adopts a tutor-like role)
    "Forced Choice": "T", "Repetition": "T", "Prompt": "T",
}

# In-context examples for each dialogue move (transcribed from Table I,
# "Coding Schemes for Dialogue Moves"). Keyed by the move name; useful as
# few-shot references for coding/labeling.
EXAMPLES = {
    # --- Tutor Moves ---
    "Comprehension Gauging Question": '"Do you understand", "You remember?"',
    "Attribution Acknowledgment": '"You\'re just making a mental block."',
    "Conversational Ok": '"Ok." "Alright."',
    "Counter Example": '"And I don\'t mean chunks like in pieces of potato."',
    "Direct Instruction": '"Well the lysosomes contain digestive enzymes."',
    "Example": '"Most picked FedEx as an analogy for Golgi body."',
    "Forced Choice": '"Would that be random, uniformed, or clumped?"',
    "Motivational Statement": '"See, you get it now." "Wow, you\'re so good."',
    "Hint": '"It\'s a family tree…it\'s just a regular family tree."',
    "Humor": '"You never see a rose bush jump on a person."',
    "Negative Feedback": '"No." "Uh uh."',
    "Negative Feedback Elaborated": '"Before that, you\'re jumping too far ahead."',
    "Neutral Feedback": '"Well, that\'s not quite it."',
    "Neutral Feedback Elaborated": '"Uh, ATP is gonna be one of the players."',
    "Paraphrase": '[S] "It ………....." [T] "It\'s inside the wall."',
    "Off topic": '"I\'m cooking dinner—I\'m really excited."',
    "Pose New Problem": '"Difference between prokaryotes and eukaryotes."',
    "Pose Simplified Problem": '"What is the outside layer?"',
    "Positive Feedback": '"Correct." "Right." "Exactly."',
    "Positive Feedback Elaborated": '"It\'s still random, exactly."',
    "Preview": '"Chloroplast, we haven\'t talked about that yet."',
    "Prompt": '"So 200 times one is what?"',
    "Provide Correct Answer": '"Code next to it."',
    "Pump": '"Which is?" "And?" "Then?"',
    "Repetition": 'S: "Is it commensalism?" T: "Commensalism."',
    "Solidarity Statement": '"Now think with me." "Alright, here we go."',
    "Summary": '"Alright, so we\'ve talked about…"',
    # --- Student Moves ---
    "Common Ground Question": '"Aren\'t they more lined up, like more in order?"',
    "Conversational Acknowledgment": '"Ok." "No sir." "Yes ma\'am."',
    "Correct Answer": '"In meiosis it starts out the same with 1 diploid."',
    "Error Ridden Answer": '"Prokaryotes are human, eukaryotes are bacteria."',
    "Gripe": '"Ugh." "<groans>"',
    "Knowledge Deficit Question": '"What do you mean by it doesn\'t have a skeleton?"',
    "Metacomment": '"I don\'t know." "Yes, I understand."',
    "Misconception": '"I always used to get diploid and haploid mixed up."',
    "No Answer": '"Umm." "Mmm."',
    "Off topic (Tutor)": '"Did you all have a quiz today."',
    "Partial Answer": '"It has to do with the cells."',
    "Read Aloud": '"Question 7: Plot growth pattern."',
    "Social Coordination Action": '"No, I didn\'t hear about that."',
    "Student Works Silently": "",   # no example given in the table
    "Think Aloud": '"500 equals 50 and 50 divided by 500 gives 10."',
    "Vague Answer": '"Because it helps to, umm, you know."',
    # Not from Table I — added for this setting (open request for the answer/move):
    "Solution Request": '"What\'s the best move?" "What now?" "Just tell me what to play." "Any advice for the next move?"',
}

TAXONOMY = ""
for k in SCHEME:
    TAXONOMY = TAXONOMY + "DIALOGUE ACT: " + k + ", EXAMPLES: " + EXAMPLES[k] + "\n"

# Constrain outputs to exactly the acts in SCHEME. Literal[...] makes dspy's
# underlying Pydantic model validate each item against this closed set (and the
# allowed values are surfaced to the LM in the output schema).
DialogueAct = Literal['Think Aloud', 'Conversational Acknowledgment', 'Knowledge Deficit Question', 'Common Ground Question', 'Metacomment', 'Solution Request']


class DialogueActClassifierSignature(dspy.Signature):
    """Given an utterance from a user talking to an AI assistant while solving a Connect Four puzzle, first split it into clauses, then classify each clause into the dialogue acts that apply per the taxonomy provided. Output all applicable acts.

Disambiguating the question-type acts (these are easily confused — read carefully):
- "Solution Request": an OPEN request for the assistant to supply the move/answer, with NO specific candidate move proposed by the user. E.g. "what's the best move?", "what now?", "any advice?", "just tell me what to play".
- "Common Ground Question": the user proposes a SPECIFIC candidate move (or their own read of the position) and asks the assistant to confirm/evaluate it. E.g. "is column 3 good?", "was my last move good?", "would 3/4 be better?".
- "Knowledge Deficit Question": asks about a concept, rule, or the board (e.g. threats), or clarifies something the assistant said — NOT a request for the move itself. E.g. "why not column 2?", "are there any threats?", "what do you mean by that?".
"""
    utterance = dspy.InputField(desc="The utterance to be classified.")
    taxonomy = dspy.InputField(desc="Taxonomy of dialogue acts, with example utterances.")
    dialogue_acts: list[DialogueAct] = dspy.OutputField(desc="The list of dialogue acts applicable to this utterance.")

class DialogueActSuite(dspy.Module):
    def __init__(self, arbiter=None, callbacks=None, lm_timeout=120, wall_timeout=150):
        """arbiter: an optional dspy.LM used to break ties. When no act reaches a
        majority, the arbiter re-labels the utterance and its labels are used as
        `final` (the item is still flagged needs_review). If None, ties fall back
        to the union of the panel.

        lm_timeout:   per-model API timeout (seconds) enforced by dspy/litellm on
                      the HTTP call.
        wall_timeout: wall-clock backstop for the whole panel in forward(). Even
                      if a model's own timeout fails to fire and its call hangs,
                      forward() stops waiting after this many seconds and drops
                      the straggler from the vote (fail-open). Keep it a bit above
                      lm_timeout so the per-model timeout normally wins."""
        super().__init__(callbacks)
        self.wall_timeout = wall_timeout
        # Panel of annotators (mirrors prompt_filter.JudgeSuite). Each model
        # independently labels the utterance; we then majority-vote per act.
        # The unit of agreement is a LABEL SET, not a 1-4 score, so aggregation
        # is a per-act vote rather than a mean.
        # timeout caps a slow/straggling model; on timeout that annotator
        # errors, is dropped, and the panel votes among the survivors (fail-open).
        # These are OFFLINE labels, so latency isn't a constraint -> generous
        # timeouts. One model per provider (Meta / OpenAI / Anthropic) so the
        # votes stay independent, which is what the majority rule assumes.
        # Gemma-4-31B held the third seat until its serving latency drifted to a
        # p50 of ~100s (tail past 300s), past wall_timeout, so it kept getting
        # dropped from the vote; Sonnet measures p50 ~3s on the same turns.
        # Sonnet takes no temperature: it 400s on any non-default sampling
        # param, and 0.0 is not the default. Thinking is on by default and
        # shares the max_tokens budget with the reply, so disable it -- this
        # signature only emits a short label list.
        specs = {
            "llama": dspy.LM("together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo", temperature=0.0, max_tokens=2048, timeout=lm_timeout),
            "gpt": dspy.LM("openai/gpt-5.4-mini", max_tokens=2048, timeout=lm_timeout),
            "sonnet": dspy.LM("anthropic/claude-sonnet-5", max_tokens=2048, timeout=lm_timeout),
        }
        self.annotators = {}
        for name, lm in specs.items():
            a = dspy.ChainOfThought(DialogueActClassifierSignature)
            a.set_lm(lm)
            self.annotators[name] = a

        self.arbiter = None
        if arbiter is not None:
            self.arbiter = dspy.ChainOfThought(DialogueActClassifierSignature)
            self.arbiter.set_lm(arbiter)

    def _annotate(self, name, annotator, utterance):
        try:
            acts = annotator(utterance=utterance, taxonomy=TAXONOMY).dialogue_acts
            # dedup + drop anything off-scheme, keep canonical SCHEME order
            valid = set(acts)
            return name, [k for k in SCHEME if k in valid]
        except Exception as e:
            # Log the reason so a dead/misconfigured model surfaces instead of
            # silently shrinking the panel.
            _log.warning("dialogue-act annotator %r failed: %s: %s", name, type(e).__name__, e)
            return name, None       # this model failed; it's dropped from the vote

    def forward(self, utterance, min_votes=None):
        """Annotate `utterance` with every model in parallel, then majority-vote.

        Returns a dict:
          - final:        the labels to use (consensus; or arbiter's labels /
                          union when there's no majority) in SCHEME order
          - consensus:    acts predicted by >= min_votes annotators
          - votes:        {act: n_annotators that predicted it}
          - confidence:   {act: votes/n_valid} — share of the panel that agreed
          - per_model:    {name: [acts] or None if that annotator failed}
          - n_valid:      number of annotators that succeeded
          - min_votes:    threshold used
          - no_majority:  True if no act cleared the threshold
          - needs_review: True if no_majority or too few annotators succeeded
          - arbiter_used: True if the arbiter was called to break a tie
        min_votes defaults to a strict majority of the annotators that succeeded.
        """
        per_model = {}
        # Wall-clock backstop: collect results as they complete, but stop waiting
        # after self.wall_timeout. A model whose call hangs past that (its own
        # per-request timeout failed to fire) is dropped from the vote instead of
        # blocking forever. We DON'T use the `with` form because its __exit__ calls
        # shutdown(wait=True), which would re-introduce the hang; shutdown(wait=False)
        # lets a stuck network thread die in the background.
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
                if name not in per_model:            # never finished within wall_timeout
                    fut.cancel()
                    _log.warning("dialogue-act annotator %r exceeded wall timeout %ss; "
                                 "dropping from vote", name, self.wall_timeout)
                    per_model[name] = None
            ex.shutdown(wait=False)
        valid = [acts for acts in per_model.values() if acts is not None]
        n_valid = len(valid)
        votes = Counter(act for acts in valid for act in acts)
        if min_votes is None:
            min_votes = n_valid // 2 + 1 if n_valid else 1   # strict majority
        consensus = [k for k in SCHEME if votes[k] >= min_votes]
        confidence = {act: votes[act] / n_valid for act in votes} if n_valid else {}

        # Any act that at least one model raised but that didn't clear the
        # threshold => the panel split. No consensus label at all is the hard case.
        no_majority = not consensus and bool(votes)
        needs_review = no_majority or n_valid < 2

        final, arbiter_used = consensus, False
        if no_majority:
            if self.arbiter is not None:
                _, arb_acts = self._annotate("arbiter", self.arbiter, utterance)
                if arb_acts is not None:
                    final, arbiter_used = arb_acts, True
                else:                                   # arbiter also failed -> union
                    final = [k for k in SCHEME if votes[k] >= 1]
            else:                                       # no arbiter -> union, flagged
                final = [k for k in SCHEME if votes[k] >= 1]

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


# ---------------------------------------------------------------------------
# Panel agreement metrics (run offline over a batch of `per_model` records).
# These are pure-python (no dspy / no API) so they're cheap to test.
# ---------------------------------------------------------------------------
def _complete_cases(records, raters):
    """Keep only items where every rater produced labels (drop items with a
    failed/None annotator, since κ needs the same rater set on every item)."""
    return [{r: set(rec[r]) for r in raters}
            for rec in records
            if all(rec.get(r) is not None for r in raters)]


def mean_pairwise_jaccard(records, raters=None):
    """Average, over items and over rater pairs, of the Jaccard overlap between
    two raters' label sets. 1.0 = identical sets; two empty sets count as 1.0.
    An intuitive set-agreement number to complement κ."""
    raters = raters or sorted({r for rec in records for r in rec})
    items = _complete_cases(records, raters)
    if not items or len(raters) < 2:
        return float("nan")
    pair_scores = []
    for it in items:
        for i in range(len(raters)):
            for j in range(i + 1, len(raters)):
                a, b = it[raters[i]], it[raters[j]]
                if not a and not b:
                    pair_scores.append(1.0)
                else:
                    pair_scores.append(len(a & b) / len(a | b))
    return sum(pair_scores) / len(pair_scores)


def fleiss_kappa(records, raters=None):
    """Multi-label Fleiss' κ: treat each dialogue act as an independent
    binary (present/absent) rating, compute Fleiss' κ per act, and macro-average
    over the acts that actually occur. Returns (macro_kappa, kappa_by_act).

    records: list of per_model dicts, e.g. [{'llama': [...], 'gpt': [...]}, ...]
    """
    raters = raters or sorted({r for rec in records for r in rec})
    items = _complete_cases(records, raters)
    n = len(raters)
    if len(items) < 2 or n < 2:
        return float("nan"), {}

    kappa_by_act = {}
    for act in SCHEME:
        # per-item count of raters who assigned this act
        counts = [sum(act in items[i][r] for r in raters) for i in range(len(items))]
        if all(c == 0 for c in counts) or all(c == n for c in counts):
            continue                      # act never/always used -> κ undefined
        # P_i = observed agreement among rater pairs on item i. For binary
        # (present/absent) categories the sum of squared counts is c^2 + (n-c)^2.
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

    # single-model classifier
    predicter = dspy.ChainOfThought(DialogueActClassifierSignature)
    dspy.configure(lm=dspy.LM("gpt-5.5"))
    print(predicter(utterance="I don't know", taxonomy=TAXONOMY))

    # panel of annotators with per-act majority vote (+ optional arbiter)
    suite = DialogueActSuite(arbiter=dspy.LM("gpt-5.5"))
    utterances = ["I don't know", "So 200 times one is what?", "Is it commensalism?"]
    records = []
    for u in utterances:
        r = suite(utterance=u)
        records.append(r["per_model"])
        flag = "  <needs review>" if r["needs_review"] else ""
        print(f"\n{u!r}{flag}")
        print("  final:     ", r["final"])
        print("  confidence:", {k: round(v, 2) for k, v in r["confidence"].items()})
        print("  per model: ", r["per_model"])

    # panel agreement over the batch
    macro, by_act = fleiss_kappa(records)
    print(f"\nmean pairwise Jaccard: {mean_pairwise_jaccard(records):.3f}")
    print(f"Fleiss' kappa (macro): {macro:.3f}")
    print("kappa by act:", {k: round(v, 2) for k, v in by_act.items()})
