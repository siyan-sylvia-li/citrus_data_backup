"""Scheme v3: hierarchical subtypes NESTED UNDER the frozen v1 acts.

Design constraint that drives everything here: the v1 act labels are not
re-annotated. They are read from participant_utterances.csv and used verbatim as
the PARENT level, so every v1-level quantity (including the Think Aloud share
that carries the published R2 effect) is reproduced bit-for-bit by aggregation.
The subtype pass can only ever refine within a parent -- it cannot move a turn
between parents, and it cannot change a parent's count.

This is the opposite of what v2 did. v2 re-drew the act boundaries, which both
crowded articulation out (v2.0) and shifted what counted as articulation at all
(v2.1) -- and the effect did not survive either. Here the boundary that produced
the result is frozen and the new resolution is added strictly below it.

Two things v1 needed and this fixes:
  * UNUSED ACTS. v1's SCHEME carried 13 acts that never reached the classifier's
    closed output set (Misconception, Vague Answer, Partial Answer, Read Aloud,
    Social Coordination Action, Forced Choice, Repetition, Prompt, ...). They are
    retired below rather than left as dead weight in the instrument.
  * OVERLOADED ACTS. Three parents hold 84% of all labels -- Solution Request
    (118), Common Ground Question (107), Think Aloud (104) -- so at v1 resolution
    most of the corpus is three buckets. Each gets subtypes.

NOTE ON WORDING: the subtype definitions are a first draft for the study owner to
edit. They are the measurement instrument, not implementation detail.
"""
import concurrent.futures
import logging
import re
from collections import Counter
from typing import Literal

import dspy

_log = logging.getLogger("dialogue_act_subtypes")

# Retired from v1's SCHEME: present in the source tutoring scheme but never in the
# classifier's closed output set, so never applied to a single turn.
RETIRED_V1_ACTS = ["Misconception", "Vague Answer", "Partial Answer", "Read Aloud",
                   "Social Coordination Action", "Correct Answer", "Forced Choice",
                   "Repetition", "Prompt"]

# Parent act -> {subtype code: definition + examples}. Codes are stable strings;
# the analysis code refers to them, so renaming one is a breaking change.
#
# REDUCED from the 22-subtype first pass to 13 (see RETIRED_SUBTYPES / MERGE_INTO).
# Retirement rule: a subtype goes if it is BOTH too thin to support analysis (n < 10
# of 405 labels) AND has a sibling that conceptually contains it, so retiring it is a
# deterministic relabel rather than a re-annotation. Two thin subtypes are kept in
# spite of the rule -- CA-answer and KDQ-interface -- because the only siblings they
# could merge into would re-create the exact v1 blind spots they were split out to
# fix (no answer act; task confusion polluting the knowledge-question act). They are
# marked DESCRIPTIVE-ONLY: report them, do not regress on them.
SUBTYPES = {
    "Think Aloud": {
        "TA-assess": 'Reads or evaluates the POSITION without naming a candidate move -- what the '
                     'opponent threatens, who stands better, what is possible. '
                     '"They nearly have a row here" / "It looks like neither one of us can win in '
                     'one move" / "I\'m not seeing any big yellow threats"',
        "TA-move": 'Talks about their OWN move -- one they are considering, one they are '
                   'committing to, or one they have just played -- with or without a reason, and '
                   'including the result. '
                   '"Column 4" / "I was thinking column 2" / "If I go there, they can block my '
                   'diagonal three, so I will avoid that" / "Okay, done" / "it worked. I was able '
                   'to connect 4"',
    },
    # The other five parents are NOT subtyped. Empirical reason: for each of them the
    # parent share predicts R2 at least as well as any of its subtypes did, on both DVs
    # -- subdividing spent per-participant variance without buying discrimination.
    # (parent p, flat R2 / selectivity, vs the best subtype's p:)
    #   Solution Request        .075 / .119   -> best subtype SR-move .155 / .199
    #   Conv. Acknowledgment    .067 / .043   -> best subtype CA-uptake .354 / .229
    #   Common Ground Question  .570 / .414   -> best subtype CGQ-retrospect .247 / .171
    #   Knowledge Deficit Q     .248 / .460   -> all subtypes null
    #   Metacomment             .163 / .194   -> parent is only n=12
    # Think Aloud is the exception and the reason this module exists: the parent's effect
    # (p=.005) concentrates in TA-assess (p=.023) while TA-move is weaker (p=.080).
    "Common Ground Question": {},
    "Solution Request": {},
    "Knowledge Deficit Question": {},
    "Conversational Acknowledgment": {},
    "Metacomment": {},
}

# Definitions of retired codes, preserved because this folder is not under version
# control and participant_utterances_v3.csv still carries first-pass labels in its
# `subtype` column -- these are the instrument for those labels.
RETIRED_SUBTYPE_DEFS = {
    "TA-intent": "candidate/intended move named, no reason given",
    "TA-justified": "reason, consequence, or lookahead stated for a move",
    "TA-report": "reports a move already played, or its result",
    "CGQ-confirm": "ONE candidate move (or general belief) put forward for confirmation",
    "CGQ-compare": "asks the assistant to choose between two or more named options",
    "CGQ-retrospect": "asks about a move ALREADY played",
    "CGQ-principle": "checks a general belief about play rather than a specific move",
    "SR-move": "asks for the best move for THIS position",
    "SR-sequence": "asks for several moves or a whole winning path",
    "SR-general": "asks for strategy beyond this board (openings, future games)",
    "SR-broad": "SR-sequence + SR-general: asks for more than the immediate move",
    "SR-open": "unspecific request for help with no content of its own",
    "KDQ-board": "asks about the state of the board -- threats, what is playable",
    "KDQ-rules": "asks about a rule or strategic concept",
    "KDQ-clarify": "asks the assistant to clarify something IT said",
    "KDQ-concept": "KDQ-rules + KDQ-clarify",
    "KDQ-interface": "asks about notation, the UI, or turn mechanics -- not about Connect Four",
    "CA-uptake": "signals the assistant's point landed, or commits to acting on it",
    "CA-answer": "answers a question the ASSISTANT asked -- yes/no, or a board fact",
    "CA-social": "pure social acknowledgment or thanks, no content",
    "MC-uncertain": "voices doubt about their own read",
    "MC-concede": "concedes the position is lost or that they cannot see a way",
    "MC-task": "comments on the task or interface rather than the game",
}

# Retired subtype -> (where its turns go, why). Applying MERGE_INTO to first-pass
# labels reproduces the reduced scheme exactly; None means "no subtype" (the parent
# is the unit of analysis). Counts are from the 405-label first pass.
RETIRED_SUBTYPES = {
    # -> TA-move: the intent/justified boundary was the least reliable in the scheme
    # (kappa 0.741, 75% panel agreement) and TA-justified predicted nothing (p=.83).
    # The reliable, predictive contrast is board-reading vs own-move talk.
    "TA-intent":      ("TA-move", "n=35; unreliable + null boundary, see module notes"),
    "TA-justified":   ("TA-move", "n=37; see TA-intent"),
    "TA-report":      ("TA-move", "n=8; reporting a move played is own-move talk"),
    # -> None: parent-level analysis. These five parents predict at least as well
    # undivided as any subtype of them did (numbers in the SUBTYPES comment above).
    "CGQ-confirm":    (None, "n=86, but p=.948/.915 -- 80% of its parent and less "
                             "informative than it"),
    "CGQ-compare":    (None, "n=11; null"),
    "CGQ-retrospect": (None, "n=10; best CGQ subtype (p=.247) but still weaker than useful "
                             "and far below usable n"),
    "CGQ-principle":  (None, "n=3"),
    "SR-move":        (None, "n=83, p=.155 vs parent p=.075 -- the split weakens the "
                             "delegation signal"),
    "SR-broad":       (None, "n=12; see SR-move"),
    "SR-sequence":    (None, "n=7"),
    "SR-general":     (None, "n=5; theoretically the transfer-relevant ask, but unusable at n=5"),
    "SR-open":        (None, "n=23, p=.847"),
    "KDQ-board":      (None, "n=14; parent itself is only n=22"),
    "KDQ-concept":    (None, "n=4"),
    "KDQ-rules":      (None, "n=2"),
    "KDQ-clarify":    (None, "n=2"),
    "KDQ-interface":  (None, "n=4; the non-task-confusion signal is real but far too small to "
                             "carry a category -- report the turns individually"),
    "CA-uptake":      (None, "n=29, p=.354 vs parent p=.067/.043 -- the split destroys a "
                             "signal that exists at parent level"),
    "CA-answer":      (None, "n=8; see CA-uptake"),
    "CA-social":      (None, "n=5; see CA-uptake"),
    "MC-uncertain":   (None, "n=7; parent is only 12 labels total"),
    "MC-concede":     (None, "n=3; see MC-uncertain"),
    "MC-task":        (None, "n=2; see MC-uncertain"),
}
MERGE_INTO = {old: new for old, (new, _why) in RETIRED_SUBTYPES.items()}

# Both surviving subtypes clear the n>=20 threshold this corpus needs for a
# per-participant share to be a usable regressor (TA-assess 24, TA-move 80), so
# nothing is descriptive-only any more. Keep the list: anything added back below 20
# belongs here rather than in a regression.
DESCRIPTIVE_ONLY = []


def reduce_subtype(code):
    """Map a first-pass subtype code onto the reduced scheme (identity if kept)."""
    if code in MERGE_INTO:
        return MERGE_INTO[code]
    return code if any(code in subs for subs in SUBTYPES.values()) else None

ALL_SUBTYPES = [c for parent in SUBTYPES.values() for c in parent]
Subtype = Literal[tuple(ALL_SUBTYPES)]           # closed set; validated per parent in code
PARENT_OF = {c: p for p, subs in SUBTYPES.items() for c in subs}


def options_block(parent):
    """The subtype menu for one parent, as shown to the annotator."""
    return "".join(f"SUBTYPE: {c}, DEFINITION AND EXAMPLES: {d}\n" for c, d in SUBTYPES[parent].items())


class SubtypeSignature(dspy.Signature):
    """A participant talking to an AI assistant while solving a Connect Four puzzle. Their utterance has ALREADY been assigned the dialogue act `parent_act` by a prior coding pass. That assignment is fixed and correct -- do not question it, and do not consider any other act.

Your only job is to pick the ONE subtype of `parent_act` that best fits, from the `options` list. Choose only from that list.

If the utterance does several things, judge only the part that made it an instance of `parent_act`. (An utterance may carry other acts too; those are coded separately and are not your concern here.) Judge only what is written -- do not infer unstated reasoning. If two subtypes seem to fit, pick the one matching the utterance's main clause; if the utterance is too terse to distinguish them, pick the more conservative (less elaborated) subtype.
"""
    utterance = dspy.InputField(desc="The participant utterance.")
    assistant_context = dspy.InputField(desc="The assistant's immediately preceding reply (may be empty).")
    parent_act = dspy.InputField(desc="The already-assigned v1 dialogue act. Fixed.")
    options = dspy.InputField(desc="The subtypes available for this parent act, with definitions.")
    subtype: Subtype = dspy.OutputField(desc="Exactly one subtype code from the options list.")


_CELL = re.compile(r"\b(?:c\s?\d|col(?:umn)?\s*\d|row\s*\d|\d\s*[-/,:xv]\s*\d)", re.I)

def responsive(assistant_context):
    """Did the assistant's preceding reply pose a question?"""
    return "?" in (assistant_context or "")

def grounded(utterance):
    """Does the turn cite a specific cell/column/row?"""
    return bool(_CELL.search(utterance or ""))


class SubtypeSuite(dspy.Module):
    """Same 3-provider panel as the v1 act classifier, voting on the subtype.

    Votes for subtypes outside the given parent's menu are discarded before the
    vote (the parent is fixed, so an off-menu answer is a rater error, not a
    disagreement about the parent). Ties fall to the subtype the panel's first
    surviving annotator chose, and the item is flagged needs_review.
    """

    def __init__(self, callbacks=None, lm_timeout=120, wall_timeout=150):
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
            a = dspy.ChainOfThought(SubtypeSignature)
            a.set_lm(lm)
            self.annotators[name] = a

    def _annotate(self, name, annotator, utterance, context, parent):
        try:
            r = annotator(utterance=utterance, assistant_context=context,
                          parent_act=parent, options=options_block(parent))
            s = r.subtype
            if s not in SUBTYPES[parent]:        # off-menu -> rater error, discard
                _log.warning("subtype %r not valid for parent %r (annotator %r)", s, parent, name)
                return name, None
            return name, s
        except Exception as e:
            _log.warning("subtype annotator %r failed: %s: %s", name, type(e).__name__, e)
            return name, None

    def forward(self, utterance, parent_act, assistant_context=""):
        if not SUBTYPES.get(parent_act):
            # Parent carries no subtypes (Metacomment): nothing to ask the panel. Return
            # a no-op record rather than letting every answer fail menu validation.
            return {"final": None, "parent": parent_act, "votes": {}, "per_model": {},
                    "n_valid": 0, "confidence": 0.0, "unanimous": True, "majority": True,
                    "needs_review": False, "not_subtyped": True}
        per_model = {}
        ex = concurrent.futures.ThreadPoolExecutor(max_workers=len(self.annotators))
        futs = {ex.submit(self._annotate, n, a, utterance, assistant_context, parent_act): n
                for n, a in self.annotators.items()}
        try:
            for fut in concurrent.futures.as_completed(futs, timeout=self.wall_timeout):
                name, s = fut.result()
                per_model[name] = s
        except concurrent.futures.TimeoutError:
            pass
        finally:
            for fut, name in futs.items():
                if name not in per_model:
                    fut.cancel()
                    _log.warning("subtype annotator %r exceeded wall timeout %ss; dropping",
                                 name, self.wall_timeout)
                    per_model[name] = None
            ex.shutdown(wait=False)

        valid = [s for s in per_model.values() if s is not None]
        votes = Counter(valid)
        top = votes.most_common()
        unanimous = bool(top) and top[0][1] == len(valid)
        majority = bool(top) and top[0][1] > len(valid) / 2
        final = top[0][0] if top else None
        return {"final": final, "parent": parent_act, "votes": dict(votes),
                "per_model": per_model, "n_valid": len(valid),
                "confidence": (top[0][1] / len(valid)) if top else 0.0,
                "unanimous": unanimous, "majority": majority,
                "needs_review": not majority}


def subtype_kappa(records, reduced=True):
    """Fleiss' κ over subtype choice, computed WITHIN each parent (the parent is
    given, so cross-parent agreement is not the panel's achievement). Returns
    (macro κ over parents, {parent: κ}, {parent: exact-agreement rate}).

    reduced=True maps each rater's vote through MERGE_INTO first, so κ describes the
    scheme now in SUBTYPES. Records stored by the first pass carry retired codes, and
    scoring those against the reduced menus without mapping them mixes the two schemes.
    Note that merging necessarily RAISES κ -- votes that differed only on a retired
    distinction stop counting as disagreement -- so a reduced κ is not evidence that
    the raters got better. Pass reduced=False with the first-pass menus to see the
    agreement the panel actually achieved.
    """
    by_parent, exact = {}, {}
    for parent, subs in SUBTYPES.items():
        if not subs:
            continue                       # parent carries no subtypes (Metacomment)
        items = []
        for r in records:
            if r["parent"] != parent or not r["per_model"] or \
                    any(v is None for v in r["per_model"].values()):
                continue
            votes = {k: (reduce_subtype(v) if reduced else v) for k, v in r["per_model"].items()}
            if any(v is None for v in votes.values()):
                continue                   # vote maps outside the current scheme
            items.append({"per_model": votes})
        if len(items) < 2:
            continue
        cats = list(subs)
        n = len(next(iter(items))["per_model"])
        if any(len(it["per_model"]) != n for it in items):
            items = [it for it in items if len(it["per_model"]) == n]
        if len(items) < 2 or n < 2:
            continue
        P, pj = [], {c: 0 for c in cats}
        for it in items:
            c = Counter(it["per_model"].values())
            P.append((sum(v * v for v in c.values()) - n) / (n * (n - 1)))
            for cat in cats:
                pj[cat] += c[cat]
        Pbar = sum(P) / len(P)
        Pe = sum((v / (len(items) * n)) ** 2 for v in pj.values())
        by_parent[parent] = 1.0 if Pe == 1 else (Pbar - Pe) / (1 - Pe)
        exact[parent] = sum(len(set(it["per_model"].values())) == 1 for it in items) / len(items)
    macro = sum(by_parent.values()) / len(by_parent) if by_parent else float("nan")
    return macro, by_parent, exact
