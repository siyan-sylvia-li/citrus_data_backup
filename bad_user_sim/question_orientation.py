"""Temporal orientation of a participant utterance, by POS features rather than an LLM.

The AutoTutor participant acts do not separate the turns that elicit Board Report from
the turns that elicit Move Verdict -- "was it a bad move to choose a1?" and "is a1 the
best move?" are both Common Ground Question and get opposite assistant replies. What
separates them is what the utterance is oriented toward:

    RETRO   what already happened      "why did white flip those?"  "what did I do wrong?"
    STATE   what is currently true     "evaluate the current board"  "how's the score now?"
    PROSP   what to do next            "should I play a1?"  "G2?"  "which is better?"

These are grammatical distinctions -- tense, modality, interrogative type -- so a POS
tagger resolves most of them. Deterministic and reproducible, unlike a panel.

Multi-label on purpose: "OK, I played b7. I'm thinking of playing g2 now" is genuinely
both RETRO and PROSP, and forcing one label would discard that.

    python question_orientation.py --demo        # label a sample and print it
    python question_orientation.py --eval        # agreement vs the regex baseline
"""
from __future__ import annotations

import argparse
import re
import sys

from nltk import pos_tag, word_tokenize

# Othello squares (a1-h8) and Connect Four columns; a bare coordinate with no verb is a
# move proposal, which no tense feature would catch.
COORD = re.compile(r"\b(?:[a-h] ?[1-8]|[1-8] ?[a-h]|col(?:umn)? ?[1-7])\b", re.I)

PAST_CUE = {"already", "just", "earlier", "previously", "last"}
FUTURE_CUE = {"next", "now", "then", "instead", "another"}
STATE_NOUN = {"board", "state", "position", "score", "situation", "setup"}
# Verbs that ask the assistant to describe rather than decide.
DESCRIBE_VERB = {"evaluate", "describe", "show", "explain", "tell", "assess", "read"}
ADVICE_NOUN = {"move", "square", "option", "choice", "play", "suggestion"}


def _tags(text):
    return pos_tag(word_tokenize(text))


def orient(text):
    """-> set of {'RETRO','STATE','PROSP'}; may be empty or hold several."""
    tags = _tags(text)
    words = [w.lower() for w, _ in tags]
    pos = [p for _, p in tags]
    out = set()

    # --- RETRO: past tense, perfect aspect, past-referring first person -------
    if any(p == "VBD" for p in pos):
        out.add("RETRO")
    if "did" in words or "was" in words or "were" in words:
        out.add("RETRO")
    # "have/had played", "has flipped"
    for i, p in enumerate(pos[:-1]):
        if p in ("VBP", "VBZ", "VBD") and words[i] in ("have", "has", "had") \
                and pos[i + 1] == "VBN":
            out.add("RETRO")
    if PAST_CUE & set(words):
        out.add("RETRO")
    # counterfactuals: "should/could I have ...", "would have"
    for i, w in enumerate(words[:-1]):
        if w in ("should", "could", "would", "might") and "have" in words[i + 1:i + 3]:
            out.add("RETRO")

    # --- PROSP: modal + base verb, bare move proposal, advice nouns ----------
    for i, p in enumerate(pos):
        if p == "MD" and "RETRO" not in out:
            out.add("PROSP")
        # "thinking of/considering <move>"
        if words[i] in ("considering", "thinking") and COORD.search(text):
            out.add("PROSP")
    if FUTURE_CUE & set(words) and not (PAST_CUE & set(words)):
        out.add("PROSP")
    if ADVICE_NOUN & set(words) and any(p == "JJS" or p == "JJR" for p in pos):
        out.add("PROSP")          # "best move", "better square"
    # bare proposal: a coordinate, short, and no lexical verb at all
    if COORD.search(text) and not any(p.startswith("VB") for p in pos) \
            and len(words) <= 12:
        out.add("PROSP")

    # --- STATE: present-tense description request, current-position query ----
    if DESCRIBE_VERB & set(words) and (STATE_NOUN & set(words)):
        out.add("STATE")
    if STATE_NOUN & set(words) and not (PAST_CUE & set(words)) \
            and any(p in ("VBZ", "VBP") for p in pos):
        out.add("STATE")
    # "what/where can I play", "which squares are legal" -- legality of the position
    if any(w in ("legal", "allowed", "playable", "available") for w in words):
        out.add("STATE")
    # imperative describe with no subject: "evaluate the current board state"
    if pos and pos[0] == "VB" and words[0] in DESCRIBE_VERB:
        out.add("STATE")

    return out


# Baseline the POS classifier is meant to replace, kept for the --eval comparison.
_RE_RETRO = re.compile(r"\b(did|was|were|happened|just (made|played|took)|i chose|"
                       r"i played|i moved|why did|how did|went wrong|do wrong|"
                       r"my last move|already|lost|won|ended|is over|could i have|"
                       r"should i have|would have)\b", re.I)
_RE_PROSP = re.compile(r"\b(should i|shall i|what (should|do|would) i|next move|now\?|"
                       r"considering|thinking of|is .{1,12} (a )?(good|best|better)|"
                       r"which (move|square|one)|suggest|recommend|what about|try)\b",
                       re.I)


def regex_orient(text):
    out = set()
    if _RE_RETRO.search(text):
        out.add("RETRO")
    if _RE_PROSP.search(text):
        out.add("PROSP")
    return out


def combined_orient(text):
    """Lexical cue AND POS feature must agree. The recommended classifier.

    POS is used to FILTER the regex, not to extend it: everything POS adds on its own
    is noise (union and back-off both fit worse than the regex alone). The gain is
    almost entirely on the prospective side -- the tagger removes regex false positives
    like "is a1 the best move?" matching the good/best/better pattern with no modal and
    no proposal structure. Othello, 621 turns, Board Report base rate .39:

        scheme      coverage   BR|RETRO   BR|PROSP   AIC
        acts only          -          -          -   759.3
        regex           37.8%       0.68       0.27   742.6
        POS             73.4%       0.60       0.30   745.5
        union           78.3%       0.60       0.32   747.8
        intersection    33.0%       0.68       0.21   739.5   <-- this

    Deliberately low recall. Turns matching neither are UNCLASSIFIED, not prospective,
    and they sit at the base rate (.39) -- i.e. unclassified noise rather than a third
    group. Exclude them; do not treat them as a category.
    """
    return orient(text) & regex_orient(text)


def _load(pattern, limit=None):
    import glob
    import json
    rows = []
    for fp in sorted(glob.glob(pattern)):
        for line in open(fp):
            if not line.strip():
                continue
            t = json.loads(line)
            u = (t.get("user") or "").strip()
            aa = t.get("annotation_assistant_fine")
            if not u or not aa:
                continue
            rows.append((u, set(aa.get("final") or [])))
            if limit and len(rows) >= limit:
                return rows
    return rows


OTH = ("/Users/siyanli/Documents/CITRUS/game_user_study_phase_othello/"
       "recordings-download/*/annotated_conversation_poc20260727.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    rows = _load(OTH, a.limit)

    if a.demo:
        import random
        random.seed(0)
        for u, acts in random.sample(rows, min(30, len(rows))):
            lab = ",".join(sorted(orient(u))) or "-"
            br = "BR" if "Board Report" in acts else "  "
            print(f"  [{br}] {lab:<16} {u[:96]}")
        return 0

    if a.eval:
        import numpy as np
        both = [(orient(u), regex_orient(u), combined_orient(u),
                 "Board Report" in acts) for u, acts in rows]
        print(f"n={len(both)}\n")
        for name, idx in (("POS", 0), ("regex", 1), ("combined", 2)):
            lab = [b[idx] for b in both]
            cov = 100 * np.mean([bool(x) for x in lab])
            print(f"{name:<6} coverage {cov:5.1f}%")
            for cat in ("RETRO", "STATE", "PROSP"):
                sel = [b[3] for b, l in zip(both, lab) if cat in l]
                if sel:
                    print(f"         {cat:<6} n={len(sel):4}  Board Report {np.mean(sel):.2f}")
            none = [b[3] for b, l in zip(both, lab) if not l]
            if none:
                print(f"         {'none':<6} n={len(none):4}  Board Report {np.mean(none):.2f}")
            print()
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
