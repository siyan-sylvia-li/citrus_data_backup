"""Build counterfactual stimuli for the replay experiment. READ-ONLY on study data.

Each real consultation is its own control: the participant's own utterance, at the board
they actually faced, paired against a minimal rewrite that flips ONE dimension. Nothing
is invented from scratch and no synthetic condition is introduced -- only utterances
that naturally occurred are rewritten, and only into their own complement.

    request_type   candidate evaluation  <->  open solution request
    orientation    prospective           <->  retrospective

Both directions of both contrasts, so the estimate is not driven by one rewrite
direction being easier than the other.

WRITES NOTHING INTO recordings-download. Every output goes to replay_experiment/, and
the stimuli file is written once and refused if it already exists -- rewrites cost money
and silently regenerating them would make an in-flight experiment irreproducible.

Two mechanical checks reject a rewrite before it is ever used:
  1. the target dimension actually flipped, per the same deterministic classifier the
     analysis uses (question_orientation / the participant act labels)
  2. the set of board squares named in the utterance is UNCHANGED -- a rewrite that
     invents or drops a square has changed the content, not the form

The rewriter is deliberately NOT a fine-panel seat (those are gpt / llama / sonnet). A
rewriter that shares a model with its own validator optimises for what that validator
detects.

    python build_stimuli.py --dry-run           # pools and pairing, no API calls
    python build_stimuli.py --n 60              # 60 pairs per direction
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import dotenv

BASE = Path(__file__).resolve().parent
dotenv.load_dotenv(BASE / ".env")

OUT_DIR = BASE / "replay_experiment"
STIMULI = OUT_DIR / "stimuli.jsonl"

# Not a fine-panel seat (gpt / llama / sonnet) -- see module docstring.
REWRITER = "openai/gpt-5.6-luna"

SQUARE = re.compile(r"\b([a-h])\s?([1-8])\b", re.I)

CONTRASTS = {
    "request_type": {
        "candidate_eval": "open_request",
        "open_request": "candidate_eval",
    },
    "orientation": {
        "PROSP": "RETRO",
        "RETRO": "PROSP",
    },
}

# Legality queries are state questions, not requests for a recommendation. Left in the
# open_request pole they produce rewrites like "is a7 legal?", which asks about a rule
# rather than about a move's quality -- a different construct from candidate evaluation.
LEGALITY = re.compile(r"\b(legal|allowed|playable|available|can i (even )?(play|go|move))\b",
                      re.I)

# Flipping orientation on an utterance whose subject is a pronoun leaves the pronoun
# pointing at nothing: "which one is the best choice?" -> "how did b7 turn out?" drops
# the antecedent that lived in the previous turn. Only self-contained utterances flip.
ANAPHORIC = re.compile(r"\b(which one|which of (these|those|them)|that one|the other one|"
                       r"^(it|that|this)\b)", re.I | re.M)
INTERROGATIVE = re.compile(r"\?|^\s*(what|which|where|why|how|when|should|could|would|"
                           r"can|do|does|did|is|are|was|were)\b", re.I)


def self_contained(text):
    """A question that carries its own referent, so an orientation flip stays coherent."""
    return bool(INTERROGATIVE.search(text)) and not ANAPHORIC.search(text)

INSTRUCTION = {
    ("request_type", "open_request"):
        "Rewrite it as an OPEN request for the assistant's recommendation, without "
        "naming a candidate move of your own.",
    ("request_type", "candidate_eval"):
        "Rewrite it so the participant proposes a SPECIFIC candidate square and asks "
        "the assistant to evaluate that candidate, instead of asking what to play.",
    ("orientation", "RETRO"):
        "Rewrite it to ask about the move the participant HAS ALREADY PLAYED and how it "
        "turned out, instead of about what to do next. The move already played is "
        "{last_move} -- ask about that, not about any move being considered.",
    ("orientation", "PROSP"):
        "Rewrite it to ask about what to do NEXT, instead of about what already "
        "happened.",
}

PROMPT = """You are preparing stimuli for a dialogue experiment.

Rewrite the participant's message below. {instruction}

Hard rules:
- Change as little as possible. This is a minimal edit, not a paraphrase.
- Keep the participant's own register: same rough length, same informality, same
  typos and lowercase if present. These are real messages from a study, not clean prose.
{square_rule}
- Output ONLY the rewritten message. No quotes, no explanation.

Participant's message:
{utterance}"""

# The square rule is per contrast. For orientation the squares must not move, because
# tense flips without changing what is asked about. For request_type the square set is
# the manipulation itself, so it has to move -- and the candidate named must be one the
# participant could actually play, or the coach answers about legality instead.
# Orientation cannot preserve the referent either. A prospective utterance names a
# CANDIDATE (not yet played); a retrospective one must name a move that WAS played.
# Preserving the square forced rewrites like "i think my next move should be a7" ->
# "why did my move to a7 turn out as it did?", asserting a7 was played when it was not.
# So each side names the appropriate real object at the identical board: the last
# committed move for RETRO, a legal candidate for PROSP.
SQUARE_RULE = {
    ("orientation", "RETRO"): "- Name exactly the square just played, {last_move}. "
                              "Mention no other square.",
    ("orientation", "PROSP"): "- Name exactly ONE candidate square, chosen from the "
                              "moves that are legal here: {legal}. Mention no other.",
    ("request_type", "candidate_eval"):
        "- Name exactly ONE candidate square, chosen from the moves that are legal "
        "here: {legal}. Mention no other square.",
    ("request_type", "open_request"):
        "- Name NO board square at all. Ask for the recommendation in general terms.",
}


def squares(text):
    return {f"{m.group(1).lower()}{m.group(2)}" for m in SQUARE.finditer(text)}


def classify(turn):
    """Which pole of each contrast this real utterance sits on (or None).

    Both poles are deliberately strict. Reading the loose version's members showed
    real contamination: "Legal moves please" landed in open_request, and an utterance
    containing "i playyed c8" landed in PROSP because the typo dodged the regex.

      request_type  a candidate evaluation must actually NAME a candidate; an open
                    request must not name one.
      orientation   unambiguous under BOTH instruments, not merely their intersection
                    -- if the union contains the opposite pole, the utterance is mixed
                    and is excluded rather than assigned.
    """
    import question_orientation as qo
    acts, u = turn["acts"], turn["user"]
    sq = squares(u)
    out = {}
    cg, sr = "Common Ground Question" in acts, "Solution Request" in acts
    if cg and not sr and sq:
        out["request_type"] = "candidate_eval"
    elif sr and not cg and not sq and not LEGALITY.search(u):
        out["request_type"] = "open_request"
    strict, loose = qo.combined_orient(u), qo.orient(u) | qo.regex_orient(u)
    if strict == {"RETRO"} and "PROSP" not in loose:
        out["orientation"] = "RETRO"
    elif strict == {"PROSP"} and "RETRO" not in loose:
        out["orientation"] = "PROSP"
    return out


SIM_GLOB = "test_agentic_run__human_style*"


def load_sim_pool():
    """Aligned simulated consultations. Same schema, same puzzle, same coach.

    Kept as a SEPARATE stratum rather than pooled with the human turns: simulated
    utterances are cleaner and more formal, cluster at later plies (endgame review
    rather than midgame play), and come from a student whose act composition differs
    from a human's. Pooling would average over all three.
    """
    import glob as _glob
    import os as _os

    rows = []
    for conv in sorted(_glob.glob(f"{SIM_GLOB}/*/annotated_conversation_poc20260727.jsonl")):
        cell = _os.path.dirname(conv)
        trace_f = _os.path.join(cell, "game_trace_poc20260727.json")
        if not _os.path.exists(trace_f):
            continue
        # The sim's user_ts values are FABRICATED -- written after the run, all after
        # the final move and spaced exactly 15.000s apart -- so timestamp alignment put
        # every simulated consultation at a terminal board (100% with no legal moves).
        # game_trace records the true interleaving as an ordered list of {"question"}
        # and {"move"} entries, so the ply is simply how many moves precede a question.
        trace = json.load(open(trace_f))
        ply_of, moves, seen = [], [], 0
        for e in trace:
            if not isinstance(e, dict):
                continue
            if "move" in e:
                moves.append(e["move"])
                seen += 1
            elif "question" in e:
                ply_of.append(seen)
        turns = [json.loads(l) for l in open(conv) if l.strip()]
        turns = [t for t in turns
                 if (t.get("user") or "").strip() and (t.get("assistant") or "").strip()
                 and t.get("annotation_user") and t.get("annotation_assistant_fine")]
        if len(ply_of) < len(turns):
            continue                      # trace and transcript disagree; skip the cell
        prior = []
        for i, t in enumerate(turns):
            u = t["user"].strip()
            rows.append(dict(pid=_os.path.basename(cell), ply=ply_of[i], user=u,
                             assistant=t["assistant"].strip(),
                             recorded=set((t["annotation_assistant_fine"].get("final") or [])),
                             acts=set((t["annotation_user"].get("final") or [])),
                             moves=moves, history=list(prior), source="sim"))
            prior.append((u, t["assistant"].strip()))
    return rows


def load_pool():
    """Every aligned HUMAN consultation, with its acts. Read-only."""
    import replay_probe as rp
    rows = []
    for d in sorted(rp.REC.iterdir()):
        if not d.is_dir():
            continue
        conv = rp.REC / d.name / f"annotated_conversation_{rp.TAG}.jsonl"
        if not conv.exists():
            continue
        acts_by_text = {}
        for line in open(conv):
            if not line.strip():
                continue
            t = json.loads(line)
            u = (t.get("user") or "").strip()
            au = t.get("annotation_user") or {}
            if u:
                acts_by_text[u] = set(au.get("final") or [])
        for t in rp.align(d.name):
            t["acts"] = acts_by_text.get(t["user"], set())
            t["source"] = "human"
            rows.append(t)
    return rows


def legal_squares(turn):
    """Squares Black may actually play at this turn's board."""
    import replay_probe as rp
    s = rp.board_at(turn["moves"], turn["ply"])
    E = __import__("engine")
    return {E.to_notation(m) for m in E.legal_moves(s.board, E.BLACK)}


def check(orig, new, contrast, target, turn):
    """[] means the rewrite is usable.

    Validation is PER CONTRAST, because the two contrasts differ in whether content
    is allowed to move:

      orientation   tense flips without touching content, so the set of named squares
                    must be identical. A rewrite that renames a square has changed
                    what is being asked about, not how.

      request_type  naming a candidate IS what makes an utterance a candidate
                    evaluation, so the square set MUST change -- open->candidate adds
                    one, candidate->open drops one. Requiring preservation here would
                    reject every rewrite. What is checked instead is that any square
                    named is LEGAL at that board, so the coach is answering about the
                    request form rather than about an unplayable square.
    """
    import question_orientation as qo
    problems = []
    if not new.strip():
        problems.append("empty rewrite")
        return problems
    if new.strip().lower() == orig.strip().lower():
        problems.append("unchanged")

    elif contrast == "orientation" and target == "RETRO":
        last = {turn["moves"][turn["ply"] - 1].lower()} if turn["ply"] >= 1 else set()
        if squares(new) != last:
            problems.append(f"names {sorted(squares(new))}, want the played move "
                            f"{sorted(last)}")
        got = qo.combined_orient(new)
        if got != {"RETRO"}:
            problems.append(f"orientation is {sorted(got) or 'unclassified'}, want RETRO")

    elif contrast == "orientation" and target == "PROSP":
        named = squares(new)
        legal = legal_squares(turn)
        if not named:
            problems.append("prospective rewrite names no candidate")
        elif named - legal:
            problems.append(f"names unplayable {sorted(named - legal)}; "
                            f"legal are {sorted(legal)}")
        got = qo.combined_orient(new)
        if got != {"PROSP"}:
            problems.append(f"orientation is {sorted(got) or 'unclassified'}, want PROSP")

    elif contrast == "request_type":
        named = squares(new)
        if target == "candidate_eval" and not named:
            problems.append("candidate evaluation names no candidate")
        if target == "open_request" and named:
            problems.append(f"open request still names {sorted(named)}")
        if named:
            legal = legal_squares(turn)
            bad = named - legal
            if bad:
                problems.append(f"names unplayable square(s) {sorted(bad)}; "
                                f"legal here are {sorted(legal)}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="pairs per direction")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--contrast", choices=sorted(CONTRASTS), default=None)
    ap.add_argument("--source", choices=("human", "sim", "both"), default="both")
    a = ap.parse_args()

    if STIMULI.exists() and not a.dry_run:
        print(f"{STIMULI} already exists -- refusing to overwrite.\n"
              f"Move it aside if you really mean to regenerate.", file=sys.stderr)
        return 1

    pool = []
    if a.source in ("human", "both"):
        pool += load_pool()
    if a.source in ("sim", "both"):
        pool += load_sim_pool()
    n_h = sum(1 for t in pool if t["source"] == "human")
    print(f"{len(pool)} aligned consultations (read-only): "
          f"{n_h} human, {len(pool) - n_h} simulated\n")

    want = [a.contrast] if a.contrast else sorted(CONTRASTS)
    buckets = {}
    skipped_ply0 = skipped_referent = 0
    for turn in pool:
        for contrast, pole in classify(turn).items():
            if contrast not in want:
                continue
            # A prospective utterance at ply 0 cannot be flipped to retrospective:
            # nothing has been played yet for it to ask about.
            if contrast == "orientation" and pole == "PROSP" and turn["ply"] < 1:
                skipped_ply0 += 1
                continue
            # Orientation flips only work on referent-free, self-contained questions --
            # see ANAPHORIC. request_type is unaffected: it changes the referent by
            # design, so it does not need one to survive the edit.
            if contrast == "orientation":
                if squares(turn["user"]) or not self_contained(turn["user"]):
                    skipped_referent += 1
                    continue
            buckets.setdefault((contrast, pole, turn["source"]), []).append(turn)

    print(f"{'contrast':<14}{'source':<8}{'from':<16}{'->':<4}{'to':<16}{'available':>10}")
    for (contrast, pole, src), turns in sorted(buckets.items()):
        print(f"{contrast:<14}{src:<8}{pole:<16}{'->':<4}"
              f"{CONTRASTS[contrast][pole]:<16}{len(turns):>10}")
    if skipped_referent:
        print(f"\n({skipped_referent} orientation turn(s) excluded: name a square or "
              f"are not self-contained questions)")
    if skipped_ply0:
        print(f"\n({skipped_ply0} prospective turn(s) at ply 0 excluded: nothing "
              f"played yet to ask about retrospectively)")
    if a.dry_run:
        print("\nsample utterances per pole:")
        for (contrast, pole, src), turns in sorted(buckets.items()):
            print(f"\n  {contrast} / {pole} / {src}")
            for t in turns[:3]:
                print(f"    ply {t['ply']}  {t['user'][:78]}")
        return 0

    import dspy
    lm = dspy.LM(REWRITER, max_tokens=2048)
    OUT_DIR.mkdir(exist_ok=True)
    kept = Counter()
    rejected = Counter()
    with open(STIMULI, "w") as w:
        for (contrast, pole, src), turns in sorted(buckets.items()):
            target = CONTRASTS[contrast][pole]
            for t in turns[:a.n]:
                sq = ", ".join(sorted(squares(t["user"]))) or "(none)"
                legal = ", ".join(sorted(legal_squares(t))) or "(none)"
                last = t["moves"][t["ply"] - 1] if t["ply"] >= 1 else "(none)"
                fmt = dict(squares=sq, legal=legal, last_move=last)
                rule = SQUARE_RULE[(contrast, target)].format(**fmt)
                prompt = PROMPT.format(
                    instruction=INSTRUCTION[(contrast, target)].format(**fmt),
                    square_rule=rule, utterance=t["user"])
                try:
                    new = lm(prompt)[0].strip().strip('"')
                except Exception as e:
                    rejected[f"{contrast}:{pole}:{src}:api"] += 1
                    print(f"  api error: {e}", file=sys.stderr)
                    continue
                problems = check(t["user"], new, contrast, target, t)
                if problems:
                    rejected[f"{contrast}:{pole}:{src}"] += 1
                    print(f"  reject [{contrast}/{pole}/{src}] {problems[0][:66]}")
                    continue
                kept[f"{contrast}:{pole}:{src}"] += 1
                w.write(json.dumps(dict(
                    pid=t["pid"], ply=t["ply"], moves=t["moves"], history=t["history"],
                    contrast=contrast, source_pole=pole, target_pole=target,
                    source=src,
                    original=t["user"], rewrite=new,
                    recorded_acts=sorted(t["recorded"]),
                    participant_acts=sorted(t["acts"]),
                    rewriter=REWRITER)) + "\n")
    print(f"\nwrote {sum(kept.values())} pair(s) to {STIMULI}")
    print(f"  kept    {dict(kept)}")
    print(f"  rejected{dict(rejected)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
