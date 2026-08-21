"""Replay real consultations at their real board positions, to validate the pipeline.

The counterfactual experiment rewrites a participant's utterance and asks the coach the
rewritten version at the board the participant actually faced. Before any rewriting is
worth doing, the replay itself has to reproduce reality: same board, same coach, same
message should give the same kind of reply.

That is checkable for free, because every one of these turns already has the real
assistant reply on record, annotated with the same fine panel.

Board reconstruction is exact -- replaying each participant's moves through
PuzzleSession reproduces the study's opponent moves, disc counts, disc loss and
optimality flags on 124/124 participants and 516/516 decisions, with no API calls. This
script checks the half that costs money: whether the coach's REPLY matches.

Perfect agreement is not expected. The coach samples, and act labels are a 3-seat panel
with its own noise. The bar is that replayed replies look like the recorded ones at the
act level -- especially on Board Report and Move Verdict, which the experiment measures.

    python replay_probe.py --n 20                 # probe, then report
    python replay_probe.py --n 20 --dry-run       # alignment only, no API calls
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import dotenv

BASE = Path(__file__).resolve().parent
dotenv.load_dotenv(BASE / ".env")

STUDY = Path("/Users/siyanli/Documents/CITRUS/game_user_study_phase_othello")
REC = STUDY / "recordings-download"
PUZZLE, TAG = "oc20260727", "poc20260727"
FIELD = "annotation_assistant_fine"
# The study's own coach: app.py CHAT_MODEL, default gpt-5.5, reasoning_effort from
# assistant_agent's game config (medium for Othello).
COACH = "openai/gpt-5.5"

sys.path.insert(0, str(STUDY / "games" / "othello"))


def _ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def align(pid):
    """[(ply, user, assistant, recorded_acts)] for one participant.

    ply = how many decisions were already committed when the participant sent the
    message, so replaying that many moves reproduces the board they were looking at.
    """
    moves_f = REC / pid / f"moves_{TAG}.jsonl"
    conv_f = REC / pid / f"annotated_conversation_{TAG}.jsonl"
    if not moves_f.exists() or not conv_f.exists():
        return []          # screened out, or consulted without finishing the round
    mv = [json.loads(l) for l in open(moves_f) if l.strip()]
    mv = [m for m in mv if m.get("attempt") == 1]
    if not mv:
        return []
    out, prior = [], []
    for line in open(conv_f):
        if not line.strip():
            continue
        t = json.loads(line)
        u, a = (t.get("user") or "").strip(), (t.get("assistant") or "").strip()
        rec = t.get(FIELD)
        if not u or not a or not rec or not t.get("user_ts"):
            continue
        when = _ts(t["user_ts"])
        ply = sum(1 for m in mv if _ts(m["ts"]) < when)
        out.append(dict(pid=pid, ply=ply, user=u, assistant=a,
                        recorded=set(rec.get("final") or []),
                        moves=[m["move"] for m in mv],
                        history=list(prior)))          # turns before this one
        prior.append((u, a))
    return out


def board_at(moves, ply):
    """A PuzzleSession advanced to the position after `ply` committed decisions."""
    from llm_eval import PuzzleSession
    s = PuzzleSession(PUZZLE)
    for mv in moves[:ply]:
        if s.done:
            break
        s.play(mv)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="turns to probe")
    ap.add_argument("--dry-run", action="store_true", help="alignment only, no API")
    a = ap.parse_args()

    turns = []
    for d in sorted(REC.iterdir()):
        if not d.is_dir():
            continue
        turns.extend(align(d.name))
        if len(turns) >= a.n * 3:
            break
    # spread across plies rather than taking the first n, which would be all opening
    turns.sort(key=lambda t: (t["ply"], t["pid"]))
    step = max(1, len(turns) // a.n)
    sample = turns[::step][:a.n]
    print(f"{len(turns)} aligned turns available; probing {len(sample)}")
    print(f"ply distribution: {dict(sorted(Counter(t['ply'] for t in sample).items()))}\n")
    if a.dry_run:
        for t in sample[:8]:
            print(f"  ply {t['ply']}  {t['pid'][:10]}  {t['user'][:70]}")
        return 0

    from assistant_acts_othello import OthelloFineAssistantSuite
    from assistant_agent import AssistantAgent
    suite = OthelloFineAssistantSuite()

    rows, used_image = [], []
    for i, t in enumerate(sample, 1):
        try:
            s = board_at(t["moves"], t["ply"])
            coach = AssistantAgent(COACH, s, game="othello")
            # The study's /api/chat sends the participant's PRIOR turns before the
            # current question (parse_conversation_history in app.py). Without them
            # the coach answers every question cold, which is not what it did.
            for pu, pa in t["history"]:
                coach.conversation_history.append({"role": "user", "content": pu})
                coach.conversation_history.append({"role": "assistant", "content": pa})
            reply = coach.answer(t["user"])
            used_image.append(getattr(coach, "used_image", None))
            lab = suite(utterance=reply.strip())
            got = set(lab.get("final") or [])
        except Exception as e:
            print(f"  [{i}] {type(e).__name__}: {e}")
            continue
        rows.append((t, got, reply))
        j = len(t["recorded"] & got) / len(t["recorded"] | got) if (t["recorded"] | got) else 1.0
        print(f"  [{i}/{len(sample)}] ply {t['ply']}  Jaccard {j:.2f}", flush=True)

    if not rows:
        print("no successful replays"); return 1

    print(f"\n{'=' * 72}\nREPLAY FIDELITY  ({len(rows)} turns)\n{'=' * 72}")
    print(f"board sent as image on {sum(1 for x in used_image if x)}/{len(used_image)} "
          f"turns (text fallback would diverge from the study)")
    print(f"history seeded: {sum(len(t['history']) for t, _, _ in rows)} prior turn(s) "
          f"across the sample")
    jac = [len(t["recorded"] & g) / len(t["recorded"] | g) if (t["recorded"] | g) else 1.0
           for t, g, _ in rows]
    print(f"mean pairwise Jaccard, recorded vs replayed acts: {sum(jac)/len(jac):.3f}")
    print(f"\n{'act':<24}{'recorded':>10}{'replayed':>10}{'agree':>8}")
    acts = sorted({x for t, g, _ in rows for x in (t["recorded"] | g)})
    for act in acts:
        r = sum(act in t["recorded"] for t, _, _ in rows)
        p = sum(act in g for _, g, _ in rows)
        agree = sum((act in t["recorded"]) == (act in g) for t, g, _ in rows)
        star = "  <==" if act in ("Board Report", "Move Verdict") else ""
        print(f"{act:<24}{r:>10}{p:>10}{100*agree/len(rows):>7.0f}%{star}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
