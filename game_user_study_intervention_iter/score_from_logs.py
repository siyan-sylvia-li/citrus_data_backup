"""Recompute every participant's scores from the move logs. `python score_from_logs.py`

The app writes a per-round summary (oth_score_*.json) as it goes, but that file
is a convenience, not the source of truth — moves_*.jsonl is. Every field the
summary reports is derivable from the per-decision records, which means:

  - rounds that ended badly still yield scores (a timeout used to write no
    summary at all, and the participant who played best in the first batch was
    invisible in the score files as a result);
  - the scoring rule can change during analysis without re-collecting anything;
  - one pass covers participants collected before and after any app change.

What is NOT recoverable for an unfinished round is the game's OUTCOME. If the
clock stopped play with squares still empty, there is no final margin — the disc
counts at the buzzer are a position, not a result, and are reported as such.

Output: one row per participant-round, plus a CSV for analysis. Each row also
carries the participant's TRANSFER-BLOCK label (see `transfer_label`), repeated
on every one of their rounds so a single groupby gets you Solvers vs Strugglers.

Run:
    python score_from_logs.py                          # recordings-download/
    python score_from_logs.py --dir recordings-local --out scores.csv
    python score_from_logs.py --exclude Fede test2     # drop pilots/colleagues
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

FIRST_N = 3          # the subscore window, matching OTH_FIRST_N in app.py

# (puzzle, label, has AI, is part of the transfer block) — mirrors OTH_ROUNDS in
# app.py. R1 is the assisted round; R2+R3 are the unaided transfer block (two
# small X-square puzzles back to back, answers at g7 and b2).
#
# The last entry is the pre-redesign single transfer puzzle. A participant folder
# has either it or the R2/R3 pair, never both, so one pass covers every batch;
# the label is kept distinct ("R2old") so pooled analyses can filter it out.
ROUNDS = [
    ("oc20260727", "R1", True, False),
    ("b220260706", "R2", False, True),
    ("bg20260726", "R3", False, True),
    ("oc20260713", "R2old", False, False),
]

# --- The Solvers / Strugglers split -----------------------------------------
# Labelled on the UNAIDED transfer block only: R1 is the treatment, so putting it
# in the label would condition the outcome on the assistance being studied.
#
# Solver = made most of their transfer-block decisions optimally, which is the
# Connect Four phase's rule (2 of 3 optimal moves) carried over. POOLED across
# the two puzzles — total optimal / total decisions — not the mean of the two
# per-puzzle rates: the puzzles offer 2 or 3 Black decisions depending on how
# the participant played, so averaging rates would weight a 2-decision puzzle
# as heavily as a 3-decision one.
#
# Two properties of this measure to keep in mind when reading the output:
#   - `optimal` credits the best move from wherever the participant already is,
#     so recovering well from a position they ruined scores like never erring.
#     Some Solvers under this rule lost both puzzles outright.
#   - the cut sits on the mode. Eight of 31 participants land exactly on 0.60.
# `transfer_wins`, `transfer_solved` and `transfer_openers` are emitted
# alongside so the grouping can be re-run against outcome-based rules.
SOLVER_POOLED_RATE = 0.6


def read_jsonl(path: Path) -> list[dict]:
    try:
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    except FileNotFoundError:
        return []


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def score_round(moves: list[dict]) -> dict:
    """Everything scoreable about one round, from its decision records alone."""
    if not moves:
        return {}
    # Multi-attempt rounds: score the best attempt, like the app does.
    by_attempt: dict[int, list[dict]] = {}
    for m in moves:
        by_attempt.setdefault(m.get("attempt", 1), []).append(m)

    def one(records: list[dict]) -> dict:
        optimal = [bool(m["optimal"]) for m in records]
        kept = [bool(m["kept_win"]) for m in records]
        losses = [m["disc_loss"] for m in records if m["disc_loss"] is not None]
        times = [datetime.fromisoformat(m["ts"]) for m in records]
        last = records[-1]
        return {
            "decisions": len(records),
            "score": sum(optimal),
            "decisions_first_n": len(optimal[:FIRST_N]),
            "score_first_n": sum(optimal[:FIRST_N]),
            "kept_win_score": sum(kept),
            "kept_win_first_n": sum(kept[:FIRST_N]),
            "opener_optimal": optimal[0],
            "disc_loss_total": sum(losses),
            "disc_loss_per_decision": round(sum(losses) / len(records), 1),
            # Wall-clock from first to last decision. Excludes the time before
            # the opening move, which in this study is substantial (orientation).
            "seconds": round((times[-1] - times[0]).total_seconds()),
            "completed": bool(last.get("game_over")),
            "black_discs": last.get("black_discs"),
            "white_discs": last.get("white_discs"),
            "margin": (None if not last.get("game_over")
                       else last["black_discs"] - last["white_discs"]),
            # Blank, not False, for an unfinished round: the buzzer disc counts
            # are a position, not a result, so "did they win" has no answer.
            "won": (None if not last.get("game_over")
                    else last["black_discs"] > last["white_discs"]),
            "moves": " ".join(m["move"] + ("" if m["optimal"] else "!") for m in records),
        }

    scored = {a: one(r) for a, r in by_attempt.items()}
    best_attempt = max(scored, key=lambda a: (scored[a]["score"], scored[a]["score_first_n"]))
    out = dict(scored[best_attempt])
    out["attempts"] = len(scored)
    out["solved"] = out["completed"] and out["score"] == out["decisions"]
    return out


def transfer_summary(rounds: list[dict]) -> dict:
    """One participant's transfer-block columns, from their R2/R3 rows.

    Returns blanks unless BOTH transfer rounds are present: a label built from
    one puzzle is not the same measure as one built from two, and silently
    mixing the two would put half-measured participants in the same column.
    """
    blank = {"transfer_rounds": len(rounds), "transfer_score": None,
             "transfer_decisions": None, "transfer_rate": None, "transfer_wins": None,
             "transfer_solved": None, "transfer_openers": None,
             "transfer_disc_loss": None, "transfer_label": ""}
    if len(rounds) != 2:
        return blank
    score = sum(r["score"] for r in rounds)
    decisions = sum(r["decisions"] for r in rounds)
    rate = score / decisions
    return {
        "transfer_rounds": 2,
        "transfer_score": score,                                      # optimal decisions, R2+R3
        "transfer_decisions": decisions,                              # decisions offered, R2+R3
        "transfer_rate": round(rate, 3),                              # the label's basis
        "transfer_wins": sum(bool(r["won"]) for r in rounds),
        "transfer_solved": sum(r["score"] == r["decisions"] for r in rounds),
        "transfer_openers": sum(bool(r["opener_optimal"]) for r in rounds),
        # Continuous move-quality cost across the block. Use this as the DV in
        # models rather than binarising it — the distribution is chunky (a 5-way
        # tie sits on the median), which is exactly what a median split ruins.
        "transfer_disc_loss": sum(r["disc_loss_total"] for r in rounds),
        "transfer_label": "Solvers" if rate >= SOLVER_POOLED_RATE else "Strugglers",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="recordings-download")
    ap.add_argument("--out", default="scores.csv")
    ap.add_argument("--exclude", nargs="*", default=["test2", "Fede"],
                    help="participant folders to skip (pilots, colleagues)")
    args = ap.parse_args()

    rows = []
    for d in sorted(Path(args.dir).iterdir()):
        if not d.is_dir() or d.name in args.exclude:
            continue
        demo = read_json(d / "demographics.json") or {}
        gate = read_json(d / "gate_poc20260727.json") or {}
        conv = read_jsonl(d / "conversation_poc20260727.jsonl")
        mine, transfer = [], []
        for puzzle, label, has_ai, is_transfer in ROUNDS:
            moves = read_jsonl(d / f"moves_p{puzzle}.jsonl")
            if not moves:
                continue
            sub = read_json(d / f"game_submission_p{puzzle}.json") or {}
            row = {
                "pid": d.name, "round": label, "puzzle": puzzle, "ai": has_ai,
                "transfer": is_transfer,
                "skill_rating": demo.get("skill_rating"), "age": demo.get("age"),
                "education": demo.get("education"), "genai_usage": demo.get("genai_usage"),
                # The unaided probe: a pre-test of the same X-square motif the
                # transfer block tests, taken before they ever see the assistant.
                "gate_correct": gate.get("first_move_optimal"),
                "gate_confidence": gate.get("confidence"),
                "ai_turns": len(conv) if has_ai else 0,
                "end_reason": sub.get("end_reason"),
                **score_round(moves),
            }
            mine.append(row)
            if is_transfer:
                transfer.append(row)
        # The label is a property of the PARTICIPANT, stamped on each of their
        # rounds so grouping needs no join.
        summary = transfer_summary(transfer)
        for row in mine:
            row.update(summary)
        rows += mine

    fields = list(rows[0].keys()) if rows else []
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"{'pid':<10}{'rnd':<6}{'gate':>6}{'turns':>6}{'score':>8}{'first3':>8}"
          f"{'loss/dec':>10}{'secs':>6}{'outcome':>12}{'group':>12}  moves")
    for r in rows:
        outcome = ("unfinished" if not r["completed"]
                   else f"{r['margin']:+d} {'WIN' if r['won'] else 'loss'}")
        score = f"{r['score']}/{r['decisions']}"
        first_n = f"{r['score_first_n']}/{r['decisions_first_n']}"
        group = r["transfer_label"] or "-"
        print(f"{r['pid'][:8]:<10}{r['round']:<6}"
              f"{('ok' if r['gate_correct'] else 'x'):>6}{r['ai_turns']:>6}"
              f"{score:>8}{first_n:>8}"
              f"{r['disc_loss_per_decision']:>10}{r['seconds']:>6}{outcome:>12}"
              f"{group:>12}  {r['moves']}")

    labelled = {r["pid"]: r["transfer_label"] for r in rows if r["transfer_label"]}
    n_solv = sum(v == "Solvers" for v in labelled.values())
    print(f"\nwrote {args.out} ({len(rows)} participant-rounds; "
          f"'!' marks a suboptimal move)")
    print(f"transfer-block label ({len(labelled)} participants with both R2 and R3): "
          f"{n_solv} Solvers / {len(labelled) - n_solv} Strugglers "
          f"= pooled optimal-move rate >= {SOLVER_POOLED_RATE:.2f} across R2+R3")
    rates = sorted(r["transfer_rate"] for r in rows if r["transfer_label"] and r["round"] == "R2")
    print(f"pooled rates: {rates}")
    unlabelled = {r["pid"] for r in rows} - set(labelled)
    if unlabelled:
        print(f"unlabelled (transfer block incomplete): {len(unlabelled)} "
              f"-> {', '.join(sorted(p[:8] for p in unlabelled))}")


if __name__ == "__main__":
    main()
