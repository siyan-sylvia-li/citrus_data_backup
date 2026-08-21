"""Transfer analysis on the WINNING-LINE (strict) metric — round 1 (AI) vs
round 2 (no AI), for the game_user_study_phase_1 two-round design.

Why winning-line: the option-2 "best move from current position" score is
forgiving and gets compressed high (partial credit after a slip), which masks
transfer. The winning-line score ("did the move keep Red's forced win") is the
strict measure, and the cleanest transfer readout is:

    R1 winning-line (AI-assisted)   vs   R2 FIRST-attempt winning-line (unaided, "cold")

R2's best-of-2 is also shown, but that reflects the retry, not carried-over skill.
It also aligns each participant's round-1 chat depth/style and GenAI-usage with
their transfer, to look for conversation patterns (e.g. AI-dependence).

Usage:
    python analysis/transfer_winning_line.py                     # ./recordings-download
    python analysis/transfer_winning_line.py <recordings_dir>
    python analysis/transfer_winning_line.py --include-tester
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics as st
from pathlib import Path

GENAI_RANK = {"never": 0, "less_than_monthly": 1, "about_monthly": 2, "few_times_month": 3,
              "weekly": 4, "few_times_week": 5, "daily": 6, "several_times_day": 7}
TESTER_HINTS = ("tester", "test234", "testing", "unknown", "totally-real")


def loadj(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def loadl(p):
    try:
        return [json.loads(l) for l in open(p) if l.strip()]
    except Exception:
        return []


def pearson(xs, ys):
    pts = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pts) < 2:
        return None
    xs, ys = zip(*pts)
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in pts)
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else None


def round2_score_file(pdir: str):
    """The round-2 cf_score file (cf_score_p<puzzle>.json that isn't p15)."""
    for f in os.listdir(pdir):
        if f.startswith("cf_score_p") and f != "cf_score_p15.json":
            return os.path.join(pdir, f)
    return None


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("recordings", nargs="?", default=str(here.parent / "recordings-download"),
                    help="recordings dir (default: ../recordings-download)")
    ap.add_argument("--include-tester", action="store_true", help="include internal test participants")
    args = ap.parse_args()
    root = Path(args.recordings)

    rows, puzzles = [], set()
    for d in sorted(glob.glob(str(root / "*") + os.sep)):
        pid = os.path.basename(d.rstrip(os.sep))
        if not args.include_tester and any(h in pid.lower() for h in TESTER_HINTS):
            continue
        s1 = loadj(os.path.join(d, "cf_score_p15.json"))
        s2f = round2_score_file(d)
        s2 = loadj(s2f) if s2f else None
        if not (s1 and s2):
            continue                       # completed both rounds only
        puzzles.add(s2.get("puzzle"))
        att = s2.get("attempt_winning_line_scores") or []
        att_opt = s2.get("attempt_scores") or []
        conv = loadl(os.path.join(d, "conversation_p15.jsonl"))
        demo = loadj(os.path.join(d, "demographics.json")) or {}
        rows.append({
            "pid": pid,
            "r1_wl": s1.get("winning_line_score"),
            "r1_opt": s1.get("score"),
            "r2_cold_wl": att[0] if att else None,        # first unaided attempt (strict)
            "r2_best_wl": s2.get("winning_line_score"),    # best across attempts (strict)
            "r2_cold_opt": att_opt[0] if att_opt else None,  # first unaided attempt (option-2)
            "r2_best_opt": s2.get("score"),                # best across attempts (option-2)
            "turns": len(conv),
            "genai": demo.get("genai_usage"),
            "skill": demo.get("skill_rating"),
        })

    if not rows:
        print(f"No completers (both rounds) found in {root}")
        return

    def transfer(r):
        if r["r1_wl"] in (None,) or r["r2_cold_wl"] is None:
            return "n/a"
        if r["r1_wl"] == 0:
            return "R1=0"
        return "transferred" if r["r2_cold_wl"] >= r["r1_wl"] else "dropped"

    n = len(rows)
    print(f"Completers (both rounds): {n}   round-2 puzzle(s): {sorted(p for p in puzzles if p)}")
    if len(puzzles) > 1:
        print("  !! mixed round-2 puzzles in this dir — results not directly comparable")
    print()
    # _c = cold (1st unaided attempt), _b = best-of-two; wl = winning-line (strict), op = option-2
    hdr = (f"{'participant':>14} | {'R1wl':>4} | {'R2wl_c':>6} | {'R2wl_b':>6} | "
           f"{'R2op_c':>6} | {'R2op_b':>6} | {'turns':>5} | {'genai':>16} | transfer")
    print(hdr); print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: (-(r["r2_cold_wl"] or -1), -(r["r1_wl"] or -1))):
        print(f"{r['pid'][:14]:>14} | {str(r['r1_wl']):>4} | {str(r['r2_cold_wl']):>6} | "
              f"{str(r['r2_best_wl']):>6} | {str(r['r2_cold_opt']):>6} | {str(r['r2_best_opt']):>6} | "
              f"{r['turns']:>5} | {str(r['genai']):>16} | {transfer(r)}")

    def col(k):
        return [r[k] for r in rows if r[k] is not None]
    print(f"\n=== winning-line (strict) means (n={n}) ===")
    print(f"  R1 (AI):               {st.mean(col('r1_wl')):.2f} / 3")
    print(f"  R2 cold, 1st unaided:  {st.mean(col('r2_cold_wl')):.2f} / 3   <-- cleanest transfer measure")
    print(f"  R2 best-of-attempts:   {st.mean(col('r2_best_wl')):.2f} / 3   (retry, not cold transfer)")
    print(f"\n=== option-2 'optimal moves' (best move from position) means ===")
    print(f"  R1 (AI):               {st.mean(col('r1_opt')):.2f} / 3")
    print(f"  R2 cold, 1st unaided:  {st.mean(col('r2_cold_opt')):.2f} / 3   <-- cold transfer, option-2")
    print(f"  R2 best-of-two:        {st.mean(col('r2_best_opt')):.2f} / 3   (credits self-correction after a slip)")
    for lab, k in [("R1", "r1_wl"), ("R2 cold", "r2_cold_wl"), ("R2 best", "r2_best_wl")]:
        c = col(k)
        print(f"  fully solved (3/3) {lab:>7}: {sum(v == 3 for v in c)}/{len(c)}")

    grp = {g: [r for r in rows if transfer(r) == g] for g in ("transferred", "dropped", "R1=0")}
    print("\n=== transfer classification ===")
    for g in ("transferred", "dropped", "R1=0"):
        gr = grp[g]
        if not gr:
            continue
        turns = [r["turns"] for r in gr]
        gens = [r["genai"] for r in gr if r["genai"]]
        print(f"  {g:<12} n={len(gr):<2}  mean R1-turns={st.mean(turns):.1f}  genai={gens}")

    print("\n=== correlations (winning-line, n varies) ===")
    r_tc = pearson([r["turns"] for r in rows], [r["r2_cold_wl"] for r in rows])
    r_11 = pearson([r["r1_wl"] for r in rows], [r["r2_cold_wl"] for r in rows])
    r_gc = pearson([GENAI_RANK.get(r["genai"]) for r in rows],
                   [r["r2_cold_wl"] for r in rows])
    print(f"  R1 chat-turns   vs R2-cold wl : r = {r_tc if r_tc is None else round(r_tc,2)}   (negative => more reliance, worse transfer)")
    print(f"  R1 wl           vs R2-cold wl : r = {r_11 if r_11 is None else round(r_11,2)}   (transfer consistency)")
    print(f"  GenAI-usage     vs R2-cold wl : r = {r_gc if r_gc is None else round(r_gc,2)}   (negative => heavier AI use, worse transfer)")
    print("\nNote: small n is noisy; treat as directional until ~15-20 completers on one config.")


if __name__ == "__main__":
    main()
