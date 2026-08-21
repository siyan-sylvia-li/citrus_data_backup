#!/usr/bin/env python3
"""
judge_screener.py

Run the JudgeSuite multi-model panel on the ORIGINAL Google-form screener prompts
and compare the panel's scores to the human "Evaluation" column.

Input CSV columns: 'Prolific ID', 'Created Prompt', 'Evaluation' (human 1-4).
For each prompt we run the panel under the current (scenario-anchored) rubric,
then report agreement between the human eval and the panel:
  - Pearson r (human eval vs panel mean)
  - exact / within-1 agreement (human eval vs rounded panel mean)
  - ICC(2,1) absolute agreement
  - quadratic-weighted Cohen's kappa

Usage:
    python analysis/judge_screener.py [csv_path]

Default csv_path points at game_user_study/post_processed_data/Pre-Screener (Responses) - Sheet2.csv.
Requires TOGETHER_API_KEY and OPENAI_API_KEY (loaded from .env). ~3 LLM calls/prompt.
"""
import argparse
import csv
import math
import os
import sys
from collections import Counter

import dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
dotenv.load_dotenv(os.path.join(ROOT, ".env"))

from prompt_filter import JudgeSuite  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(
    os.path.dirname(ROOT), "game_user_study", "post_processed_data",
    "Pre-Screener (Responses) - Sheet2.csv",
)


def half_up(x):
    return int(math.floor(x + 0.5))


def icc_2_1(pairs):
    """Two-way random, absolute agreement, single measures."""
    n, k = len(pairs), 2
    grand = sum(a + b for a, b in pairs) / (2 * n)
    rowm = [(a + b) / 2 for a, b in pairs]
    colm = [sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n]
    sst = sum((v - grand) ** 2 for ab in pairs for v in ab)
    ssr = k * sum((rm - grand) ** 2 for rm in rowm)
    ssc = n * sum((cm - grand) ** 2 for cm in colm)
    sse = sst - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + (k / n) * (msc - mse))


def weighted_kappa(pairs, cats=(1, 2, 3, 4)):
    n = len(pairs)
    O = Counter(pairs)
    rA = Counter(a for a, _ in pairs)
    rB = Counter(b for _, b in pairs)
    num = den = 0.0
    for i in cats:
        for j in cats:
            w = (i - j) ** 2
            o = O.get((i, j), 0) / n
            e = (rA[i] / n) * (rB[j] / n)
            num += w * o
            den += w * e
    return 1 - num / den if den else float("nan")


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / math.sqrt(vx * vy) if vx and vy else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path", nargs="?", default=DEFAULT_CSV)
    args = ap.parse_args()

    rows_in = []
    with open(args.csv_path, newline="") as f:
        for r in csv.DictReader(f):
            pid = (r.get("Prolific ID") or "").strip()
            prompt = (r.get("Created Prompt") or "").strip()
            ev = (r.get("Evaluation") or "").strip()
            if pid and prompt and ev.isdigit():
                rows_in.append((pid, prompt, int(ev)))

    suite = JudgeSuite()
    out = []
    print(f"Scoring {len(rows_in)} screener prompts...\n")
    for pid, prompt, human in rows_in:
        mean, scores = suite(prompt=prompt)
        out.append({
            "pid": pid, "human": human, "mean": mean, "scores": scores,
            "panel_round": half_up(mean) if mean is not None else None,
        })
        print(f"{pid[:24]:<25} human={human}  panel_mean={round(mean,2) if mean is not None else None}  {scores}")

    # write CSV
    csv_path = os.path.join(HERE, "screener_panel_scores.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pid", "human_eval", "panel_mean", "panel_round", "qwen", "llama", "gpt"])
        for o in out:
            s = o["scores"]
            w.writerow([o["pid"], o["human"], round(o["mean"], 3) if o["mean"] is not None else "",
                        o["panel_round"], s.get("qwen"), s.get("llama"), s.get("gpt")])

    # agreement (only rows with a valid panel mean)
    usable = [o for o in out if o["mean"] is not None]
    human = [o["human"] for o in usable]
    pmean = [o["mean"] for o in usable]
    pround = [o["panel_round"] for o in usable]
    pairs = list(zip(human, pround))
    exact = sum(1 for a, b in pairs if a == b)
    within1 = sum(1 for a, b in pairs if abs(a - b) <= 1)

    print(f"\n=== Agreement: human eval vs panel (n={len(usable)}) ===")
    print(f"  Pearson r (eval vs panel mean) : {pearson(human, pmean):.3f}")
    print(f"  Exact agreement (eval vs round): {exact}/{len(pairs)} = {exact/len(pairs):.0%}")
    print(f"  Within 1 point                 : {within1}/{len(pairs)} = {within1/len(pairs):.0%}")
    print(f"  ICC(2,1) absolute agreement    : {icc_2_1(pairs):.3f}")
    print(f"  Weighted kappa (quadratic)     : {weighted_kappa(pairs):.3f}")
    print(f"\n  human eval mean = {sum(human)/len(human):.2f} | panel mean = {sum(pmean)/len(pmean):.2f}")
    print(f"  Output CSV: {csv_path}")


if __name__ == "__main__":
    main()
