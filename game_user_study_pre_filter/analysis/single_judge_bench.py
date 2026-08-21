#!/usr/bin/env python3
"""
single_judge_bench.py

Benchmark a SINGLE-model judge (one grader, no panel) against the human
screener Evaluation, with timing. Reuses the live JUDGE_RUBRIC / JudgeSignature.

Usage:
    python analysis/single_judge_bench.py [model_id] [predict|cot]

Defaults: model_id = openai/gpt-5.4-mini, grader = cot.
Requires the relevant API key in .env.
"""
import csv
import math
import os
import statistics
import sys
import time
from collections import Counter

import dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
dotenv.load_dotenv(os.path.join(ROOT, ".env"))
import dspy  # noqa: E402
from prompt_filter import JUDGE_RUBRIC, JudgeSignature  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-5.4-mini"
GRADER_KIND = (sys.argv[2] if len(sys.argv) > 2 else "cot").lower()
CSV = os.path.join(os.path.dirname(ROOT), "game_user_study", "post_processed_data",
                   "Pre-Screener (Responses) - Sheet2.csv")


def half_up(x): return int(math.floor(x + 0.5))


def icc21(pairs):
    n, k = len(pairs), 2
    g = sum(a + b for a, b in pairs) / (2 * n)
    rm = [(a + b) / 2 for a, b in pairs]
    cm = [sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n]
    sst = sum((v - g) ** 2 for ab in pairs for v in ab)
    ssr = k * sum((r - g) ** 2 for r in rm)
    ssc = n * sum((c - g) ** 2 for c in cm)
    sse = sst - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + (k / n) * (msc - mse))


def wkappa(pairs, cats=(1, 2, 3, 4)):
    n = len(pairs); O = Counter(pairs)
    rA = Counter(a for a, _ in pairs); rB = Counter(b for _, b in pairs)
    num = den = 0.0
    for i in cats:
        for j in cats:
            w = (i - j) ** 2
            num += w * O.get((i, j), 0) / n
            den += w * (rA[i] / n) * (rB[j] / n)
    return 1 - num / den if den else float("nan")


def pearson(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs); vy = sum((y - my) ** 2 for y in ys)
    return cov / math.sqrt(vx * vy) if vx and vy else float("nan")


lm = dspy.LM(MODEL, max_tokens=2048)
grader = (dspy.ChainOfThought if GRADER_KIND == "cot" else dspy.Predict)(JudgeSignature)
grader.set_lm(lm)

rows = []
with open(CSV, newline="") as f:
    for r in csv.DictReader(f):
        p = (r.get("Created Prompt") or "").strip()
        ev = (r.get("Evaluation") or "").strip()
        if p and ev.isdigit():
            rows.append((p, int(ev)))

print(f"Single judge: {MODEL}  grader={GRADER_KIND}  on {len(rows)} screener prompts...\n")
human, panel, times, miss = [], [], [], 0
for prompt, ev in rows:
    t0 = time.perf_counter()
    try:
        s = int(grader(prompt_judged=prompt, rubric=JUDGE_RUBRIC).output_score)
        s = max(1, min(4, s))
    except Exception as e:
        miss += 1
        print(f"  miss: {str(e)[:80]}")
        continue
    times.append(time.perf_counter() - t0)
    human.append(ev); panel.append(s)

pairs = list(zip(human, panel))
exact = sum(1 for a, b in pairs if a == b)
within1 = sum(1 for a, b in pairs if abs(a - b) <= 1)
m = statistics.mean(times); sd = statistics.stdev(times) if len(times) > 1 else 0.0
half = 1.96 * sd / math.sqrt(len(times))

print(f"\n=== {MODEL} ({GRADER_KIND}) vs human, n={len(pairs)} (missing={miss}) ===")
print(f"  ICC(2,1)             : {icc21(pairs):.3f}")
print(f"  Weighted kappa (quad): {wkappa(pairs):.3f}")
print(f"  Pearson r            : {pearson(human, panel):.3f}")
print(f"  Exact agreement      : {exact}/{len(pairs)} = {exact/len(pairs):.0%}")
print(f"  Within 1 point       : {within1}/{len(pairs)} = {within1/len(pairs):.0%}")
print(f"  human mean={statistics.mean(human):.2f}  judge mean={statistics.mean(panel):.2f}")
print(f"  time/prompt: mean {m:.2f}s  95% CI [{m-half:.2f}, {m+half:.2f}]")
