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

Agreement statistics come from statsmodels/scipy rather than hand-rolled formulas:
  - scipy.stats.pearsonr
  - statsmodels.stats.inter_rater.cohens_kappa (quadratic weights, with 95% CI)
  - ICC(2,1) from a two-way statsmodels ANOVA (score ~ C(subject) + C(rater))

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

import dotenv
import numpy as np
import pandas as pd
from scipy import stats as sps
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.inter_rater import cohens_kappa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
dotenv.load_dotenv(os.path.join(ROOT, ".env"))

from prompt_filter import JudgeSuite  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = "pre_screener.csv"
CATS = [1, 2, 3, 4]


def half_up(x):
    return int(math.floor(x + 0.5))


def icc_2_1(rater_a, rater_b):
    """ICC(2,1): two-way random effects, absolute agreement, single measures.

    Mean squares come from a two-way ANOVA fit by statsmodels on the long-format
    ratings; the ICC itself is the standard McGraw & Wong combination of them.
    """
    n, k = len(rater_a), 2
    long = pd.DataFrame({
        "score": np.concatenate([np.asarray(rater_a, float), np.asarray(rater_b, float)]),
        "subject": np.tile(np.arange(n), k),
        "rater": np.repeat(np.arange(k), n),
    })
    tbl = anova_lm(smf.ols("score ~ C(subject) + C(rater)", data=long).fit(), typ=2)
    ms = tbl["sum_sq"] / tbl["df"]
    msr, msc, mse = ms["C(subject)"], ms["C(rater)"], ms["Residual"]
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def contingency(rater_a, rater_b, cats=CATS):
    """Square cats x cats count table, keeping categories nobody used."""
    return pd.crosstab(
        pd.Categorical(rater_a, categories=cats),
        pd.Categorical(rater_b, categories=cats),
        dropna=False,
    ).to_numpy()


def weighted_kappa(rater_a, rater_b, cats=CATS):
    """Quadratic-weighted Cohen's kappa; returns the full statsmodels result."""
    return cohens_kappa(contingency(rater_a, rater_b, cats), wt="quadratic")


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
        w.writerow(["pid", "human_eval", "panel_mean", "panel_round", "nemotron", "llama", "gpt"])
        for o in out:
            s = o["scores"]
            w.writerow([o["pid"], o["human"], round(o["mean"], 3) if o["mean"] is not None else "",
                        o["panel_round"], s.get("nemotron"), s.get("llama"), s.get("gpt")])

    # agreement (only rows with a valid panel mean)
    usable = [o for o in out if o["mean"] is not None]
    human = [o["human"] for o in usable]
    pmean = [o["mean"] for o in usable]
    pround = [o["panel_round"] for o in usable]
    n = len(usable)
    diff = np.abs(np.asarray(human) - np.asarray(pround))
    exact, within1 = int((diff == 0).sum()), int((diff <= 1).sum())
    r = sps.pearsonr(human, pmean)
    kappa = weighted_kappa(human, pround)

    print(f"\n=== Agreement: human eval vs panel (n={n}) ===")
    print(f"  Pearson r (eval vs panel mean) : {r.statistic:.3f} (p={r.pvalue:.3g})")
    print(f"  Exact agreement (eval vs round): {exact}/{n} = {exact/n:.0%}")
    print(f"  Within 1 point                 : {within1}/{n} = {within1/n:.0%}")
    print(f"  ICC(2,1) absolute agreement    : {icc_2_1(human, pround):.3f}")
    print(f"  Weighted kappa (quadratic)     : {kappa.kappa:.3f} "
          f"[95% CI {kappa.kappa_low:.3f}, {kappa.kappa_upp:.3f}]")
    print(f"\n  human eval mean = {np.mean(human):.2f} | panel mean = {np.mean(pmean):.2f}")
    print(f"  Output CSV: {csv_path}")


if __name__ == "__main__":
    main()
