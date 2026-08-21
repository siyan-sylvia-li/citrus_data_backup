#!/usr/bin/env python3
"""Average NASA-TLX and AI-Likert post-survey ratings, split by puzzle outcome.

Outcome rule (consistent with the rest of the analysis):
  solved   = participant made >= 5 optimal moves
  unsolved = fewer than 5 optimal moves
Excluded: participant with > 8 total moves (retried & succeeded) and the
'test-test' scratch account.  -> 6 solved / 4 unsolved.
"""
import json
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(__file__), "recordings-download_batch_1")
TLX = ["mental_demand", "physical_demand", "temporal_demand",
       "performance", "effort", "frustration"]
LIKERT = ["ai_consistent", "ai_easy_to_use"]

SOLVED = "Solved (≥5 optimal)"
UNSOLVED = "Unsolved"
EXCLUDE = {"test-test"}


def outcome(d):
    """SOLVED / UNSOLVED / None (excluded)."""
    pid = os.path.basename(d.rstrip("/"))
    if pid in EXCLUDE:
        return None
    mv = os.path.join(d, "moves.jsonl")
    if not os.path.exists(mv):
        return None
    moves = [json.loads(l) for l in open(mv) if l.strip()]
    if not moves or len(moves) > 8:            # >8 total = retried & succeeded -> excluded
        return None
    return SOLVED if sum(1 for m in moves if m.get("optimal")) >= 5 else UNSOLVED


def n_turns(d):
    p = os.path.join(d, "conversation.jsonl")
    return sum(1 for l in open(p) if l.strip()) if os.path.exists(p) else 0


labels = [SOLVED, UNSOLVED]
groups = {k: [] for k in labels}
turns = {k: [] for k in labels}
for d in sorted(glob.glob(os.path.join(BASE, "*/"))):
    ps = os.path.join(d, "post_survey.json")
    o = outcome(d)
    if o is None or not os.path.exists(ps):
        continue
    groups[o].append(json.load(open(ps)))
    turns[o].append(n_turns(d))


# two-sided t critical values, alpha=0.05, by degrees of freedom (n-1); 1.96 for df>30
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
       8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
       15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086}


def t_crit(n):
    return T95.get(n - 1, 1.96)


def mean(surveys, section, field):
    vals = [su[section][field] for su in surveys if field in su.get(section, {})]
    return np.mean(vals) if vals else np.nan

def ci95(surveys, section, field):
    """Half-width of the t-based 95% CI for the mean: t(.975, n-1) * s / sqrt(n)."""
    vals = [su[section][field] for su in surveys if field in su.get(section, {})]
    n = len(vals)
    if n < 2:
        return 0.0
    return t_crit(n) * np.std(vals, ddof=1) / np.sqrt(n)

counts = {k: len(v) for k, v in groups.items()}
print("Group sizes:", counts)

tlx_means = {k: [mean(v, "nasa_tlx", f) for f in TLX] for k, v in groups.items()}
tlx_ci = {k: [ci95(v, "nasa_tlx", f) for f in TLX] for k, v in groups.items()}
likert_means = {k: [mean(v, "ai_likert", f) for f in LIKERT] for k, v in groups.items()}
likert_ci = {k: [ci95(v, "ai_likert", f) for f in LIKERT] for k, v in groups.items()}

for k in labels:
    print(f"\n{k} (n={counts[k]})")
    for f, m in zip(TLX, tlx_means[k]):
        print(f"  TLX  {f:16} {m:.2f}")
    for f, m in zip(LIKERT, likert_means[k]):
        print(f"  AI   {f:16} {m:.2f}")

COLORS = {SOLVED: "#2a9d8f", UNSOLVED: "#e76f51"}


def grouped_bar(ax, fields, means_by_group, ci_by_group, title, ymax):
    x = np.arange(len(fields))
    w = 0.38
    for i, g in enumerate(labels):
        offs = (i - 0.5) * w
        bars = ax.bar(x + offs, means_by_group[g], w, yerr=ci_by_group[g], capsize=3,
                      label=f"{g} (n={counts[g]})", color=COLORS[g],
                      error_kw=dict(lw=1))
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_", "\n") for f in fields], fontsize=9)
    ax.set_ylabel("Mean rating")
    ax.set_ylim(0, ymax)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)


turns_means = {k: np.mean(v) if v else np.nan for k, v in turns.items()}
for k in labels:
    print(f"\n{k}: avg conversation turns = {turns_means[k]:.2f} (values {turns[k]})")

fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                         gridspec_kw={"width_ratios": [3, 1.2, 0.9]})
grouped_bar(axes[0], TLX, tlx_means, tlx_ci, "NASA-TLX (1–7) by solve outcome", 8.5)
grouped_bar(axes[1], LIKERT, likert_means, likert_ci, "AI Likert (1–5) by solve outcome", 6.0)

ax = axes[2]
turns_ci = {k: (t_crit(len(v)) * np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
            for k, v in turns.items()}
bars = ax.bar(range(len(labels)), [turns_means[k] for k in labels],
              yerr=[turns_ci[k] for k in labels], capsize=3, error_kw=dict(lw=1),
              color=[COLORS[k] for k in labels], width=0.6)
ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=9)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels([f"{k}\n(n={counts[k]})" for k in labels], fontsize=8)
ax.set_ylabel("Mean # turns")
ax.set_ylim(0, (max(turns_means.values()) + max(turns_ci.values())) * 1.2)
ax.set_title("Avg conversation turns", fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

fig.suptitle("Post-survey averages: solved (6) vs. unsolved (4)   [error bars = 95% CI, t-based]",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(os.path.dirname(__file__), "post_survey_averages.png")
fig.savefig(out, dpi=150)
print("\nSaved:", out)
