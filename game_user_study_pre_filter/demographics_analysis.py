#!/usr/bin/env python3
"""Demographic breakdown by puzzle outcome.

Outcome rule (consistent with the rest of the analysis):
  solved   = participant made >= 5 optimal moves
  unsolved = fewer than 5 optimal moves
Excluded: participant with > 8 total moves (retried & succeeded) and 'test-test'.
-> 6 solved / 4 unsolved.  Numeric panels show t-based 95% CIs.
"""
import json, glob, os, statistics
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(__file__), "recordings-download_batch_1")
SOLVED, UNSOLVED = "Solved (≥5 optimal)", "Unsolved"
EXCLUDE = {"test-test"}

# two-sided t critical values, alpha=0.05, by df (n-1); 1.96 for df>20
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
       8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
       15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086}


def t_crit(n):
    return T95.get(n - 1, 1.96)


def outcome(d):
    pid = os.path.basename(d.rstrip("/"))
    if pid in EXCLUDE:
        return None
    mv = os.path.join(d, "moves.jsonl")
    if not os.path.exists(mv):
        return None
    moves = [json.loads(l) for l in open(mv) if l.strip()]
    if not moves or len(moves) > 8:
        return None
    return SOLVED if sum(1 for m in moves if m.get("optimal")) >= 5 else UNSOLVED


labels = [SOLVED, UNSOLVED]
groups = {k: [] for k in labels}
for d in sorted(glob.glob(os.path.join(BASE, "*/"))):
    o = outcome(d)
    dem = os.path.join(d, "demographics.json")
    if o is None or not os.path.exists(dem):
        continue
    groups[o].append(json.load(open(dem)))

counts = {k: len(v) for k, v in groups.items()}
COLORS = {SOLVED: "#2a9d8f", UNSOLVED: "#e76f51"}

NUM = [("age", "Age", 70),
       ("game_familiarity", "Game familiarity (1–5)", 6.0),
       ("skill_rating", "Self-rated skill (1–5)", 6.0)]


def vals_of(rows, f):
    return [r[f] for r in rows if isinstance(r.get(f), (int, float))]

def mean(rows, f):
    v = vals_of(rows, f)
    return statistics.mean(v) if v else np.nan

def ci95(rows, f):
    v = vals_of(rows, f)
    n = len(v)
    if n < 2:
        return 0.0
    return t_crit(n) * statistics.stdev(v) / np.sqrt(n)


fig, axes = plt.subplots(1, 4, figsize=(20, 6),
                         gridspec_kw={"width_ratios": [1, 1, 1, 1.4]})

for ax, (f, title, ymax) in zip(axes[:3], NUM):
    m = [mean(groups[k], f) for k in labels]
    e = [ci95(groups[k], f) for k in labels]
    bars = ax.bar(range(len(labels)), m, yerr=e, capsize=4, error_kw=dict(lw=1),
                  color=[COLORS[k] for k in labels], width=0.6)
    ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=10)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"{k}\n(n={counts[k]})" for k in labels], fontsize=9)
    ax.set_ylim(0, ymax)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

# Education distribution (stacked counts)
ax = axes[3]
edu_levels = sorted({r.get("education") for v in groups.values() for r in v if r.get("education")})
edu_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(edu_levels)))
bottoms = np.zeros(len(labels))
for lvl, c in zip(edu_levels, edu_colors):
    heights = [Counter(r.get("education") for r in groups[k])[lvl] for k in labels]
    ax.bar(range(len(labels)), heights, bottom=bottoms, label=lvl, color=c, width=0.6)
    bottoms += heights
ax.set_xticks(range(len(labels)))
ax.set_xticklabels([f"{k}\n(n={counts[k]})" for k in labels], fontsize=9)
ax.set_ylabel("# participants")
ax.set_title("Education", fontsize=12, fontweight="bold")
ax.legend(fontsize=8, loc="upper right")

fig.suptitle(f"Demographics: solved ({counts[SOLVED]}) vs. unsolved ({counts[UNSOLVED]})   "
             "[error bars = 95% CI, t-based]", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(os.path.dirname(__file__), "demographics_breakdown.png")
fig.savefig(out, dpi=150)
print("Saved:", out)

# Text summary
for k in labels:
    print(f"\n{k} (n={counts[k]})")
    for f, title, _ in NUM:
        print(f"  {title:24} mean={mean(groups[k], f):.2f}  (95% CI ±{ci95(groups[k], f):.2f})")
    print(f"  played_connect4_before  {dict(Counter(r.get('played_connect4_before') for r in groups[k]))}")
    print(f"  education                {dict(Counter(r.get('education') for r in groups[k]))}")
    print(f"  occupation               {dict(Counter(r.get('occupation') for r in groups[k]))}")
