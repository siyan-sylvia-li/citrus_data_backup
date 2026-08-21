#!/usr/bin/env python3
"""
Correlate per-participant dialogue-act profiles with move efficiency (optimal/total moves).
Continuous outcome -> Spearman (rank) correlation, robust to the clumped efficiency values.
Excludes the >8-move participant and test-test. n=10.
"""
import json, glob, os, csv
from collections import Counter, defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from scipy.stats import spearmanr, pearsonr
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "recordings-download_batch_1")
OUT_PNG = os.path.join(HERE, "analysis_charts", "efficiency_vs_dialogue_acts.png")
OUT_CSV = os.path.join(BASE, "efficiency_features.csv")

def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]

# ---- move efficiency + solved ----
eff, solved = {}, {}
for f in sorted(glob.glob(os.path.join(BASE, "*", "moves.jsonl"))):
    pid = os.path.basename(os.path.dirname(f))
    if pid == "test-test":
        continue
    m = load_jsonl(f)
    total = len(m)
    opt = sum(1 for x in m if x.get("optimal"))
    if total > 8:
        continue  # excluded
    eff[pid] = opt / total
    solved[pid] = opt >= 5

# ---- dialogue-act features ----
acts_by_pid = defaultdict(Counter)
turns_by_pid = defaultdict(int)
for f in sorted(glob.glob(os.path.join(BASE, "*", "conversation_annotated.jsonl"))):
    pid = os.path.basename(os.path.dirname(f))
    if pid not in eff:
        continue
    for d in load_jsonl(f):
        turns_by_pid[pid] += 1
        for a in d.get("user_dialogue_acts", []):
            acts_by_pid[pid][a["act"]] += 1

REASONING = {"Think Aloud", "Metacomment", "Partial Answer", "Common Ground Question"}
HELP = {"Knowledge Deficit Question", "Forced Choice"}

pids = sorted(eff, key=lambda p: eff[p])
feat = {}
for p in pids:
    t = turns_by_pid[p]
    c = acts_by_pid[p]
    reasoning = sum(c[a] for a in REASONING)
    help_ = sum(c[a] for a in HELP)
    feat[p] = {
        "efficiency": eff[p],
        "solved": solved[p],
        "n_turns": t,
        "total_acts": sum(c.values()),
        "acts_per_turn": sum(c.values()) / t,
        "KDQ_rate": c["Knowledge Deficit Question"] / t,
        "ThinkAloud_rate": c["Think Aloud"] / t,
        "reasoning_rate": reasoning / t,
        "help_rate": help_ / t,
        "reasoning_share": reasoning / (reasoning + help_) if (reasoning + help_) else np.nan,
    }

# ---- correlations ----
FEATURES = ["n_turns", "total_acts", "acts_per_turn", "KDQ_rate", "ThinkAloud_rate",
            "reasoning_rate", "help_rate", "reasoning_share"]
y = np.array([feat[p]["efficiency"] for p in pids])

def spearman_manual(a, b):
    ra, rb = _rank(a), _rank(b)
    return _pearson(ra, rb)
def spearman_perm(a, b, B=50000, seed=0):
    """Spearman r + two-sided permutation p (exact-null via resampled rank correlation)."""
    ra, rb = _rank(a), _rank(b)
    r = _pearson(ra, rb)
    rng = np.random.default_rng(seed)
    ra_c = ra - ra.mean()
    rb_c = rb - rb.mean()
    perms = np.argsort(rng.random((B, len(a))), axis=1)      # random permutations
    stats = (rb_c[perms] * ra_c).sum(axis=1)                  # ∝ correlation (denominators constant)
    obs = abs((rb_c * ra_c).sum())
    p = (np.sum(np.abs(stats) >= obs - 1e-9) + 1) / (B + 1)
    return r, p
def _rank(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x)); r[order] = np.arange(len(x))
    # average ties
    vals = np.array(x)
    out = np.empty(len(x))
    for v in np.unique(vals):
        idx = np.where(vals == v)[0]
        out[idx] = r[idx].mean()
    return out
def _pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1])

rows = []
print(f"{'feature':16} {'spearman_r':>11} {'p':>8}   {'pearson_r':>10} {'p':>8}   direction")
for fname in FEATURES:
    x = np.array([feat[p][fname] for p in pids], float)
    mask = ~np.isnan(x)
    xx, yy = x[mask], y[mask]
    sr, sp = spearman_perm(xx, yy)
    pr = _pearson(xx, yy)
    pp = pearsonr(xx, yy)[1] if HAVE_SCIPY else float("nan")
    direction = "↑ higher eff" if sr > 0 else "↓ higher eff"
    print(f"{fname:16} {sr:>11.3f} {sp:>8.3f}   {pr:>10.3f} {pp:>8.3f}   {direction}")
    rows.append({"feature": fname, "spearman_r": round(sr, 3), "spearman_p": round(sp, 4),
                 "pearson_r": round(pr, 3), "pearson_p": round(pp, 4)})

# ---- CSV: per-participant features + correlation summary ----
with open(OUT_CSV, "w", newline="") as cf:
    w = csv.writer(cf)
    w.writerow(["prolific_id", "efficiency", "solved"] + FEATURES)
    for p in pids:
        w.writerow([p, round(feat[p]["efficiency"], 3), feat[p]["solved"]] +
                   [round(feat[p][k], 3) if isinstance(feat[p][k], float) else feat[p][k]
                    for k in FEATURES])
    w.writerow([])
    w.writerow(["feature", "spearman_r", "spearman_p", "pearson_r", "pearson_p"])
    for r in rows:
        w.writerow([r["feature"], r["spearman_r"], r["spearman_p"], r["pearson_r"], r["pearson_p"]])
print("\nwrote", OUT_CSV)

# ---- scatter grid for the most interpretable features ----
PANELS = ["KDQ_rate", "ThinkAloud_rate", "reasoning_share", "n_turns"]
TITLES = {"KDQ_rate": "Knowledge Deficit Q rate (help-seeking)",
          "ThinkAloud_rate": "Think-Aloud rate (self-explanation)",
          "reasoning_share": "Reasoning share  reasoning/(reasoning+help)",
          "n_turns": "Number of user turns (engagement)"}
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
for ax, fname in zip(axes.ravel(), PANELS):
    x = np.array([feat[p][fname] for p in pids], float)
    mask = ~np.isnan(x); xx, yy = x[mask], y[mask]
    cols = ["#2a9d8f" if feat[p]["solved"] else "#e76f51" for p, m in zip(pids, mask) if m]
    ax.scatter(xx, yy, c=cols, s=80, edgecolor="k", zorder=3)
    if len(xx) > 1 and np.ptp(xx) > 0:
        b, a = np.polyfit(xx, yy, 1)
        xs = np.linspace(xx.min(), xx.max(), 50)
        ax.plot(xs, b * xs + a, "--", color="gray", zorder=2)
    sr, sp = spearman_perm(xx, yy)
    sub = f"Spearman r={sr:.2f}, perm p={sp:.2f}"
    ax.set_title(f"{TITLES[fname]}\n{sub}", fontsize=10)
    ax.set_xlabel(fname); ax.set_ylabel("move efficiency"); ax.grid(alpha=0.3)
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([0],[0],marker="o",color="w",markerfacecolor="#2a9d8f",markeredgecolor="k",label="solved"),
                    Line2D([0],[0],marker="o",color="w",markerfacecolor="#e76f51",markeredgecolor="k",label="unsolved")],
           loc="upper right", ncol=2)
fig.suptitle("Move efficiency vs user dialogue-act features (n=10; small sample, directional)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print("wrote", OUT_PNG)
print("scipy:", HAVE_SCIPY)
