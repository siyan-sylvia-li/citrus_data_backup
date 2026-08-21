#!/usr/bin/env python3
"""
Grouped bar chart of user-turn dialogue acts, solved vs unsolved participants.
Outcome rule: solved = >=5 optimal moves; exclude >8-total-moves participant and test-test.
Two panels: raw counts and per-turn frequency (fairer, since group sizes differ).
"""
import json, glob, os
from collections import Counter, defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings-download_batch_1")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_charts",
                   "dialogue_acts_solved_vs_unsolved.png")

def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]

# classify participants
outcome = {}
for f in sorted(glob.glob(os.path.join(BASE, "*", "moves.jsonl"))):
    pid = os.path.basename(os.path.dirname(f))
    if pid == "test-test":
        continue
    moves = load_jsonl(f)
    if len(moves) > 8:
        continue
    outcome[pid] = "solved" if sum(1 for m in moves if m.get("optimal")) >= 5 else "unsolved"

# collect acts (flat list of acts per pid) and count user turns per pid
acts_by_pid = defaultdict(list)
turns_by_pid = defaultdict(int)
SCHEME = {}
for f in sorted(glob.glob(os.path.join(BASE, "*", "conversation_annotated.jsonl"))):
    pid = os.path.basename(os.path.dirname(f))
    for d in load_jsonl(f):
        turns_by_pid[pid] += 1  # one user turn per line
        for a in d.get("user_dialogue_acts", []):
            acts_by_pid[pid].append(a["act"])
            SCHEME[a["act"]] = a["scheme"]

def group_pids(group):
    return [p for p in outcome if outcome[p] == group]

def counts(group):
    pids = group_pids(group)
    n_turns = sum(turns_by_pid[p] for p in pids)
    return Counter(a for p in pids for a in acts_by_pid[p]), n_turns, len(pids)

def per_participant_rates(group, act):
    """acts-per-turn of `act` for each participant in group."""
    return np.array([Counter(acts_by_pid[p])[act] / turns_by_pid[p]
                     for p in group_pids(group)])

def bootstrap_ci(rates, B=10000, seed=0):
    """Mean per-participant rate and 95% CI by resampling participants."""
    rng = np.random.default_rng(seed)
    n = len(rates)
    if n == 0:
        return 0.0, 0.0, 0.0
    means = rates[rng.integers(0, n, size=(B, n))].mean(axis=1)
    return rates.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)

s_c, s_turns, s_n = counts("solved")
u_c, u_turns, u_n = counts("unsolved")

# order acts: student first then tutor, by combined frequency
acts = sorted(set(s_c) | set(u_c), key=lambda a: (SCHEME[a] != "student", -(s_c[a] + u_c[a])))
x = np.arange(len(acts))
w = 0.4
S_COL, U_COL = "#2a9d8f", "#e76f51"

# per-participant means + bootstrap CIs
def group_stats(group):
    means, los, his = [], [], []
    for a in acts:
        m, lo, hi = bootstrap_ci(per_participant_rates(group, a))
        means.append(m); los.append(lo); his.append(hi)
    means = np.array(means)
    yerr = np.vstack([np.clip(means - np.array(los), 0, None),
                      np.clip(np.array(his) - means, 0, None)])
    return means, yerr

s_mean, s_err = group_stats("solved")
u_mean, u_err = group_stats("unsolved")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10))

# panel 1: raw counts
ax1.bar(x - w/2, [s_c[a] for a in acts], w, label=f"Solved (n={s_n}, {s_turns} turns)", color=S_COL)
ax1.bar(x + w/2, [u_c[a] for a in acts], w, label=f"Unsolved (n={u_n}, {u_turns} turns)", color=U_COL)
ax1.set_ylabel("Count (user turns)")
ax1.set_title("User dialogue acts by outcome: raw counts")
ax1.legend()

# panel 2: per-participant mean acts-per-turn with 95% bootstrap CI
ax2.bar(x - w/2, s_mean, w, yerr=s_err, capsize=3, label="Solved", color=S_COL,
        error_kw=dict(ecolor="#1d6f66", lw=1.2))
ax2.bar(x + w/2, u_mean, w, yerr=u_err, capsize=3, label="Unsolved", color=U_COL,
        error_kw=dict(ecolor="#a84a34", lw=1.2))
ax2.set_ylabel("Acts per user turn (per-participant mean)")
ax2.set_title("Per-participant mean acts per turn, 95% bootstrap CI (resampling participants)")
ax2.legend()

for ax in (ax1, ax2):
    ax.set_xticks(x)
    labels = [f"{a}\n({'T' if SCHEME[a]=='tutor' else 'S'})" for a in acts]
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("Dialogue acts on user turns: solved (6) vs unsolved (4) [S=student move, T=tutor move]\n"
             "(Small sample size, directional only)",
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("wrote", OUT)
