"""Test the pre-registered prediction of the refined assistant scheme.

Recorded in dialogue_act_annotation_assistant_fine.py BEFORE the re-annotation ran:

    General Principle and Worked Line predict unassisted transfer.
    Board Report and Move Verdict do not.

Metric is TURN SHARE — the fraction of a run's assistant turns carrying the act —
matching assistant_act_analysis.py so the numbers sit alongside the coarse ones.
Outcome is the truncated solo score already stored in the shim (0-3, first three
decisions, the human scale).

Two columns, for the reason established earlier in this project: the pooled
correlation is the estimate comparable to the human analysis and is deliberately
NOT adjusted for student model (models are how this simulator spans participant
ability, and partialling them out deletes the intended variance). But an
association carried by a single engine is that engine's house style, so
rho_min(LOO) reports the weakest correlation after dropping any one model.

    python test_fine_prediction.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

BASE = Path(__file__).resolve().parent
ROOT = BASE / "sim_assistant_acts_cf" / "recordings-download"
ANN = "annotated_conversation_p15.jsonl"
FIELD = "annotation_assistant_fine"

sys.path.insert(0, str(BASE))
import dialogue_act_annotation_assistant_fine as fine  # noqa: E402

# See note in othello_assistant_acts.py: gate on share SD, not prevalence.
MIN_SHARE_SD, RARE_BELOW = 1.0, 2.0


def load():
    """{run: (turn_share dict, outcome, model)} over runs with fine coding."""
    out = {}
    for d in sorted(ROOT.iterdir()):
        if not d.is_dir() or not (d / ANN).exists():
            continue
        acts, n_turns = Counter(), 0
        for line in open(d / ANN):
            if not line.strip():
                continue
            t = json.loads(line)
            a = t.get(FIELD)
            if not a or a.get("n_valid", 0) < 1:
                continue
            n_turns += 1
            for act in dict.fromkeys(a["final"]):
                acts[act] += 1
        if not n_turns:
            continue
        score_f = d / "cf_score_selectivity_pw4p6.json"
        demo_f = d / "demographics.json"
        if not score_f.exists():
            continue
        y = json.load(open(score_f)).get("flat_score")
        if y is None:
            continue
        model = json.load(open(demo_f)).get("student_model") if demo_f.exists() else None
        out[d.name] = ({k: 100 * v / n_turns for k, v in acts.items()},
                       float(y), model, n_turns)
    return out


def loo_min(x, y, groups):
    """Weakest Spearman rho after dropping any one student model."""
    worst = (np.inf, np.nan, None)
    for m in set(groups):
        keep = [i for i, g in enumerate(groups) if g != m]
        if len(keep) < 10:
            continue
        xs, ys = x[keep], y[keep]
        if xs.std() == 0 or ys.std() == 0:
            continue
        r, p = stats.spearmanr(xs, ys)
        if abs(r) < abs(worst[0]):
            worst = (r, p, m)
    return worst if np.isfinite(worst[0]) else (np.nan, np.nan, None)


def bh(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj, prev = np.empty(len(p)), 1.0
    for rank, idx in enumerate(order[::-1]):
        prev = min(prev, p[idx] * len(p) / (len(p) - rank))
        adj[idx] = prev
    return adj


def main():
    data = load()
    if len(data) < 10:
        print(f"only {len(data)} runs carry fine coding — let the annotation finish")
        return 1
    runs = sorted(data)
    y = np.array([data[r][1] for r in runs])
    models = [data[r][2] for r in runs]
    n_turns = np.array([data[r][3] for r in runs])

    print(f"{len(runs)} runs | outcome = truncated solo score 0-3 "
          f"(mean {y.mean():.2f}, sd {y.std(ddof=1):.2f}) | "
          f"{int(n_turns.sum())} coded assistant turns")

    rows = []
    for act in fine.ROLLUP:
        x = np.array([data[r][0].get(act, 0.0) for r in runs])
        prev = float((x > 0).mean() * 100)
        if x.std() == 0:
            rows.append([act, prev, np.nan, np.nan, np.nan, None, "no variance", 0.0])
            continue
        note = ("no spread" if x.std(ddof=1) < MIN_SHARE_SD
                else "rare" if prev < RARE_BELOW else "")
        r, p = stats.spearmanr(x, y)
        rl, _, wm = loo_min(x, y, models)
        rows.append([act, prev, r, p, rl, wm, note, float(x.std(ddof=1))])

    testable = [r for r in rows if np.isfinite(r[2]) and not r[6]]
    q = bh([r[3] for r in testable]) if testable else []
    qmap = {r[0]: qq for r, qq in zip(testable, q)}

    print(f"\n{'act':>32}{'prev%':>7}{'sd':>6}{'rho':>8}{'p':>8}{'q':>7}{'rho_min':>9}  tag")
    for act, prev, r, p, rl, wm, note, sd in rows:
        tag = note
        if act in fine.TRANSFER_BEARING:
            tag = (tag + " " if tag else "") + "<PREDICTED +>"
        elif act in fine.TRANSFER_INERT:
            tag = (tag + " " if tag else "") + "<PREDICTED null>"
        f = lambda v: f"{v:+.3f}" if np.isfinite(v) else "    n/a"
        qs = f"{qmap[act]:.3f}" if act in qmap else "    -"
        print(f"{act:>32}{prev:>7.1f}{sd:>6.1f}{f(r):>8}{p if np.isfinite(p) else float('nan'):>8.3f}"
              f"{qs:>7}{f(rl):>9}  {tag}")

    # The prediction as a single contrast: do the two transfer-bearing codes carry a
    # larger association than the two inert ones?
    def mean_rho(names):
        vals = [r[2] for r in rows if r[0] in names and np.isfinite(r[2])]
        return float(np.mean(vals)) if vals else np.nan

    tb, ti = mean_rho(fine.TRANSFER_BEARING), mean_rho(fine.TRANSFER_INERT)
    print(f"\nPRE-REGISTERED CONTRAST")
    print(f"  transfer-bearing {fine.TRANSFER_BEARING}: mean rho = {tb:+.3f}")
    print(f"  inert            {fine.TRANSFER_INERT}: mean rho = {ti:+.3f}")
    print(f"  difference = {tb - ti:+.3f}  (prediction: clearly positive)")
    print("\n  A saturated code is excluded from FDR: with no between-run variance it")
    print("  cannot correlate with anything, which is a property of the assistant's")
    print("  fixed system prompt rather than a finding about transfer.")
    print("  rho is unadjusted for model on purpose; rho_min is the single-engine check.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
