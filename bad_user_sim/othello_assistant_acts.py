"""Annotate Othello assistant turns with the fine tutor-move scheme, then test the
pre-registered transfer prediction on the second game.

Only the FINE panel is run: coarse labels are derived through ROLLUP, which halves
the API cost and makes the two levels consistent by construction rather than by two
independent panel runs agreeing.

Outcome is the Othello solo margin sum (ceiling +16) — graded, unlike Connect Four's
0-3 count, so it has more resolution for detecting an assistant effect if one exists.

    python othello_assistant_acts.py              # annotate what is missing, then test
    python othello_assistant_acts.py --report-only
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

BASE = Path(__file__).resolve().parent
import dotenv  # noqa: E402
dotenv.load_dotenv(BASE / ".env")

import dialogue_act_annotation_assistant_fine as fine  # noqa: E402
from dialogue_act_annotation import mean_pairwise_jaccard  # noqa: E402

ROOT = BASE / "test_agentic_run"
AI, SOLO = "oc20260727", ["b220260706", "bg20260726"]
ANN = f"annotated_conversation_p{AI}.jsonl"
CONVO = f"conversation_p{AI}.jsonl"
FIELD = "annotation_assistant_fine"
# A code cannot correlate with anything only if its SHARE is near-constant. An
# earlier version tested PREVALENCE (does the act occur in every run) and excluded
# the two largest associations in the Othello table on that basis -- a code can be
# present in 100% of runs while its share ranges from 50% to 100%, which is ample
# variance. Prevalence is still reported, but only share SD gates the test.
MIN_SHARE_SD, RARE_BELOW = 1.0, 2.0


def margin(d):
    total = 0
    for pz in SOLO:
        f = d / f"summary_p{pz}.json"
        if not f.exists():
            return None
        v = json.load(open(f)).get("final_margin")
        if v is None:
            return None
        total += v
    return float(total)


def model_of(name):
    import re
    m = re.match(r"^preset_(\d+)(.+)$", name)
    return m.group(2) if m else name


def annotate(limit):
    from assistant_acts_othello import OthelloFineAssistantSuite
    suite = OthelloFineAssistantSuite()
    done = 0
    for d in sorted(ROOT.iterdir()):
        if not d.is_dir() or done >= limit:
            continue
        src = d / ANN if (d / ANN).exists() and (d / ANN).stat().st_size else d / CONVO
        if not src.exists() or not src.stat().st_size:
            continue
        rows = [json.loads(l) for l in open(src) if l.strip()]
        if all(FIELD in t for t in rows):
            continue
        for t in rows:
            if done >= limit:
                break
            if FIELD in t or not (t.get("assistant") or "").strip():
                continue
            t[FIELD] = suite(utterance=t["assistant"].strip())
            done += 1
            if done % 25 == 0:
                print(f"  {done} turns", flush=True)
        tmp = (d / ANN).with_suffix(".jsonl.tmp")
        with open(tmp, "w") as w:
            for t in rows:
                w.write(json.dumps(t) + "\n")
        tmp.replace(d / ANN)
    print(f"annotated {done} turn(s)")


def load():
    out, recs = {}, []
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
            recs.append(a["per_model"])
            for act in dict.fromkeys(a["final"]):
                acts[act] += 1
        y = margin(d)
        if n_turns and y is not None:
            out[d.name] = ({k: 100 * v / n_turns for k, v in acts.items()},
                           y, model_of(d.name), n_turns)
    return out, recs


def loo_min(x, y, groups):
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


def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj, prev = np.empty(len(p)), 1.0
    for rank, idx in enumerate(order[::-1]):
        prev = min(prev, p[idx] * len(p) / (len(p) - rank))
        adj[idx] = prev
    return adj


def report():
    data, recs = load()
    if len(data) < 10:
        print(f"only {len(data)} runs coded — let the annotation finish")
        return 1
    runs = sorted(data)
    y = np.array([data[r][1] for r in runs])
    models = [data[r][2] for r in runs]
    turns = int(sum(data[r][3] for r in runs))

    print(f"\nOTHELLO | {len(runs)} runs | {turns} coded assistant turns | "
          f"outcome = solo margin sum (mean {y.mean():+.1f}, sd {y.std(ddof=1):.1f})")

    macro, by_act = fine.fleiss_kappa(recs)
    print(f"reliability: Jaccard {mean_pairwise_jaccard(recs):.3f}  Fleiss kappa {macro:.3f}")

    rows = []
    for act in fine.ROLLUP:
        x = np.array([data[r][0].get(act, 0.0) for r in runs])
        prev = float((x > 0).mean() * 100)
        if x.std() == 0:
            rows.append([act, prev, np.nan, np.nan, np.nan, "no variance", by_act.get(act), 0.0])
            continue
        note = ("no spread" if x.std(ddof=1) < MIN_SHARE_SD
                else "rare" if prev < RARE_BELOW else "")
        r, p = stats.spearmanr(x, y)
        rl, _, _ = loo_min(x, y, models)
        rows.append([act, prev, r, p, rl, note, by_act.get(act), float(x.std(ddof=1))])

    testable = [r for r in rows if np.isfinite(r[2]) and not r[5]]
    q = bh([r[3] for r in testable]) if testable else []
    qmap = {r[0]: v for r, v in zip(testable, q)}

    print(f"\n{'act':>32}{'prev%':>7}{'sd':>6}{'kappa':>7}{'rho':>8}{'p':>8}{'q':>7}{'rho_min':>9}  tag")
    for act, prev, r, p, rl, note, k, sd in rows:
        tag = note
        if act in fine.TRANSFER_BEARING:
            tag = (tag + " " if tag else "") + "<PREDICTED +>"
        elif act in fine.TRANSFER_INERT:
            tag = (tag + " " if tag else "") + "<PREDICTED null>"
        f = lambda v: f"{v:+.3f}" if np.isfinite(v) else "    n/a"
        ks = f"{k:.2f}" if k is not None else "  -"
        qs = f"{qmap[act]:.3f}" if act in qmap else "    -"
        print(f"{act:>32}{prev:>7.1f}{sd:>6.1f}{ks:>7}{f(r):>8}"
              f"{p if np.isfinite(p) else float('nan'):>8.3f}{qs:>7}{f(rl):>9}  {tag}")

    def mean_rho(names):
        v = [r[2] for r in rows if r[0] in names and np.isfinite(r[2])]
        return float(np.mean(v)) if v else np.nan

    tb, ti = mean_rho(fine.TRANSFER_BEARING), mean_rho(fine.TRANSFER_INERT)
    print(f"\nPRE-REGISTERED CONTRAST (second game)")
    print(f"  transfer-bearing {fine.TRANSFER_BEARING}: mean rho = {tb:+.3f}")
    print(f"  inert            {fine.TRANSFER_INERT}: mean rho = {ti:+.3f}")
    print(f"  difference = {tb - ti:+.3f}  (prediction: clearly positive)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1200)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if not a.report_only:
        annotate(a.limit)
    return report()


if __name__ == "__main__":
    sys.exit(main())
