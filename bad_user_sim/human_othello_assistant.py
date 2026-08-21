"""Fine tutor-move coding of the HUMAN Othello assistant turns, and the sim comparison.

The question this exists to answer: does the assistant produce portable, generalisable
content for human participants, when it produced almost none that predicted transfer
for simulated ones? On the simulated side `General Principle` was present on 71.9% of
turns but showed rho = -0.017 with the solo margin, and no student act elicited it
(every lift ~1.0; Knowledge Deficit Question actually suppressed it at 0.71).

Same Othello-worded fine panel as the simulated runs, so prevalence and reliability
sit on one scale. Outcome is the same construct too: `final_margin` summed over the
two solo puzzles, read from oth_score_p*.json.

    python human_othello_assistant.py                # annotate, then report
    python human_othello_assistant.py --report-only
    python human_othello_assistant.py --compare      # human vs simulated prevalence
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

HUMAN = Path("/data/home/siyanli/agentic_sim/othello_conversations/recordings-download")
SIM = BASE / "test_agentic_run"
AI, SOLO = "poc20260727", ["pb220260706", "pbg20260726"]
CONVO = f"conversation_{AI}.jsonl"
ANN = f"annotated_conversation_{AI}.jsonl"
FIELD = "annotation_assistant_fine"
MIN_SHARE_SD, RARE_BELOW = 1.0, 2.0


def human_margin(d):
    total = 0
    for pz in SOLO:
        f = d / f"oth_score_{pz}.json"
        if not f.exists():
            return None
        v = json.load(open(f)).get("final_margin")
        if v is None:
            return None
        total += v
    return float(total)


def annotate(limit):
    from assistant_acts_othello import OthelloFineAssistantSuite
    suite = OthelloFineAssistantSuite()
    done = 0
    for d in sorted(HUMAN.iterdir()):
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
        tmp.replace(d / ANN)          # atomic; preserves annotation_user already there
    print(f"annotated {done} human turn(s)")


def shares(root, ann_name, outcome_fn):
    """{id: (share dict, outcome, n_turns)} plus the raw per_model records."""
    out, recs = {}, []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not (d / ann_name).exists():
            continue
        acts, n = Counter(), 0
        for line in open(d / ann_name):
            if not line.strip():
                continue
            t = json.loads(line)
            a = t.get(FIELD)
            if not a or a.get("n_valid", 0) < 1:
                continue
            n += 1
            recs.append(a["per_model"])
            for act in dict.fromkeys(a["final"]):
                acts[act] += 1
        y = outcome_fn(d)
        if n:
            out[d.name] = ({k: 100 * v / n for k, v in acts.items()}, y, n)
    return out, recs


def sim_margin(d):
    total = 0
    for pz in ["b220260706", "bg20260726"]:
        f = d / f"summary_p{pz}.json"
        if not f.exists():
            return None
        v = json.load(open(f)).get("final_margin")
        if v is None:
            return None
        total += v
    return float(total)


def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj, prev = np.empty(len(p)), 1.0
    for rank, idx in enumerate(order[::-1]):
        prev = min(prev, p[idx] * len(p) / (len(p) - rank))
        adj[idx] = prev
    return adj


def report(compare=False):
    data, recs = shares(HUMAN, ANN, human_margin)
    if len(data) < 10:
        print(f"only {len(data)} participants coded — let the annotation finish")
        return 1
    ids = sorted(data)
    graded = [i for i in ids if data[i][1] is not None]
    y = np.array([data[i][1] for i in graded])
    turns = int(sum(data[i][2] for i in ids))

    macro, by_act = fine.fleiss_kappa(recs)
    print(f"\nHUMAN OTHELLO | {len(ids)} participants | {turns} coded assistant turns")
    print(f"  {len(graded)} have both solo scores (outcome mean {y.mean():+.1f}, "
          f"sd {y.std(ddof=1):.1f})")
    print(f"  reliability: Jaccard {mean_pairwise_jaccard(recs):.3f}  "
          f"Fleiss kappa {macro:.3f}")

    rows = []
    for act in fine.ROLLUP:
        xs = np.array([data[i][0].get(act, 0.0) for i in graded])
        prev = float(np.mean([data[i][0].get(act, 0.0) > 0 for i in ids]) * 100)
        if xs.std() == 0:
            rows.append([act, prev, 0.0, by_act.get(act), np.nan, np.nan, "no variance"])
            continue
        note = ("no spread" if xs.std(ddof=1) < MIN_SHARE_SD
                else "rare" if prev < RARE_BELOW else "")
        r, p = stats.spearmanr(xs, y)
        rows.append([act, prev, float(xs.std(ddof=1)), by_act.get(act), r, p, note])

    testable = [r for r in rows if np.isfinite(r[4]) and not r[6]]
    q = bh([r[5] for r in testable]) if testable else []
    qmap = {r[0]: v for r, v in zip(testable, q)}

    print(f"\n{'act':>32}{'prev%':>7}{'sd':>6}{'kappa':>7}{'rho':>8}{'p':>8}{'q':>7}  tag")
    for act, prev, sd, k, r, p, note in rows:
        tag = note
        if act in fine.TRANSFER_BEARING:
            tag = (tag + " " if tag else "") + "<PREDICTED +>"
        f = lambda v: f"{v:+.3f}" if np.isfinite(v) else "    n/a"
        ks = f"{k:.2f}" if k is not None else "  -"
        qs = f"{qmap[act]:.3f}" if act in qmap else "    -"
        print(f"{act:>32}{prev:>7.1f}{sd:>6.1f}{ks:>7}{f(r):>8}"
              f"{p if np.isfinite(p) else float('nan'):>8.3f}{qs:>7}  {tag}")

    if compare:
        sim, _ = shares(SIM, f"annotated_conversation_p{AI[1:]}.jsonl"
                        if AI.startswith("p") else ANN, sim_margin)
        if sim:
            print(f"\n{'act':>32}{'human%':>9}{'sim%':>8}{'diff':>8}")
            for act in fine.ROLLUP:
                h = np.mean([data[i][0].get(act, 0.0) for i in ids])
                s = np.mean([v[0].get(act, 0.0) for v in sim.values()])
                print(f"{act:>32}{h:>9.1f}{s:>8.1f}{h - s:>+8.1f}")
            print("  share = mean % of a conversation's assistant turns carrying the act")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1200)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    if not HUMAN.exists():
        print(f"missing {HUMAN}")
        return 1
    if not a.report_only:
        annotate(a.limit)
    return report(compare=a.compare)


if __name__ == "__main__":
    sys.exit(main())
