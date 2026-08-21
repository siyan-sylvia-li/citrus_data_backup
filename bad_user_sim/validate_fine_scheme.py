"""Score the refined assistant scheme on turns already coded under the coarse one.

Three checks, run before the scheme is trusted for analysis:

  1. PREVALENCE SPREAD  no code above ~90% or below ~2%. This is the defect the
                        coarse scheme has (Direct Instruction 99.1%, Comprehension
                        Gauging 0%), and the reason it cannot explain outcome
                        differences.
  2. RELIABILITY        per-code Fleiss kappa >= 0.5. Splitting a category usually
                        costs agreement, so this is the real limit on how fine the
                        scheme can go -- a distinction that cannot be coded
                        reliably is unusable however good the theory behind it is.
  3. ROLL-UP FIDELITY   collapsing fine labels via ROLLUP should reproduce the
                        coarse labels already stored on the same turns. Anything
                        else means the schemes are not comparable and the existing
                        human coding is stranded.

    python validate_fine_scheme.py --limit 150     # annotate a sample, then report
    python validate_fine_scheme.py --report-only   # score what is already stored
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
import dotenv  # noqa: E402
dotenv.load_dotenv(BASE / ".env")

import dialogue_act_annotation_assistant_fine as fine  # noqa: E402
from dialogue_act_annotation import mean_pairwise_jaccard  # noqa: E402
from dialogue_act_annotation_assistant import ASSISTANT_SCHEME  # noqa: E402

# Coded on the shim, which is where the coarse assistant annotation already lives.
ROOT = BASE / "sim_assistant_acts_cf" / "recordings-download"
ANN = "annotated_conversation_p15.jsonl"
FIELD = "annotation_assistant_fine"
COARSE_FIELD = "annotation_assistant"


def turn_files():
    return sorted(d / ANN for d in ROOT.iterdir() if d.is_dir() and (d / ANN).exists())


def annotate(limit):
    suite = fine.FineAssistantActSuite()
    done = 0
    for f in turn_files():
        rows = [json.loads(l) for l in open(f) if l.strip()]
        changed = False
        for t in rows:
            if done >= limit:
                break
            if FIELD in t or not (t.get("assistant") or "").strip():
                continue
            t[FIELD] = suite(utterance=t["assistant"].strip())
            done += 1
            changed = True
            if done % 20 == 0:
                print(f"  {done}/{limit} turns", flush=True)
        if changed:
            tmp = f.with_suffix(".jsonl.tmp")
            with open(tmp, "w") as w:
                for t in rows:
                    w.write(json.dumps(t) + "\n")
            tmp.replace(f)
        if done >= limit:
            break
    print(f"annotated {done} turn(s)")


def report():
    fine_recs, n_turns = [], 0
    fine_counts, coarse_counts = Counter(), Counter()
    rollup_ok = rollup_tot = 0
    mismatches = []
    for f in turn_files():
        for line in open(f):
            if not line.strip():
                continue
            t = json.loads(line)
            a = t.get(FIELD)
            if not a or a.get("n_valid", 0) < 1:
                continue
            n_turns += 1
            fine_recs.append(a["per_model"])
            for act in a["final"]:
                fine_counts[act] += 1
            c = t.get(COARSE_FIELD)
            if c and c.get("n_valid", 0) >= 1:
                got = set(fine.to_coarse(a["final"]))
                want = set(c["final"])
                # Pose Simplified Problem was deliberately merged into Prompt, so it
                # cannot be reproduced; exclude it from the fidelity test rather than
                # counting a known, documented merge as a failure.
                want.discard("Pose Simplified Problem")
                rollup_tot += 1
                rollup_ok += (got == want)
                if got != want and len(mismatches) < 6:
                    mismatches.append((sorted(want - got), sorted(got - want)))
                for act in want:
                    coarse_counts[act] += 1

    if not n_turns:
        print("no fine annotations found — run without --report-only first")
        return 1

    print(f"\n{'=' * 78}\n1. PREVALENCE  ({n_turns} turns)\n{'=' * 78}")
    print(f"{'fine code':>32}{'turns':>8}{'prevalence':>12}   flag")
    for act in fine.ROLLUP:
        n = fine_counts[act]
        pct = 100 * n / n_turns
        flag = ("SATURATED" if pct > 90 else "too rare" if pct < 2 else "")
        star = " *" if act in fine.TRANSFER_BEARING else ""
        print(f"{act:>32}{n:>8}{pct:>11.1f}%   {flag}{star}")
    print("  * pre-registered as transfer-bearing")

    macro, by_act = fine.fleiss_kappa(fine_recs)
    print(f"\n{'=' * 78}\n2. RELIABILITY\n{'=' * 78}")
    print(f"mean pairwise Jaccard: {mean_pairwise_jaccard(fine_recs):.3f}")
    print(f"Fleiss kappa (macro) : {macro:.3f}")
    for act in fine.ROLLUP:
        k = by_act.get(act)
        mark = "" if k is None or k >= 0.5 else "   BELOW 0.5"
        print(f"{act:>32}{'      n/a' if k is None else f'{k:>9.3f}'}{mark}")

    print(f"\n{'=' * 78}\n3. ROLL-UP FIDELITY\n{'=' * 78}")
    if rollup_tot:
        print(f"fine labels collapse to the stored coarse labels on "
              f"{rollup_ok}/{rollup_tot} turns ({100 * rollup_ok / rollup_tot:.0f}%)")
        for miss, extra in mismatches:
            print(f"  coarse-only: {miss or '-'}    fine-only: {extra or '-'}")
        print("  Disagreement is not automatically a defect: the two schemes were coded")
        print("  in separate panel runs, so some of it is ordinary annotator variance.")
    else:
        print("no turn carries both codings — cannot test")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if not ROOT.exists():
        print(f"missing {ROOT}")
        return 1
    if not args.report_only:
        annotate(args.limit)
    return report()


if __name__ == "__main__":
    sys.exit(main())
