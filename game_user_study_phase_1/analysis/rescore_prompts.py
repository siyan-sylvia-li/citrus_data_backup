#!/usr/bin/env python3
"""
rescore_prompts.py  (phase_1)

Re-score stored phase_1 prompt submissions with the CURRENT JudgeSuite panel
(Llama-3.3 + gpt-5.4-mini + Nemotron-3-Ultra). Phase_1's live gate used the
MEDIAN of valid judge scores >= threshold (fail-open when no valid scores), so
this rescorer applies the same rule for an apples-to-apples decision comparison.

For each participant we re-run the panel on the saved `created_prompt`, write
`prompt_task_rescored.json` NEXT TO the original (never overwriting the live
prompt_task.json), and print an old-vs-new comparison. Key-agnostic: it records
whatever model keys the panel returns.

Usage:
    python analysis/rescore_prompts.py [recordings_dir] [--threshold 3.0]

Defaults: recordings_dir = "recordings-download", threshold = 3.0.
Requires TOGETHER_API_KEY and OPENAI_API_KEY (loaded from .env).
"""
import argparse
import csv
import datetime
import json
import math
import os
import statistics
import sys

import dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
dotenv.load_dotenv(os.path.join(ROOT, ".env"))

from prompt_filter import JudgeSuite  # noqa: E402  (phase_1's panel)

HERE = os.path.dirname(os.path.abspath(__file__))


def decide(scores, threshold, fail_open=True):
    """Phase_1 live rule: median of valid scores >= threshold; fail-open if none."""
    valid = [s for s in scores.values() if s is not None]
    if not valid:
        return fail_open, None
    med = statistics.median(valid)
    return bool(med >= threshold), med


def old_scores(d):
    scores = {}
    for j in d.get("judges") or []:
        if isinstance(j, dict):
            scores[j.get("model")] = j.get("score")
    return scores, d.get("mean_score"), d.get("median_score"), d.get("passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recordings_dir", nargs="?", default="recordings-download")
    ap.add_argument("--threshold", type=float, default=3.0)
    args = ap.parse_args()

    base = args.recordings_dir if os.path.isabs(args.recordings_dir) else os.path.join(ROOT, args.recordings_dir)
    paths = sorted(
        os.path.join(base, d, "prompt_task.json")
        for d in os.listdir(base)
        if os.path.isfile(os.path.join(base, d, "prompt_task.json"))
    )

    suite = JudgeSuite()
    rows = []
    print(f"Re-scoring {len(paths)} phase_1 prompts (median >= {args.threshold})...\n", flush=True)

    for p in paths:
        pid = os.path.basename(os.path.dirname(p))
        try:
            orig = json.load(open(p))
        except Exception:
            continue
        prompt = orig.get("created_prompt", "") or ""
        o_scores, o_mean, o_med, o_pass = old_scores(orig)

        t0 = datetime.datetime.now()
        mean, scores = suite(prompt=prompt)
        new_pass, new_med = decide(scores, args.threshold)
        dt = (datetime.datetime.now() - t0).total_seconds()

        out = {
            "created_prompt": prompt,
            "judges": [{"model": n, "score": s} for n, s in scores.items()],
            "n_judges": sum(1 for s in scores.values() if s is not None),
            "mean_score": round(mean, 3) if mean is not None else None,
            "median_score": new_med,
            "threshold": args.threshold,
            "passed": bool(new_pass),
            "original": {"judges": o_scores, "mean_score": o_mean,
                         "median_score": o_med, "passed": o_pass},
            "time_taken": round(dt, 3),
        }
        with open(os.path.join(os.path.dirname(p), "prompt_task_rescored.json"), "w") as f:
            json.dump(out, f, indent=2)

        changed = (o_pass is not None) and (bool(o_pass) != bool(new_pass))
        rows.append({
            "participant": pid, "old_med": o_med, "old_pass": o_pass, "old_scores": o_scores,
            "new_med": new_med, "new_pass": new_pass, "new_scores": scores,
            "decision_changed": changed, "time_taken": dt,
        })
        flag = "  <-- DECISION CHANGED" if changed else ""
        print(f"{pid[:24]:<25} old med={o_med} ({'pass' if o_pass else 'fail'}) {o_scores}"
              f"  ->  new med={new_med} ({'pass' if new_pass else 'fail'}) {scores}{flag}", flush=True)

    # union of model keys across old+new for stable CSV columns
    keys = []
    for r in rows:
        for k in list(r["old_scores"]) + list(r["new_scores"]):
            if k and k not in keys:
                keys.append(k)
    csv_path = os.path.join(HERE, "rescore_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["participant", "old_med", "old_pass"] + [f"old_{k}" for k in keys]
                   + ["new_med", "new_pass"] + [f"new_{k}" for k in keys]
                   + ["decision_changed", "time_taken"])
        for r in rows:
            os_, ns = r["old_scores"], r["new_scores"]
            w.writerow([r["participant"], r["old_med"], r["old_pass"]] + [os_.get(k) for k in keys]
                       + [r["new_med"], r["new_pass"]] + [ns.get(k) for k in keys]
                       + [r["decision_changed"], round(r["time_taken"], 3)])

    n = len(rows)
    chg = sum(1 for r in rows if r["decision_changed"])
    o_passers = sum(1 for r in rows if r["old_pass"])
    n_passers = sum(1 for r in rows if r["new_pass"])
    flips_pf = sum(1 for r in rows if r["old_pass"] and not r["new_pass"])
    flips_fp = sum(1 for r in rows if (not r["old_pass"]) and r["new_pass"])
    print(f"\n{n} prompts re-scored.  Pass count: {o_passers} (live) -> {n_passers} (new panel).", flush=True)
    print(f"Decisions changed: {chg}  (pass->fail: {flips_pf}, fail->pass: {flips_fp}).", flush=True)
    print(f"Comparison CSV: {csv_path}", flush=True)
    times = [r["time_taken"] for r in rows]
    if times:
        m = statistics.mean(times); sd = statistics.stdev(times) if len(times) > 1 else 0.0
        half = 1.96 * sd / math.sqrt(len(times))
        print(f"Time per prompt: mean {m:.2f}s  95% CI [{m-half:.2f}, {m+half:.2f}]  "
              f"(total {sum(times):.1f}s)", flush=True)


if __name__ == "__main__":
    main()
