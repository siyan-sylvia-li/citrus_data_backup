#!/usr/bin/env python3
"""
rescore_prompts.py

Re-score the stored prompt-writing submissions with the CURRENT JudgeSuite rubric
(scenario-anchored). For each participant we re-run the multi-model panel on the
saved `created_prompt`, write `prompt_task_rescored.json` NEXT TO the original
(never overwriting it), and print an old-vs-new comparison.

Usage:
    python analysis/rescore_prompts.py [recordings_dir] [--threshold 3.0]

Defaults: recordings_dir = "recordings-download", threshold = 3.0 (pass if mean >= threshold).
Requires TOGETHER_API_KEY and OPENAI_API_KEY (loaded from .env). Makes ~3 LLM
calls per participant.
"""
import argparse
import csv
import json
import os
import sys

import dotenv
import datetime
import math
import statistics


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
dotenv.load_dotenv(os.path.join(ROOT, ".env"))

from prompt_filter import JudgeSuite  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def old_scores(d):
    """Pull {model: score} and mean from an existing prompt_task.json."""
    scores = {}
    for j in d.get("judges") or []:
        if isinstance(j, dict):
            scores[j.get("model")] = j.get("score")
    return scores, d.get("mean_score"), d.get("passed")


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
    print(f"Re-scoring {len(paths)} prompts (threshold mean >= {args.threshold})...\n")

    for p in paths:
        pid = os.path.basename(os.path.dirname(p))
        try:
            orig = json.load(open(p))
        except Exception:
            continue
        prompt = orig.get("created_prompt", "") or ""
        o_scores, o_mean, o_pass = old_scores(orig)

        begin_time = datetime.datetime.now()
        mean, scores = suite(prompt=prompt)
        new_pass = mean is not None and mean >= args.threshold
        new_time = datetime.datetime.now()

        out = {
            "created_prompt": prompt,
            "judges": [{"model": n, "score": s} for n, s in scores.items()],
            "n_judges": sum(1 for s in scores.values() if s is not None),
            "mean_score": round(mean, 3) if mean is not None else None,
            "threshold": args.threshold,
            "passed": bool(new_pass),
            "original": {"judges": o_scores, "mean_score": o_mean, "passed": o_pass},
            "time_taken": round((new_time - begin_time).total_seconds(), 3),
        }
        with open(os.path.join(os.path.dirname(p), "prompt_task_rescored.json"), "w") as f:
            json.dump(out, f, indent=2)

        changed = (o_pass is not None) and (bool(o_pass) != bool(new_pass))
        rows.append({
            "participant": pid,
            "old_mean": o_mean, "old_pass": o_pass, "old_scores": o_scores,
            "new_mean": out["mean_score"], "new_pass": new_pass, "new_scores": scores,
            "decision_changed": changed,
            "time_taken": out["time_taken"],
        })
        flag = "  <-- DECISION CHANGED" if changed else ""
        print(f"{pid[:24]:<25} old {o_mean} ({'pass' if o_pass else 'fail'}) {o_scores}"
              f"  ->  new {out['mean_score']} ({'pass' if new_pass else 'fail'}) {scores}{flag}")

    # CSV
    csv_path = os.path.join(HERE, "rescore_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["participant", "old_mean", "old_pass", "old_qwen", "old_llama", "old_gpt",
                    "new_mean", "new_pass", "new_qwen", "new_llama", "new_gpt", "decision_changed", "time_taken"])
        for r in rows:
            os_, ns = r["old_scores"], r["new_scores"]
            w.writerow([r["participant"], r["old_mean"], r["old_pass"],
                        os_.get("qwen"), os_.get("llama"), os_.get("gpt"),
                        r["new_mean"], r["new_pass"],
                        ns.get("qwen"), ns.get("llama"), ns.get("gpt"), r["decision_changed"], r["time_taken"]])

    n = len(rows)
    chg = sum(1 for r in rows if r["decision_changed"])
    o_passers = sum(1 for r in rows if r["old_pass"])
    n_passers = sum(1 for r in rows if r["new_pass"])
    print(f"\n{n} prompts re-scored.  Pass count: {o_passers} (old) -> {n_passers} (new).  "
          f"Decisions changed: {chg}.")
    print(f"Comparison CSV: {csv_path}")
    times = [r["time_taken"] for r in rows]
    if times:
        m = statistics.mean(times)
        sd = statistics.stdev(times) if len(times) > 1 else 0.0
        half = 1.96 * sd / math.sqrt(len(times))   # ~95% CI for the mean (normal approx)
        print(f"Time per prompt: mean {m:.2f}s  95% CI [{m - half:.2f}, {m + half:.2f}]  "
              f"(sd {sd:.2f}, n {len(times)}, total {sum(times):.1f}s)")


if __name__ == "__main__":
    main()
