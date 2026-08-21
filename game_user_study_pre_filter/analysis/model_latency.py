#!/usr/bin/env python3
"""Measure per-model judge latency (cold cache) to see which model is the
wall-clock bottleneck in the parallel panel.

Times each grader (qwen / llama / gpt) individually on a sample of stored
prompts, sequentially, and reports mean latency + 95% CI per model.

Usage: python analysis/model_latency.py [n_prompts]   (default 10)
Clear ~/.dspy_cache first for real timings.
"""
import json
import math
import os
import statistics
import sys
import time

import dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
dotenv.load_dotenv(os.path.join(ROOT, ".env"))
from prompt_filter import JudgeSuite, JUDGE_RUBRIC  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
base = os.path.join(ROOT, "recordings-download")

prompts = []
for d in sorted(os.listdir(base)):
    p = os.path.join(base, d, "prompt_task.json")
    if os.path.isfile(p):
        try:
            txt = (json.load(open(p)).get("created_prompt") or "").strip()
            if txt:
                prompts.append(txt)
        except Exception:
            pass
    if len(prompts) >= N:
        break

suite = JudgeSuite()
times = {name: [] for name in suite.graders}
print(f"Timing {len(suite.graders)} models on {len(prompts)} prompts (sequential, cold cache)...\n")
for prompt in prompts:
    for name, g in suite.graders.items():
        t0 = time.perf_counter()
        try:
            g(prompt_judged=prompt, rubric=JUDGE_RUBRIC)
        except Exception as e:
            print(f"  {name} error: {e}")
            continue
        times[name].append(time.perf_counter() - t0)

print(f"{'model':<8}{'mean':>8}{'95% CI':>20}{'min':>8}{'max':>8}")
for name, ts in times.items():
    if not ts:
        print(f"{name:<8}  (no successful calls)"); continue
    m = statistics.mean(ts)
    sd = statistics.stdev(ts) if len(ts) > 1 else 0.0
    half = 1.96 * sd / math.sqrt(len(ts))
    print(f"{name:<8}{m:>7.2f}s   [{m-half:>5.2f}, {m+half:>5.2f}]{min(ts):>7.2f}s{max(ts):>7.2f}s")

# panel wall-clock = slowest model per prompt (parallel)
per_prompt_max = [max(times[n][i] for n in times if i < len(times[n])) for i in range(len(prompts))]
print(f"\nPanel wall-clock (max of 3, parallel): mean {statistics.mean(per_prompt_max):.2f}s")
