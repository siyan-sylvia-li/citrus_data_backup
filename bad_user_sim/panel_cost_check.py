"""Can the Anthropic panel seat be a cheaper model without changing the labels?

The 3-model panel (llama / gpt / sonnet) is one seat per provider so the votes stay
independent, which is what the per-act majority rule assumes. Nothing about that
argument requires the Anthropic seat to be Sonnet — this is short multi-label
classification against a fixed taxonomy, not a reasoning task. Sonnet has taken
~8,500 calls across this project.

Re-labels a sample of turns ALREADY annotated by the sonnet panel, swapping only that
seat for haiku, and reports:
  * per-seat agreement with the sonnet seat it replaces (does haiku label like sonnet?)
  * whether the panel's FINAL consensus changes (the thing that actually matters --
    a seat can disagree and still be outvoted into the same answer)

Reads existing per_model records rather than re-running the whole panel, so only the
haiku calls are new.

    python panel_cost_check.py --n 50
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
import dotenv  # noqa: E402
dotenv.load_dotenv(BASE / ".env")

import dspy  # noqa: E402
from dialogue_act_annotation import (SCHEME, TAXONOMY,  # noqa: E402
                                     DialogueActClassifierSignature)

SRC = BASE / "test_agentic_run"
ANN = "annotated_conversation_poc20260727.jsonl"
FIELD = "annotation_user"
HAIKU = "anthropic/claude-haiku-4-5-20251001"


def consensus(per_model):
    """The panel's per-act majority vote, in canonical order — same rule as the suite."""
    valid = [v for v in per_model.values() if v is not None]
    if not valid:
        return []
    votes = Counter(a for acts in valid for a in acts)
    need = len(valid) // 2 + 1
    return [k for k in SCHEME if votes[k] >= need]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    pool = []
    for f in sorted(SRC.glob(f"*/{ANN}")):
        for line in open(f):
            if not line.strip():
                continue
            t = json.loads(line)
            a, utt = t.get(FIELD), (t.get("user") or "").strip()
            if a and utt and a.get("per_model", {}).get("sonnet") is not None:
                pool.append((utt, a["per_model"]))
    if len(pool) < args.n:
        print(f"only {len(pool)} usable turns")
        return 1
    random.seed(0)
    sample = random.sample(pool, args.n)
    print(f"{len(pool)} annotated turns available; testing {args.n}\n")

    lm = dspy.LM(HAIKU, max_tokens=2048, timeout=120)
    clf = dspy.ChainOfThought(DialogueActClassifierSignature)
    clf.set_lm(lm)

    seat_match = jac = 0.0
    same_final = changed = failed = 0
    examples = []
    for i, (utt, per) in enumerate(sample, 1):
        try:
            acts = clf(utterance=utt, taxonomy=TAXONOMY).dialogue_acts
            hk = [k for k in SCHEME if k in set(acts)]
        except Exception as e:
            failed += 1
            print(f"  [{i}] haiku failed: {type(e).__name__}")
            continue
        sn = set(per["sonnet"])
        h = set(hk)
        seat_match += (h == sn)
        jac += 1.0 if not h and not sn else len(h & sn) / len(h | sn)
        before = consensus(per)
        after = consensus({**per, "sonnet": hk})
        if before == after:
            same_final += 1
        else:
            changed += 1
            if len(examples) < 5:
                examples.append((utt[:70], before, after))
        if i % 10 == 0:
            print(f"  {i}/{args.n}", flush=True)

    n = args.n - failed
    if not n:
        print("every haiku call failed")
        return 1
    print(f"\n{'=' * 74}\nHAIKU vs SONNET as the Anthropic panel seat  (n={n})\n{'=' * 74}")
    print(f"seat-level exact label-set match : {100 * seat_match / n:.0f}%")
    print(f"seat-level mean Jaccard          : {jac / n:.3f}")
    print(f"panel CONSENSUS unchanged        : {100 * same_final / n:.0f}%  "
          f"({same_final}/{n})")
    if examples:
        print("\nturns where the consensus moved:")
        for u, b, a in examples:
            print(f"  {u!r}\n     sonnet-panel {b}\n     haiku-panel  {a}")
    print("\n  Consensus stability is the decision criterion, not seat agreement: a seat")
    print("  can disagree and still be outvoted into the same final label, and the final")
    print("  label is what every downstream analysis uses.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
