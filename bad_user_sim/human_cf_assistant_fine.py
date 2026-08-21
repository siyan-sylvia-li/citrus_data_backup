"""Fine assistant coding of the HUMAN Connect Four turns.

CF's human conversations carry only the coarse assistant scheme, and its rollup is
lossy precisely where the Othello result lives:

    Move Verdict  -> Provide Correct Answer                       (1:1)
    Board Report  -> Direct Instruction, with Local Justification,
                     Worked Line and General Principle            (blended)

Direct Instruction sits near ceiling because General Principle does, so a Board
Report signal inside it is unrecoverable -- every participant act came out at lift
1.00 against it (human_elicitation.py --game connect_four --coarse). This adds the
12-code layer so CF can test the Board Report half of the chain the same way
Othello did.

Same panel as the Othello fine pass (llama / gpt-5.4-mini / sonnet), deliberately:
the point of running CF at all is cross-game comparability, and a cheaper seat here
would mean the two games' labels differ in provenance as well as in game.

Writes `annotation_assistant_fine` into the existing annotated files, preserving
`annotation_user` and `annotation_assistant` already there.

    python human_cf_assistant_fine.py --limit 50     # probe
    python human_cf_assistant_fine.py                # all remaining
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
import dotenv  # noqa: E402
dotenv.load_dotenv(BASE / ".env")

HUMAN = Path("/data/home/siyanli/agentic_sim/connect_four_conversations/recordings-download")
ANN = "annotated_conversation_p15.jsonl"
FIELD = "annotation_assistant_fine"


def usable(rec):
    """A record whose every seat failed is a FILE FULL OF EMPTY LABELS, not data.

    This bit me on an earlier arm: load_dotenv() found no .env, every annotator
    raised on the missing key, and the suite dutifully wrote complete annotation
    files of nothing. Checked here so a bad run is caught on turn 1, not after 399.
    """
    per = (rec or {}).get("per_model") or {}
    return any(v is not None for v in per.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10_000)
    args = ap.parse_args()

    from dialogue_act_annotation_assistant_fine import FineAssistantActSuite
    suite = FineAssistantActSuite()

    done = skipped = 0
    for d in sorted(HUMAN.iterdir()):
        if not d.is_dir() or done >= args.limit:
            continue
        src = d / ANN
        if not src.exists() or not src.stat().st_size:
            continue
        rows = [json.loads(l) for l in open(src) if l.strip()]
        todo = [t for t in rows
                if FIELD not in t and (t.get("assistant") or "").strip()]
        if not todo:
            skipped += 1
            continue
        for t in rows:
            if done >= args.limit:
                break
            if FIELD in t or not (t.get("assistant") or "").strip():
                continue
            rec = suite(utterance=t["assistant"].strip())
            if not usable(rec):
                print("every panel seat failed on turn 1 -- check .env / keys; "
                      "refusing to write empty labels", file=sys.stderr)
                return 1
            t[FIELD] = rec
            done += 1
            if done % 25 == 0:
                print(f"  {done} turns", flush=True)
        tmp = src.with_suffix(".jsonl.tmp")
        with open(tmp, "w") as w:
            for t in rows:
                w.write(json.dumps(t) + "\n")
        tmp.replace(src)              # atomic; keeps the two existing layers intact
    print(f"annotated {done} CF assistant turn(s); {skipped} conversation(s) already done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
