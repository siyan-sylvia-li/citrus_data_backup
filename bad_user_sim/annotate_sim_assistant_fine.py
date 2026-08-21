"""Fine assistant-act annotation over simulated arms, for any arm and all puzzles.

Generalises asst_panel_kdq.py, which was pinned to two arms and one puzzle. Same
12-code Othello-worded panel and the same three seats as the human pass, so
prevalences from sim and human sit on one scale.

Writes `annotation_assistant_fine` per turn into
`annotated_conversation_<puzzle>.jsonl`, preserving `annotation_user`. Turns that
already carry the field are skipped, so re-running tops up rather than re-spending.

    python annotate_sim_assistant_fine.py --dry-run          # count, spend nothing
    python annotate_sim_assistant_fine.py                    # all human_style arms
    python annotate_sim_assistant_fine.py --arms A B         # named arms
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import sys
import threading
from pathlib import Path

import dotenv

BASE = Path(__file__).resolve().parent
dotenv.load_dotenv(BASE / ".env")

FIELD = "annotation_assistant_fine"
DEFAULT_GLOB = "test_agentic_run__human_style*"
_lock = threading.Lock()
_abort = threading.Event()


def usable(rec):
    """All seats failed -> a file of empty labels, not data. Caught on turn 1."""
    per = (rec or {}).get("per_model") or {}
    return any(v is not None for v in per.values())


def jobs(arms):
    """(annotated_path, source_path) for every conversation with unlabelled turns."""
    out = []
    for arm in arms:
        for raw in sorted((BASE / arm).glob("*/conversation_*.jsonl")):
            puzzle = raw.name[len("conversation_"):-len(".jsonl")]
            ann = raw.parent / f"annotated_conversation_{puzzle}.jsonl"
            src = ann if ann.exists() and ann.stat().st_size else raw
            todo = 0
            for line in open(src):
                if not line.strip():
                    continue
                t = json.loads(line)
                if FIELD not in t and (t.get("assistant") or "").strip():
                    todo += 1
            if todo:
                out.append((ann, src, todo))
    return out


def do_file(suite, ann, src, seats):
    """Annotate one conversation; rewrite atomically so annotation_user survives."""
    rows = [json.loads(l) for l in open(src) if l.strip()]
    n = 0
    for t in rows:
        if _abort.is_set():
            return 0
        if FIELD in t or not (t.get("assistant") or "").strip():
            continue
        rec = suite(utterance=t["assistant"].strip())
        # usable() only requires ONE seat, so a mid-run seat outage would quietly
        # mix 1-seat labels into a 3-seat panel. Track it and report at the end;
        # per_model keeps the evidence so the analysis can filter on 3/3.
        with _lock:
            per = (rec or {}).get("per_model") or {}
            seats[sum(v is not None for v in per.values())] += 1
            for name, v in per.items():
                if v is None:
                    seats[f"fail:{name}"] += 1
        if not usable(rec):
            with _lock:
                print("every panel seat failed -- check .env; refusing to write empty "
                      f"labels ({ann.parent.name}/{ann.name})", file=sys.stderr)
            _abort.set()
            return 0
        t[FIELD] = rec
        n += 1
    if n:
        tmp = ann.with_suffix(".jsonl.tmp")
        with open(tmp, "w") as w:
            for t in rows:
                w.write(json.dumps(t) + "\n")
        tmp.replace(ann)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=None,
                    help=f"arm directory names; default every {DEFAULT_GLOB}")
    ap.add_argument("--dry-run", action="store_true", help="count turns, spend nothing")
    ap.add_argument("--workers", type=int, default=8,
                    help="conversations in flight; each turn fans out to 3 seats")
    # Othello turns name squares (a1-h8), so the Othello-worded subclass is right for
    # Othello and for the simulator. Connect Four must use the game-neutral base, the
    # same one human_cf_assistant_fine.py runs, or the two games' labels differ in
    # provenance as well as in game.
    ap.add_argument("--suite", choices=("othello", "generic"), default="othello",
                    help="panel wording: othello (default) or generic (Connect Four)")
    args = ap.parse_args()

    arms = args.arms or sorted(d.name for d in BASE.glob(DEFAULT_GLOB) if d.is_dir())
    work = jobs(arms)
    total = sum(t for _, _, t in work)
    print(f"{len(work)} conversation(s) across {len(arms)} arm(s), "
          f"{total} assistant turn(s) to label -> ~{total * 3} panel calls")
    if args.dry_run or not total:
        return 0

    if args.suite == "othello":
        from assistant_acts_othello import OthelloFineAssistantSuite
        suite = OthelloFineAssistantSuite()
    else:
        from dialogue_act_annotation_assistant_fine import FineAssistantActSuite
        suite = FineAssistantActSuite()

    done = 0
    seats = collections.Counter()
    with concurrent.futures.ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(do_file, suite, ann, src, seats): ann
                for ann, src, _ in work}
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            done += fut.result()
            print(f"  [{i}/{len(work)}] {futs[fut].parent.name[:44]} "
                  f"-> {done}/{total} turns", flush=True)
    if _abort.is_set():
        print("aborted -- panel failure", file=sys.stderr)
        return 1

    print(f"\nannotated {done} assistant turn(s)")
    print("seat coverage:", {k: v for k, v in sorted(seats.items(), key=str)
                             if not str(k).startswith("fail:")})
    fails = {k: v for k, v in seats.items() if str(k).startswith("fail:")}
    if fails:
        print("seat failures:", fails)
    if seats[3] < done:
        print(f"WARNING: {done - seats[3]} turn(s) labelled by fewer than 3 seats; "
              "filter on per_model before pooling with the human pass", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
