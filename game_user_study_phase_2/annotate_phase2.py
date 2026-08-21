"""Annotate phase-2 participant utterances with the AutoTutor dialogue-act scheme.

Writes annotated_conversation_<puzzle>.jsonl ALONGSIDE the raw conversation file, never
modifying it -- same convention as phase 1, so the analysis notebook reads the same shape
of data from both studies. Turns already carrying `annotation_user` are skipped, so this
is cheap to re-run as the sample fills.

Same three-seat panel as every other annotation layer in this project, so act rates from
phase 1, phase 2 and the simulator sit on one scale.

    python annotate_phase2.py --dry-run     # count what would be annotated
    python annotate_phase2.py               # annotate what is missing
"""
from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import os
import sys
import threading
from pathlib import Path

import dotenv

BASE = Path(__file__).resolve().parent
dotenv.load_dotenv(BASE / ".env")
REC = BASE / "recordings-download"
FIELD = "annotation_user"
_lock = threading.Lock()


def jobs(root: Path):
    out = []
    for raw in sorted(root.glob("*/conversation_*.jsonl")):
        puzzle = raw.name[len("conversation_"):-len(".jsonl")]
        ann = raw.parent / f"annotated_conversation_{puzzle}.jsonl"
        src = ann if ann.exists() and ann.stat().st_size else raw
        todo = sum(1 for l in open(src) if l.strip()
                   and FIELD not in json.loads(l)
                   and (json.loads(l).get("user") or "").strip())
        if todo:
            out.append((ann, src, todo))
    return out


def do_file(suite, ann, src):
    rows = [json.loads(l) for l in open(src) if l.strip()]
    n = 0
    for t in rows:
        if FIELD in t or not (t.get("user") or "").strip():
            continue
        t[FIELD] = suite(utterance=t["user"].strip())
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
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--root", default=str(REC))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    work = jobs(Path(a.root))
    total = sum(t for _, _, t in work)
    print(f"{len(work)} conversation(s), {total} utterance(s) to annotate "
          f"-> ~{total * 3} panel calls")
    if a.dry_run or not total:
        return 0

    from dialogue_act_annotation import DialogueActSuite
    suite = DialogueActSuite()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(do_file, suite, ann, src): ann for ann, src, _ in work}
        for i, f in enumerate(concurrent.futures.as_completed(futs), 1):
            done += f.result()
            with _lock:
                print(f"  [{i}/{len(work)}] {futs[f].parent.name[:12]} -> {done}/{total}",
                      flush=True)
    print(f"annotated {done} utterance(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
