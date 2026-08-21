"""Fine tutor-move coding of the PHASE-2 assistant turns.

Same Othello-worded fine panel as `human_othello_assistant.py` runs on the phase-1
Othello study, so prevalence and reliability sit on one scale and the two studies can be
compared directly. Writes `annotation_assistant_fine` into the existing
annotated_conversation file, leaving `annotation_user` untouched.

    python phase2_assistant_fine.py --limit 3        # probe a few participants
    python phase2_assistant_fine.py                  # annotate everything missing
    python phase2_assistant_fine.py --report-only
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import threading
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
import dotenv  # noqa: E402
dotenv.load_dotenv(BASE / ".env")

# Default kept for backwards compatibility; --root points it at any Othello recordings
# directory. Every study uses the same assisted puzzle id, so one annotator covers all.
REC = Path("/Users/siyanli/Documents/CITRUS/game_user_study_phase_2/recordings-download")
AI = "poc20260727"
CONVO, ANN = f"conversation_{AI}.jsonl", f"annotated_conversation_{AI}.jsonl"
FIELD = "annotation_assistant_fine"
_lock = threading.Lock()


def src_for(d: Path):
    a, r = d / ANN, d / CONVO
    if a.exists() and a.stat().st_size:
        return a, a
    return (r, a) if r.exists() and r.stat().st_size else (None, None)


def collect(limit):
    """Global work list: (path_out, rows, index) so the model pool stays full."""
    convos, todo = {}, []
    for d in sorted(REC.iterdir()):
        if not d.is_dir() or len(convos) >= limit:
            continue
        src, out = src_for(d)
        if src is None:
            continue
        rows = [json.loads(l) for l in open(src) if l.strip()]
        need = [i for i, t in enumerate(rows)
                if FIELD not in t and (t.get("assistant") or "").strip()]
        if not need:
            continue
        convos[d.name] = (out, rows)
        todo += [(d.name, i) for i in need]
    return convos, todo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="recordings directory (default: phase 2's)")
    ap.add_argument("--limit", type=int, default=10_000, help="max participants")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()

    global REC
    if a.root:
        REC = Path(a.root).resolve()
        if not REC.is_dir():
            raise SystemExit(f"no such directory: {REC}")
    print(f"root: {REC}")
    convos, todo = collect(a.limit)
    print(f"{len(convos)} participants, {len(todo)} assistant turns to annotate")
    if a.report_only or not todo:
        return

    from assistant_acts_othello import OthelloFineAssistantSuite
    suite = OthelloFineAssistantSuite()
    done = [0]

    def work(job):
        pid, i = job
        _, rows = convos[pid]
        try:
            ann = suite(utterance=rows[i]["assistant"].strip())
        except Exception as e:                                  # keep the run going
            return pid, i, None, repr(e)[:120]
        return pid, i, ann, None

    errs = Counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for pid, i, ann, err in ex.map(work, todo):
            with _lock:
                done[0] += 1
                if err:
                    errs[err] += 1
                else:
                    convos[pid][1][i][FIELD] = ann
                if done[0] % 40 == 0 or done[0] == len(todo):
                    print(f"  {done[0]}/{len(todo)} turns  ({len(errs)} distinct errors)",
                          flush=True)

    for pid, (out, rows) in convos.items():                     # atomic per participant
        tmp = out.with_suffix(".jsonl.tmp")
        with open(tmp, "w") as f:
            for t in rows:
                f.write(json.dumps(t) + "\n")
        tmp.replace(out)
    print(f"wrote {len(convos)} files")
    if errs:
        print("errors:")
        for e, n in errs.most_common(5):
            print(f"  {n:>4}  {e}")


if __name__ == "__main__":
    main()
