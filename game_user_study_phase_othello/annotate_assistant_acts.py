"""Annotate every Round-1 (assisted, oc20260727) ASSISTANT reply with the tutor-move panel.

Reads `annotated_conversation_poc20260727.jsonl` when it exists (so the participant-side
`annotation_user` written by the notebook is preserved) and otherwise falls back to
the raw `conversation_poc20260727.jsonl`. Adds an `annotation_assistant` field per turn and
rewrites the annotated file atomically.

Skips participants already carrying `annotation_assistant` unless OVERWRITE=1.

    python annotate_assistant_acts.py            # annotate what's missing
    OVERWRITE=1 python annotate_assistant_acts.py
"""

import concurrent.futures
import json
import os
import sys
from pathlib import Path

import dotenv

dotenv.load_dotenv()

from dialogue_act_annotation_assistant import AssistantDialogueActSuite  # noqa: E402

RECORDINGS_BASE = Path("recordings-download")
RAW = "conversation_poc20260727.jsonl"
ANN = "annotated_conversation_poc20260727.jsonl"
OVERWRITE = os.environ.get("OVERWRITE") == "1"
# Turns in flight across ALL participants; each turn fans out to 3 models, so this is
# ~3x that many concurrent API calls. Participants average ~3 turns, so pooling turns
# globally (rather than per participant) is what actually keeps the pool full.
TURN_WORKERS = int(os.environ.get("TURN_WORKERS", 16))


def load_turns(pid):
    """Prefer the annotated file (keeps annotation_user), fall back to the raw log."""
    ann_f, raw_f = RECORDINGS_BASE / pid / ANN, RECORDINGS_BASE / pid / RAW
    src = ann_f if ann_f.exists() and ann_f.stat().st_size > 0 else raw_f
    if not src.exists():
        return None
    return [json.loads(l) for l in open(src) if l.strip()]


def main():
    suite = AssistantDialogueActSuite()
    pids = sorted(p.parent.name for p in RECORDINGS_BASE.glob(f"*/{RAW}")
                  if "test" not in p.parent.name.lower())
    print(f"{len(pids)} participants with a Round-1 conversation")

    # Build one global work list first. Per-participant pools starved (median ~3 turns
    # per participant), so the run was serialized on the slowest model per participant.
    convos, todo = {}, []
    for pid in pids:
        turns = load_turns(pid)
        if not turns:
            continue
        if not OVERWRITE and all("annotation_assistant" in t for t in turns):
            continue
        convos[pid] = turns
        todo += [(pid, i) for i, t in enumerate(turns)
                 if (t.get("assistant") or "").strip()
                 and (OVERWRITE or "annotation_assistant" not in t)]
    print(f"{len(todo)} assistant turns to annotate across {len(convos)} participants "
          f"({len(pids) - len(convos)} already done)", flush=True)

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=TURN_WORKERS) as ex:
        futs = {ex.submit(suite, utterance=convos[pid][i]["assistant"].strip()): (pid, i)
                for pid, i in todo}
        for fut in concurrent.futures.as_completed(futs):
            pid, i = futs[fut]
            convos[pid][i]["annotation_assistant"] = fut.result()
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(todo)} turns", flush=True)

    for pid, turns in convos.items():
        for t in turns:                                   # empty replies get an explicit None
            t.setdefault("annotation_assistant", None)
        out_f = RECORDINGS_BASE / pid / ANN
        tmp_f = out_f.with_suffix(".jsonl.tmp")
        with open(tmp_f, "w") as w:
            for t in turns:
                w.write(json.dumps(t) + "\n")
        tmp_f.replace(out_f)

    print(f"done — wrote {len(convos)} files")


if __name__ == "__main__":
    sys.exit(main())
