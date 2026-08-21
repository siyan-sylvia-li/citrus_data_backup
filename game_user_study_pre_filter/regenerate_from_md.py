#!/usr/bin/env python3
"""
Treat utterances_by_act.md as the SOURCE OF TRUTH for dialogue-act annotations.

Workflow: after hand-deleting inappropriate act bullets/sections from the .md,
run this to rewrite each conversation_annotated.jsonl so its user_dialogue_acts
match exactly the assignments that survive in the .md. Then re-run the stats and
plot scripts (this file does that at the end).

Membership rule: a turn is assigned act A iff a bullet for that (pid, turn)
remains under A's "## A (...)" section. The "_[+ co-acts]_" brackets are display
only and are ignored — section membership is authoritative.
"""
import json, glob, os, re, subprocess
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "recordings-download_batch_1")
MD   = os.path.join(BASE, "utterances_by_act.md")
VENV_PY = os.path.join(HERE, "..", "document_study_og", ".venv", "bin", "python")

HEADER_RE = re.compile(r"^## (?P<act>.+?)\s+\((?P<scheme>Student move|TUTOR move)\)")
BULLET_RE = re.compile(r"^- \*\*\[(?P<grp>[^\]]+)\]\s+(?P<pid8>\S+)\s+t(?P<turn>\d+)\*\*")

# map truncated pid (first 8 chars, as printed in the md) -> full pid
pid_full = {}
for f in glob.glob(os.path.join(BASE, "*", "conversation.jsonl")):
    p = os.path.basename(os.path.dirname(f))
    pid_full[p[:8]] = p

# parse md -> surviving acts per (full_pid, turn); also scheme per act
surviving = defaultdict(set)         # (pid, turn) -> {act, ...}
scheme_of = {}
cur_act = cur_scheme = None
with open(MD) as fh:
    for line in fh:
        h = HEADER_RE.match(line)
        if h:
            cur_act = h.group("act").strip()
            cur_scheme = "student" if h.group("scheme") == "Student move" else "tutor"
            scheme_of[cur_act] = cur_scheme
            continue
        b = BULLET_RE.match(line)
        if b and cur_act:
            pid = pid_full.get(b.group("pid8"), b.group("pid8"))
            surviving[(pid, int(b.group("turn")))].add(cur_act)

# keep existing per-turn rationale text (rationale is per turn, not per act)
rationale_of = {}
for f in glob.glob(os.path.join(BASE, "*", "conversation_annotated.jsonl")):
    pid = os.path.basename(os.path.dirname(f))
    for i, line in enumerate(open(f)):
        line = line.strip()
        if line:
            d = json.loads(line)
            rationale_of[(pid, i)] = d.get("user_dialogue_acts_rationale", "")

# rewrite conversation_annotated.jsonl from ORIGINAL conversation.jsonl + surviving acts
EXCLUDE = {"test-test"}
n_turns = n_assign = 0
for f in sorted(glob.glob(os.path.join(BASE, "*", "conversation.jsonl"))):
    pid = os.path.basename(os.path.dirname(f))
    if pid in EXCLUDE:
        continue
    out = os.path.join(os.path.dirname(f), "conversation_annotated.jsonl")
    with open(f) as fh, open(out, "w") as oh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            acts = sorted(surviving.get((pid, i), set()),
                          key=lambda a: (scheme_of.get(a, "student") != "student", a))
            d["user_dialogue_acts"] = [
                {"act": a, "scheme": scheme_of.get(a, "student")} for a in acts
            ]
            d["user_dialogue_acts_rationale"] = rationale_of.get((pid, i), "")
            oh.write(json.dumps(d, ensure_ascii=False) + "\n")
            n_turns += 1
            n_assign += len(acts)

print(f"Rewrote annotations from {os.path.basename(MD)}: "
      f"{n_turns} turns, {n_assign} act assignments, {len(scheme_of)} distinct acts kept")
print("Acts kept:", ", ".join(sorted(scheme_of)))

# regenerate downstream artifacts
for script in ("dialogue_acts_by_outcome.py", "list_utterances_by_act.py",
               "plot_dialogue_acts_by_outcome.py"):
    print(f"\n>>> {script}")
    subprocess.run([VENV_PY, os.path.join(HERE, script)], check=True)
