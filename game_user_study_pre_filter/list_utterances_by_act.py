#!/usr/bin/env python3
"""
List every user utterance grouped by assigned dialogue act, for manual verification.
Reads the annotated conversations produced by annotate_dialogue_acts.py.
Writes a human-readable report (.md) and a flat CSV.
"""
import json, glob, os, csv
from collections import defaultdict

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings-download_batch_1")
MD   = os.path.join(BASE, "utterances_by_act.md")
CSV  = os.path.join(BASE, "utterances_by_act.csv")

def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]

# outcome (for context tags)
outcome = {}
for f in sorted(glob.glob(os.path.join(BASE, "*", "moves.jsonl"))):
    pid = os.path.basename(os.path.dirname(f))
    if pid == "test-test":
        continue
    m = load_jsonl(f)
    if len(m) > 8:
        outcome[pid] = "excluded(>8)"
    else:
        outcome[pid] = "solved" if sum(1 for x in m if x.get("optimal")) >= 5 else "unsolved"

# gather (act -> list of records) and flat rows
by_act = defaultdict(list)
scheme_of = {}
rows = []
for f in sorted(glob.glob(os.path.join(BASE, "*", "conversation_annotated.jsonl"))):
    pid = os.path.basename(os.path.dirname(f))
    if pid == "test-test":
        continue
    grp = outcome.get(pid, "?")
    for i, d in enumerate(load_jsonl(f)):
        text = d.get("user", "").replace("\n", " ").strip()
        acts = d.get("user_dialogue_acts", [])
        rationale = d.get("user_dialogue_acts_rationale", "")
        act_names = [a["act"] for a in acts]
        for a in acts:
            scheme_of[a["act"]] = a["scheme"]
            by_act[a["act"]].append({
                "pid": pid, "grp": grp, "turn": i, "text": text,
                "co_acts": [x for x in act_names if x != a["act"]],
                "rationale": rationale,
            })
        rows.append({"prolific_id": pid, "outcome": grp, "turn_index": i,
                     "user_text": text, "dialogue_acts": "; ".join(act_names),
                     "rationale": rationale})

# ---- markdown report ----
order = sorted(by_act, key=lambda a: (scheme_of[a] != "student", -len(by_act[a]), a))
with open(MD, "w") as fh:
    fh.write("# User utterances grouped by dialogue act\n\n")
    fh.write("Each utterance is a **user turn**. `co-acts` are other acts also assigned to the "
             "same turn (multi-label). Outcome shown per participant for context.\n\n")
    for a in order:
        recs = by_act[a]
        sch = "Student move" if scheme_of[a] == "student" else "TUTOR move"
        fh.write(f"## {a}  ({sch}) — {len(recs)} turn(s)\n\n")
        for r in recs:
            co = f"  _[+ {', '.join(r['co_acts'])}]_" if r["co_acts"] else ""
            fh.write(f"- **[{r['grp']}] {r['pid'][:8]} t{r['turn']}** — \"{r['text']}\"{co}\n")
            if r["rationale"]:
                fh.write(f"    - rationale: {r['rationale']}\n")
        fh.write("\n")

# ---- flat CSV (one row per act assignment) ----
with open(CSV, "w", newline="") as cf:
    w = csv.writer(cf)
    w.writerow(["dialogue_act", "scheme", "outcome", "prolific_id", "turn_index",
                "user_text", "co_acts", "rationale"])
    for a in order:
        for r in by_act[a]:
            w.writerow([a, scheme_of[a], r["grp"], r["pid"], r["turn"],
                        r["text"], "; ".join(r["co_acts"]), r["rationale"]])

print("wrote", MD)
print("wrote", CSV)
print(f"\n{len(rows)} user turns, {sum(len(v) for v in by_act.values())} act assignments, "
      f"{len(by_act)} distinct acts")
