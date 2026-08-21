#!/usr/bin/env python3
"""
Split dialogue-act statistics by puzzle outcome (solved vs unsolved).

Outcome rule (from moves.jsonl):
  - solved   = participant made >= 5 optimal moves
  - unsolved = fewer than 5 optimal moves
Exclusions:
  - any participant with > 8 total moves (retried and succeeded)
  - the 'test-test' scratch account
"""
import json, glob, os, csv
from collections import Counter, defaultdict

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings-download_batch_1")

def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]

# ---- classify participants by outcome ----
outcome = {}       # pid -> "solved"/"unsolved"
move_stats = {}    # pid -> (total, optimal)
excluded = {}
for f in sorted(glob.glob(os.path.join(BASE, "*", "moves.jsonl"))):
    pid = os.path.basename(os.path.dirname(f))
    if pid == "test-test":
        excluded[pid] = "test account"
        continue
    moves = load_jsonl(f)
    total = len(moves)
    opt = sum(1 for m in moves if m.get("optimal"))
    move_stats[pid] = (total, opt)
    if total > 8:
        excluded[pid] = f"{total} total moves (retried & succeeded)"
        continue
    outcome[pid] = "solved" if opt >= 5 else "unsolved"

# ---- load user-turn dialogue acts from annotated conversations ----
# key: pid -> list of act-lists (one per user turn)
acts_by_pid = defaultdict(list)
SCHEME = {}
for f in sorted(glob.glob(os.path.join(BASE, "*", "conversation_annotated.jsonl"))):
    pid = os.path.basename(os.path.dirname(f))
    for d in load_jsonl(f):
        acts = d.get("user_dialogue_acts", [])
        acts_by_pid[pid].append([a["act"] for a in acts])
        for a in acts:
            SCHEME[a["act"]] = a["scheme"]

# ---- aggregate per group ----
def summarize(group):
    pids = [p for p in outcome if outcome[p] == group]
    n_part = len(pids)
    n_turns = sum(len(acts_by_pid[p]) for p in pids)
    act_counts = Counter(a for p in pids for turn in acts_by_pid[p] for a in turn)
    return pids, n_part, n_turns, act_counts

print("="*70)
print("PARTICIPANT CLASSIFICATION")
print("="*70)
print(f"{'pid':42} {'total':>5} {'opt':>4}  {'outcome'}")
for pid in sorted(move_stats):
    total, opt = move_stats[pid]
    tag = excluded.get(pid) or outcome.get(pid, "?")
    print(f"{pid:42} {total:5d} {opt:4d}  {tag}")
for pid, why in excluded.items():
    if pid not in move_stats:
        print(f"{pid:42} {'-':>5} {'-':>4}  EXCLUDED ({why})")

all_acts = sorted(set(a for p in outcome for turn in acts_by_pid[p] for a in turn),
                  key=lambda a: (SCHEME[a], a))

for group in ("solved", "unsolved"):
    pids, n_part, n_turns, ac = summarize(group)
    print("\n" + "="*70)
    print(f"{group.upper()}  —  {n_part} participants, {n_turns} user turns")
    print("="*70)
    print(f"  avg user turns/participant: {n_turns/n_part:.2f}")
    print(f"  {'act':32} {'scheme':>7} {'count':>6} {'/turn':>7} {'/part':>7}")
    for a in all_acts:
        cnt = ac.get(a, 0)
        if cnt == 0:
            continue
        print(f"  {a:32} {SCHEME[a]:>7} {cnt:6d} {cnt/n_turns:7.3f} {cnt/n_part:7.2f}")

# ---- write CSV: act x outcome matrix ----
out_csv = os.path.join(BASE, "dialogue_acts_by_outcome.csv")
sp, snp, snt, sac = summarize("solved")
up, unp, unt, uac = summarize("unsolved")
with open(out_csv, "w", newline="") as cf:
    w = csv.writer(cf)
    w.writerow(["act", "scheme",
                "solved_count", "solved_per_turn", "solved_per_participant",
                "unsolved_count", "unsolved_per_turn", "unsolved_per_participant"])
    for a in all_acts:
        w.writerow([a, SCHEME[a],
                    sac.get(a,0), round(sac.get(a,0)/snt,3), round(sac.get(a,0)/snp,2),
                    uac.get(a,0), round(uac.get(a,0)/unt,3), round(uac.get(a,0)/unp,2)])
    w.writerow([])
    w.writerow(["_participants", "", snp, "", "", unp, "", ""])
    w.writerow(["_user_turns", "", snt, "", "", unt, "", ""])
print(f"\nwrote {out_csv}")
print(f"solved participants:   {sorted(sp)}")
print(f"unsolved participants: {sorted(up)}")
