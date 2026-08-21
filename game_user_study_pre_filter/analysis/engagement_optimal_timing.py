#!/usr/bin/env python3
"""
engagement_optimal_timing.py

Cross-reference AI-chat timestamps with move timestamps to test whether
participants are more likely to make an OPTIMAL Connect Four move *after*
engaging with the AI assistant.

Logic
-----
Each chat exchange in conversation.jsonl carries `assistant_ts` (when the AI
reply finished) — the moment the advice became available. Each move in
moves.jsonl carries `ts` and `optimal`.

For every move we find the most recent AI reply that preceded it and flag the
move as **post_chat** when BOTH:
  1. that chat is the most recent event before the move (no other move happened
     between the chat and this move -> they consulted, then moved), and
  2. the gap is within --window seconds (default 180).

We then compare the optimal-move rate for post_chat moves vs. all other moves,
per participant and pooled.

Usage
-----
    python analysis/engagement_optimal_timing.py [recordings_dir] [--window SECONDS]

Defaults: recordings_dir = "recordings-download", window = 180s.

Outputs (written next to this script)
-------------------------------------
    moves_with_engagement.csv   one row per move (with post_chat flag)
    engagement_summary.csv      one row per participant
and prints a pooled summary to stdout.

Caveats
-------
- Correlation, not causation: a move "after chat" may still ignore the advice;
  and skill is confounded with both engaging and playing well.
- Replays: moves.jsonl can concatenate multiple sessions (moves_left resets),
  while conversation.jsonl is cleared on re-entry, so some early moves legitimately
  have no preceding chat. Timestamp ordering handles this correctly, but per-move
  attribution across a replay boundary is approximate.
- Tiny n: treat as directional.
"""
import argparse
import csv
import json
import os
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_ts(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def load_jsonl(path):
    try:
        return [json.loads(line) for line in open(path) if line.strip()]
    except FileNotFoundError:
        return []


def analyze_participant(pdir, window):
    """Return list of per-move dicts with engagement flags."""
    convo = load_jsonl(os.path.join(pdir, "conversation.jsonl"))
    moves = load_jsonl(os.path.join(pdir, "moves.jsonl"))

    # AI-reply completion times (fall back to user_ts if assistant_ts is missing).
    chat_ts = sorted(
        t for t in (parse_ts(c.get("assistant_ts")) or parse_ts(c.get("user_ts")) for c in convo)
        if t is not None
    )

    moves_ts = sorted(
        ((parse_ts(m.get("ts")), m) for m in moves if parse_ts(m.get("ts"))),
        key=lambda x: x[0],
    )

    rows = []
    prev_move_ts = None
    for t, m in moves_ts:
        prior_chats = [c for c in chat_ts if c < t]
        last_chat = prior_chats[-1] if prior_chats else None
        secs_since_chat = (t - last_chat).total_seconds() if last_chat else None
        # The chat is the most recent event before this move (no move since the chat).
        chat_is_latest = last_chat is not None and (prev_move_ts is None or last_chat > prev_move_ts)
        within = secs_since_chat is not None and secs_since_chat <= window
        post_chat = bool(chat_is_latest and within)
        rows.append({
            "ts": t.isoformat(),
            "optimal": bool(m.get("optimal")),
            "status": m.get("status"),
            "secs_since_chat": round(secs_since_chat, 1) if secs_since_chat is not None else "",
            "chat_is_latest_event": chat_is_latest,
            "post_chat": post_chat,
        })
        prev_move_ts = t
    return rows


def is_participant_dir(name):
    # 24-char hex Prolific IDs, plus testers/anon for completeness.
    return (len(name) == 24 and all(c in "0123456789abcdef" for c in name)) or name.startswith(
        ("tester", "test", "anon")
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recordings_dir", nargs="?", default="recordings-download")
    ap.add_argument("--window", type=float, default=180.0, help="max seconds between chat and move (default 180)")
    args = ap.parse_args()

    base = args.recordings_dir
    parts = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)) and is_participant_dir(d))

    per_move_rows = []
    summary_rows = []
    # pooled tallies
    pool = {"post": [0, 0], "other": [0, 0]}  # [optimal, total]

    for pid in parts:
        pdir = os.path.join(base, pid)
        rows = analyze_participant(pdir, args.window)
        if not rows:
            continue
        n_chats = len(load_jsonl(os.path.join(pdir, "conversation.jsonl")))
        solved = any(r["status"] == "solved" for r in rows)
        post = [r for r in rows if r["post_chat"]]
        other = [r for r in rows if not r["post_chat"]]
        post_opt = sum(r["optimal"] for r in post)
        other_opt = sum(r["optimal"] for r in other)
        any_opt_after_chat = any(r["optimal"] and r["post_chat"] for r in rows)

        pool["post"][0] += post_opt
        pool["post"][1] += len(post)
        pool["other"][0] += other_opt
        pool["other"][1] += len(other)

        for r in rows:
            per_move_rows.append({"participant": pid, **r})
        summary_rows.append({
            "participant": pid,
            "solved": solved,
            "n_chats": n_chats,
            "n_moves": len(rows),
            "n_post_chat_moves": len(post),
            "post_chat_optimal": post_opt,
            "post_chat_optimal_rate": round(post_opt / len(post), 2) if post else "",
            "other_optimal": other_opt,
            "other_optimal_rate": round(other_opt / len(other), 2) if other else "",
            "any_optimal_after_chat": any_opt_after_chat,
        })

    # write CSVs
    mv_path = os.path.join(HERE, "moves_with_engagement.csv")
    with open(mv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["participant", "ts", "optimal", "status",
                                          "secs_since_chat", "chat_is_latest_event", "post_chat"])
        w.writeheader(); w.writerows(per_move_rows)
    sm_path = os.path.join(HERE, "engagement_summary.csv")
    with open(sm_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        w.writeheader(); w.writerows(summary_rows)

    # pooled summary
    def rate(pair):
        return f"{pair[0]}/{pair[1]} = {pair[0]/pair[1]:.0%}" if pair[1] else "n/a"

    print(f"Participants analyzed: {len(summary_rows)}  (window = {args.window:.0f}s)")
    print(f"Outputs: {mv_path}")
    print(f"         {sm_path}\n")
    print("POOLED optimal-move rate:")
    print(f"  Moves right after AI chat : {rate(pool['post'])}")
    print(f"  All other moves           : {rate(pool['other'])}")
    n_any = sum(1 for s in summary_rows if s["any_optimal_after_chat"])
    print(f"\nParticipants with >=1 optimal move right after a chat: {n_any}/{len(summary_rows)}")
    print("\nPer participant (solved | chats | moves | post-chat opt rate | other opt rate):")
    for s in summary_rows:
        print(f"  {s['participant'][:12]:<13} {str(s['solved']):<5} "
              f"chats={s['n_chats']:<2} moves={s['n_moves']:<2} "
              f"post={s['post_chat_optimal']}/{s['n_post_chat_moves']:<2} "
              f"({s['post_chat_optimal_rate']})  other={s['other_optimal']}/"
              f"{s['n_moves']-s['n_post_chat_moves']} ({s['other_optimal_rate']})")


if __name__ == "__main__":
    main()
