#!/usr/bin/env python3
"""Pull conversation traces for participants who rated the AI assistant poorly.

Reads assistant_helpfulness_othello.csv, flags anyone who answered Neutral or
worse on either Likert item, and dumps their full AI-condition conversation
(plus puzzle outcome) to a markdown file for qualitative review.
"""
import csv, json, os, sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
SURVEY = BASE / "assistant_helpfulness_othello.csv"
REC = BASE / "recordings-download"
OUT_MD = BASE / "low_helpfulness_traces.md"
OUT_CSV = BASE / "low_helpfulness_summary.csv"

# lower = worse
LIKERT = {"strongly disagree": 1, "disagree": 2, "neutral": 3,
          "agree": 4, "strongly agree": 5}
THRESHOLD = 3  # Neutral or worse on either item

Q_INTENT = "The AI assistant's responses were consistent with the intentions of my questions."
Q_EASE = "The AI assistant was easy to use."
Q_FREE = "Any additional feedback you would like to share about the AI assistant?"


def score_of(val):
    return LIKERT.get((val or "").strip().lower())


def load_survey():
    with SURVEY.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def conv_files(pid):
    d = REC / pid
    if not d.is_dir():
        return []
    # prefer annotated version when present (same turns + dialogue-act labels)
    convs = sorted(d.glob("conversation_*.jsonl"))
    out = []
    for c in convs:
        ann = d / f"annotated_{c.name}"
        out.append(ann if ann.exists() else c)
    return out


def read_jsonl(p):
    rows = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def puzzle_from_conv(name):
    # conversation_poc20260727.jsonl -> poc20260727
    stem = name.replace("annotated_", "").replace("conversation_", "").replace(".jsonl", "")
    return stem


def load_score(pid, puzzle):
    p = REC / pid / f"oth_score_{puzzle}.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def moves_table(pid, puzzle):
    """Per-decision play vs. the solved-puzzle oracle, as markdown rows."""
    p = REC / pid / f"moves_{puzzle}.jsonl"
    if not p.exists():
        return []
    ms = read_jsonl(p)
    out = ["", "| # | played | optimal? | oracle best | value/best | White reply | discs B–W |",
           "|---|---|---|---|---|---|---|"]
    for m in ms:
        out.append(
            f"| {m['decision_number']} | `{m['move']}` | "
            f"{'yes' if m['optimal'] else '**NO**'} | "
            f"{', '.join(f'`{b}`' for b in m['best_moves'])} | "
            f"{m['played_value']}/{m['best_value']} | "
            f"{', '.join(m['ai_moves']) or '—'} | "
            f"{m['black_discs']}–{m['white_discs']} |")
    out.append("")
    return out


def acts(turn, key):
    a = turn.get(key) or {}
    final = a.get("final") if isinstance(a, dict) else None
    if isinstance(final, list) and final:
        return ", ".join(str(x) for x in final)
    return ""


def main():
    rows = load_survey()
    flagged = []
    for r in rows:
        pid = (r.get("Prolific ID?") or "").strip()
        s_intent, s_ease = score_of(r.get(Q_INTENT)), score_of(r.get(Q_EASE))
        if s_intent is None or s_ease is None:
            print(f"[warn] unparsed Likert for {pid}: {r.get(Q_INTENT)!r}/{r.get(Q_EASE)!r}", file=sys.stderr)
            continue
        if min(s_intent, s_ease) <= THRESHOLD:
            flagged.append((r, pid, s_intent, s_ease))
    flagged.sort(key=lambda t: (min(t[2], t[3]), t[2] + t[3]))

    md = ["# Othello study — conversation traces for low assistant-helpfulness raters",
          "",
          f"Source survey: `{SURVEY.name}` ({len(rows)} respondents).",
          f"Flagged: **{len(flagged)}** participants who answered Neutral or worse on either Likert item.",
          ""]

    summary = []
    for r, pid, s_intent, s_ease in flagged:
        md += ["---", "", f"## {pid}", "",
               f"- **Intent-consistency:** {r[Q_INTENT]} ({s_intent}/5)",
               f"- **Ease of use:** {r[Q_EASE]} ({s_ease}/5)",
               f"- **Submitted:** {r['Timestamp']}"]
        free = (r.get(Q_FREE) or "").strip()
        md.append(f"- **Free-text:** {free if free else '_(none)_'}")

        convs = conv_files(pid)
        if not convs:
            md += ["", "_No conversation file found in recordings-download._", ""]
            summary.append({"prolific_id": pid, "intent": r[Q_INTENT], "ease": r[Q_EASE],
                            "turns": 0, "puzzle": "", "solved": "", "final_margin": "",
                            "end_reason": "", "free_text": free})
            continue

        for c in convs:
            puzzle = puzzle_from_conv(c.name)
            turns = read_jsonl(c)
            sc = load_score(pid, puzzle)
            if sc:
                md += ["", f"- **Puzzle `{puzzle}`:** solved={sc.get('solved')}, "
                           f"result={sc.get('result')}, margin={sc.get('final_margin')}, "
                           f"end_reason={sc.get('end_reason')}, score={sc.get('score')}/"
                           f"{sc.get('decisions')} decisions"]
            md += moves_table(pid, puzzle)
            md += ["", f"### Conversation — `{c.name}` ({len(turns)} turns)", ""]
            for i, t in enumerate(turns, 1):
                ua, aa = acts(t, "annotation_user"), acts(t, "annotation_assistant")
                md.append(f"**[{i}] User**{f' _({ua})_' if ua else ''} — {t.get('user_ts','')}")
                md += ["", "> " + (t.get("user") or "").replace("\n", "\n> "), ""]
                md.append(f"**[{i}] Assistant**{f' _({aa})_' if aa else ''} — {t.get('assistant_ts','')}")
                md += ["", "> " + (t.get("assistant") or "").replace("\n", "\n> "), ""]

            summary.append({"prolific_id": pid, "intent": r[Q_INTENT], "ease": r[Q_EASE],
                            "turns": len(turns), "puzzle": puzzle,
                            "solved": sc.get("solved") if sc else "",
                            "final_margin": sc.get("final_margin") if sc else "",
                            "end_reason": sc.get("end_reason") if sc else "",
                            "free_text": free})

    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print(f"{len(flagged)} flagged participants -> {OUT_MD.name}, {OUT_CSV.name}")
    for s in summary:
        print(f"  {s['prolific_id']}  intent={s['intent']:<17} ease={s['ease']:<17} "
              f"turns={s['turns']:<3} solved={s['solved']} margin={s['final_margin']}")


if __name__ == "__main__":
    main()
