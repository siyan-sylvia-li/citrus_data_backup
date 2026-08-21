#!/usr/bin/env python3
"""
Annotate user turns in recordings-download/*/conversation.jsonl with dialogue acts
from Rus et al. (ERIC EJ1115376), Table I "Coding Schemes for Dialogue Moves".

Only USER turns are encoded. Because the human player switches roles during the
Socratic tutoring dialogue, each user turn is eligible for BOTH the Tutor-move and
Student-move taxonomies; multiple acts per turn are allowed.

The act labels below are human/LLM-assigned judgments (not rule-derived). Outputs:
  - <dir>/conversation_annotated.jsonl  (each original line + "user_dialogue_acts")
  - recordings-download/dialogue_acts_summary.csv  (one row per user turn)
Originals are never modified.
"""
import json, glob, os, csv

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings-download")

# Scheme membership for the acts actually used (T = Tutor move, S = Student move)
SCHEME = {
    # Student moves
    "Think Aloud": "S", "Conversational Acknowledgment": "S",
    "Knowledge Deficit Question": "S", "Misconception": "S",
    "Common Ground Question": "S", "Vague Answer": "S", "Partial Answer": "S",
    "Social Coordination Action": "S", "Metacomment": "S", "Read Aloud": "S",
    "Correct Answer": "S",
    # Tutor moves (used where the user adopts a tutor-like role)
    "Forced Choice": "T", "Repetition": "T", "Prompt": "T",
}

# acts keyed by (prolific_id, turn_index) -> (acts, rationale)
ANN = {
 ("p673", 0): (["Think Aloud"], "Reasons aloud about the board and an intended blocking move."),
 ("p673", 1): (["Conversational Acknowledgment", "Think Aloud"], "'I see' acknowledges, then reasons aloud about building/blocking."),
 ("p673", 2): (["Conversational Acknowledgment", "Think Aloud"], "'Okay' + verbalized decision to block."),
 ("p673", 3): (["Think Aloud"], "Verbalizes a two-pronged plan."),

 ("p640", 0): (["Knowledge Deficit Question"], "Asks the tutor whether a proposed move is good."),
 ("p640", 1): (["Think Aloud"], "Self-evaluates own idea ('it would not work')."),
 ("p640", 2): (["Think Aloud", "Misconception"], "Proposes deceptive plan; assumes a disc reaches row 3 (gravity misconception)."),
 ("p640", 3): (["Knowledge Deficit Question"], "Directly requests a suggestion."),
 ("p640", 4): (["Conversational Acknowledgment", "Social Coordination Action"], "'done' coordinates that an action was completed."),
 ("p640", 5): (["Knowledge Deficit Question"], "Asks a factual rule question (how many discs to align)."),
 ("p640", 6): (["Knowledge Deficit Question"], "'and now?' requests next-step guidance."),
 ("p640", 7): (["Think Aloud", "Metacomment"], "Comments on own search ('the only one was the one I tried')."),
 ("p640", 8): (["Knowledge Deficit Question"], "'now?' requests next-step guidance."),
 ("p640", 9): (["Vague Answer"], "Tentative, underspecified reply to a forced choice ('the first one?')."),

 ("p255", 0): (["Knowledge Deficit Question"], "Asks whether a column move works."),
 ("p255", 1): (["Knowledge Deficit Question"], "Asks why a move would not work."),
 ("p255", 2): (["Knowledge Deficit Question"], "Asks whether a column helps."),
 ("p255", 3): (["Knowledge Deficit Question"], "Asks about another column option."),

 ("p503", 0): (["Knowledge Deficit Question", "Common Ground Question"], "Clarifies the win condition ('exactly 5' vs 'less than 5')."),
 ("p503", 1): (["Misconception", "Knowledge Deficit Question"], "Believes the move makes 4-in-a-row and asks why it's wrong."),

 ("p414", 0): (["Read Aloud", "Knowledge Deficit Question"], "Restates the posed problem, then asks how to think it through."),
 ("p414", 1): (["Think Aloud", "Partial Answer"], "Works through candidate forcing lines aloud."),

 ("p46", 0): (["Forced Choice", "Knowledge Deficit Question"], "Poses an offense-vs-defense binary to the tutor (tutor-style forced choice)."),
 ("p46", 1): (["Think Aloud"], "States board observation about both sides having three."),
 ("p46", 2): (["Think Aloud", "Common Ground Question"], "Reasons aloud and checks understanding ('they would just put one on top?')."),

 ("p139", 0): (["Knowledge Deficit Question"], "Asks where the biggest threat is."),
 ("p139", 1): (["Knowledge Deficit Question"], "Asks where own strongest position is."),
 ("p139", 2): (["Knowledge Deficit Question", "Repetition"], "Repeats the strongest-position question verbatim."),
 ("p139", 3): (["Knowledge Deficit Question"], "Asks where own weakest position is."),

 ("p668", 0): (["Knowledge Deficit Question"], "Asks whether to play the left corner."),
 ("p668", 1): (["Think Aloud"], "States current goal/intent (blocking the yellow diagonal)."),
 ("p668", 2): (["Knowledge Deficit Question"], "Asks what to do next."),

 ("p613", 0): (["Knowledge Deficit Question"], "Asks which slots to avoid."),

 ("p74", 0): (["Forced Choice", "Knowledge Deficit Question"], "Poses defense-vs-attack binary to the tutor."),
 ("p74", 1): (["Think Aloud"], "Notes a spot to drop a third red, reasoning aloud."),

 ("p580", 0): (["Knowledge Deficit Question"], "Asks for the best winning-chance column."),
 ("p580", 1): (["Conversational Acknowledgment", "Think Aloud"], "'okay' + reasons that a block is forced."),
 ("p580", 2): (["Knowledge Deficit Question"], "Asks why a move isn't allowed/advisable."),
 ("p580", 3): (["Knowledge Deficit Question"], "Asks which moves permit a win (without the exact answer)."),
}


def main():
    rows = []
    files = sorted(glob.glob(os.path.join(BASE, "*", "conversation.jsonl")))
    missing = []
    for f in files:
        pid = os.path.basename(os.path.dirname(f))
        out = os.path.join(os.path.dirname(f), "conversation_annotated.jsonl")
        with open(f) as fh, open(out, "w") as oh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                key = (pid, i)
                if key not in ANN:
                    missing.append(key)
                    acts, rationale = [], ""
                else:
                    acts, rationale = ANN[key]
                d["user_dialogue_acts"] = [
                    {"act": a, "scheme": {"T": "tutor", "S": "student"}[SCHEME[a]]}
                    for a in acts
                ]
                d["user_dialogue_acts_rationale"] = rationale
                oh.write(json.dumps(d, ensure_ascii=False) + "\n")
                rows.append({
                    "prolific_id": pid,
                    "turn_index": i,
                    "user_text": d.get("user", ""),
                    "dialogue_acts": "; ".join(acts),
                    "schemes": "; ".join(SCHEME[a] for a in acts),
                    "rationale": rationale,
                })
        print(f"wrote {out}")

    csv_path = os.path.join(BASE, "dialogue_acts_summary.csv")
    with open(csv_path, "w", newline="") as cf:
        w = csv.DictWriter(cf, fieldnames=["prolific_id", "turn_index", "user_text",
                                           "dialogue_acts", "schemes", "rationale"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {csv_path}  ({len(rows)} user turns)")
    if missing:
        print("WARNING: unannotated turns:", missing)

    # quick distribution
    from collections import Counter
    c = Counter(a for r in rows for a in (r["dialogue_acts"].split("; ") if r["dialogue_acts"] else []))
    print("\nAct distribution (user turns):")
    for a, n in c.most_common():
        print(f"  {n:3d}  {a} [{ {'T':'tutor','S':'student'}[SCHEME[a]] }]")


if __name__ == "__main__":
    main()
