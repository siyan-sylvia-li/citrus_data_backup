"""Descriptive dialogue-act distribution comparison: controlled (phase-1 participants)
vs wild (ShareChat tutoring, judged-pedagogical), SEPARATED into openers (first user
turn) vs follow-ups (all later user turns). Multi-label: rates = share of turns carrying
each act, so columns need not sum to 100%. Descriptive only — no success conditioning.
"""
from act_sequence_analysis import load_phase1, load_sharechat, ACTS, SHORT


def rates_by_position(convos):
    openers, followups, allt = [], [], []
    for turns, _ in convos:
        openers.append(turns[0])
        followups.extend(turns[1:])
        allt.extend(turns)

    def rate(ts):
        n = len(ts)
        return ({a: (sum(a in t for t in ts) / n if n else 0.0) for a in ACTS}, n)

    return {"opener": rate(openers), "followup": rate(followups), "all": rate(allt)}


def main():
    import sys
    wild_path = sys.argv[1] if len(sys.argv) > 1 else "acts_tutoring_or_teaching_pedagogical_panel.json"
    print(f"controlled: participant_utterances.csv | wild: {wild_path}")
    P = rates_by_position(load_phase1())
    S = rates_by_position(load_sharechat(wild_path))

    for pos in ("opener", "followup", "all"):
        (rp, np_), (rs, ns) = P[pos], S[pos]
        title = {"opener": "OPENERS (first user turn)",
                 "followup": "FOLLOW-UPS (later user turns)",
                 "all": "ALL user turns"}[pos]
        print(f"\n{'='*58}\n{title}   controlled n={np_}t   wild n={ns}t\n{'='*58}")
        print(f"{'act':<14}{'phase1':>9}{'wild':>9}{'Δ(w-p1)':>10}")
        for a in sorted(ACTS, key=lambda a: -rp[a]):
            print(f"{SHORT[a]:<14}{rp[a]:>9.0%}{rs[a]:>9.0%}{rs[a]-rp[a]:>+10.0%}")

    # compact "engagement vs request" summary (openers vs follow-ups, both corpora)
    ENGAGE = ["Common Ground Question", "Think Aloud", "Knowledge Deficit Question"]
    print(f"\n{'='*58}\nENGAGEMENT (CommonGrd+ThinkAloud+KnowDef) vs REQUEST (SolReq)\n{'='*58}")
    print(f"{'':<20}{'engage':>9}{'request':>9}")
    for name, corpus in (("phase1", P), ("wild", S)):
        for pos in ("opener", "followup"):
            r, _ = corpus[pos]
            eng = sum(r[a] for a in ENGAGE)
            print(f"{name+' '+pos:<20}{eng:>9.0%}{r['Solution Request']:>9.0%}")


if __name__ == "__main__":
    main()
