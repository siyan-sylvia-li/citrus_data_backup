"""Dialogue-act SEQUENCE comparison: controlled phase-1 participants vs wild ShareChat
(tutoring, judged-pedagogical). Models each corpus as a first-order Markov / bigram over
dialogue acts, optionally conditioned on success, so we can read off "strategies".

Both corpora share the same 6-act vocabulary and are multi-label per turn; we handle
multi-label with fractional counts (each turn is a uniform distribution over its acts, so
every turn-pair contributes total weight 1 to the bigram counts). START/END states capture
openers and closers. Rows are Laplace-smoothed because the data is sparse.

Inputs:
  participant_utterances.csv               (phase-1: pid, utt, utt_type, utt_ind, outcome)
  acts_tutoring_or_teaching_pedagogical.json (ShareChat: url, message_index, acts)
  [optional] sharechat_success.json          ({url: "success"|"failure"|...}) to condition ShareChat
"""
import sys, json, math
from collections import defaultdict, Counter
import pandas as pd

ACTS = ["Solution Request", "Common Ground Question", "Knowledge Deficit Question",
        "Think Aloud", "Conversational Acknowledgment"]     # Metacomment dropped (unreliable)
FROM_STATES = ["START"] + ACTS
TO_STATES = ACTS + ["END"]
SHORT = {"Solution Request": "SolReq", "Common Ground Question": "CommonGrd",
         "Knowledge Deficit Question": "KnowDef", "Think Aloud": "ThinkAloud",
         "Conversational Acknowledgment": "Ack"}


# ---------- load sequences: each convo -> ([set_of_acts per turn in order], outcome) ----------
def load_phase1(path="participant_utterances.csv"):
    d = pd.read_csv(path)
    d = d[d["utt_type"].isin(ACTS)]
    out = []
    for pid, g in d.groupby("pid"):
        turns = [set(t["utt_type"]) for _, t in sorted(g.groupby("utt_ind"))]
        turns = [t for t in turns if t]
        if turns:
            out.append((turns, g["outcome"].iloc[0]))
    return out


def load_sharechat(path="acts_tutoring_or_teaching_pedagogical.json", labels=None):
    rows = [r for r in json.load(open(path)) if r.get("acts")]
    by_url = defaultdict(list)
    for r in rows:
        by_url[r["url"]].append((r["message_index"], [a for a in r["acts"] if a in ACTS]))
    out = []
    for url, msgs in by_url.items():
        turns = [set(a) for _, a in sorted(msgs) if a]
        if turns:
            out.append((turns, (labels or {}).get(url, "ALL")))
    return out


# ---------- bigram model ----------
def bigram_counts(convos):
    """Fractional bigram counts over FROM_STATES x TO_STATES."""
    c = defaultdict(float)
    for turns, _ in convos:
        # START -> first turn
        for a in turns[0]:
            c[("START", a)] += 1.0 / len(turns[0])
        # turn t -> turn t+1
        for A, B in zip(turns, turns[1:]):
            for a in A:
                for b in B:
                    c[(a, b)] += 1.0 / (len(A) * len(B))
        # last turn -> END
        for a in turns[-1]:
            c[(a, "END")] += 1.0 / len(turns[-1])
    return c


def prob_matrix(counts, alpha=0.5):
    """Row-normalized transition probs with add-alpha smoothing."""
    P = {}
    for s in FROM_STATES:
        row = {t: counts.get((s, t), 0.0) for t in TO_STATES}
        tot = sum(row.values()) + alpha * len(TO_STATES)
        P[s] = {t: (row[t] + alpha) / tot for t in TO_STATES}
    return P


def unigram_rate(convos):
    """Share of TURNS carrying each act (multi-label), for a group of convos."""
    n_turns = sum(len(turns) for turns, _ in convos)
    c = Counter(a for turns, _ in convos for t in turns for a in t)
    return {a: c[a] / n_turns for a in ACTS}, n_turns


def print_matrix(P, title):
    print(f"\n{title}")
    hdr = "from\\to     " + " ".join(f"{SHORT[t] if t in SHORT else t:>10}" for t in TO_STATES)
    print(hdr)
    for s in FROM_STATES:
        cells = " ".join(f"{P[s][t]:>10.2f}" for t in TO_STATES)
        print(f"{(SHORT.get(s, s)):<11} {cells}")


def enriched(P_pos, P_neg, k=8):
    """Transitions most enriched in POS vs NEG by log-ratio (with prob floor)."""
    rows = []
    for s in FROM_STATES:
        for t in TO_STATES:
            lr = math.log(P_pos[s][t] / P_neg[s][t])
            rows.append((lr, s, t, P_pos[s][t], P_neg[s][t]))
    rows.sort(reverse=True)
    return rows[:k], rows[-k:]


if __name__ == "__main__":
    p1 = load_phase1()
    sc_labels = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else None
    sc = load_sharechat(labels=sc_labels)

    print("=" * 70)
    print(f"PHASE-1 (controlled): {len(p1)} participants; "
          f"outcomes {dict(Counter(o for _, o in p1))}")
    print(f"SHARECHAT (wild):     {len(sc)} conversations"
          + (f"; outcomes {dict(Counter(o for _, o in sc))}" if sc_labels else " (unconditioned)"))

    # ---- robust summary 1: unigram act rate by phase-1 outcome ----
    won = [c for c in p1 if c[1] == "won"]
    lost = [c for c in p1 if c[1] == "lost"]
    rw, nw = unigram_rate(won)
    rl, nl = unigram_rate(lost)
    rsc, nsc = unigram_rate(sc)
    print(f"\n--- act rate per turn (WON n={nw}t / LOST n={nl}t / ShareChat n={nsc}t) ---")
    print(f"{'act':<16}{'p1-won':>8}{'p1-lost':>9}{'won-lost':>10}{'sharechat':>11}")
    for a in ACTS:
        print(f"{SHORT[a]:<16}{rw[a]:>8.0%}{rl[a]:>9.0%}{rw[a]-rl[a]:>+10.0%}{rsc[a]:>11.0%}")

    # ---- transition matrices ----
    print_matrix(prob_matrix(bigram_counts(won)), "PHASE-1 WON  P(next|cur)")
    print_matrix(prob_matrix(bigram_counts(lost)), "PHASE-1 LOST  P(next|cur)")
    print_matrix(prob_matrix(bigram_counts(sc)), "SHARECHAT (all)  P(next|cur)")

    # ---- success-enriched transitions in phase-1 ----
    Pw, Pl = prob_matrix(bigram_counts(won)), prob_matrix(bigram_counts(lost))
    top, bot = enriched(Pw, Pl)
    print("\n--- transitions ENRICHED in WON vs LOST (phase-1) ---")
    for lr, s, t, pp, pn in top:
        print(f"  {SHORT.get(s,s):>10} -> {SHORT.get(t,t):<11} log-ratio {lr:+.2f}  (won {pp:.2f} vs lost {pn:.2f})")
    print("--- transitions enriched in LOST ---")
    for lr, s, t, pp, pn in reversed(bot):
        print(f"  {SHORT.get(s,s):>10} -> {SHORT.get(t,t):<11} log-ratio {lr:+.2f}  (won {pp:.2f} vs lost {pn:.2f})")
