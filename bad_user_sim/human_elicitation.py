"""Which PARTICIPANT act elicits the assistant acts that predict human transfer?

    python human_elicitation.py                      # Othello, fine assistant codes
    python human_elicitation.py --game connect_four  # CF, COARSE codes only -- see below

Step 2 of the causal design. Step 1 is settled on the human Othello data
(human_othello_assistant.py): Board Report is the strongest positive correlate of
the solo-puzzle margin (rho = +0.231, p = .014) and Move Verdict the strongest
negative (rho = -0.198, p = .037), while both PRE-REGISTERED transfer-bearing acts
came in null (General Principle +0.061, Worked Line +0.062). The pre-registration
was falsified; this reads the result rather than the prediction.

The gate arms cannot target Board Report directly -- the participant taxonomy is
the 14 AutoTutor codes and none of them is "asks about board state", so board
queries currently ride under Knowledge Deficit Question. Rather than bolt a new
act onto the scheme that every human/sim comparison in this project runs on, this
asks which EXISTING participant act already pulls Board Report out of the
assistant, so the amplification arm can target something the whole chain
(participant act -> assistant act -> transfer) is expressed in.

Costs nothing to run: both annotation layers are already on the same rows, so
pairing is within-row (a user message and the reply it drew), not a guess at
adjacency.

CONNECT FOUR IS NOT THE SAME TEST. Its human turns carry only the coarse
assistant scheme, and the rollup is lossy in exactly the place that matters:

    Move Verdict  -> Provide Correct Answer                      (1:1, testable)
    Board Report  -> Direct Instruction, together with Local Justification,
                     Worked Line and General Principle           (blended, NOT testable)

So CF can replicate the Solution Request -> Move Verdict link exactly, and can say
nothing clean about Board Report until the fine panel is run over its assistant
turns. Participants here were also not required to use five consultations, so
conversations are shorter and many contribute nothing to a within-conversation
shuffle.
"""
from __future__ import annotations

import json
from collections import Counter
from glob import glob
from pathlib import Path

import numpy as np

N_PERM, SEED = 5000, 0
MIN_OBS = 10          # below this the lift is one or two conversations talking

GAMES = {
    "othello": {
        "root": Path("/data/home/siyanli/agentic_sim/othello_conversations/recordings-download"),
        "convo": "annotated_conversation_poc20260727.jsonl",
        "field": "annotation_assistant_fine",
        # Reported because step 1 found them, not because they were predicted.
        "focus": ("Board Report", "Move Verdict"),
    },
    "connect_four": {
        "root": Path("/data/home/siyanli/agentic_sim/connect_four_conversations/recordings-download"),
        "convo": "annotated_conversation_p15.jsonl",
        "field": "annotation_assistant_fine",
        "focus": ("Board Report", "Move Verdict"),
        # --coarse reproduces the pre-fine-panel run, where Board Report was
        # unrecoverable inside Direct Instruction.
        "coarse_field": "annotation_assistant",
        "coarse_focus": ("Provide Correct Answer", "Direct Instruction"),
    },
}


def load(cfg):
    """[(conversation_id, [participant acts], [assistant acts])] per paired row."""
    rows = []
    for f in sorted(glob(str(cfg["root"] / "*" / cfg["convo"]))):
        cid = Path(f).parent.name
        for line in open(f):
            if not line.strip():
                continue
            t = json.loads(line)
            u, a = t.get("annotation_user"), t.get(cfg["field"])
            if not u or not a:
                continue
            ua = u.get("final") or u.get("consensus") or []
            aa = a.get("final") or a.get("consensus") or []
            if ua and aa:
                rows.append((cid, list(ua), list(aa)))
    return rows


def lift_table(rows, rng):
    """Lift of every (participant act -> assistant act) pair, with a paired-shuffle p.

    The null shuffles WHICH REPLY GOES WITH WHICH MESSAGE inside a conversation, so
    both speakers' own act rates and each conversation's composition are held fixed
    and only the pairing is randomised. A global shuffle would also break the fact
    that some participants simply talk more, and would call that an effect.
    """
    p_acts = sorted({a for _, ua, _ in rows for a in ua})
    a_acts = sorted({a for _, _, aa in rows for a in aa})
    pi = {a: i for i, a in enumerate(p_acts)}
    ai = {a: i for i, a in enumerate(a_acts)}

    def contingency(assign):
        m = np.zeros((len(p_acts), len(a_acts)))
        for (_, ua, _), aa in zip(rows, assign):
            for p in ua:
                for a in aa:
                    m[pi[p], ai[a]] += 1
        return m

    obs = contingency([aa for _, _, aa in rows])

    by_conv = {}
    for i, (cid, _, _) in enumerate(rows):
        by_conv.setdefault(cid, []).append(i)

    ge = np.zeros_like(obs)
    assign = [aa for _, _, aa in rows]
    for _ in range(N_PERM):
        shuf = list(assign)
        for idx in by_conv.values():
            if len(idx) > 1:
                perm = rng.permutation(len(idx))
                for k, j in enumerate(idx):
                    shuf[j] = assign[idx[perm[k]]]
        ge += (contingency(shuf) >= obs)
    p = (ge + 1) / (N_PERM + 1)

    # Expected from each speaker's own marginal rate over the same rows.
    n = len(rows)
    p_rate = np.array([sum(a in ua for _, ua, _ in rows) for a in p_acts]) / n
    a_rate = np.array([sum(a in aa for _, _, aa in rows) for a in a_acts]) / n
    exp = np.outer(p_rate, a_rate) * n
    return p_acts, a_acts, obs, exp, p


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", choices=sorted(GAMES), default="othello")
    ap.add_argument("--coarse", action="store_true",
                    help="use the coarse assistant scheme (connect_four only)")
    ap.add_argument("--exposure", action="append", default=None,
                    help="transpose: show every assistant act THIS participant act "
                         "elicits, instead of the per-target view. Repeatable.")
    args = ap.parse_args()
    cfg = dict(GAMES[args.game])
    if args.coarse:
        if "coarse_field" not in cfg:
            ap.error(f"{args.game} has no coarse layer")
        cfg["field"], cfg["focus"] = cfg["coarse_field"], cfg["coarse_focus"]

    rows = load(cfg)
    by_conv = Counter(c for c, _, _ in rows)
    convs = len(by_conv)
    # A conversation with one paired turn cannot be shuffled against itself, so it
    # contributes to the lift but nothing to the p-value. Under CF's un-forced turn
    # count that is a large share of the sample, and hiding it would overstate power.
    singletons = sum(1 for v in by_conv.values() if v < 2)
    print(f"{args.game}: {len(rows)} paired human turns across {convs} conversations "
          f"({len(rows)/convs:.1f} per conversation)")
    print(f"  {singletons} conversations have a single paired turn -- "
          f"informative for lift, inert under the shuffle\n")

    rng = np.random.default_rng(SEED)
    p_acts, a_acts, obs, exp_m, pv = lift_table(rows, rng)

    for exp in (args.exposure or []):
        if exp not in p_acts:
            print(f"[{exp}] not present in this game's participant labels\n")
            continue
        r = p_acts.index(exp)
        print(f"=== what {exp} elicits (all assistant acts) ===")
        print(f"{'assistant act':<32}{'obs':>6}{'exp':>8}{'lift':>7}{'p':>8}")
        recs = [(aa, obs[r, c], exp_m[r, c], obs[r, c] / exp_m[r, c] if exp_m[r, c] else np.inf,
                 pv[r, c]) for c, aa in enumerate(a_acts) if obs[r, c] >= MIN_OBS]
        for aa, o, e, l, pp in sorted(recs, key=lambda x: -x[3]):
            print(f"{aa:<32}{int(o):>6}{e:>8.1f}{l:>7.2f}{pp:>8.3f} {'*' if pp < .05 else ''}")
        print()
    if args.exposure:
        print("  p is one-sided (lift > 1); a suppression shows as p near 1.0.")
        return

    for target in cfg["focus"]:
        if target not in a_acts:
            continue
        c = a_acts.index(target)
        print(f"=== what elicits {target} ===")
        print(f"{'participant act':<32}{'obs':>6}{'exp':>8}{'lift':>7}{'p':>8}")
        recs = []
        for r, pa in enumerate(p_acts):
            if obs[r, c] < MIN_OBS:
                continue
            recs.append((pa, obs[r, c], exp_m[r, c],
                         obs[r, c] / exp_m[r, c] if exp_m[r, c] else np.inf, pv[r, c]))
        for pa, o, e, l, p in sorted(recs, key=lambda x: -x[3]):
            star = "*" if p < .05 else ""
            print(f"{pa:<32}{int(o):>6}{e:>8.1f}{l:>7.2f}{p:>8.3f} {star}")
        print()

    print("  lift > 1 = the assistant produced this act more often after this participant")
    print("  act than either speaker's own rate would predict. p is a within-conversation")
    print("  pairing shuffle, so it tests the coupling and not either speaker's volume.")
    print(f"  Rows with fewer than {MIN_OBS} co-occurrences are dropped.")


if __name__ == "__main__":
    main()
