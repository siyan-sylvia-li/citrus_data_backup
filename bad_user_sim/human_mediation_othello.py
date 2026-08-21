"""Does a participant act help via the assistant, or on its own?

Two mechanisms produce the same participant-side correlation with transfer:

  MEDIATED  participant act -> assistant produces a helpful act -> transfer
  DIRECT    participant act -> the participant thinks harder -> transfer
            (self-explanation; the assistant is incidental)

The elicitation table (human_elicitation_othello.py) already hints they differ by
act: Knowledge Deficit Question pulls Board Report hard (lift 1.51, p = .002),
while Think Aloud barely does (1.13, p = .076) and instead suppresses Move Verdict
(0.61). If Think Aloud still predicts transfer once the assistant's act shares are
in the model, its benefit is not something the assistant handed over.

Everything is per participant on the human Othello data: act shares over that
person's own turns, outcome = final_margin summed over the two solo puzzles. No
LLM calls -- both annotation layers are already on the rows.

    python human_mediation_othello.py
"""
from __future__ import annotations

import json
from glob import glob
from pathlib import Path

import numpy as np
from scipy import stats

HUMAN = Path("/data/home/siyanli/agentic_sim/othello_conversations/recordings-download")
CONVO = "annotated_conversation_poc20260727.jsonl"
SOLO = ["pb220260706", "pbg20260726"]
EXPOSURES = ("Think Aloud", "Knowledge Deficit Question", "Solution Request")
MEDIATORS = ("Board Report", "Move Verdict")
N_BOOT, SEED = 5000, 0


def margin(d):
    total = 0.0
    for pz in SOLO:
        f = d / f"oth_score_{pz}.json"
        if not f.exists():
            return None
        v = json.load(open(f)).get("final_margin")
        if v is None:
            return None
        total += v
    return total


def load():
    """Per participant: participant-act shares, assistant-act shares, transfer margin."""
    recs = []
    for f in sorted(glob(str(HUMAN / "*" / CONVO))):
        d = Path(f).parent
        y = margin(d)
        if y is None:
            continue
        u_turns, a_turns = [], []
        for line in open(f):
            if not line.strip():
                continue
            t = json.loads(line)
            u, a = t.get("annotation_user"), t.get("annotation_assistant_fine")
            if u and (u.get("final") or u.get("consensus")):
                u_turns.append(set(u.get("final") or u.get("consensus")))
            if a and (a.get("final") or a.get("consensus")):
                a_turns.append(set(a.get("final") or a.get("consensus")))
        if not u_turns or not a_turns:
            continue
        recs.append({
            "y": y,
            "n_turns": len(u_turns),
            **{f"U:{k}": sum(k in s for s in u_turns) / len(u_turns) for k in EXPOSURES},
            **{f"A:{k}": sum(k in s for s in a_turns) / len(a_turns) for k in MEDIATORS},
        })
    return recs


def z(x):
    x = np.asarray(x, float)
    s = x.std(ddof=1)
    return (x - x.mean()) / s if s > 0 else x * 0.0


def ols(y, X):
    """Standardised coefficients; X columns are already z-scored, intercept added."""
    A = np.column_stack([np.ones(len(y))] + list(X))
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta[1:]


def main():
    recs = load()
    y = z([r["y"] for r in recs])
    turns = z([r["n_turns"] for r in recs])
    print(f"{len(recs)} human participants with both solo scores and both annotation layers")
    print(f"outcome = summed solo final_margin (raw mean {np.mean([r['y'] for r in recs]):+.1f}, "
          f"sd {np.std([r['y'] for r in recs], ddof=1):.1f})\n")

    med = {m: z([r[f"A:{m}"] for r in recs]) for m in MEDIATORS}
    for m in MEDIATORS:
        rho, p = stats.spearmanr([r[f"A:{m}"] for r in recs], [r["y"] for r in recs])
        print(f"  mediator {m:<14} share vs margin: rho {rho:+.3f}  p {p:.3f}")
    print()

    rng = np.random.default_rng(SEED)
    for e in EXPOSURES:
        x = z([r[f"U:{e}"] for r in recs])
        rho, p = stats.spearmanr([r[f"U:{e}"] for r in recs], [r["y"] for r in recs])
        # Turn count is in both models: someone who talks more has more chances to
        # show any act AND more assistant replies, which would fake a mediation.
        c = ols(y, [x, turns])[0]
        full = ols(y, [x, turns] + [med[m] for m in MEDIATORS])
        c_prime = full[0]
        indirect = c - c_prime

        boot = []
        idx = np.arange(len(y))
        for _ in range(N_BOOT):
            b = rng.choice(idx, len(idx), replace=True)
            try:
                cb = ols(y[b], [x[b], turns[b]])[0]
                cpb = ols(y[b], [x[b], turns[b]] + [med[m][b] for m in MEDIATORS])[0]
                boot.append(cb - cpb)
            except np.linalg.LinAlgError:
                continue
        lo, hi = np.percentile(boot, [2.5, 97.5])

        share = 100 * indirect / c if abs(c) > 1e-9 else float("nan")
        print(f"=== {e} ===")
        print(f"  raw association with margin      rho {rho:+.3f}  p {p:.3f}")
        print(f"  total effect        (c)          {c:+.3f}")
        print(f"  direct, mediators in (c')        {c_prime:+.3f}")
        print(f"  indirect via {'/'.join(MEDIATORS)}   {indirect:+.3f}  "
              f"95% CI [{lo:+.3f}, {hi:+.3f}]{'  *' if lo * hi > 0 else ''}")
        print(f"  -> {share:.0f}% of the association runs through the assistant's acts\n")

    print("  A CI excluding zero means the assistant's act mix carries part of it.")
    print("  A c' that stays large with a null indirect path means the participant is")
    print("  getting the benefit directly -- self-explanation, not tutoring received.")
    print("  Correlational: nobody was assigned to think aloud. The gate arms are the")
    print("  experiment; this says which mechanism they should be testing for.")


if __name__ == "__main__":
    main()
