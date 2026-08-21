"""Elicitation odds ratios over EVERY Othello conversation we have.

The question: which participant acts summon which assistant acts. One logit per assistant
act on all six participant acts simultaneously, standard errors clustered by participant.
The adjustment matters -- participant acts co-occur within a turn, so an unadjusted lift
for one act partly reflects the others.

Scope: all five Othello datasets that have conversations (1274 assistant turns). The two
baseline studies are absent because R1 was unassisted there -- no conversations exist.

Scheme follows bad_user_sim/elicitation_analysis.ipynb exactly, so numbers sit on one
scale with the existing analysis:
  * the three affective codes collapse to one `Feedback`
  * `Comprehension Gauging Question` and `Paraphrase` are dropped
  * both applied to EACH SEAT before the majority re-vote, not to `final` after it, so two
    seats that disagree on the flavour of feedback still agree that feedback occurred
  * six assistant codes clear kappa .6 and may carry a claim; `Hint` (.38) and `Feedback`
    (.47) stay in the tables marked, so their absence is visible rather than silent

These associations are OBSERVATIONAL. The participant's act is not randomised, so an
elicitation association is not a causal effect -- and forcing an act does not reproduce it
(in the simulator's gate-require_kdq arm the act was classifier-enforced and the link it was
supposed to carry collapsed).

    from elicitation import load_turns, or_table
    T = load_turns()                      # one row per assistant turn
    or_table(T, "ALL OTHELLO")            # pooled
    or_table(T[T.source == "phase2"], …)  # or any subset
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

CITRUS = Path(__file__).resolve().parent.parent
AI = "poc20260727"
ANN = f"annotated_conversation_{AI}.jsonl"
FA, FP = "annotation_assistant_fine", "annotation_user"

PACTS = ["Knowledge Deficit Question", "Solution Request", "Think Aloud",
         "Common Ground Question", "Conversational Acknowledgment", "Metacomment"]
SHORT = {"Knowledge Deficit Question": "KDQ", "Solution Request": "SolReq",
         "Think Aloud": "ThinkAloud", "Common Ground Question": "CGQ",
         "Conversational Acknowledgment": "ConvAck", "Metacomment": "Metacomment"}
FOCUS = ["Think Aloud", "Solution Request", "Knowledge Deficit Question"]
ALL_A = ["Board Report", "Move Verdict", "General Principle", "Local Justification",
         "Worked Line", "Prompt", "Hint", "Feedback"]
RELIABLE = {"Move Verdict", "Prompt", "Board Report", "General Principle",
            "Worked Line", "Local Justification"}

# |log OR| above this is treated as logistic separation rather than an estimate
# (exp(5) ~ 148, exp(-5) ~ 0.007 -- far outside anything this data can support).
SEPARATION_LOGIT = 5.0

SEATS = ["gpt", "llama", "sonnet"]
COLLAPSE = {"Positive Feedback": "Feedback", "Negative Feedback": "Feedback",
            "Neutral Feedback": "Feedback"}
DROP = {"Comprehension Gauging Question", "Paraphrase"}

# Which population each source recruited, from the condition map in
# scores_adopter_non_adopter.ipynb. phase_othello is the ADOPTERS study; every other
# source with conversations is non-adopters (phase 2 screens for non-adoption, and the
# iteration deployments inherit that screen). The two baseline studies have no
# conversations at all, so they cannot appear here.
POPULATION = {
    "phase1_othello": "adopters",
    "phase2":         "non-adopters",
    "johnny":         "non-adopters",
    "modelled_ta":    "non-adopters",
    "esr":            "non-adopters",
    "3arm":           "non-adopters",
}

# source -> recordings directory. Add a row here when a new deployment lands.
SOURCES = {
    "phase1_othello": "game_user_study_phase_othello/recordings-download",
    "phase2":         "game_user_study_phase_2/recordings-download",
    "johnny":         "game_user_study_intervention_iter/recordings-download",
    "modelled_ta":    "game_user_study_intervention_iter/recordings-modelled-ta",
    "esr":            "game_user_study_intervention_iter/recordings-esr",
    "3arm":           "game_user_study_intervention_iter/recordings-3arm",
}


def remap(labels):
    return {COLLAPSE.get(c, c) for c in labels if c not in DROP}


def revote(rec, min_votes=2):
    """Re-derive the panel label under the revised scheme; fall back to stored `final`."""
    per = {k: v for k, v in (rec.get("per_model") or {}).items() if v is not None}
    if len(per) < len(SEATS):
        return remap(rec.get("final") or []), len(per)
    cnt = Counter()
    for s in SEATS:
        cnt.update(remap(per[s]))
    return {c for c, n in cnt.items() if n >= min_votes}, len(per)


def _condition(d: Path, source: str) -> str:
    """Arm/form label where one exists, so a subset can be taken without a join."""
    iv = d / "intervention.json"
    if source == "phase2":
        if not iv.is_file():
            return "vanilla" if (d / "filler.json").is_file() else "unknown"
        return json.load(open(iv)).get("variant") or "unknown"
    if source == "3arm":
        # One directory, three arms: vanilla has no intervention.json.
        if not iv.is_file():
            return "3arm_vanilla"
        j = json.load(open(iv))
        return "3arm_" + (j.get("variant") or j.get("form") or "unknown")
    if iv.is_file():
        return json.load(open(iv)).get("form") or source
    return source


def load_turns(sources=None) -> pd.DataFrame:
    """One row per ASSISTANT turn carrying both annotation layers. Cluster = pid."""
    rows = []
    for source, rel in (sources or SOURCES).items():
        root = CITRUS / rel
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            f = d / ANN
            if not d.is_dir() or not f.exists():
                continue
            cond = _condition(d, source)
            for line in open(f):
                if not line.strip():
                    continue
                t = json.loads(line)
                if FA not in t or not (t.get("assistant") or "").strip():
                    continue
                asst, nseat = revote(t[FA])
                p = set((t.get(FP) or {}).get("final") or [])
                r = {"source": source, "population": POPULATION.get(source, "unknown"),
                     "condition": cond, "pid": d.name,
                     "asst": asst, "n_seats": nseat,
                     "asst_words": len(t["assistant"].split()),
                     "has_user_layer": bool(t.get(FP))}
                for a in PACTS:
                    r[SHORT[a]] = int(a in p)
                rows.append(r)
    return pd.DataFrame(rows)


def prevalence(T: pd.DataFrame) -> pd.DataFrame:
    """Assistant-act share of turns, by source."""
    out = {}
    for src, g in T.groupby("source"):
        col = {a: 100.0 * g.asst.apply(lambda s, t=a: t in s).mean() for a in ALL_A}
        col["turns"] = len(g)
        col["participants"] = g.pid.nunique()
        out[src] = col
    col = {a: 100.0 * T.asst.apply(lambda s, t=a: t in s).mean() for a in ALL_A}
    col["turns"], col["participants"] = len(T), T.pid.nunique()
    out["ALL"] = col
    return pd.DataFrame(out).T[["participants", "turns"] + ALL_A].round(1)


def or_table(T: pd.DataFrame, label: str, focus=None, source_fe=False, bh=True):
    """Print the OR table and return {(assistant_act, participant_act): (OR, p)}.

    source_fe adds source dummies, so a pooled estimate is not driven by between-study
    differences in how often each act occurs.
    """
    focus = focus or FOCUS
    print("=" * (26 + 22 * len(focus)))
    print(f"{label} — odds ratio for each assistant act, adjusted for all participant acts"
          + ("  [+ source fixed effects]" if source_fe else ""))
    print(f"  {T.pid.nunique()} participants, {len(T)} assistant turns")
    print("=" * (26 + 22 * len(focus)))
    print(f"{'assistant act':<22}" + "".join(f"{SHORT[a]:>22}" for a in focus)
          + f"{'kappa>=.6':>11}")
    X0 = T[[SHORT[p] for p in PACTS]].astype(float)
    if source_fe and T.source.nunique() > 1:
        X0 = pd.concat([X0, pd.get_dummies(T.source, prefix="src", drop_first=True)
                        .astype(float)], axis=1)
    res, pvals = {}, []
    for aa in ALL_A:
        y = T.asst.apply(lambda s, t=aa: int(t in s))
        if y.nunique() < 2:
            print(f"{aa:<22}  (no variation)")
            continue
        try:
            m = sm.Logit(y, sm.add_constant(X0)).fit(disp=0, cov_type="cluster",
                                                     cov_kwds={"groups": T.pid})
        except Exception as e:
            print(f"{aa:<22}  (fit failed: {type(e).__name__})")
            continue
        converged = bool((m.mle_retvals or {}).get("converged", True))
        line = f"{aa:<22}"
        for pa in focus:
            k = SHORT[pa]
            b, pv = float(m.params[k]), float(m.pvalues[k])
            # SEPARATION GUARD. In a 63-turn subset an act can perfectly predict the
            # outcome, and the MLE then runs off to +/-inf -- statsmodels reports odds
            # ratios like 1e18, which are an artefact of the fit, not an estimate. Those
            # cells are reported as "sep" and stored as NaN so nothing downstream averages
            # them in.
            if (not converged) or abs(b) > SEPARATION_LOGIT:
                res[(aa, k)] = (float("nan"), float("nan"))
                line += f"{'sep':>17} {'':<4}"
                continue
            o = float(np.exp(b))
            res[(aa, k)] = (o, pv)
            pvals.append(pv)
            star = "***" if pv < .001 else "**" if pv < .01 else "*" if pv < .05 else ""
            line += f"{o:>17.2f} {star:<4}"
        print(line + f"{'yes' if aa in RELIABLE else 'NO':>11}")
    n_sep = sum(1 for v in res.values() if np.isnan(v[0]))
    if n_sep:
        print(f"\n  {n_sep} cell(s) marked 'sep': the fit separated, so no odds ratio is "
              f"estimable. Expected in the small batches; not a result either way.")
    if bh and pvals:
        p = np.sort(np.array(pvals))
        q = np.minimum.accumulate((p * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
        print(f"\n  {len(pvals)} tests in this table; Benjamini-Hochberg q for the smallest "
              f"p is {q[0]:.4f}; {int((q < .05).sum())} cells survive q<.05")
    print()
    return res
