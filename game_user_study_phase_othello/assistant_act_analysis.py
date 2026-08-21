"""Othello: assistant (tutor-move) dialogue acts vs the participant's transfer margin,
plus the dynamics between assistant acts and participant acts.

Outcome convention follows `othello_margin_correlations.ipynb`, not the 2-group notebook:
one row per participant, DV = the SIGNED disc margin summed over the two unassisted
rounds, `y = margin(R2) + margin(R3)`, read from scores.csv. No Solvers/Strugglers
split. `r1_margin` (the assisted round) is the control, so a correlation is not just
picking up who was already good at Othello.

Predictors are the assistant's acts in the R1 conversation, measured two ways:
  * SHARE    -- % of a participant's R1 turns whose assistant reply carried the act
  * PRESENCE -- did the assistant ever do it at all (binary; with a median of ~3 turns
                these are close, and presence is what "acts associated with the best
                transfer" means most directly)

Sections:
  0. coding reliability (3-model panel)
  1. prevalence
  2. correlation with the transfer margin -- Pearson + Fisher CI, Spearman, BH-FDR
  3. OLS: does an act survive the R1-margin control, and then the participant's own acts
  4. same-exchange coupling   participant act (t)  -> assistant act (t)
  5. lagged coupling          assistant act (t)    -> participant act (t+1)
  6. direction contrast
  7. assistant act vs participant Think Aloud share

Run:  document_study_og/.venv/bin/python assistant_act_analysis.py
"""

import csv
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

RECORDINGS_BASE = Path("recordings-download")
CONVO = "conversation_poc20260727.jsonl"
ANN = "annotated_conversation_poc20260727.jsonl"
SCORES = "scores.csv"

N_PERM = 20000
SEED = 0
GENAI_OPTS = {"never": 0, "less_than_monthly": 1, "about_monthly": 2, "few_times_month": 3,
              "weekly": 4, "few_times_week": 5, "daily": 6, "several_times_day": 7}
# Same relabels the Othello notebooks apply to the participant side.
USER_ACT_REMAP = {"Correct Answer": "Think Aloud", "Social Coordination Action": "Metacomment",
                  "Forced Choice": "Metacomment"}
REASONING_ACTS = ["Think Aloud"]


def hr(title):
    print(f"\n{'=' * 96}\n{title}\n{'=' * 96}")


# ---------------------------------------------------------------------------
# Outcome: summed transfer margin, and the R1 margin control (from scores.csv)
# ---------------------------------------------------------------------------
def load_margins():
    by_pid = defaultdict(dict)
    for r in csv.DictReader(open(SCORES)):
        by_pid[r["pid"]][r["round"]] = r

    def m(row):
        v = (row or {}).get("margin")
        return float(v) if v not in (None, "") else None

    margin, r1_margin, incomplete = {}, {}, []
    for pid, rd in by_pid.items():
        if "R2" in rd and "R3" in rd:
            m2, m3 = m(rd["R2"]), m(rd["R3"])
            if m2 is not None and m3 is not None:
                margin[pid] = m2 + m3
            else:
                incomplete.append(pid)
        m1 = m(rd.get("R1"))
        if m1 is not None:
            r1_margin[pid] = m1
    return margin, r1_margin, incomplete


# ---------------------------------------------------------------------------
# Long frames: one row per (pid, turn, act) for each speaker
# ---------------------------------------------------------------------------
def normalize_ann(ann):
    if ann is None:
        return None
    if isinstance(ann, dict):
        return ann
    consensus, votes, per_model = ann
    n_valid = sum(v is not None for v in per_model.values())
    return {"final": consensus or list(votes), "votes": votes, "per_model": per_model,
            "confidence": {a: votes[a] / n_valid for a in votes} if n_valid else {},
            "needs_review": (not consensus and bool(votes)) or n_valid < 2}


def load_long(pids):
    urows, arows, records = [], [], []
    for pid in pids:
        f = RECORDINGS_BASE / pid / ANN
        if not f.exists():
            continue
        for i, line in enumerate(open(f)):
            if not line.strip():
                continue
            t = json.loads(line)

            ua = normalize_ann(t.get("annotation_user"))
            uacts = [USER_ACT_REMAP.get(a, a) for a in (ua["final"] if ua else [])]
            for act in (list(dict.fromkeys(uacts)) or [None]):
                urows.append(dict(pid=pid, utt_ind=i, utt_type=act, text=t.get("user")))

            aa = normalize_ann(t.get("annotation_assistant"))
            if aa and aa.get("per_model"):
                records.append(aa["per_model"])
            aacts = list(dict.fromkeys(aa["final"] if aa else []))
            for act in (aacts or [None]):
                arows.append(dict(pid=pid, utt_ind=i, act_type=act, text=t.get("assistant"),
                                  needs_review=bool(aa and aa.get("needs_review"))))
    return pd.DataFrame(urows), pd.DataFrame(arows), records


def _pids_with_user_acts(pids):
    """Participants whose annotated file actually carries participant-side labels."""
    out = []
    for pid in pids:
        f = RECORDINGS_BASE / pid / ANN
        if not f.exists():
            continue
        if any(json.loads(l).get("annotation_user") is not None
               for l in open(f) if l.strip()):
            out.append(pid)
    return out


def turn_share(long_df, act_col, pids):
    """pid x act, 100 * (turns carrying the act) / (all that pid's turns)."""
    tot = long_df.groupby("pid")["utt_ind"].nunique()
    coded = long_df.dropna(subset=[act_col])
    cnt = (coded.groupby(["pid", act_col])["utt_ind"].nunique()
           .unstack(fill_value=0).reindex(pids, fill_value=0))
    return 100 * cnt.div(tot.reindex(pids), axis=0).fillna(0)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------
def fisher_ci(r, n, alpha=0.05):
    if n < 4 or not np.isfinite(r) or abs(r) >= 1:
        return (np.nan, np.nan)
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    lo, hi = z - stats.norm.ppf(1 - alpha / 2) * se, z + stats.norm.ppf(1 - alpha / 2) * se
    return np.tanh(lo), np.tanh(hi)


def bh_fdr(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        k = m - rank
        prev = min(prev, p[idx] * m / k)
        adj[idx] = prev
    return adj


# ---------------------------------------------------------------------------
# 0-1. Reliability and prevalence
# ---------------------------------------------------------------------------
def report_reliability(records, asst_df):
    from dialogue_act_annotation_assistant import fleiss_kappa, ASSISTANT_SCHEME
    from dialogue_act_annotation import mean_pairwise_jaccard

    hr("0. ASSISTANT CODING RELIABILITY (3-model panel, per-act majority vote)")
    macro, by_act = fleiss_kappa(records)
    print(f"items with all 3 raters: {sum(all(v is not None for v in r.values()) for r in records)}"
          f" / {len(records)}")
    print(f"mean pairwise Jaccard : {mean_pairwise_jaccard(records):.3f}")
    print(f"Fleiss' kappa (macro) : {macro:.3f}\n")
    for a in ASSISTANT_SCHEME:
        k = by_act.get(a)
        print(f"  {a:<32}{'   n/a (never used)' if k is None else f'{k:>8.3f}'}")
    flag = asst_df.drop_duplicates(["pid", "utt_ind"])["needs_review"].mean()
    print(f"\n{flag:.0%} of assistant turns flagged needs_review")


def report_prevalence(asst_df, share_a, presence_a):
    hr("1. ASSISTANT ACT PREVALENCE")
    n_turns = asst_df.groupby("pid")["utt_ind"].nunique()
    coded = asst_df.dropna(subset=["act_type"])
    print(f"{asst_df['pid'].nunique()} participants | {int(n_turns.sum())} exchanges | "
          f"{len(coded)} act labels ({len(coded)/max(1,int(n_turns.sum())):.2f} per assistant turn)\n")
    order = share_a.mean().sort_values(ascending=False).index.tolist()
    print(f"{'act':<32}{'mean % of turns':>17}{'% of participants who saw it':>30}")
    for a in order:
        print(f"{a:<32}{share_a[a].mean():>17.1f}{100*presence_a[a].mean():>30.1f}")


# ---------------------------------------------------------------------------
# 2. Correlation with the transfer margin
# ---------------------------------------------------------------------------
def report_margin_corr(P, predictors, label):
    hr(f"2{label[0]}. ASSISTANT ACT {label[1]} vs R2+R3 TRANSFER MARGIN")
    rows = []
    for v in predictors:
        d = P[[v, "y"]].dropna()
        if len(d) < 4 or d[v].std(ddof=1) == 0:
            rows.append(dict(predictor=v, n=len(d)))
            continue
        r, rp = stats.pearsonr(d[v], d["y"])
        rho, rhop = stats.spearmanr(d[v], d["y"])
        lo, hi = fisher_ci(r, len(d))
        rows.append(dict(predictor=v, n=len(d), r=r, lo=lo, hi=hi, p_r=rp,
                         rho=rho, p_rho=rhop))
    T = pd.DataFrame(rows)
    ok = T["p_rho"].notna()
    T.loc[ok, "q_rho"] = bh_fdr(T.loc[ok, "p_rho"].values)
    T = T.sort_values("rho", key=lambda s: s.abs(), ascending=False)

    print(f"BH-FDR over {int(ok.sum())} predictors. Read Spearman: these shares are "
          f"bounded and zero-heavy.\n")
    print(f"  {'predictor':<32}{'n':>4}{'r':>7}{'95% CI':>18}{'p':>8}{'rho':>7}{'p':>8}{'q':>7}")
    for _, x in T.iterrows():
        if pd.isna(x.get("rho")):
            print(f"  {x['predictor']:<32}{int(x['n']):>4}   -- too few / no variance --")
            continue
        star = ("***" if x.p_rho < .001 else "**" if x.p_rho < .01
                else "*" if x.p_rho < .05 else "." if x.p_rho < .10 else "")
        ci = f"[{x.lo:+.2f}, {x.hi:+.2f}]"
        print(f"  {x['predictor']:<32}{int(x['n']):>4}{x.r:>+7.2f}{ci:>18}{x.p_r:>8.3f}"
              f"{x.rho:>+7.2f}{x.p_rho:>8.3f}{x.q_rho:>7.3f} {star}")
    return T


# ---------------------------------------------------------------------------
# 2c. Other continuous parameterizations of "how much the assistant did act X"
# ---------------------------------------------------------------------------
def report_alt_scales(P, asst_df, acts, pids):
    """`acts per turn` is ALREADY section 2a: the panel returns a SET per turn, so an
    act is labelled at most once per turn and (count of act) / (turns) is identical to
    the turn share. The parameterizations that actually differ are:

      label-share  count of act / that participant's TOTAL act labels
                   -- the CF notebook's old "A" spec. Denominator is labels, so it is
                   deflated by however many other acts a turn happened to earn; the
                   acts are also forced to trade off against each other (shares sum to 1).
      density      total act labels / turns, one scalar per participant -- how much the
                   assistant did overall, not which act.
      raw count    count of act, unnormalized. Confounded with conversation length, so
                   it is reported with `turns` controlled in an OLS rather than raw.
    """
    hr("2c. OTHER CONTINUOUS SCALES (acts/turn is already 2a -- see docstring)")
    turns = asst_df.groupby("pid")["utt_ind"].nunique().reindex(pids)
    coded = asst_df.dropna(subset=["act_type"])
    cnt = (coded.groupby(["pid", "act_type"]).size().unstack(fill_value=0)
           .reindex(pids, fill_value=0))
    total_labels = cnt.sum(axis=1)

    Q = pd.DataFrame(index=pids)
    Q["y"] = P["y"]
    Q["turns"] = turns
    Q["density"] = total_labels / turns
    for a in acts:
        Q["LS_" + a] = 100 * cnt[a] / total_labels.replace(0, np.nan)
        Q["N_" + a] = cnt[a]

    d = Q[["density", "turns", "y"]].dropna()
    for v in ["density", "turns"]:
        rho, p = stats.spearmanr(d[v], d["y"])
        r, pr = stats.pearsonr(d[v], d["y"])
        print(f"  {v:<32} n={len(d)}  r={r:+.2f} (p={pr:.3f})  rho={rho:+.2f} (p={p:.3f})")

    print(f"\n  label-share (act / all act labels), and raw count with `turns` controlled:")
    print(f"  {'act':<32}{'LS rho':>8}{'p':>8}{'q':>7}   {'N b|turns':>10}{'p':>8}")
    rows = []
    for a in acts:
        dd = Q[["LS_" + a, "y"]].dropna()
        if len(dd) < 4 or dd["LS_" + a].std(ddof=1) == 0:
            continue
        rho, p = stats.spearmanr(dd["LS_" + a], dd["y"])
        sub = Q[["N_" + a, "turns", "y"]].dropna().rename(columns={"N_" + a: "n_act"})
        f = smf.ols("y ~ n_act + turns", sub).fit()
        rows.append((a, rho, p, f.params["n_act"], f.pvalues["n_act"]))
    q = bh_fdr([r[2] for r in rows])
    rows = [r + (qq,) for r, qq in zip(rows, q)]
    rows.sort(key=lambda r: abs(r[1]), reverse=True)
    for a, rho, p, b, pb, qq in rows:
        star = "*" if qq < .05 else ("." if p < .05 else " ")
        print(f"  {a:<32}{rho:>+8.2f}{p:>8.3f}{qq:>7.3f}   {b:>+10.3f}{pb:>8.3f} {star}")
    print("\n  BH-FDR over the label-share column only.")
    return Q


# ---------------------------------------------------------------------------
# 3. OLS with controls
# ---------------------------------------------------------------------------
def report_ols(P, acts):
    hr("3. OLS: does an assistant act survive the R1-margin control, then the "
       "participant's own acts?")
    M = P.rename(columns={a: "a_" + a.replace(" ", "_") for a in acts})
    M = M.rename(columns={"Think Aloud_u": "u_think_aloud",
                          "Solution Request_u": "u_solution_req"})
    have = [c for c in ["u_think_aloud", "u_solution_req", "r1_margin"] if c in M.columns]
    base = M.dropna(subset=["y"] + have)
    print(f"complete-case n={len(base)}   controls available: {', '.join(have)}\n")
    print(f"  {'act':<32}{'b alone':>9}{'p':>8}{'+R1mgn':>9}{'p':>8}"
          f"{'+partic.acts':>14}{'p':>8}")
    rows = []
    for a in acts:
        v = "a_" + a.replace(" ", "_")
        if base[v].std(ddof=1) == 0:
            continue
        f1 = smf.ols(f"y ~ {v}", base).fit()
        f2 = smf.ols(f"y ~ {v} + r1_margin", base).fit()
        rhs3 = f"{v} + r1_margin + " + " + ".join(c for c in have if c != "r1_margin")
        f3 = smf.ols(f"y ~ {rhs3}", base).fit()
        rows.append((a, f1.params[v], f1.pvalues[v], f2.params[v], f2.pvalues[v],
                     f3.params[v], f3.pvalues[v]))
    rows.sort(key=lambda r: r[2])
    for a, b1, p1, b2, p2, b3, p3 in rows:
        star = " *" if p3 < .05 else ("  ." if p3 < .10 else "")
        print(f"  {a:<32}{b1:>+9.3f}{p1:>8.3f}{b2:>+9.3f}{p2:>8.3f}"
              f"{b3:>+14.3f}{p3:>8.3f}{star}")
    print("\n  b = discs of R2+R3 margin per 1 percentage point of the act's turn share.")
    print("  Uncorrected p; with 10 acts treat as descriptive.")
    return rows


# ---------------------------------------------------------------------------
# 4-6. Dynamics (same machinery as the Connect Four phase)
# ---------------------------------------------------------------------------
def turn_sets(user_df, asst_df):
    u = (user_df.dropna(subset=["utt_type"]).groupby(["pid", "utt_ind"])["utt_type"]
         .apply(set).to_dict())
    a = (asst_df.dropna(subset=["act_type"]).groupby(["pid", "utt_ind"])["act_type"]
         .apply(set).to_dict())
    idx = sorted(set(user_df.groupby(["pid", "utt_ind"]).groups) |
                 set(asst_df.groupby(["pid", "utt_ind"]).groups))
    seqs = {}
    for pid, i in idx:
        seqs.setdefault(pid, []).append((u.get((pid, i), set()), a.get((pid, i), set())))
    return seqs


def coupling(seqs, lag, user_acts, asst_acts, n_perm=2000, seed=SEED):
    """lag=0: user act (t) -> assistant act (t), i.e. how the assistant RESPONDS.
    lag=1: assistant act (t) -> user act (t+1), i.e. how it SHAPES the next turn.
    Null shuffles the responding speaker's label sets WITHIN participant, holding
    each person's own act rates fixed."""
    if lag == 0:
        by_pid = {p: list(s) for p, s in seqs.items()}
        rows, cols = user_acts, asst_acts
    else:
        by_pid = {p: [(s[i][1], s[i + 1][0]) for i in range(len(s) - 1)]
                  for p, s in seqs.items()}
        rows, cols = asst_acts, user_acts
    by_pid = {p: v for p, v in by_pid.items() if v}

    pairs, blocks, start = [], [], 0
    for p, v in by_pid.items():
        pairs += v
        blocks.append((start, start + len(v)))
        start += len(v)
    n = len(pairs)
    C = np.array([[r in cue for r in rows] for cue, _ in pairs], float)
    R = np.array([[c in resp for c in cols] for _, resp in pairs], float)
    obs = C.T @ R

    rng = np.random.default_rng(seed)
    null = np.empty((n_perm, len(rows), len(cols)))
    idx = np.arange(n)
    for b in range(n_perm):
        perm = idx.copy()
        for a, z in blocks:
            if z - a > 1:
                perm[a:z] = a + rng.permutation(z - a)
        null[b] = C.T @ R[perm]

    exp = null.mean(axis=0)
    p = (np.minimum((null >= obs[None]).sum(0), (null <= obs[None]).sum(0)) + 1) / (n_perm + 1)
    return dict(rows=rows, cols=cols, obs=obs, exp=exp, p=np.minimum(1.0, 2 * p), n=n,
                n_shufflable=sum(z - a for a, z in blocks if z - a > 1))


def report_coupling(res, title, cue_label, resp_label, top=14):
    hr(title)
    rows, cols, obs, exp, p = res["rows"], res["cols"], res["obs"], res["exp"], res["p"]
    print(f"{res['n']} ordered pairs ({res['n_shufflable']} shufflable). exp = mean count "
          f"under within-participant shuffling of the {resp_label.lower()} labels.\n")
    recs = []
    for ri, r in enumerate(rows):
        for ci, c in enumerate(cols):
            if obs[ri, ci] + exp[ri, ci] < 3:
                continue
            lift = obs[ri, ci] / exp[ri, ci] if exp[ri, ci] > 0 else np.inf
            recs.append([r, c, obs[ri, ci], exp[ri, ci], lift, p[ri, ci]])
    if not recs:
        print("  (no cell with enough mass to test)")
        return []
    q = bh_fdr([x[5] for x in recs])
    recs = [x + [qq] for x, qq in zip(recs, q)]
    recs.sort(key=lambda x: x[5])
    print(f"{cue_label:<30}{resp_label:<32}{'obs':>5}{'exp':>7}{'lift':>7}{'p':>8}{'q':>7}")
    for r, c, o, e, lift, pp, qq in recs[:top]:
        star = "*" if qq < 0.05 else ("." if pp < 0.05 else " ")
        print(f"{r:<30}{c:<32}{int(o):>5}{e:>7.1f}{lift:>7.2f}{pp:>8.3f}{qq:>7.3f} {star}")
    print("\n  lift > 1 = pairing more often than chance given both speakers' own rates.")
    return recs


def report_direction(seqs, user_acts, asst_acts):
    hr("6. WHICH DIRECTION IS THE STRONGER DEPENDENCY?")
    for lag, name in [(0, "user(t) -> assistant(t)   [assistant reacting]"),
                      (1, "assistant(t) -> user(t+1) [assistant steering]")]:
        res = coupling(seqs, lag, user_acts, asst_acts, n_perm=1000, seed=SEED)
        obs, exp = res["obs"], res["exp"]
        mask = (obs + exp) >= 3
        chi = np.sum((obs[mask] - exp[mask]) ** 2 / np.maximum(exp[mask], 1e-9))
        cells = int(mask.sum())
        print(f"{name:<44} mean chi2 per tested cell = {chi/max(cells,1):>6.2f} "
              f"({cells} cells, {res['n']} pairs)")
    print("\n  Same statistic and null in both rows, so they are comparable; descriptive, "
          "not a test of one direction against the other.")


def report_vs_think_aloud(P, acts, ta="Think Aloud_u"):
    hr("7. ASSISTANT ACT SHARE vs PARTICIPANT THINK ALOUD SHARE")
    if ta not in P.columns:
        print("  (participant Think Aloud not present)")
        return
    rows = []
    for a in acts:
        d = P[[a, ta]].dropna()
        if d[a].std(ddof=1) == 0:
            continue
        r, p = stats.pearsonr(d[a], d[ta])
        rows.append((a, r, p, len(d)))
    q = bh_fdr([r[2] for r in rows])
    rows = sorted(zip(rows, q), key=lambda t: t[0][2])
    print(f"  {'act':<32}{'r':>8}{'p':>9}{'n':>6}")
    for (a, r, p, n), qq in rows:
        star = "*" if qq < .05 else ("." if p < .05 else " ")
        print(f"  {a:<32}{r:>+8.3f}{p:>9.4f}{n:>6} {star}")
    print("\n  Same-round and correlational: cannot separate 'the assistant elicited it' "
          "from 'the participant invited it'.")


# ---------------------------------------------------------------------------
def main():
    margin, r1_margin, incomplete = load_margins()
    pids_with_convo = sorted(p.parent.name for p in RECORDINGS_BASE.glob(f"*/{ANN}"))
    user_df, asst_df, records = load_long(pids_with_convo)
    have = sorted(asst_df["pid"].unique())

    print(f"{len(have)} participants with an annotated R1 conversation")
    print(f"DV present for {len(margin)} participants "
          f"({len(incomplete)} had an R2/R3 round end at the buzzer with no final margin)")

    report_reliability(records, asst_df)

    share_a = turn_share(asst_df, "act_type", have)
    share_u = turn_share(user_df, "utt_type", have)
    presence_a = (share_a > 0).astype(float)
    report_prevalence(asst_df, share_a, presence_a)

    acts = list(share_a.columns)
    P = pd.DataFrame(index=share_a.index)
    P["y"] = [margin.get(p, np.nan) for p in P.index]
    P["r1_margin"] = [r1_margin.get(p, np.nan) for p in P.index]
    for a in acts:
        P[a] = share_a[a]
        P[a + " (present)"] = presence_a[a]
    for a in ["Think Aloud", "Solution Request"]:
        if a in share_u.columns:
            P[a + "_u"] = share_u[a]
    y_n = P["y"].notna().sum()
    print(f"\nusable rows (act frame ∩ DV): {y_n}")

    report_margin_corr(P, acts, ("a", "SHARE = acts/turns (% of turns)"))
    report_margin_corr(P, [a + " (present)" for a in acts], ("b", "PRESENCE (ever/never)"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Q = report_alt_scales(P, asst_df, acts, have)
        report_ols(P, acts)
    Q.to_csv("assistant_alt_scales.csv")

    # Sections 4-7 compare the two speakers, so they need BOTH codings. The driver
    # annotated every assistant turn, but the notebook's participant pass covered
    # fewer participants; the rest would enter as turns with an empty user act set,
    # diluting the counts and the null. Restrict rather than silently include them.
    both = sorted(_pids_with_user_acts(have))
    dropped = len(have) - len(both)
    print(f"\ndynamics sections use the {len(both)} participants with BOTH codings "
          f"({dropped} have assistant acts but no participant acts)")
    u_both = user_df[user_df.pid.isin(both)]
    a_both = asst_df[asst_df.pid.isin(both)]
    share_u_both = turn_share(u_both, "utt_type", both)

    seqs = turn_sets(u_both, a_both)
    r0 = coupling(seqs, 0, list(share_u_both.columns), acts)
    report_coupling(r0, "4. SAME EXCHANGE: participant act -> assistant reply",
                    "participant act (t)", "assistant act (t)")
    r1 = coupling(seqs, 1, list(share_u_both.columns), acts)
    report_coupling(r1, "5. NEXT TURN: assistant reply -> what the participant did next",
                    "assistant act (t)", "participant act (t+1)")
    report_direction(seqs, list(share_u_both.columns), acts)
    report_vs_think_aloud(P.loc[both], acts)

    asst_df.to_csv("assistant_utterances.csv", index=False)
    share_a.to_csv("assistant_act_turn_shares.csv")
    P.to_csv("assistant_margin_frame.csv")
    print("\nwrote assistant_utterances.csv, assistant_act_turn_shares.csv, "
          "assistant_margin_frame.csv")


if __name__ == "__main__":
    main()
