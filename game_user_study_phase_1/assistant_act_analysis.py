"""Assistant (tutor-move) dialogue acts: effect on transfer, and their dynamics with
the participant's (student-move) acts.

Mirrors the participant-side analysis in `data_analysis_2group.ipynb`:
  * same population and Solvers/Strugglers split (R2 flat_score >= 2 vs <= 1),
  * same TURN-share metric (fraction of a participant's R1 turns containing an act),
  * same permutation / Mann-Whitney / Cohen's d battery,
  * same ordinal-logit + AIC/BIC model comparison on the R2 optimal-move count.

Then four dynamics analyses that the participant-only view can't give:
  A. same-exchange coupling      user act (t)      -> assistant act (t)
  B. lagged coupling             assistant act (t) -> user act (t+1)
  C. direction contrast          which of A / B is the stronger dependency
  D. does the assistant act still predict transfer once Think Aloud is controlled

Run:  document_study_og/.venv/bin/python assistant_act_analysis.py
"""

import itertools
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel

RECORDINGS_BASE = Path("recordings-download")
SCORE_R1 = "cf_score_p15.json"
SCORE_R2 = "cf_score_selectivity_pw4p6.json"
ANN = "annotated_conversation_p15.jsonl"

N_PERM = 20000
SEED = 0
GENAI_OPTS = {"never": 0, "less_than_monthly": 1, "about_monthly": 2, "few_times_month": 3,
              "weekly": 4, "few_times_week": 5, "daily": 6, "several_times_day": 7}
# Same manual relabels the notebook applies to the participant side.
USER_ACT_REMAP = {"Correct Answer": "Think Aloud", "Social Coordination Action": "Metacomment",
                  "Forced Choice": "Metacomment"}


def hr(title):
    print(f"\n{'=' * 92}\n{title}\n{'=' * 92}")


# ---------------------------------------------------------------------------
# Population: identical to notebook cells 3 + 8
# ---------------------------------------------------------------------------
def build_population():
    r1 = {p.parent.name for p in RECORDINGS_BASE.glob(f"*/{SCORE_R1}")}
    r2 = {p.parent.name for p in RECORDINGS_BASE.glob(f"*/{SCORE_R2}")}
    cands = r1 & r2
    cands.discard("p789")
    cands -= {p for p in cands if "test" in p.lower()}

    good, bad = [], []
    for pid in sorted(cands):
        d = RECORDINGS_BASE / pid
        if not all((d / f).exists() for f in ("post_survey_pw4p6.json", "demographics.json",
                                              "conversation_p15.jsonl")):
            continue
        try:
            s = json.load(open(d / SCORE_R2))["flat_score"]
            if s >= 2:
                good.append(pid)
            elif s <= 1:
                bad.append(pid)
        except (TypeError, KeyError):
            continue
    return good, bad


# ---------------------------------------------------------------------------
# Long frames: one row per (pid, turn, act) for each speaker
# ---------------------------------------------------------------------------
def normalize_ann(ann):
    """Canonicalize a stored annotation (current dict form or legacy list form)."""
    if ann is None:
        return None
    if isinstance(ann, dict):
        return ann
    consensus, votes, per_model = ann
    n_valid = sum(v is not None for v in per_model.values())
    return {"final": consensus or list(votes), "votes": votes, "per_model": per_model,
            "confidence": {a: votes[a] / n_valid for a in votes} if n_valid else {},
            "needs_review": (not consensus and bool(votes)) or n_valid < 2}


def load_long(pids, group_of):
    """Returns (user_df, asst_df, per_model_records). Both frames carry one row per
    (pid, utt_ind, act); a turn with no act contributes a single act=None row so the
    turn still counts in the denominator."""
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
            uacts = list(dict.fromkeys(uacts))
            for act in (uacts or [None]):
                urows.append(dict(pid=pid, utt_ind=i, utt_type=act, text=t.get("user"),
                                  needs_review=bool(ua and ua.get("needs_review")),
                                  outcome=group_of.get(pid)))

            aa = normalize_ann(t.get("annotation_assistant"))
            if aa and aa.get("per_model"):
                records.append(aa["per_model"])
            aacts = list(dict.fromkeys(aa["final"] if aa else []))
            for act in (aacts or [None]):
                arows.append(dict(pid=pid, utt_ind=i, act_type=act, text=t.get("assistant"),
                                  needs_review=bool(aa and aa.get("needs_review")),
                                  outcome=group_of.get(pid)))
    return pd.DataFrame(urows), pd.DataFrame(arows), records


def turn_share(long_df, act_col, pids):
    """pid x act table of 100 * (turns containing the act) / (turns). Rows for every pid."""
    tot = long_df.groupby("pid")["utt_ind"].nunique()
    coded = long_df.dropna(subset=[act_col])
    cnt = (coded.drop_duplicates(["pid", "utt_ind", act_col])
           .groupby(["pid", act_col]).size().unstack(fill_value=0)
           .reindex(pids, fill_value=0))
    return 100 * cnt.div(tot.reindex(pids), axis=0).fillna(0)


# ---------------------------------------------------------------------------
# Stats helpers (same as the notebook's cell 33)
# ---------------------------------------------------------------------------
def cohend(x, z):
    nx, nz = len(x), len(z)
    sp = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (nz - 1) * np.var(z, ddof=1)) / (nx + nz - 2))
    return (np.mean(x) - np.mean(z)) / sp if sp > 0 else np.nan


def perm_test(x, z, n_perm=N_PERM, seed=SEED):
    """Two-sided label-permutation test on the difference in group means."""
    obs = np.mean(x) - np.mean(z)
    pooled = np.concatenate([x, z])
    nx = len(x)
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        if abs(perm[:nx].mean() - perm[nx:].mean()) >= abs(obs) - 1e-12:
            ge += 1
    return obs, (ge + 1) / (n_perm + 1)


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values."""
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
# 0. Coding reliability
# ---------------------------------------------------------------------------
def report_reliability(records, asst_df):
    from dialogue_act_annotation_assistant import fleiss_kappa, ASSISTANT_SCHEME
    from dialogue_act_annotation import mean_pairwise_jaccard

    hr("0. ASSISTANT CODING RELIABILITY (3-model panel, per-act majority vote)")
    macro, by_act = fleiss_kappa(records)
    print(f"items with all 3 raters: {sum(all(v is not None for v in r.values()) for r in records)}"
          f" / {len(records)}")
    print(f"mean pairwise Jaccard : {mean_pairwise_jaccard(records):.3f}")
    print(f"Fleiss' kappa (macro) : {macro:.3f}")
    print("\nkappa by act (multi-label: each act as an independent present/absent rating):")
    for a in ASSISTANT_SCHEME:
        k = by_act.get(a)
        print(f"  {a:<32}{'   n/a (never used)' if k is None else f'{k:>8.3f}'}")
    flag = asst_df.drop_duplicates(["pid", "utt_ind"])["needs_review"].mean()
    print(f"\n{flag:.0%} of assistant turns flagged needs_review (no majority / <2 raters)")


# ---------------------------------------------------------------------------
# 1. What the assistant did, by outcome group
# ---------------------------------------------------------------------------
def report_distribution(asst_df, good, bad, share_a):
    hr("1. ASSISTANT ACT PREVALENCE (share of a participant's R1 turns carrying the act)")
    n_turns = asst_df.groupby("pid")["utt_ind"].nunique()
    coded = asst_df.dropna(subset=["act_type"])
    print(f"{asst_df['pid'].nunique()} participants | {int(n_turns.sum())} exchanges | "
          f"{len(coded)} assistant act labels "
          f"({len(coded) / max(1, int(n_turns.sum())):.2f} acts per assistant turn)")

    order = share_a.mean().sort_values(ascending=False).index.tolist()
    print(f"\n{'act':<32}{'overall%':>9}{'Solvers%':>10}{'Strug%':>9}{'diff':>8}")
    for a in order:
        w, l = share_a.loc[good, a], share_a.loc[bad, a]
        print(f"{a:<32}{share_a[a].mean():>9.1f}{w.mean():>10.1f}{l.mean():>9.1f}"
              f"{w.mean() - l.mean():>+8.1f}")


def report_group_tests(share_a, good, bad, label="ASSISTANT"):
    hr(f"2. {label} ACTS vs OUTCOME — permutation test on turn shares "
       f"(Solvers n={len(good)} vs Strugglers n={len(bad)})")
    rows = []
    for a in share_a.columns:
        x, z = share_a.loc[good, a].values, share_a.loc[bad, a].values
        if x.std() == 0 and z.std() == 0:
            continue
        obs, p = perm_test(x, z)
        _, pu = stats.mannwhitneyu(x, z, alternative="two-sided")
        rows.append([a, x.mean(), z.mean(), obs, p, pu, cohend(x, z)])
    rows.sort(key=lambda r: r[4])
    q = bh_fdr([r[4] for r in rows])
    alpha_bonf = 0.05 / len(rows)
    print(f"Bonferroni alpha = {alpha_bonf:.4f} over {len(rows)} acts; "
          f"BH-FDR q also shown\n")
    print(f"{'act':<32}{'Solv%':>7}{'Strug%':>8}{'diff':>7}{'perm_p':>8}{'MWU_p':>8}"
          f"{'q':>7}{'d':>7}")
    for (a, w, l, dff, p, pu, d), qq in zip(rows, q):
        star = "*" if p < alpha_bonf else ("." if p < 0.05 else " ")
        print(f"{a:<32}{w:>7.1f}{l:>8.1f}{dff:>+7.1f}{p:>8.3f}{pu:>8.3f}{qq:>7.3f}{d:>+7.2f} {star}")
    print("\n  * p < Bonferroni alpha    . p < 0.05 (uncorrected)")
    return rows


# ---------------------------------------------------------------------------
# 3. Ordinal logit on the R2 optimal-move count
# ---------------------------------------------------------------------------
def build_model_frame(good, bad, share_a, share_u, user_df):
    rows = []
    for pid in sorted(set(good) | set(bad)):
        d = RECORDINGS_BASE / pid
        f1, f2 = d / SCORE_R1, d / SCORE_R2
        if not (f1.exists() and f2.exists()):
            continue
        j2 = json.load(open(f2))
        if not j2.get("gradable") or j2.get("flat_score") is None:
            continue
        j1 = json.load(open(f1))
        if j1.get("score") is None:
            continue
        fd = d / "demographics.json"
        gv = json.load(open(fd)).get("genai_usage") if fd.exists() else None
        rows.append(dict(pid=pid, genai=float(GENAI_OPTS.get(gv, np.nan)),
                         r1_score=float(j1["score"]), y=int(j2["flat_score"])))
    M = pd.DataFrame(rows).set_index("pid")
    M["n_turns"] = user_df.groupby("pid")["utt_ind"].nunique().reindex(M.index).fillna(0)
    for a in share_a.columns:
        M["A_" + a] = share_a[a].reindex(M.index).fillna(0) / 100.0
    for a in share_u.columns:
        M["U_" + a] = share_u[a].reindex(M.index).fillna(0) / 100.0
    return M[M.n_turns >= 1].dropna(subset=["genai", "y"]).copy()


def _ordfit(rhs, frame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return OrderedModel.from_formula(f"y ~ 0 + {rhs}", frame, distr="logit").fit(
            method="bfgs", disp=False)


def report_ordinal(M, acts, base="genai + r1_score", title=None, extra=None):
    """One single-predictor ordinal model per act on a fixed complete-case frame.
    `extra` adds a further control term (e.g. Think Aloud) to every model incl. M0."""
    hr(title or "3. ORDINAL LOGIT: assistant act turn-share -> R2 optimal moves (0-3)")
    cols = ["A_" + a for a in acts] + (["U_Think Aloud"] if extra else [])
    A = M.dropna(subset=["genai", "y", "r1_score"] + cols).copy()
    A["y"] = A.y.astype(int)
    keep, zname = [], {}
    for i, a in enumerate(acts):
        c = "A_" + a
        if A[c].std(ddof=1) == 0:
            continue
        # patsy formulas can't hold spaces. Index the name rather than initialising it:
        # initialisms collide (Neutral/Negative Feedback -> zNF, Prompt/Paraphrase -> zP),
        # which silently makes two acts share one column and fit the identical model.
        zname[a] = f"z{i}"
        A[zname[a]] = (A[c] - A[c].mean()) / A[c].std(ddof=1)
        keep.append(a)
    rhs_base = base + (" + zTA" if extra else "")
    if extra:
        A["zTA"] = (A["U_Think Aloud"] - A["U_Think Aloud"].mean()) / A["U_Think Aloud"].std(ddof=1)
    n = len(A)
    print(f"complete-case n={n}   DV distribution: {dict(A.y.value_counts().sort_index())}")
    print(f"covariates: {rhs_base}\n")

    m0 = _ordfit(rhs_base, A)
    out = []
    for a in keep:
        v = zname[a]
        f = _ordfit(f"{v} + {rhs_base}", A)
        lr = 2 * (f.llf - m0.llf)
        out.append(dict(act=a, b=f.params[v], se=f.bse[v], z=f.tvalues[v], p=f.pvalues[v],
                        AIC=f.aic, BIC=-2 * f.llf + len(f.params) * np.log(n),
                        LRp=stats.chi2.sf(lr, 1)))
    T = pd.DataFrame(out)
    aic0 = m0.aic
    bic0 = -2 * m0.llf + len(m0.params) * np.log(n)
    T["dAIC"] = T.AIC - min(T.AIC.min(), aic0)
    T = T.sort_values("AIC")
    print(f"{'model':<34}{'b(1SD)':>9}{'SE':>7}{'z':>7}{'p':>8}{'OR':>7}{'AIC':>8}"
          f"{'dAIC':>7}{'LR p':>8}")
    print(f"{'M0  covariates only':<34}{'':>9}{'':>7}{'':>7}{'':>8}{'':>7}{aic0:>8.1f}"
          f"{aic0 - min(T.AIC.min(), aic0):>7.2f}")
    for _, r in T.iterrows():
        star = " *" if r.p < 0.05 else ("  ." if r.p < 0.10 else "")
        print(f"{'+ ' + r.act:<34}{r.b:>+9.3f}{r.se:>7.3f}{r.z:>7.2f}{r.p:>8.4f}"
              f"{np.exp(r.b):>7.2f}{r.AIC:>8.1f}{r.dAIC:>7.2f}{r.LRp:>8.4f}{star}")
    print(f"\n  BIC of M0 = {bic0:.1f}. b is the log-odds shift per 1 SD of the act's turn share.")
    print("  * p<0.05   . p<0.10   (uncorrected; treat as descriptive across 10 acts)")
    return T


# ---------------------------------------------------------------------------
# 4. Dynamics
# ---------------------------------------------------------------------------
def turn_sets(user_df, asst_df):
    """{pid: [(user_acts:set, asst_acts:set), ...]} ordered by turn index."""
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
    """Association between a user act and an assistant act.

    lag=0  : user act at turn t   -> assistant act in the SAME exchange (the assistant
             is replying to that user turn), i.e. how the assistant RESPONDS.
    lag=+1 : assistant act at t   -> user act at t+1, i.e. how the assistant SHAPES
             what the participant does next.

    Null: shuffle the *responding* speaker's label sets WITHIN each participant. That
    holds each participant's rate of every act fixed and destroys only the pairing, so
    a significant result cannot be manufactured by between-participant differences in
    how much anyone talked or which acts they favour. Participants with a single usable
    pair contribute nothing to the null (nothing to shuffle) but still count in obs --
    that is the conservative direction, since their pairing is held fixed.
    """
    if lag == 0:
        by_pid = {p: [(u, a) for u, a in s] for p, s in seqs.items()}
        rows, cols = user_acts, asst_acts
    else:
        by_pid = {p: [(s[i][1], s[i + 1][0]) for i in range(len(s) - 1)]
                  for p, s in seqs.items()}
        rows, cols = asst_acts, user_acts
    by_pid = {p: v for p, v in by_pid.items() if v}

    # indicator matrices: C = cue acts (n x |rows|), R = response acts (n x |cols|)
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
    # two-sided empirical p on the cell count
    p = (np.minimum((null >= obs[None]).sum(0), (null <= obs[None]).sum(0)) + 1) / (n_perm + 1)
    p = np.minimum(1.0, 2 * p)
    return dict(rows=rows, cols=cols, obs=obs, exp=exp, p=p, n=n,
                base_cue=C.mean(0), base_res=R.mean(0),
                n_shufflable=sum(z - a for a, z in blocks if z - a > 1))


def report_coupling(res, title, cue_label, resp_label, top=14):
    hr(title)
    rows, cols, obs, exp, p = res["rows"], res["cols"], res["obs"], res["exp"], res["p"]
    print(f"{res['n']} ordered pairs ({res['n_shufflable']} of them from participants with "
          f">1 pair, i.e. actually shufflable).\nexp = mean count under within-participant "
          f"shuffling of the {resp_label.lower()} labels.\n")
    recs = []
    for ri, r in enumerate(rows):
        for ci, c in enumerate(cols):
            if obs[ri, ci] + exp[ri, ci] < 3:
                continue
            lift = obs[ri, ci] / exp[ri, ci] if exp[ri, ci] > 0 else np.inf
            recs.append((r, c, obs[ri, ci], exp[ri, ci], lift, p[ri, ci]))
    if not recs:
        print("  (no cell with enough mass to test)")
        return []
    q = bh_fdr([x[5] for x in recs])
    recs = [(*x, qq) for x, qq in zip(recs, q)]
    recs.sort(key=lambda x: x[5])
    print(f"{cue_label:<30}{resp_label:<30}{'obs':>5}{'exp':>7}{'lift':>7}{'p':>8}{'q':>7}")
    for r, c, o, e, lift, pp, qq in recs[:top]:
        star = "*" if qq < 0.05 else ("." if pp < 0.05 else " ")
        print(f"{r:<30}{c:<30}{int(o):>5}{e:>7.1f}{lift:>7.2f}{pp:>8.3f}{qq:>7.3f} {star}")
    print("\n  lift > 1 = the pairing happens more than chance given both speakers' own rates.")
    print("  * BH-FDR q < 0.05    . uncorrected p < 0.05")
    return recs


def report_direction(seqs, user_acts, asst_acts):
    """Overall dependency strength in each direction, as a single number, so 'the
    assistant is just reacting' can be told apart from 'the assistant is steering'."""
    hr("6. WHICH DIRECTION IS THE STRONGER DEPENDENCY?")
    out = {}
    for lag, name in [(0, "user(t) -> assistant(t)   [assistant reacting]"),
                      (1, "assistant(t) -> user(t+1) [assistant steering]")]:
        res = coupling(seqs, lag, user_acts, asst_acts, n_perm=1000, seed=SEED)
        obs, exp = res["obs"], res["exp"]
        mask = (obs + exp) >= 3
        chi = np.sum((obs[mask] - exp[mask]) ** 2 / np.maximum(exp[mask], 1e-9))
        cells = int(mask.sum())
        out[name] = (chi / max(cells, 1), cells, res["n"])
        print(f"{name:<44} mean chi2 per tested cell = {chi / max(cells,1):>6.2f} "
              f"({cells} cells, {res['n']} pairs)")
    print("\n  Higher = the pairing departs further from the within-participant shuffled null.")
    print("  The two are on the same scale (same statistic, same null), so they can be compared,")
    print("  but this is descriptive: it is not a formal test of one direction against the other.")
    return out


def report_mediation(M, acts, ta_col="U_Think Aloud"):
    """Does an assistant act move the participant's Think Aloud share (the established
    predictor of transfer), and does the act's own effect survive controlling for it?"""
    hr("7. ASSISTANT ACT -> PARTICIPANT THINK ALOUD (the established transfer predictor)")
    print(f"{'act':<32}{'r with TA share':>17}{'p':>9}{'n':>6}")
    rows = []
    for a in acts:
        c = "A_" + a
        x, y = M[c].values, M[ta_col].values
        if np.std(x) == 0:
            continue
        r, p = stats.pearsonr(x, y)
        rows.append((a, r, p))
    q = bh_fdr([r[2] for r in rows])
    rows = sorted(zip(rows, q), key=lambda t: t[0][2])
    for (a, r, p), qq in rows:
        star = "*" if qq < 0.05 else ("." if p < 0.05 else " ")
        print(f"{a:<32}{r:>+17.3f}{p:>9.4f}{len(M):>6} {star}")
    print("\n  * BH-FDR q < 0.05   . uncorrected p < 0.05")
    print("  Correlational and same-round: an assistant that scaffolds and a participant who")
    print("  reasons aloud co-occur within the same exchanges, so this cannot separate")
    print("  'the assistant elicited it' from 'the participant invited it'. Section 5's lagged")
    print("  coupling is the part that carries any temporal ordering.")


# ---------------------------------------------------------------------------
def main():
    pd.set_option("display.width", 200)
    good, bad = build_population()
    group_of = {**{p: "won" for p in good}, **{p: "lost" for p in bad}}
    pids = sorted(set(good) | set(bad))

    user_df, asst_df, records = load_long(pids, group_of)
    have = sorted(asst_df["pid"].unique())
    good = [p for p in good if p in have]
    bad = [p for p in bad if p in have]

    print(f"population: {len(good)} Solvers (R2 optimal >= 2) + {len(bad)} Strugglers "
          f"= {len(good) + len(bad)} with an annotated R1 conversation")

    report_reliability(records, asst_df)

    share_a = turn_share(asst_df, "act_type", have)
    share_u = turn_share(user_df, "utt_type", have)
    report_distribution(asst_df, good, bad, share_a)
    report_group_tests(share_a, good, bad)

    M = build_model_frame(good, bad, share_a, share_u, user_df)
    acts = list(share_a.columns)
    report_ordinal(M, acts)
    report_ordinal(M, acts, extra=True,
                   title="4. SAME MODELS, CONTROLLING FOR THE PARTICIPANT'S THINK ALOUD SHARE")

    seqs = turn_sets(user_df, asst_df)
    user_acts = [a for a in share_u.columns]
    asst_acts = [a for a in share_a.columns]
    r0 = coupling(seqs, 0, user_acts, asst_acts)
    report_coupling(r0, "5a. SAME EXCHANGE: what the participant said -> how the assistant replied",
                    "participant act (t)", "assistant act (t)")
    r1 = coupling(seqs, 1, user_acts, asst_acts)
    report_coupling(r1, "5b. NEXT TURN: how the assistant replied -> what the participant did next",
                    "assistant act (t)", "participant act (t+1)")
    report_direction(seqs, user_acts, asst_acts)
    report_mediation(M, acts)

    # persist tidy outputs for the notebook / figures
    asst_df.to_csv("assistant_utterances.csv", index=False)
    share_a.to_csv("assistant_act_turn_shares.csv")
    print("\nwrote assistant_utterances.csv, assistant_act_turn_shares.csv")


if __name__ == "__main__":
    main()
