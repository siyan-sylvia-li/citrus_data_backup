"""Test-retest of the v1 scheme: re-annotate the same 295 utterances with the
UNCHANGED v1 taxonomy (dialogue_act_annotation.py) and a fresh panel pass.

This is the one check that doesn't involve re-drawing any coding boundary. It
separates two explanations for the v2 nulls:
  * if v1's Think Aloud -> R2 effect reproduces on fresh v1 labels, the effect is
    real and merely sensitive to how "articulation" is operationalized;
  * if it does not, the original p=.0045 was specific to that annotation run.

Writes annotations_v1_replication.jsonl (+ .csv) and reports:
  - panel agreement within the replication run,
  - item-level stability between the ORIGINAL v1 labels and the replication,
  - the OLS re-run on replication labels, side by side with the original.
"""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import dotenv
import dspy
import numpy as np
import pandas as pd

import dialogue_act_annotation as V1

BASE = Path("recordings-download")
SCORE_R2 = "cf_score_selectivity_pw4p6.json"
OUT = Path("annotations_v1_replication.jsonl")


def dvs_for(pids):
    """pid -> (flat, difficulty-weighted selectivity, winning-line) for participants
    with a gradable R2, matching the notebook's filter."""
    out = {}
    for pid in pids:
        f = BASE / pid / SCORE_R2
        if not f.exists():
            continue
        j = json.load(open(f))
        if not j.get("gradable") or j.get("score_selectivity_raw") is None:
            continue
        out[pid] = (j.get("flat_score"), j.get("score_selectivity_raw"),
                    j.get("flat_winning_line_score"))
    return out


def ols1(x, y):
    """Simple OLS y ~ 1 + x with t test (no statsmodels dependency)."""
    from scipy import stats
    m = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[m], y[m]
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - 2
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    se = np.sqrt(np.diag((ss_res / dof) * np.linalg.inv(X.T @ X)))
    t = beta / se
    return dict(n=n, coef=beta[1], se=se[1], t=t[1],
                p=2 * (1 - stats.t.cdf(np.abs(t[1]), dof)),
                r2=1 - ss_res / ss_tot if ss_tot else float("nan"))


def main():
    dotenv.load_dotenv()
    # CRITICAL: dspy caches completions on disk (~/.dspy_cache) by default. The v1
    # labels were produced with this same signature, so a cached run replays the
    # ORIGINAL completions and "replicates" perfectly by construction -- the first
    # attempt at this script returned per-item Jaccard 1.000 on all 295 items for
    # exactly that reason. Both caches must be off for this to be a real test.
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
    d1 = pd.read_csv("participant_utterances.csv")
    items = (d1.groupby(["pid", "utt_ind"]).agg(utt=("utt", "first")).reset_index()
               .sort_values(["pid", "utt_ind"]).to_dict("records"))

    done = set()
    if OUT.exists():
        done = {(json.loads(l)["pid"], json.loads(l)["utt_ind"]) for l in open(OUT)}
    todo = [it for it in items if (it["pid"], it["utt_ind"]) not in done]
    print(f"{len(done)} done | {len(todo)} to go", flush=True)

    suite = V1.DialogueActSuite()          # same panel, same signature, same taxonomy
    fh = open(OUT, "a")
    with ThreadPoolExecutor(max_workers=6) as ex:
        for it, r in zip(todo, ex.map(lambda i: suite(utterance=str(i["utt"])), todo)):
            fh.write(json.dumps(dict(pid=it["pid"], utt_ind=int(it["utt_ind"]), utt=str(it["utt"]),
                                     acts=r["final"], votes=r["votes"], per_model=r["per_model"],
                                     n_valid=r["n_valid"], needs_review=r["needs_review"])) + "\n")
            fh.flush()
    fh.close()

    recs = [json.loads(l) for l in open(OUT)]
    ACT_REMAP = {"Correct Answer": "Think Aloud", "Social Coordination Action": "Metacomment",
                 "Forced Choice": "Metacomment"}          # same remap the notebook applies
    for r in recs:
        r["acts"] = list(dict.fromkeys(ACT_REMAP.get(a, a) for a in r["acts"]))
    rows = [{"pid": r["pid"], "utt": r["utt"], "utt_type": a, "utt_ind": r["utt_ind"],
             "needs_review": r["needs_review"]}
            for r in recs for a in (r["acts"] or [None])]
    rep = pd.DataFrame(rows)
    rep.to_csv(OUT.with_suffix(".csv"), index=False)

    macro, by_act = V1.fleiss_kappa([r["per_model"] for r in recs])
    print(f"\nreplication panel: Fleiss macro-kappa={macro:.3f} | "
          f"Jaccard={V1.mean_pairwise_jaccard([r['per_model'] for r in recs]):.3f} | "
          f"needs_review {sum(r['needs_review'] for r in recs)}/{len(recs)}")
    print("kappa by act:", {k: round(v, 2) for k, v in sorted(by_act.items(), key=lambda x: -x[1])})

    # --- run-to-run stability of the labels themselves ---
    orig = d1.dropna(subset=["utt_type"]).groupby(["pid", "utt_ind"])["utt_type"].apply(frozenset)
    new = rep.dropna(subset=["utt_type"]).groupby(["pid", "utt_ind"])["utt_type"].apply(frozenset)
    keys = [k for k in orig.index if k in new.index]
    jac = np.mean([len(orig[k] & new[k]) / len(orig[k] | new[k]) if (orig[k] | new[k]) else 1.0
                   for k in keys])
    print(f"\noriginal vs replication: mean per-item Jaccard = {jac:.3f}  (n={len(keys)})")
    print(f"{'act':<32}{'orig n':>8}{'repl n':>8}{'both':>7}{'F1':>7}")
    for act in ["Think Aloud", "Common Ground Question", "Solution Request",
                "Conversational Acknowledgment", "Knowledge Deficit Question", "Metacomment"]:
        a = {k for k in keys if act in orig[k]}
        b = {k for k in keys if act in new[k]}
        f1 = 2 * len(a & b) / (len(a) + len(b)) if (a or b) else float("nan")
        print(f"{act:<32}{len(a):>8}{len(b):>8}{len(a & b):>7}{f1:>7.2f}")

    # --- the regression, original labels vs replication labels ---
    def share(df):
        a = df.dropna(subset=["utt_type"])
        tot = a.groupby("pid").size()
        cnt = (a.groupby(["pid", "utt_type"]).size().unstack(fill_value=0)
                 .reindex(sorted(df["pid"].unique()), fill_value=0))
        return cnt.div(tot, axis=0).fillna(0)

    s_orig, s_rep = share(d1), share(rep)
    dv = dvs_for(sorted(s_orig.index))
    print(f"\nOLS on ORIGINAL vs REPLICATION labels (n gradable R2 = {len(dv)})")
    print(f"{'IV':<40}{'labels':<14}{'coef':>9}{'p':>9}{'R2':>7}")
    for nm, ivs in [("CGQ + Think Aloud (pre-specified)",
                     [("original", s_orig["Common Ground Question"] + s_orig["Think Aloud"]),
                      ("replication", s_rep["Common Ground Question"] + s_rep["Think Aloud"])]),
                    ("Think Aloud alone",
                     [("original", s_orig["Think Aloud"]), ("replication", s_rep["Think Aloud"])])]:
        for lab, iv in ivs:
            pids = [p for p in iv.index if p in dv]
            x = np.array([iv[p] for p in pids], float)
            for k, dvn in enumerate(["flat R2", "selectivity", "winning-line"]):
                o = ols1(x, np.array([dv[p][k] for p in pids], float))
                star = "*" if o["p"] < .05 else ("." if o["p"] < .10 else "")
                print(f"{(nm + ' -> ' + dvn):<40}{lab:<14}{o['coef']:+9.3f}{o['p']:9.4f}{o['r2']:7.3f} {star}")


if __name__ == "__main__":
    main()
