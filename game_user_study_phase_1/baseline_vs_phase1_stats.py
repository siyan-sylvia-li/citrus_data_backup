"""Baseline vs Phase-1 group comparisons.

Reproduces the statistical tests run comparing:
  - game_user_study_baseline_pro   (both rounds unassisted)
  - game_user_study_phase_1        (R1 AI-assisted, R2 unassisted)

Tests
-----
  Game scores (R1=p15, R2=w4p6): flat `score` and `winning_line_score`  -> Mann-Whitney U
  NASA-TLX   (R1, R2): 5 subscales + an overall workload index          -> Mann-Whitney U
  Demographics: age, skill_rating -> Mann-Whitney U;
                education, occupation, genai_usage -> chi-square (+ Cramer's V)

Population
----------
  Real participants only: prolific-id dirs matching 24 hex chars (this drops test
  accounts like `tester`/`testtest` and malformed ids). Phase-1 is restricted to
  participants with conversation_p15.jsonl (the analysis set); flip REQUIRE_CONV
  to compare that.

Run:
  /Users/siyanli/Documents/CITRUS/document_study_og/.venv/bin/python baseline_vs_phase1_stats.py
"""
import os
import json
import re
from collections import Counter

import numpy as np
from scipy.stats import mannwhitneyu, chi2_contingency

BASE = "/Users/siyanli/Documents/CITRUS"
BASELINE = f"{BASE}/game_user_study_baseline_pro/recordings-download"
PHASE1 = f"{BASE}/game_user_study_phase_1/recordings-download"
HEX = re.compile(r"^[0-9a-f]{24}$")          # 24-hex prolific id -> excludes test accounts
REQUIRE_CONV = True                          # phase_1: keep only participants with conversation_p15

TLX_DIMS = ["mental_demand", "temporal_demand", "performance", "effort", "frustration"]


# --------------------------------------------------------------------------- helpers
def participant_dirs(root, require_conv=False):
    dirs = []
    for pid in sorted(os.listdir(root)):
        d = os.path.join(root, pid)
        if not os.path.isdir(d) or not HEX.match(pid):
            continue
        if require_conv and not os.path.exists(os.path.join(d, "conversation_p15.jsonl")):
            continue
        dirs.append(d)
    return dirs


def _read(d, fname):
    f = os.path.join(d, fname)
    if not os.path.exists(f):
        return None
    try:
        return json.load(open(f))
    except Exception:
        return None


def collect_score(dirs, fname, key):
    out = []
    for d in dirs:
        j = _read(d, fname)
        if j and j.get(key) is not None:
            out.append(j[key])
    return out


def collect_tlx(dirs, fname):
    data = {k: [] for k in TLX_DIMS}
    data["overall"] = []                      # mean of 5, performance reverse-scored (higher = more load)
    for d in dirs:
        j = _read(d, fname)
        t = j.get("nasa_tlx") if j else None
        if not t or any(t.get(x) is None for x in TLX_DIMS):
            continue
        for x in TLX_DIMS:
            data[x].append(t[x])
        data["overall"].append(np.mean([t["mental_demand"], t["temporal_demand"],
                                         8 - t["performance"], t["effort"], t["frustration"]]))
    return data


def collect_demo_num(dirs, key):
    out = []
    for d in dirs:
        j = _read(d, "demographics.json")
        if not j:
            continue
        x = j.get(key)
        if x in (None, ""):
            continue
        try:
            out.append(float(x))
        except (TypeError, ValueError):        # e.g. skill_rating == "dont_know"
            pass
    return out


def collect_demo_cat(dirs, key):
    out = []
    for d in dirs:
        j = _read(d, "demographics.json")
        if j and j.get(key) not in (None, ""):
            out.append(str(j[key]))
    return out


def mwu(a, b, label):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        print(f"  {label:22s}: insufficient data (n_base={len(a)}, n_ph1={len(b)})")
        return
    U, p = mannwhitneyu(a, b, alternative="two-sided")
    rbc = 1 - 2 * U / (len(a) * len(b))        # rank-biserial effect size
    print(f"  {label:22s}: base med={np.median(a):5.2f} mean={a.mean():5.2f} (n={len(a)}) | "
          f"ph1 med={np.median(b):5.2f} mean={b.mean():5.2f} (n={len(b)}) | "
          f"U={U:7.1f} p={p:.4f} rbc={rbc:+.2f}{'  *' if p < 0.05 else ''}")


def chisq(ca, cb, label):
    cats = sorted(set(ca) | set(cb))
    ct = np.array([[Counter(ca)[c] for c in cats], [Counter(cb)[c] for c in cats]])
    ct = ct[:, ct.sum(axis=0) > 0]             # drop categories absent in both
    if ct.shape[1] < 2 or ct.sum() == 0:
        print(f"  {label:22s}: insufficient categories")
        return
    chi2, p, dof, exp = chi2_contingency(ct)
    v = np.sqrt(chi2 / (ct.sum() * (min(ct.shape) - 1)))    # Cramer's V
    warn = "  (min expected < 5: chi-square approximate)" if exp.min() < 5 else ""
    print(f"  {label:22s}: chi2={chi2:6.2f} dof={dof} p={p:.4f} CramersV={v:.2f}"
          f"{'  *' if p < 0.05 else ''}{warn}")


# --------------------------------------------------------------------------- run
def main():
    B = participant_dirs(BASELINE, require_conv=False)
    P = participant_dirs(PHASE1, require_conv=REQUIRE_CONV)
    print(f"baseline n={len(B)} | phase_1 n={len(P)} "
          f"(phase_1 require conversation_p15 = {REQUIRE_CONV})\n")

    print("=== Game scores (Mann-Whitney U, two-sided; rbc = rank-biserial effect size) ===")
    for rnd, fname in [("R1 (p15, AI-assisted for phase_1)", "cf_score_p15.json"),
                       ("R2 (w4p6, unassisted both)", "cf_score_pw4p6.json")]:
        print(f" {rnd}:")
        for key in ["score", "winning_line_score"]:
            mwu(collect_score(B, fname, key), collect_score(P, fname, key), key)

    print("\n=== NASA-TLX (Mann-Whitney U) ===")
    print("   note: 'performance' is NOT reversed in its own row (higher = self-rated better);")
    print("         'overall' reverses it (8 - performance) so higher = more workload.")
    for rnd, fname in [("R1 (p15)", "post_survey_p15.json"),
                       ("R2 (w4p6)", "post_survey_pw4p6.json")]:
        print(f" {rnd}:")
        tb, tp = collect_tlx(B, fname), collect_tlx(P, fname)
        for m in TLX_DIMS + ["overall"]:
            mwu(tb[m], tp[m], m)

    print("\n=== Demographics ===")
    print(" numeric / ordinal (Mann-Whitney U):")
    for key in ["age", "skill_rating"]:
        mwu(collect_demo_num(B, key), collect_demo_num(P, key), key)
    print(" categorical (chi-square + Cramer's V):")
    for key in ["education", "occupation", "genai_usage"]:
        chisq(collect_demo_cat(B, key), collect_demo_cat(P, key), key)


if __name__ == "__main__":
    main()
