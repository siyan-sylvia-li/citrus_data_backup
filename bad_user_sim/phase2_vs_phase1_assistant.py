"""Phase 2 vs phase 1: is the assistant producing the same quality of response?

Uses the same fine tutor-move panel, the same revised scheme (affective codes collapsed,
two codes dropped) and the same cluster-robust logit as elicitation_analysis.ipynb, so
the two studies sit on one scale.
"""
from __future__ import annotations
import json, warnings
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
warnings.filterwarnings("ignore"); pd.set_option("display.width", 200)

CITRUS = Path("/Users/siyanli/Documents/CITRUS")
AI = "poc20260727"
ANN = f"annotated_conversation_{AI}.jsonl"
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
SEATS = ["gpt", "llama", "sonnet"]
COLLAPSE = {"Positive Feedback": "Feedback", "Negative Feedback": "Feedback",
            "Neutral Feedback": "Feedback"}
DROP = {"Comprehension Gauging Question", "Paraphrase"}
FA, FP = "annotation_assistant_fine", "annotation_user"


def remap(labels):
    return {COLLAPSE.get(c, c) for c in labels if c not in DROP}


def revote(rec, min_votes=2):
    per = {k: v for k, v in (rec.get("per_model") or {}).items() if v is not None}
    if len(per) < len(SEATS):
        return remap(rec.get("final") or []), len(per)
    cnt = Counter()
    for s in SEATS:
        cnt.update(remap(per[s]))
    return {c for c, n in cnt.items() if n >= min_votes}, len(per)


def jacc(rec):
    """Mean pairwise Jaccard across seats, as a reliability proxy on the revised scheme."""
    per = [remap(v) for v in (rec.get("per_model") or {}).values() if v is not None]
    if len(per) < 2:
        return np.nan
    out = []
    for i in range(len(per)):
        for j in range(i + 1, len(per)):
            u = per[i] | per[j]
            out.append(1.0 if not u else len(per[i] & per[j]) / len(u))
    return float(np.mean(out))


def load(root, arm_from=None):
    rows = []
    for d in sorted(Path(root).iterdir()):
        f = d / ANN
        if not d.is_dir() or not f.exists():
            continue
        arm = None
        if arm_from:
            arm = ("scaffolded" if (d / "intervention.json").exists()
                   else "vanilla" if (d / "filler.json").exists() else None)
        for line in open(f):
            if not line.strip():
                continue
            t = json.loads(line)
            if FA not in t or not (t.get("assistant") or "").strip():
                continue
            asst, nseat = revote(t[FA])
            p = set((t.get(FP) or {}).get("final") or [])
            r = dict(pid=d.name, arm=arm, asst=asst, nseat=nseat, jac=jacc(t[FA]),
                     awords=len(t["assistant"].split()))
            for a in PACTS:
                r[SHORT[a]] = int(a in p)
            rows.append(r)
    return pd.DataFrame(rows)


P1 = load(CITRUS / "game_user_study_phase_othello/recordings-download")
P2 = load(CITRUS / "game_user_study_phase_2/recordings-download", arm_from=True)
print(f"phase 1: {len(P1)} assistant turns, {P1.pid.nunique()} participants")
print(f"phase 2: {len(P2)} assistant turns, {P2.pid.nunique()} participants  "
      f"(arm known for {int(P2.arm.notna().sum())})")
print(f"3-seat turns: phase 1 {100*(P1.nseat==3).mean():.0f}%, phase 2 {100*(P2.nseat==3).mean():.0f}%")

print("\n" + "=" * 92)
print("1. ASSISTANT ACT PREVALENCE — is phase 2 getting the same responses?")
print("=" * 92)
print(f"{'assistant act':<22}{'phase 1 %':>11}{'phase 2 %':>11}{'diff':>8}{'p':>9}"
      f"{'reliable':>10}")
out = []
for a in ALL_A:
    x = P1.asst.apply(lambda s, t=a: int(t in s))
    y = P2.asst.apply(lambda s, t=a: int(t in s))
    tab = np.array([[x.sum(), len(x) - x.sum()], [y.sum(), len(y) - y.sum()]])
    p = stats.chi2_contingency(tab)[1] if tab.min() >= 0 and tab.sum(1).min() > 0 else np.nan
    out.append((a, 100 * x.mean(), 100 * y.mean(), p))
    star = "  *" if p < .05 else (" ." if p < .10 else "")
    print(f"{a:<22}{100*x.mean():>11.1f}{100*y.mean():>11.1f}"
          f"{100*(y.mean()-x.mean()):>+8.1f}{p:>9.4f}{'yes' if a in RELIABLE else 'NO':>10}{star}")
t = pd.DataFrame(out, columns=["a", "p1", "p2", "p"]).sort_values("p")
k = len(t)
t["q"] = np.minimum.accumulate((t.p.values * k / np.arange(1, k + 1))[::-1])[::-1]
print(f"\n  BH q for the smallest p: {t.q.min():.3f}   "
      f"({int((t.p<.05).sum())} of {k} at raw p<.05)")
print(f"\n  acts per assistant turn: phase 1 {P1.asst.apply(len).mean():.2f}, "
      f"phase 2 {P2.asst.apply(len).mean():.2f}  "
      f"(MW p {stats.mannwhitneyu(P1.asst.apply(len), P2.asst.apply(len)).pvalue:.3f})")
print(f"  assistant reply length (words): phase 1 {P1.awords.mean():.0f}, "
      f"phase 2 {P2.awords.mean():.0f}  "
      f"(MW p {stats.mannwhitneyu(P1.awords, P2.awords).pvalue:.3f})")
print(f"  panel agreement (mean pairwise Jaccard): phase 1 {P1.jac.mean():.3f}, "
      f"phase 2 {P2.jac.mean():.3f}")


def or_table(d, label):
    print("\n" + "=" * 92)
    print(f"{label} — odds ratio for each assistant act, adjusted for all participant acts")
    print("=" * 92)
    print(f"{'assistant act':<22}" + "".join(f"{SHORT[a]:>22}" for a in FOCUS))
    res = {}
    for aa in ALL_A:
        y = d.asst.apply(lambda s, t=aa: int(t in s))
        if y.nunique() < 2:
            print(f"{aa:<22}  (no variation)")
            continue
        X = sm.add_constant(d[[SHORT[p] for p in PACTS]])
        try:
            m = sm.Logit(y, X).fit(disp=0, cov_type="cluster",
                                   cov_kwds={"groups": d.pid})
        except Exception as e:
            print(f"{aa:<22}  (did not converge: {type(e).__name__})")
            continue
        line = f"{aa:<22}"
        for pa in FOCUS:
            kk = SHORT[pa]
            o, pv = np.exp(m.params[kk]), m.pvalues[kk]
            star = "***" if pv < .001 else "**" if pv < .01 else "*" if pv < .05 else ""
            line += f"{o:>17.2f} {star:<4}"
            res[(aa, kk)] = (o, pv)
        print(line)
    return res


r1 = or_table(P1, "PHASE 1 OTHELLO")
r2 = or_table(P2, "PHASE 2")

print("\n" + "=" * 92)
print("2. DO THE ELICITATION PATTERNS AGREE?  (reliable assistant acts only)")
print("=" * 92)
print(f"{'assistant act':<22}{'participant act':<14}{'phase 1 OR':>12}{'phase 2 OR':>12}"
      f"{'same side of 1':>16}")
agree = []
for aa in ALL_A:
    if aa not in RELIABLE:
        continue
    for pa in FOCUS:
        kk = SHORT[pa]
        if (aa, kk) not in r1 or (aa, kk) not in r2:
            continue
        o1, o2 = r1[(aa, kk)][0], r2[(aa, kk)][0]
        same = (o1 > 1) == (o2 > 1)
        agree.append(same)
        print(f"{aa:<22}{kk:<14}{o1:>12.2f}{o2:>12.2f}{('yes' if same else 'NO'):>16}")
print(f"\n  agreement on direction: {sum(agree)}/{len(agree)}")
lo1 = np.log([r1[k][0] for k in r1 if k in r2])
lo2 = np.log([r2[k][0] for k in r1 if k in r2])
ok = np.isfinite(lo1) & np.isfinite(lo2)
print(f"  correlation of log-ORs across all cells: "
      f"r {np.corrcoef(lo1[ok], lo2[ok])[0,1]:+.3f} (n={ok.sum()} cells)")

print("\n" + "=" * 92)
print("3. DOES THE ARM CHANGE WHAT THE ASSISTANT GIVES?  (phase 2 only)")
print("=" * 92)
A = P2[P2.arm.notna()]
print(f"{'assistant act':<22}{'scaffolded %':>14}{'vanilla %':>11}{'diff':>8}{'p':>9}")
for a in ALL_A:
    s = A.loc[A.arm == "scaffolded", "asst"].apply(lambda x, t=a: int(t in x))
    v = A.loc[A.arm == "vanilla", "asst"].apply(lambda x, t=a: int(t in x))
    tab = np.array([[s.sum(), len(s)-s.sum()], [v.sum(), len(v)-v.sum()]])
    p = stats.chi2_contingency(tab)[1] if tab.sum(1).min() > 0 else np.nan
    star = "  *" if p < .05 else (" ." if p < .10 else "")
    print(f"{a:<22}{100*s.mean():>14.1f}{100*v.mean():>11.1f}"
          f"{100*(s.mean()-v.mean()):>+8.1f}{p:>9.3f}{star}")
print(f"\n  n turns: scaffolded {int((A.arm=='scaffolded').sum())}, "
      f"vanilla {int((A.arm=='vanilla').sum())}")
