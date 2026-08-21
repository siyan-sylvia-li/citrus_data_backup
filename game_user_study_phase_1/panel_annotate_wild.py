"""Re-annotate the wild ShareChat tutoring-pedagogical turns with the 3-MODEL PANEL
(DialogueActSuite: llama-3.3-70B + gpt-5.4-mini + gemma-4-31B, per-act majority vote) —
symmetric with how phase-1 participant_utterances.csv was annotated.

Uses the EXACT taxonomy/signature/suite from check_sharechat.ipynb (exec'd from the .ipynb)
so the only change vs the single-model run is the annotator, not the prompt. Re-runs on the
same (url, message_index, text) turns saved by the single-model run.
"""
import json
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from tqdm import tqdm

# --- exec the notebook's taxonomy(1) + signature(2) + suite(3) cells for an exact match ---
nb = json.load(open("check_sharechat.ipynb"))
src = {c["id"]: "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"}
ns = {}
for cid in ("73c15c1f", "5c86d596", "738824cc"):
    exec(src[cid], ns)
SCHEME, TAXONOMY = ns["SCHEME"], ns["TAXONOMY"]
DialogueActSuite = ns["DialogueActSuite"]
fleiss_kappa, mean_pairwise_jaccard = ns["fleiss_kappa"], ns["mean_pairwise_jaccard"]

# --- same turns as the single-model run ---
turns = [r for r in json.load(open("acts_tutoring_or_teaching_pedagogical.json")) if r.get("acts") is not None]
print(f"panel-annotating {len(turns)} wild turns (3 models each)...")

suite = DialogueActSuite()

def run(r):
    res = suite(utterance=r["text"])
    return {"url": r["url"], "message_index": r["message_index"], "text": r["text"],
            "acts_single": r["acts"], "acts": res["final"],
            "per_model": res["per_model"], "confidence": res["confidence"],
            "n_valid": res["n_valid"], "needs_review": res["needs_review"]}

with ThreadPoolExecutor(max_workers=6) as ex:
    out = list(tqdm(ex.map(run, turns), total=len(turns), desc="panel"))

json.dump(out, open("acts_tutoring_or_teaching_pedagogical_panel.json", "w"), indent=2, default=str)

# --- agreement ---
recs = [r["per_model"] for r in out if r["per_model"]]
macro, by_act = fleiss_kappa(recs)
print(f"\nagreement: mean pairwise Jaccard {mean_pairwise_jaccard(recs):.3f} | Fleiss' kappa (macro) {macro:.3f}")
print("  kappa by act:", {k: round(v, 2) for k, v in by_act.items()})
print("  n_valid per turn:", dict(Counter(r["n_valid"] for r in out)),
      "| needs_review:", sum(r["needs_review"] for r in out))

# --- did the label change vs single-model? ---
changed = sum(set(r["acts"]) != set(r["acts_single"]) for r in out)
print(f"  final differs from single-model on {changed}/{len(out)} turns ({changed/len(out):.0%})")

# --- panel vs single act rate, and opener/follow-up engagement (robustness of the descriptive result) ---
n = len(out)
def rate(key):
    return {a: sum(a in r[key] for r in out) / n for a in SCHEME if any(a in r[key] for r in out)}
print("\nact rate per turn  (single -> panel):")
pr, sr = rate("acts"), rate("acts_single")
for a in sorted(set(pr) | set(sr), key=lambda a: -pr.get(a, 0)):
    print(f"  {a:<30} {sr.get(a,0):>5.0%} -> {pr.get(a,0):>5.0%}")

by_url = defaultdict(list)
for r in out:
    by_url[r["url"]].append((r["message_index"], r["acts"]))
ENG = {"Common Ground Question", "Think Aloud", "Knowledge Deficit Question"}
op, fu = [], []
for u, msgs in by_url.items():
    seq = [set(a) for _, a in sorted(msgs)]
    op.append(seq[0]); fu.extend(seq[1:])
def eng_req(ts):
    ne = sum(bool(t & ENG) for t in ts) / len(ts)
    nr = sum("Solution Request" in t for t in ts) / len(ts)
    return ne, nr
print("\nPANEL wild engagement / request (any-of):")
print(f"  opener   (n={len(op)}): engage {eng_req(op)[0]:.0%}  request {eng_req(op)[1]:.0%}")
print(f"  followup (n={len(fu)}): engage {eng_req(fu)[0]:.0%}  request {eng_req(fu)[1]:.0%}")
