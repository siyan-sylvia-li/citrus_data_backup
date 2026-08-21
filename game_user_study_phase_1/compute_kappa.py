"""Score inter-rater agreement on the kappa_sample turns, per source (controlled / wild).
Raters: panel (3-model vote, 6 acts), claude (LLM rater, 7 acts), human (kappa_sample.csv).
For each pair we score only acts BOTH raters can produce (an act one rater never uses is
degenerate -> skipped). So ProvideInfo, which the panel lacks, is scored claude-vs-human only
and reported as prevalence against the panel.
"""
import json, os
import pandas as pd

# (human sheet, panel key) pairs — merged into one gold set. Add pairs here to grow it.
SOURCES = [
           ("controlled_extra.csv", "controlled_extra_key.json")]

# Metacomment DROPPED (kappa ~0.23-0.35, unstable even after relabel). 5 core acts + ProvideInfo.
ACTS = ["Solution Request", "Common Ground Question", "Knowledge Deficit Question",
        "Think Aloud", "Conversational Acknowledgment", "Provide Information"]
CODE = {"Solution Request": "SolReq", "Common Ground Question": "CommonGrd",
        "Knowledge Deficit Question": "KnowDef", "Think Aloud": "ThinkAloud",
        "Conversational Acknowledgment": "Ack", "Provide Information": "ProvideInfo"}


def truthy(x):
    return str(x).strip() not in ("", "0", "nan", "False", "false")


def load_coded(path):
    try:
        d = pd.read_csv(path)
    except FileNotFoundError:
        return {}
    # detect coded rows via all label cols (incl. dropped Meta + NoAct); scored acts exclude Meta
    cols = [CODE[a] for a in ACTS if CODE[a] in d.columns] + [c for c in ("Meta", "NoAct") if c in d.columns]
    out = {}
    for r in d.itertuples():
        marks = {c: truthy(getattr(r, c)) for c in cols}
        if any(marks.values()):
            out[int(r.turn_id)] = {a for a in ACTS if marks.get(CODE[a])}
    return out


def cohen_binary(pairs):
    n = len(pairs)
    po = sum(h == p for h, p in pairs) / n
    ph = sum(h for h, _ in pairs) / n
    pp = sum(p for _, p in pairs) / n
    pe = ph * pp + (1 - ph) * (1 - pp)
    return None if pe >= 1 else (po - pe) / (1 - pe)


def agreement(A, B, ids):
    U = []                                   # acts both raters actually use (non-degenerate)
    for a in ACTS:
        hs = sum(a in A[t] for t in ids); ps = sum(a in B[t] for t in ids)
        if hs not in (0, len(ids)) and ps not in (0, len(ids)):
            U.append(a)
    kappas = {}
    for a in U:
        k = cohen_binary([(a in A[t], a in B[t]) for t in ids])
        if k is not None:
            kappas[a] = k
    jac, exact = [], []
    for t in ids:
        h = {a for a in U if a in A[t]}; p = {a for a in U if a in B[t]}
        jac.append(1.0 if not h and not p else len(h & p) / len(h | p))
        exact.append(h == p)
    macro = sum(kappas.values()) / len(kappas) if kappas else float("nan")
    return macro, sum(exact) / len(exact), sum(jac) / len(jac), kappas


key, human = {}, {}
for sheet_path, key_path in SOURCES:
    if os.path.exists(key_path):
        key.update({int(k): v for k, v in json.load(open(key_path)).items()})
    if os.path.exists(sheet_path):
        human.update(load_coded(sheet_path))
raters = {"panel": {t: set(key[t]["panel"]) for t in key},
          "claude": load_coded("kappa_sample_claude.csv"),
          "human": human}
raters = {n: r for n, r in raters.items() if r}
src_of = {t: key[t]["source"] for t in key}

for src in ["controlled"]:
    src_ids = [t for t in key if src_of[t] == src]
    print(f"\n{'='*58}\n{src.upper()}  ({len(src_ids)} turns)\n{'='*58}")
    names = list(raters)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            A, B = raters[names[i]], raters[names[j]]
            ids = [t for t in src_ids if t in A and t in B]
            if len(ids) < 2:
                continue
            macro, exact, jac, ka = agreement(A, B, ids)
            print(f"  {names[i]:>6} vs {names[j]:<6} (n={len(ids)}): kappa {macro:.3f} | exact {exact:.0%} | Jaccard {jac:.2f}")
            print(f"         by act: {{{', '.join(f'{CODE[a]}:{v:.2f}' for a, v in ka.items())}}}")

# ProvideInfo prevalence (the grounding asymmetry) — panel can't produce it, so show raters that can
print(f"\n{'='*58}\nProvideInfo prevalence (grounding asymmetry)\n{'='*58}")
for name in ("claude", "human"):
    if name not in raters:
        continue
    for src in ("controlled", "wild"):
        ids = [t for t in raters[name] if src_of.get(t) == src]
        pi = sum("Provide Information" in raters[name][t] for t in ids)
        print(f"  {name:>6} {src:<11}: {pi}/{len(ids)} turns ProvideInfo ({pi/len(ids):.0%})" if ids else "")
