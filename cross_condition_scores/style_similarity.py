"""Are adopters and non-adopters stylistically different, act by act?

Embeddings: StyleDistance (StyleDistance/styledistance) -- content-independent style
embeddings, RoBERTa-base with attention-masked mean pooling and max_seq_length 512, per the
model's own 1_Pooling/config.json. Loaded through AutoModel rather than sentence-transformers
so no extra dependency is needed; the pooling is reproduced exactly.

Design of the test. For one act we have utterances from adopters and from non-adopters, and
want to know whether style tracks population. Two things make a naive t-test on pairwise
cosines invalid:

  * PAIRS ARE NOT INDEPENDENT -- each utterance appears in many pairs.
  * UTTERANCES CLUSTER BY PARTICIPANT, and population is a property of the PARTICIPANT, not
    of the utterance.

So: statistic = mean(within-group cosine) - mean(cross-group cosine), and the null is built
by permuting the population label ACROSS PARTICIPANTS (the level at which it was assigned),
recomputing the statistic each time. Same-participant pairs are excluded from every mean --
a person's own utterances are stylistically alike, which would inflate the within-group term
for reasons that have nothing to do with population.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from elicitation import ANN, CITRUS, PACTS, POPULATION, SOURCES

MODEL = "StyleDistance/styledistance"
_tok = _mod = None


def _load():
    global _tok, _mod
    if _mod is None:
        _tok = AutoTokenizer.from_pretrained(MODEL)
        _mod = AutoModel.from_pretrained(MODEL).eval()
    return _tok, _mod


@torch.no_grad()
def embed(texts, batch=32):
    """Attention-masked mean pooling, matching the model's 1_Pooling config."""
    tok, mod = _load()
    out = []
    for i in range(0, len(texts), batch):
        b = tok(texts[i:i + batch], padding=True, truncation=True, max_length=512,
                return_tensors="pt")
        h = mod(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1).clamp(min=1e-9)).cpu().numpy())
    E = np.vstack(out)
    return E / np.linalg.norm(E, axis=1, keepdims=True)      # cosine == dot product


def load_utterances():
    rows = []
    for src, rel in SOURCES.items():
        root = CITRUS / rel
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            f = d / ANN
            if not d.is_dir() or not f.exists():
                continue
            for line in open(f):
                if not line.strip():
                    continue
                t = json.loads(line)
                a, u = t.get("annotation_user"), (t.get("user") or "").strip()
                if not a or not u:
                    continue
                rows.append({"pop": POPULATION[src], "pid": d.name,
                             "text": " ".join(u.split()),
                             "acts": set(a.get("final") or [])})
    return rows


def analyse(S, pop, pid, n_perm=2000, seed=0):
    """Within/cross means and a participant-level permutation p-value.

    The upper triangle, the participant codes and the same-participant mask are computed
    ONCE; each permutation then only re-derives `same_pop` by fancy-indexing a permuted
    per-participant label vector. Same-participant pairs are dropped from every mean.
    """
    n = len(pop)
    iu = np.triu_indices(n, k=1)
    v = S[iu]
    pids, pcode = np.unique(pid, return_inverse=True)
    pi, pj = pcode[iu[0]], pcode[iu[1]]
    keep = pi != pj                              # drop same-participant pairs
    v, pi, pj = v[keep], pi[keep], pj[keep]
    labs = np.array([pop[pcode == c][0] for c in range(len(pids))])
    is_ad = labs == "adopters"

    def split(L):
        same = L[pi] == L[pj]
        return float(v[same].mean()), float(v[~same].mean())

    within, cross = split(labs)
    both_a = (is_ad[pi]) & (is_ad[pj])
    both_n = (~is_ad[pi]) & (~is_ad[pj])
    wa = float(v[both_a].mean()) if both_a.any() else np.nan
    wn = float(v[both_n].mean()) if both_n.any() else np.nan

    obs = within - cross
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        w2, c2 = split(rng.permutation(labs))
        null[k] = w2 - c2
    return dict(within_adopters=wa, within_non_adopters=wn, within_pooled=within,
                cross=cross, diff=obs,
                p_perm=float((np.abs(null) >= abs(obs)).mean()),
                null_sd=float(null.std()),
                n_adopters=int((pop == "adopters").sum()),
                n_non_adopters=int((pop == "non-adopters").sum()),
                n_participants=len(pids), n_pairs=int(len(v)))
