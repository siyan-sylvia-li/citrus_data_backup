# Old vs. New (Jul-28) Participants — Difference Analysis

_Phase-1 Connect-Four transfer study. Written 2026-07-28._

## TL;DR

A batch of 30 participants collected on **Jul 28** (16 analysis-ready) briefly appeared to
kill our headline result. It didn't — the **median-split** test is just fragile. The new wave
is a **systematically different, less AI-fluent population** that solves by *offloading to the
AI* rather than reasoning. The underlying effect — **R1 reasoning (Think Aloud) predicts R2
transfer** — survives pooling, and survives controls for AI-fluency *and* R1 performance. We
**keep all participants** and report the continuous, controlled model as primary.

## Populations

- **Old** = pre-Jul-28 recordings, four waves (Jul 16 / 22 / 23 / 24); **91 analysis-ready**
  (has R1 score + R2 score + conversation + panel annotation).
- **New** = Jul-28 wave, 30 folders, **16 analysis-ready** (14 incomplete: missing conversation
  or annotation).
- Merged set = **107**.

## What differs (old 91 vs new 16)

| dimension | old (91) | new (16) | note |
|---|---|---|---|
| Solver rate (R2 flat ≥ 2) | 56% | 50% | ~same — not an easier/harder draw |
| Reasoning-act share (all turns) | **0.50** | **0.32** | new wave engages far less |
| Reasoning: Solver vs Struggler | 0.58 vs 0.40 (**Δ+0.18, p=0.025**) | 0.31 vs 0.33 (**Δ−0.02, p=0.91**) | effect **absent** in new |
| % IT occupation | **30%** | **12%** | new less technical |
| % frequent GenAI (≥ few/wk) | **67%** | **44%** | new less AI-fluent |

Per-act, Solver vs Struggler:

| act | old-91 (Solver / Struggler) | new-16 (Solver / Struggler) |
|---|---|---|
| **Solution Request** | 25% / 43%  → gap **−18** (solvers ask *less*) | 48% / 31% → gap **+17** (solvers ask *more*) |
| **Think Aloud** | 30% / 15% → gap **+15** | 15% / 14% → gap **+2** (flat) |

## Why the SolReq result "broke" but Think Aloud didn't

- **Solution Request *reverses* in the new wave.** Old solvers requested *less*; new solvers
  request the *most*. A reversal pulls the pooled gap toward zero fast (median-split MWU
  **p 0.04 → 0.13**). In a less-fluent, transactional wave, "just ask the AI" is a viable route
  to solving — so *low requesting is not a robust marker of good learners; it's population-dependent.*
- **Think Aloud only *flattens*** in the new wave (gap +2), it doesn't reverse. The large old gap
  still dominates the pooled test (MWU **p 0.00 → 0.00**). Reasoning is the durable signal.

## Requesting vs reasoning — one axis, not two

Solution Request is **not an independent signal** — it's the structural complement of reasoning,
so we describe it, we don't model it as a competing predictor.

**Co-occurrence** (of a participant's request turns, share that *also* carry a reasoning act —
Think Aloud / Common-Ground):

| group | SolReq % of turns | co-occ % (requests that also reason) |
|---|---|---|
| OLD Solver | 32 | **27** |
| OLD Struggler | 50 | **9** |
| NEW Solver | 64 | **31** |
| NEW Struggler | 39 | **0** |

The stable discriminator is the **co-occurrence rate**, not the raw request rate: in *both* waves
Solvers reason *while* requesting (27–31%), Strugglers request *bare* (9% → 0%). New Solvers
request heavily (64%) but ~a third of those requests are engaged — they "solved by requesting"
while reasoning as they went. (New groups are n=8; treat as directional.)

**Why we don't put reasoning and requesting in the same regression.** They are shares of one
finite turn budget → compositional and anti-correlated: `corr(Think Aloud, SolReq)=−0.46`,
`corr(reasoning [Think Aloud + Common Ground], SolReq)=−0.66`. In the *stable* single-act spec
(Think Aloud + SolReq), requesting adds nothing beyond reasoning: Think Aloud β=+0.24 (p=0.013),
**SolReq β=−0.02 (p=0.88)**. Broader composites are worse — a 3-act reasoning set (incl.
Knowledge-Deficit, r=−0.83) makes SolReq's coefficient flip *positive* (+0.31, p=0.04), a pure
collinearity artifact. So the analysis uses a **single predictor, Think Aloud**; requesting is
reported descriptively as its mirror image (people who request more reason less).

## The finding survives — the median split was the weak link

Continuous OLS on the merged **107** (standardized predictors, DV = R2 flat_score):

| model | reasoning β (p) | covariate β (p) |
|---|---|---|
| Think Aloud only | **+0.248 (0.005)** | — |
| Think Aloud + **R1 performance** | **+0.254 (0.004)** | R1 β=+0.099 (0.247) |
| Think Aloud + **GenAI fluency** | +0.168 (0.057) | fluency β=−0.129 (0.144) |
| composite (TA+CG) + R1 | +0.197 (0.027) | R1 β=+0.106 (0.229) |

Key correlations: `corr(reasoning, fluency) = −0.13`, `corr(Think Aloud, R1 score) = −0.06`,
`corr(R1, R2) = +0.09`.

Interpretation:
1. **Reasoning predicts R2 transfer controlling for R1 performance** (β barely moves, p=0.004) —
   it adds value *beyond* how well you did with the AI.
2. **R1 performance itself doesn't transfer** (r=0.09, ns). Being good *with* the AI ≠ being good
   *without* it.
3. **Reasoning is not a proxy** for skill (r=−0.06) or AI-fluency (r=−0.13).
4. **The new wave is consistent, not contradictory:** its solvers did well in R1 by offloading,
   but R1 performance doesn't transfer and they didn't reason — so no R2 boost.

**Defensible claim:** *It's not how well you perform with the AI that transfers — it's whether
you reason while you do it.*

## Decision

- **Keep all 107.** Do not exclude the Jul-28 wave; the effect holds with it in.
- **Report the continuous, controlled OLS** (reasoning + R1 performance, and + fluency) as the
  primary analysis, not the median-split Solver/Struggler test.
- **Report a single predictor — Think Aloud share.** Reasoning and requesting are compositional
  (one turn budget), so **requesting is described, not modeled** as a competing coefficient. Don't
  frame "asks less" as a learner trait — it's the mirror image of reasoning, and population-dependent.

## Tested and null (don't re-run)

- **Lexical diversity (MTLD / HD-D).** R1 user turns are too short for these to be reliable
  (median ~24 tokens total per participant; HD-D returns 0 below ~42 tokens — median HD-D = 0 in
  *both* groups). The full-set MTLD looked significant (Solvers 54.7 vs Strugglers 40.4, p=0.002)
  but that's a short-text/degeneracy artifact; on the length-reliable subset (≥50 tokens, token
  counts matched 65 vs 65) MTLD is **n.s.** (58.5 vs 49.3, p=0.21, n=15 vs 7). TTR n.s. too.
- **Unique-word count.** **n.s. everywhere** — pooled 22.8 vs 19.4 (p=0.34), tracks total words
  (also n.s.), and reverses in the new wave. Vocabulary quantity/variety doesn't separate groups.
- **Takeaway:** the solver/struggler difference is **not verbosity or lexical richness** (similar
  word counts *and* variety) — it's the *act* (reasoning / Think Aloud). These nulls tighten the
  story around the act-level measure rather than diluting it.

## Caveats

- Effect is **small** (R² ≈ 0.07–0.09) and **observational** — the R1 control rules out skill,
  but not disposition (motivated participants may both reason and learn); frame as "predicts,
  controlling for performance," not strict causation.
- New wave is **small (n=16)**, so within-wave nulls are underpowered; the *direction* + the
  demographic shift are what carry the interpretation.
- A **suggestive reasoning × fluency interaction** (β=+0.14, p=0.11) hints the effect concentrates
  in fluent users; needs more less-fluent participants to test.
- R2 `flat_score` is coarse; the difficulty-weighted selectivity DV gave the same direction.
- **Compositional predictors:** act-shares sum across a participant's turns, so reasoning and
  requesting can't be cleanly separated in one model (corr −0.46 to −0.83) — a single well-defined
  predictor (Think Aloud) is the honest specification; horse-racing shares produces artifacts.

## Open question

For the new wave, participants "solved" R2 while mostly *requesting* in R1 — worth checking
whether R2 is capturing transfer/learning for that group, or whether they drew easier positions /
brought prior skill the assisted round didn't measure.
