# CITRUS — Coarse Dialogue-Act Codebook (v0.1)

A deliberately **coarse** scheme for coding *participant* utterances in the
Connect Four coaching task, where the human directs an AI assistant that
**withholds the final move**. Designed for **directional observations** with high
inter-coder agreement — not fine-grained discourse analysis.

The participant is the "student"; the AI is the "tutor." We code **only the
participant's** turns (the AI's replies are not coded here). The scheme is a
coarsening of tutor–student / peer schemes (Vail & Boyer student moves; Erkens &
Janssen; loosely ISO 24617-2 dimensions), collapsed to survive the two things
that broke the fine scheme on this data: pervasive **compound utterances** and
fuzzy question-type boundaries.

---

## Unit of analysis
- **Code at the clause level**, not the whole chat message. One message often
  contains several acts (e.g. *"I cut them off. I think if I played row 3 they'd
  put one on top?"* = a NARRATING clause + a REASONING clause).
- **Multi-label is allowed.** If a clause genuinely does two things, tag both and
  mark one as **primary** (the act that carries the clause's main intent).
- *(The v0 analysis coded at the message/turn level — note this when comparing.)*

---

## Categories

### 1. REASONING — works the problem out themselves
A **statement** in which the participant reasons about the position rather than
asking the AI to do it. Two optional sub-codes:

- **R-Analysis** — assessing the board, threats, or consequences.
  - *"Yellow has three in a row now, but so do I."*
  - *"Column 2 gets immediately blocked and forces me to play column 3 next turn."*
  - *"Even if they complete the 3-diagonal they still wouldn't have an immediate threat."*
- **R-Plan** — stating what they intend to do (forward-looking intent).
  - *"I'm thinking of building off my two diagonals and blocking the stacked yellows."*
  - *"Okay, now I must block their connect four."*

**Include:** hypotheses stated as claims, lookahead, weighing options aloud.
**Exclude:** if it's phrased as a question to the AI → that's ASKING (even if it
contains reasoning, e.g. *"if I play row 3, would they just stack on top?"* is
ASKING-primary with REASONING secondary).

### 2. ASKING — queries the AI
Any **question** directed at the assistant. Two optional sub-codes:

- **A-Solution** — wants the move / the answer.
  - *"What should I do next?"*  ·  *"Is there a column that's the best chance for a win?"*
  - *"Which ones would let me win — just the possible ones."* (constraint-adapted)
- **A-Info/Verify** — wants board information or a check on the participant's own idea (NOT the final move).
  - *"Where's the biggest threat?"*  ·  *"How do I think this through?"*
  - *"Would column 4 work?"*  ·  *"Why couldn't I play 3?"*

**Tip:** A-Solution asks the AI to **decide**; A-Info/Verify asks the AI to
**inform or confirm**. When unsure between the two, default to A-Info/Verify
unless they're plainly demanding the move.

### 3. NARRATING — reports state/action, no reasoning
A statement that **reports** what they did or observed, without working anything
out.
- *"I played 4."*  ·  *"I cut them off."*  ·  *"I see a spot on the left."*

**Exclude:** if the report includes *why* or a consequence → REASONING.

### 4. META / OTHER — housekeeping
Anything not about solving the position itself:
- **Task clarification:** *"Will I win in exactly 5 steps or fewer?"*
- **Uptake / acknowledgement:** *"I see," "ok," "got it."*
- **Difficulty:** *"I'm lost," "I have no idea."*
- **Social:** *"thanks."*

---

## Decision rule (apply in order)
1. Is the clause a **question to the AI**? → **ASKING**
   - asks the AI to decide the move → *A-Solution*; asks for info or a check → *A-Info/Verify*
2. Otherwise, is it a **statement that works out the position** (analysis or plan)? → **REASONING**
   - assessing board/consequences → *R-Analysis*; stating intended action → *R-Plan*
3. Otherwise, does it just **report** a state or action? → **NARRATING**
4. Otherwise (clarify task / acknowledge / express difficulty / social) → **META**

The **question-vs-statement** split (step 1 vs 2–3) is the robust backbone and
should drive most agreement.

---

## Edge cases / conventions
- **Compound clause that reasons *and* asks** (e.g. *"if I play row 3 they'd stack
  on top?"*): tag **ASKING (primary) + REASONING (secondary)**.
- **"Should I play X?"** → ASKING / A-Info-Verify (seeking approval of a specific
  idea), *not* A-Solution, unless it's an open *"what should I do?"*.
- **Column enumeration** (*"would 4 work? what about 7?"*): each is ASKING /
  A-Info-Verify; note the pattern, but don't reclassify as REASONING.
- **Grounding / capability checks** (*"can you see the board?"*) were frequent in
  the no-intro version and vanished after the agent intro stated "I can see the
  board." If they recur, code as META (task clarification) and flag.
- **Constraint-adaptation** (*"just tell me the possible ones, not which"*): code
  as ASKING / A-Solution, and flag separately — it's a setting-specific behavior
  worth tracking.

---

## Reliability protocol
- Two coders independently code a shared subset (≥ 20% of utterances, min ~40).
- Report **Cohen's κ** on the 4 top-level categories (primary act). Target κ ≥ 0.7.
- Expect most disagreement at the **A-Solution vs A-Info/Verify** sub-split — report
  sub-split κ separately and treat sub-codes as exploratory until κ is acceptable.
- Resolve disagreements by discussion; log rule clarifications in the changelog.

## Known limitations
- Coarse by design — does not capture argument structure, repair, or the AI's moves.
- Built on a small sample (n=10 conversations / 29 utterances); category prevalence
  and the solved-vs-unsolved contrast are **directional**, not confirmatory.
- Outcome (solved/unsolved) is confounded with raw game skill, so act mix is not a
  clean cause of success.

## Provenance
Coarsening of: Vail & Boyer (tutor/student dialogue moves); Erkens & Janssen
(CSCL argumentative/epistemic/social acts); ISO 24617-2 / DIT++ dimensional
framing. No existing scheme is purpose-built for human↔(withholding)AI reasoning
assistance — this is an adaptation for that setting.

## Changelog
- **v0.1** — initial coarse scheme (REASONING / ASKING / NARRATING / META, with
  optional R- and A- sub-splits). Coded at message level in v0 analysis; codebook
  recommends clause-level going forward.
