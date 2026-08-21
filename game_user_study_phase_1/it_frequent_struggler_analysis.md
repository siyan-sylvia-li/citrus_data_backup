# Why IT Professionals & Frequent GenAI Users Tend to Be Strugglers

*Phase-1 Connect Four user study — round-1 (AI-assisted) vs round-2 (AI-free transfer) analysis.*

## Headline

**IT professionals and frequent GenAI users used the round-1 AI as an answer engine
instead of a tutor — offloading the reasoning rather than building it — so when round 2
removed the AI, they had nothing to transfer.**

The study design makes this legible:

- **Round 1 (puzzle 15):** an AI assistant was available.
- **Round 2 (puzzle w4p6):** no AI.
- "Struggler" (`flat_score <= 1`) vs "Solver" (`flat_score >= 2`) is defined by the
  **AI-free round 2** — so the question is really about *how each group used the assistant*.

## The Association Being Explained

Two correlated demographic facts both predict struggling:

| Split | Solvers | Strugglers | n |
|---|---|---|---|
| IT professionals | 41% | **59%** | 27 |
| non-IT | 62% | 38% | 64 |
| Frequent GenAI (≥ few/wk) | 46% | **54%** | 61 |
| Infrequent GenAI | 77% | 23% | 30 |

IT and frequent-GenAI-use are themselves correlated (φ = 0.30, χ² p = 0.004 within the
analysis groups; φ = 0.22, p = 0.02 over all demographics), so they are **not two
independent stories** — solution-offloading is the common thread linking both to struggling.

## Evidence Chain

### 1. They succeeded *with* the AI, then collapsed without it

| Group | Round 1 (with AI) | Round 2 (no AI) |
|---|---|---|
| All | 2.24 | 1.52 |
| IT | 2.44 | 1.37 |
| IT & Frequent | **2.50** | **1.42** |
| non-IT | 2.16 | 1.58 |
| Infrequent | 2.23 | 1.77 |
| Solvers | 2.24 | 2.22 |
| Strugglers | 2.25 | 0.62 |

IT & Frequent users scored *above* average in round 1 but *below* average in round 2.
Round-1 → round-2 score correlation across everyone is ≈ 0 (**r = 0.03**) — assisted
performance did not carry over. Solvers held steady across rounds (2.24 → 2.22);
Strugglers cratered (2.25 → 0.62).

### 2. Behavioral fingerprint: solution-offloading

Dialogue-act labels (ShareChat taxonomy) on the round-1 chats:

| Group | Solution Requests | Think Aloud |
|---|---|---|
| Solvers | 25% | 29% |
| Strugglers | **44%** | **14%** |
| IT | 41% | 20% |
| IT & Frequent | 42% | 23% |

IT/frequent users disproportionately *ask for the move*; Solvers disproportionately
*reason out loud and check their own idea*.

### 3. Per-person, behavior predicts the outcome

- corr(Solution-Request share, round-2 score) = **−0.26**
- corr(Think-Aloud share, round-2 score) = **+0.27**

Asking for the answer → worse transfer; reasoning aloud → better transfer.

### 4. The traces make the mechanism concrete

**IT & Frequent Strugglers** (100% solution-requests) — extracting answers, suppressing explanation:

- *"Help me with this move"*
- *"what is the best move now?"*
- *"tell me the best sequence of my next three moves... list only the column numbers"*
  → *"Reply with only the column number."*

**non-IT Solvers** (high think-aloud) — proposing a move *with rationale*, using the AI to verify:

- *"I was thinking of putting one in 3-3"* → *"it would prevent yellow and open up one opportunity for me"*
- *"im thinking of adding red on the 3rd column to set up the diagonal four... Is my idea fine
  or do you have a better suggestion?"*

The Strugglers' "list only the column numbers / reply with only the column number" is the
tell: an efficient, well-engineered prompt to get the answer with no explanation — great for
productivity, terrible for learning. That is a habit fluent AI users bring with them.

## Interpretation

IT professionals and frequent GenAI users are habituated to AI as a fast oracle. That habit
is optimal when the AI stays available, but here it meant they never internalized the
Connect-Four reasoning — so the AI-free transfer test exposed the gap. There is also a mild
overconfidence signal: IT rate their own skill higher (2.00 vs 1.84) yet do worse unaided.

## Caveats

- Correlational; small n (IT n = 27, IT & Frequent n = 24).
- IT ↔ frequent-use are correlated (φ = 0.30) — solution-offloading is the common thread,
  not two independent effects.
- Conversations are short (~2.9 turns/person), so dialogue-act shares are coarse per person.
- "Frequent" uses the notebook's `FREQUENT_GENAI = {few_times_week, daily, several_times_day}`
  (plain `weekly` counts as infrequent).

## Reproduction

- Data: `recordings-download/<pid>/` — `demographics.json`, `annotated_conversation_p15.jsonl`
  (dialogue-act labels under `annotation_user.final`), `cf_score_p15.json` (round 1),
  `cf_score_selectivity_pw4p6.json` (round 2, `flat_score`).
- Groups: Solver = round-2 `flat_score >= 2`; Struggler = `flat_score <= 1`
  (requires `conversation_p15.jsonl` to exist).
- Analysis notebook: `data_analysis_2group.ipynb`.
