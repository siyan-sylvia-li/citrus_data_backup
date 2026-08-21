# Measuring puzzle difficulty

Notes on how to measure the difficulty of a Connect Four puzzle (Red to move,
forced win), grounded in the puzzle-difficulty literature. Written for the
user-study puzzles; the metrics generalize across `games/*/`.

## What we have to work with

`solution.json` is the precomputed *optimal subtree* for one puzzle. Keys are
`engine.state_key(board, to_move)` (42 board digits — `0` empty, `1` Red, `2`
Yellow — plus `:1` Red-to-move / `:2` Yellow-to-move). Entries:

- Red-to-move nodes: `{"winning_columns": [...], "plies_to_win": int}`
- Yellow-to-move nodes: `{"best_defense": col}`

For the current puzzle the forced win is **9 plies (5 Red moves)**, and the
first three Red moves are each a *unique* winning column that is not an
immediate connect-4 — i.e. a chain of forced setup moves. That structure is
where the difficulty lives.

## On branching factor

Two different quantities get called "branching factor"; only one is useful for
ranking a single puzzle.

- **Game-level branching factor** — Connect Four's is ~4 (≤7 columns, dropping
  as columns fill). It's a *constant of the game*, so it can't distinguish one
  puzzle from another. Not useful here.
- **Solution-search branching** — at each decision node, how many moves must
  actually be considered / look plausible. This *is* puzzle-specific and
  meaningful (the "narrowness vs. decoys" signal).

To fold branching + depth into one number, use the **effective branching
factor** `b*` from the search literature (Nilsson; Russell & Norvig), defined
by `N = (b*)^d` where `N` = nodes a solver expands and `d` = solution depth.
Measure it over the *human-plausible* candidate tree (plausible candidate moves
per node, not all 7), across the 5-move depth.

## What the literature says

Difficulty-metric research clusters into three families, with a consistent
finding: raw combinatorial metrics are weak predictors of *human* difficulty on
their own.

1. **Combinatorial / structural metrics** (branching, state-space size,
   solution length). Allis's thesis (1994) — the work that *solved* Connect
   Four — formalizes state-space and game-tree complexity. Cheap, but weak
   predictors of human difficulty alone.

2. **Computational-model-of-solving metrics.** Pelánek's Sudoku work is the
   canonical reference (~1700 puzzles, thousands of solver-hours). His best
   predictor was a *simulated human solver* (ρ≈0.95), and he identifies two
   sources of difficulty: (a) complexity of each individual step, and (b) the
   **dependency structure among steps** — long chains of forced deductions
   where each step gates the next. Maps directly onto our three consecutive
   unique setup moves.

3. **Search-effort / engine-based metrics.** Guid & Bratko (Stoiljkovikj, Guid
   & Bratko, "A Computational Model for Estimating the Difficulty of Chess
   Problems") — most transferable to a two-player game. Core idea: run a solver
   at increasing search depths and measure **how deep you must search before
   the best move stabilizes**, and **how many alternatives look good at shallow
   depth but turn out wrong**. We already have the machinery for this in
   `precompute.py`'s bounded alpha-beta + `engine.evaluate`.

**Gold standard:** empirical — fit an Elo/IRT rating from human (or model)
solve rates (how Lichess rates tactics puzzles). Everything above is a *proxy*.

## Recommended metrics for these puzzles

Branching factor alone rates our puzzle as unremarkable (~4, like every Connect
Four position). The difficulty actually lives in the step-dependency chain
(Pelánek) and the shallow-search decoys (Guid & Bratko). Measure:

1. **Forced-line length** — consecutive unique-winning-move Red nodes from the
   root (structural, cheap; read straight from `solution.json`).
2. **Uniqueness ratio** — (# Red nodes with a unique winning column) / (# Red
   nodes).
3. **Decoy count via shallow search** — at each Red node, how many *non*-winning
   moves score well at low search depth (Guid–Bratko style), using the existing
   `search()` in `precompute.py`.
4. **Effective branching `b*`** over the human-plausible candidate tree.
5. **Immediacy** — per node, is any winning column an immediate connect-4 or
   only a setup? Setup-only nodes are harder.
6. **Empirical solve rate** (optional, strongest) — from the study, or an
   LLM-solver sweep — to validate the proxies above.

An analyzer can compute 1–5 from `solution.json` + `engine.py` and emit a
per-puzzle difficulty profile, batchable across `games/*/`.

## References

- Pelánek, R. *Difficulty Rating of Sudoku Puzzles: An Overview and
  Evaluation.* arXiv:1403.7373. https://arxiv.org/abs/1403.7373
- Pelánek, R. *Human Problem Solving: Sudoku Case Study.*
  https://www.semanticscholar.org/paper/Human-Problem-Solving-:-Sudoku-Case-Study-%E2%88%97-Pel%C3%A1nek/f0ffd2aa27b1699c7ba866371cf00f3c60e18a88
- Stoiljkovikj, S., Guid, M., Bratko, I. *A Computational Model for Estimating
  the Difficulty of Chess Problems.*
  https://www.semanticscholar.org/paper/A-Computational-Model-for-Estimating-the-Difficulty-Stoiljkovikj-Bratko/add69037c6c5f8eebe98fe7c40a588a8a2b41500
- Guid, M., Bratko, I. *Computer Analysis of World Chess Champions.*
  https://www.researchgate.net/profile/Ivan-Bratko/publication/220174548_Computer_Analysis_of_World_Chess_Champions/links/00463531ebec10d56f000000/Computer-Analysis-of-World-Chess-Champions.pdf
- Allis, L. V. (1994). *Searching for Solutions in Games and Artificial
  Intelligence.* PhD thesis (Connect Four solved; state-space & game-tree
  complexity).
- Russell, S., Norvig, P. *Artificial Intelligence: A Modern Approach* —
  effective branching factor `b*`.
