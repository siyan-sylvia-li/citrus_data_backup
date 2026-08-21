# Selecting puzzles of similar difficulty to puzzle 15

Goal: the study currently uses **puzzle 15** (a "win-in-five" Connect Four
position, identical to [puzzle_config.txt](puzzle_config.txt)). We want three
*other* puzzles of comparable difficulty so participants can be given
equivalent tasks. This documents how we chose them and what we found.

See [DIFFICULTY.md](DIFFICULTY.md) for the underlying difficulty theory and
references (Pelánek, Guid–Bratko, Allis). Reproduce with
[rank_similar_puzzles.py](rank_similar_puzzles.py).

## Source of puzzles

Doron Zeilberger's "Win-In-Five-Moves" problem set (15 problems):
https://sites.math.rutgers.edu/~zeilberg/C4/ch5/Problems.html

Each problem `n` has three pages: `P<n>` (position as a GIF), `A<n>` (the
first move), and `S<n>` (a machine-generated **detailed proof tree**). The
`S<n>` page prints the starting position and the full forced win as text grids,
`0`=empty `1`=Red `2`=Yellow — which is exactly our engine's
`EMPTY/RED/YELLOW` encoding, just top-row-first instead of bottom-row-first.

Puzzle 15's starting position parses to precisely our current
`puzzle_config.txt`, confirming #15 is the one in use.

## What the script does

`rank_similar_puzzles.py`, four steps:

1. **Scrape** the 15 starting positions from the `S<n>` pages (cached to
   `zeilberger_positions.json` so it only fetches once). Fetching is done with
   `curl` into the cache because Python 3.14's `urllib` fails TLS verification
   on this host; the script reads the cache.
2. **Rebuild each puzzle under our study's conditions.** We do *not* reuse
   Zeilberger's tree (it branches on Black's replies while fixing one Red move).
   Instead we run our own `precompute.build_solution` on each position, which
   follows *Red's winning columns* and *Yellow's single best defense* — the
   exact experience the participant faces (Red = participant, Yellow = the AI
   playing its toughest defense). This keeps every puzzle measured on the same
   footing as #15.
3. **Score** each puzzle with the difficulty metrics below.
4. **Rank by similarity.** Each puzzle becomes a vector of the metrics,
   z-normalized across the 15 puzzles, and we take the Euclidean distance to
   puzzle 15. The nearest are the most similar in difficulty.

## Why raw depth doesn't work here

All 15 problems are win-in-five, i.e. **9 plies** by construction, so solution
*depth* is constant and can't separate them. The difficulty differences live in
the **shape** of the forced-win tree, which is what the metrics capture.

## Metrics (per puzzle)

Measured on the principal variation (PV = Red's fastest-win column → Yellow's
best defense) and over the whole optimal subtree:

- **`forced_run`** — length of the leading run of unique "only-move" Red nodes
  on the PV. Pelánek's step-dependency: a long chain of forced setup moves,
  each gating the next, is hard. (Puzzle 15 = 3.)
- **`pv_only_moves`** — total Red nodes on the PV with a single winning column.
- **`uniqueness_ratio`** — fraction of *all* Red nodes in the subtree that have
  a unique winning column (narrower = harder).
- **`mean_winning_cols`** — average number of winning columns per Red node
  (lower = harder).
- **`mean_decoys`** — Guid–Bratko style: average number of non-winning legal
  Red moves per PV node that still look non-losing at a shallow (depth-3)
  search. Many plausible moves but only one wins ⇒ harder.
- **`tree_size`** — total nodes in the optimal subtree (search effort / breadth
  of the defense the participant must handle).

## Result

Ranked by distance to puzzle 15 (smaller = more similar):

| pz | dist | forced_run | pv_only_moves | uniqueness | mean_win_cols | mean_decoys | tree_size |
|----|------|-----------|---------------|-----------|---------------|-------------|-----------|
| **15** | — (target) | 3 | 3 | 0.57 | 1.57 | 3.2 | 13 |
| **1** | **0.71** | 2 | 3 | 0.57 | 1.43 | 3.2 | 13 |
| **2** | **1.17** | 3 | 3 | 0.67 | 1.33 | 2.4 | 11 |
| **7** | **1.78** | 4 | 4 | 0.80 | 1.20 | 3.8 | 9 |
| 4 | 2.08 | 1 | 4 | 0.83 | 1.17 | 3.6 | 12 |
| 14 | 2.29 | 2 | 4 | 0.78 | 1.33 | 4.8 | 17 |
| 3 | 2.37 | 1 | 2 | 0.27 | 1.80 | 2.4 | 29 |
| 9 | 2.97 | 5 | 5 | 1.0 | 1.0 | 2.8 | 9 |
| 5 | 3.05 | 1 | 1 | 0.40 | 1.6 | 5.0 | 29 |
| 10 | 3.07 | 5 | 5 | 1.0 | 1.0 | 4.0 | 9 |
| 13 | 3.08 | 1 | 1 | 0.26 | 2.0 | 3.6 | 39 |
| 11 | 3.33 | 5 | 5 | 1.0 | 1.0 | 4.6 | 9 |
| 12 | 3.42 | 1 | 3 | 0.42 | 1.92 | 4.8 | 49 |
| 8 | 3.57 | 1 | 2 | 0.31 | 2.08 | 3.8 | 54 |
| 6 | 3.70 | 5 | 5 | 1.0 | 1.0 | 5.2 | 9 |

## Recommendation: puzzles 1, 2, 7

They share puzzle 15's signature shape — a short forced stretch of only-moves at
the top, then the position opens into several winning continuations. Puzzle **1**
is almost a twin (same tree size, decoy load, and uniqueness). **2** is very
close. **7** is the best pick if you want a touch more difficulty.

The 15 puzzles fall into three natural clusters; #15 sits in the middle one, and
the recommended three come from the same cluster:

- **Fully-forced single lines** (6, 9, 10, 11): `uniqueness = 1.0`,
  `forced_run = 5`, tiny tree — *every* Red move is the unique only-move.
  Rigidly hard in a different way; unlike #15.
- **Wide, many-win puzzles** (3, 8, 12, 13): large trees (29–54 nodes), low
  uniqueness — many winning options, so easier to find *a* win. Unlike #15.
- **Mixed forced-then-open** (15, 1, 2, 7, 4, 14): #15's family — choose from
  here.

## Caveats

- This is a **model-based proxy**, appropriate for *pre-selecting* candidates.
  The gold standard is empirical: validate 1/2/7 against #15 with pilot human
  solve-rates or an LLM-solver sweep before locking them in.
- The PV walk picks Red's fastest-win column; other PV tie-break choices shift
  `forced_run` slightly but not the clustering.

## Reproduce

```
cd game_user_study_pre_filter/games/connect_four
# one-time position cache (Python's urllib fails TLS here, so use curl):
for n in $(seq 1 15); do curl -s "https://sites.math.rutgers.edu/~zeilberg/C4/ch5/S$n" -o /tmp/S$n.txt; done
# then build zeilberger_positions.json from the S<n> grids (see script header), and:
python rank_similar_puzzles.py --target 15 --k 3
```
