# Othello for the game user study

Same structure as `games/connect_four` in the phase-1 study, with one important
difference: Connect Four puzzles came from Zeilberger's book with known forced
wins, while these Othello endgames ship with **no published solution**. So the
ground truth is computed here — an endgame is small enough to search to the very
last disc, which makes every claim in `solution_*.json` exact rather than an
engine's opinion.

```
engine.py             rules, board I/O, heuristic minimax, exact endgame search
solver.py             puzzle-level analysis on top of engine: optimal moves,
                      best defense, disc-loss grading
precompute.py         walk a puzzle's optimal subtree -> solution_<id>.json
import_puzzles.py     import real puzzles (othelloclub.com), solve + rank them
generate_puzzles.py   manufacture puzzles by self-play when a custom one is needed
motif_analysis.py     classify what STRATEGY each puzzle's answer requires
verify_against_source.py  checklist for hand-checking answers against the site
test_othello.py       fast self-checks (python test_othello.py)
validate_engine.py    slower correctness pass: perft, symmetry, exhaustive search
othelloclub_*.txt     archives of imported puzzles (the source of record)
puzzle_config_<id>.txt    a puzzle's starting position
solution_<id>.json        its precomputed optimal subtree
```

## Conventions

- **The participant is Black and moves first**; the AI is White. Puzzles that
  were originally White-to-move are color-flipped on import, which changes
  nothing about the position.
- `board[row][col]`, **row 0 is the top row** — the opposite of the Connect Four
  engine, because Othello numbers rows downward. Cell values `EMPTY/BLACK/WHITE`
  = `0/1/2`, so a board is JSON-serializable straight into the session.
- Moves are `(row, col)` internally and algebraic `"d3"` on the wire
  (`to_notation` / `from_notation`; `parse_move` accepts either, plus `[r, c]`
  and a flat 0-63 index).
- A player with no legal move **passes** (`must_pass`, `side_to_move`); the game
  ends only when neither side can move, which is not always a full board.
- Value of a position = **final disc difference, Black minus White**, under
  optimal play. Empty squares are not awarded to the winner; that affects only
  the margin of a game that ends early, never who won.

## Puzzle file format

`load_puzzle(path) -> (board, to_move)`, or `load_board(path)` for the board
alone (the Connect Four-compatible signature). The parser accepts, in any mix:

```
# comments and blank lines are ignored
*, *, W, B, *, *, *, *      # ", "-separated, top row first (Connect Four style)
--------                    # or eight bare characters per row
0022220000212220...         # or one 64-character line (othelloclub's format)
to_move: B                  # optional; defaults to Black
```

Cell characters: `* . - _ 0` empty, `B X 1` black, `W O 2` white (case
insensitive). A 64-character line may carry the side to move glued on the end
(`...0111110 2`), and column-letter / row-number rulers are stripped, so a board
copy-pasted out of `board_to_text` parses back in.

## Importing puzzles

`othelloclub.com/en/puzzle_ep.php` publishes a daily endgame puzzle whose task —
"win from here" — is exactly the study's task. `othelloclub_puzzles.txt` holds a
fetched batch as `DATE|TURN|BOARD` lines.

```bash
python import_puzzles.py --dry-run                    # solve + rank the batch
python import_puzzles.py --max-empties 13 --write     # emit config + solution files
python import_puzzles.py --url "https://othelloclub.com/en/board.php?board=...&turn=1" --write
python import_puzzles.py --board <64 chars> --turn 2 --tag mine --write
```

All 20 imported puzzles are Black wins with a **unique** optimal move, solve in
about 0.1 s, and 17 of 20 punish the naive "flip the most discs now" move:

| puzzle | empties | legal | best | margin | gap | naive move costs | forced line |
|---|---|---|---|---|---|---|---|
| oc20260729 | 10 | 6 | h4 | +2 | 4 | g2: 30 | 1 |
| oc20260728 | 9 | 6 | h4 | +14 | 18 | h7: 18 | 2 |
| oc20260727 | 10 | 6 | b7 | +18 | 32 | c8: 40 | 7 |
| oc20260726 | 9 | 7 | a3 | +2 | 4 | b1: 8 | 1 |
| oc20260725 | 10 | 8 | h1 | +4 | 8 | h2: 22 | 6 |
| oc20260724 | 9 | 5 | a6 | +2 | 14 | h7: 14 | 5 |
| oc20260723 | 10 | 8 | h8 | +5 | 7 | d2: 19 | 5 |
| oc20260722 | 9 | 5 | g7 | +2 | 4 | f7: 12 | 5 |
| oc20260721 | 10 | 6 | h3 | +4 | 6 | b8: 28 | 2 |
| oc20260720 | 9 | 8 | h5 | +2 | 4 | h3: 12 | 5 |
| oc20260719 | 10 | 7 | a7 | +1 | 2 | c2: 4 | 5 |
| oc20260718 | 9 | 6 | a3 | +2 | 4 | b8: 40 | 5 |
| oc20260717 | 10 | 7 | h5 | +10 | 12 | g7: 14 | 6 |
| oc20260716 | 9 | 6 | d8 | +12 | 15 | d8: 0 | 1 |
| oc20260715 | 10 | 6 | h1 | +6 | 18 | a4: 28 | 3 |
| oc20260714 | 9 | 6 | a7 | +4 | 10 | b2: 10 | 2 |
| oc20260713 | 10 | 7 | g7 | +8 | 12 | b6: 38 | 6 |
| oc20260712 | 9 | 6 | e1 | +10 | 28 | e1: 0 | 3 |
| oc20260711 | 10 | 6 | a1 | +6 | 8 | g8: 24 | 4 |
| oc20260710 | 9 | 6 | a8 | +10 | 12 | a8: 0 | 5 |

- **margin** — final Black−White disc difference with best play by both sides.
- **gap** — discs lost by the second-best move; how close a near-miss is.
- **naive move costs** — discs lost by the max-flips move. `0` means the obvious
  move is also the right one, i.e. the puzzle does not discriminate.
- **forced line** — how many Black decisions in a row have a *single* correct
  answer, i.e. how long the puzzle keeps testing them.

For picking a round-1 / round-2 pair, `forced_line >= 5` with a nonzero naive
cost is the discriminating end (oc20260727, oc20260713, oc20260725, oc20260717),
and a short line with a small gap is the easy end.

## Working set on disk

Only the puzzles the study uses are kept here, plus a few documented alternates:

| file | role |
|---|---|
| `oc20260727` | round 1 (assisted) |
| `b220260706`, `bg20260726` | round 2, the two-puzzle transfer block |
| `b220260507` | third puzzle that passed the transfer-candidate filter |
| `oc20260713`, `oc20260725` | earlier round-2 picks, kept as swap-backs |

Everything else was deleted, because nothing is lost by doing so: the archive
files below are the source of record, and any puzzle regenerates (config +
exactly-solved solution table) in seconds.

| archive file | puzzles | prefix |
|---|---|---|
| `othelloclub_puzzles.txt` | 20 expert, page 1 | `oc` |
| `othelloclub_expert_more.txt` | 75 expert, pages 2-5 | `ep` |
| `othelloclub_beginner_puzzles.txt` | 20 beginner, page 1 | `bg` |
| `othelloclub_beginner_more.txt` | 78 beginner, pages 2-5 | `b2` |

```bash
python import_puzzles.py --source othelloclub_beginner_more.txt --prefix b2 --write
```

Note that `motif_analysis.py` and the tables below scan whatever is on disk, so
re-import the relevant archive before regenerating them.

## Motif matching (how the study's puzzles were chosen)

Difficulty numbers alone don't say whether two puzzles test the same *idea*.
Classifying each move on the optimal line by square type does: corner (a1, a8,
h1, h8), C-square (edge-adjacent to a corner: a2, b1, …), X-square (diagonally
adjacent to a corner: b2, b7, g2, g7 — the square every primer says never to
play), edge, interior.

| puzzle | answer | class | line by class | corners | greedy move costs | White passes |
|---|---|---|---|---|---|---|
| oc20260727 | b7 | **X-square** | X → edge → C → edge → corner → corner | 2, both late | 40 | 3× |
| oc20260713 | g7 | **X-square** | X → corner → C → corner → X | 2 | 38 | 1× |
| oc20260725 | h1 | **corner** | corner → C → corner → C → C → corner | 3, one at once | 22 | 1× |

Every puzzle the study uses is answered by an **X-square**, so round 2 tests
whether the round-1 insight generalised rather than whether the participant
remembers a square:

| | puzzle | answer | greedy move costs | scored decisions |
|---|---|---|---|---|
| round 1 (assisted) | oc20260727 | b7 | 40 | 6 |
| round 2, first | b220260706 | g7 | 16 | 3 |
| round 2, second | bg20260726 | b2 | 16 | 3 |

Three different coordinates, and **b7 is not even playable** in either round-2
puzzle, so round-1 coordinate memory is worth nothing. Both round-2 puzzles also
offer a *decoy* X-square, so "play next to the corner" is a coin flip and the
concept has to be applied rather than recited.

Round 2 is two small (5-empty) puzzles rather than one large one because it is
the primary outcome, and the first two batches showed a 10-empty round 2 getting
a median of 15 seconds of attention and producing 0 wins in 8 attempts. Small is
not the same as guessable: these two were picked from 98 candidates for having
the lowest chance-win rates (4.8% and 4.0%) — most beginner puzzles sit at
15-30%. See `OTH_ROUNDS` in app.py for the full reasoning.

0725 shows what a *bad* match looks like: its answer is an immediate corner grab,
the most intuitive move in the game, and its whole line stays on the corner/edge
ring. Pairing it with 0727 would test general endgame reasoning rather than
transfer — a participant could ace it having learned nothing in round 1.

Regenerate this analysis with `python motif_analysis.py` (it scans whatever
puzzles are on disk).

## Solution tables

`precompute.py` walks the subtree the study can actually reach — Black's optimal
moves crossed with White's single best reply — and tabulates it by
`engine.state_key(board, to_move)`:

```json
"…0111110:1": {"best_moves": ["h4"], "optimal_diff": 2,
               "move_values": {"f1": -2, "g2": -28, "h4": 2, …},
               "must_pass": false},
"…0111110:2": {"best_defense": "g2", "optimal_diff": 2},
"_meta":      {"empties": 10, "optimal_diff": 2, "result": "black_win",
               "root_best_moves": ["h4"], "source": {…}}
```

`move_values` on Black's nodes means a deviation can still be graded exactly from
the table, in discs lost. Once play leaves the table entirely, `engine.ai_move`
and `engine.grade_move` take over and are themselves exact at these sizes — the
table is a latency optimization, not the only source of truth.

```bash
python precompute.py --config puzzle_config_oc20260729.txt -o solution_oc20260729.json
python solver.py puzzle_config_oc20260729.txt      # one-off solve + report
python engine.py puzzle_config_oc20260729.txt      # print the board + best moves
```

## What the web layer calls

`engine.py` is loaded by `app.py` the same way `cf_engine` was
(`_load_local_module`), and offers the equivalent surface:

| Connect Four | Othello |
|---|---|
| `load_board(path)` | `load_board(path)` / `load_puzzle(path)` |
| `valid_columns(board)` | `legal_moves(board, piece)` |
| `is_valid_column(board, col)` | `is_valid_move(board, move, piece)` |
| `drop(board, col, piece)` | `apply_move(board, move, piece)` → flipped discs |
| `winning_move` / `is_full` | `status(board)` / `is_game_over(board)` |
| — | `must_pass(board, piece)`, `side_to_move(board, nominal)` |
| `ai_move(board)` | `ai_move(board)` (exact in the endgame; `None` = pass) |
| `state_key(board, to_move)` | `state_key(board, to_move)` |
| `board_to_text(board)` | `board_to_text(board, to_move=None)` |
| `render_image(board)` | `render_image(board, legal_for=None)` |
| `solver.grade(...)` | `solver.grade(...)` → adds `disc_loss` |

Two things the Connect Four flow does not have and the UI will need: a move is a
**square, not a column**, so the client must send `{"move": "d3"}` and should
show the legal squares (`legal_moves`); and either side may have to **pass**,
which the server has to detect and tell the client about rather than waiting for
a move that cannot come.

## Tests

```bash
python test_othello.py
```

Checks that the bitboard and list rule implementations agree move-for-move over
complete random games, that the exact endgame value matches a full-depth search
with the independent list minimax, that every accepted file layout parses to the
same board, that passes are handled as passes, and that all 20 shipped puzzles
play out from their own tables to exactly the margin the table promises.
