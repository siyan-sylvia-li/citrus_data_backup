"""Exact solver for the fixed Connect Four puzzle (the precompute approach).

Unlike engine.minimax (depth-limited + heuristic), this searches to the END of
the game and returns the EXACT game-theoretic result from RED's point of view,
with a distance-to-end so the AI defends as long as possible and the participant
must find the *full* winning line. A transposition table (`memo`) dedupes shared
positions, making the one-time solve tractable and later lookups O(1).

Because the puzzle is fixed, you solve once (see `build_memo` / the __main__
block), optionally pickle the table to disk, and the web layer just does lookups.

Value model (always from RED's POV)
-----------------------------------
Each position resolves to (outcome, dist):
    outcome in {WIN, DRAW, LOSS}   # for RED
    dist     = plies until the game ends under optimal play, from THIS node
We compare with a scalar key so RED maximizes and YELLOW minimizes the same
number (see _rank): win is best (fewer plies better), loss is worst (more plies
better, i.e. stall as long as possible).

You implement: solve, winning_columns, best_defense  (search for TODO(you)).
The plumbing around them (_key, _rank, grade, build_memo, __main__) is provided.
"""

from __future__ import annotations

import sys
import time

from engine import (
    RED, YELLOW, Board,
    valid_columns, drop, copy_board, winning_move, is_full, opponent, load_board,
)

WIN, DRAW, LOSS = 1, 0, -1          # outcome from RED's POV

Value = tuple[int, int]             # (outcome, dist_to_end_in_plies)


# ---------------------------------------------------------------------------
# Provided plumbing
# ---------------------------------------------------------------------------
def _key(board: Board, to_move: int):
    """Hashable transposition-table key for (position, side to move).

    Lists aren't hashable, so convert the board to a tuple of tuples.
    """
    return (tuple(tuple(row) for row in board), to_move)


def _rank(value: Value) -> tuple[int, int]:
    """Scalar sort key so RED can maximize and YELLOW can minimize the SAME number.

        win  -> (2, -dist)   draw -> (1, 0)   loss -> (0, dist)
    """
    outcome, dist = value
    if outcome == WIN:
        return (2, -dist)
    if outcome == DRAW:
        return (1, 0)
    return (0, dist)


# ---------------------------------------------------------------------------
# TODO(you) — the solver
# ---------------------------------------------------------------------------
def solve(board: Board, to_move: int, memo: dict) -> Value:
    """Exact (outcome, dist) from RED's POV for `board` with `to_move` to play.

    TODO(you): implement the recursion.

    1. Look up _key(board, to_move) in `memo`; return it if present.
    2. Terminal cases (compute, store in memo, return):
         winning_move(board, RED)    -> (WIN, 0)
         winning_move(board, YELLOW) -> (LOSS, 0)
         is_full(board)              -> (DRAW, 0)
    3. Otherwise recurse over valid_columns(board):
         child = copy_board(board); drop(child, col, to_move)
         outcome, dist = solve(child, opponent(to_move), memo)
         child_val = (outcome, dist + 1)        # one more ply from here
       Choose the child_val that's best for `to_move`:
         to_move == RED  -> the one with the MAX _rank(child_val)
         to_move == YELLOW -> the one with the MIN _rank(child_val)
       (Hint: Python's max()/min() take key=_rank.)
    4. Store the chosen value in memo and return it.

    This always terminates: every move fills a cell, so the board can't recurse
    forever, and the memo means each distinct position is solved only once.
    """
    curr_key = _key(board, to_move)
    if curr_key in memo:
        return memo[curr_key]
    # Terminal: the game is already decided at this position.
    if winning_move(board, RED):
        memo[curr_key] = (WIN, 0)
        return memo[curr_key]
    if winning_move(board, YELLOW):
        memo[curr_key] = (LOSS, 0)
        return memo[curr_key]
    if is_full(board):
        memo[curr_key] = (DRAW, 0)
        return memo[curr_key]

    # Otherwise: each move leads to a child. child_val is that child's result
    # re-expressed from HERE — one ply deeper, so dist + 1.
    options = []
    for col in valid_columns(board):
        child = copy_board(board)
        drop(child, col, to_move)
        outcome, dist = solve(child, opponent(to_move), memo)
        options.append((outcome, dist + 1))     # <- this is child_val

    # The side to move chooses: RED maximizes, YELLOW minimizes (the same _rank).
    best = max(options, key=_rank) if to_move == RED else min(options, key=_rank)
    memo[curr_key] = best
    return best


def winning_columns(board: Board, memo: dict) -> list[int]:
    """Columns RED can drop into that still lead to a forced win.

    TODO(you): for each col in valid_columns(board): drop RED into a copy, then
    solve(child, YELLOW, memo); keep col if that outcome == WIN.
    (You only need the outcome here, not the distance.)
    """
    winning_cols = []
    for col in valid_columns(board):
        child = copy_board(board)
        drop(child, col, RED)
        outcome, _ = solve(child, YELLOW, memo)
        if outcome == WIN:
            winning_cols.append(col)
    return winning_cols


def best_defense(board: Board, memo: dict) -> int | None:
    """YELLOW's optimal reply column (defend longest / win fastest), or None.

    TODO(you): same shape as the YELLOW branch of solve — among YELLOW's valid
    moves, pick the column whose resulting (outcome, dist+1) has the MIN _rank.
    """
    best_col, best_val = None, None
    for col in valid_columns(board):
        child = copy_board(board)
        drop(child, col, YELLOW)
        outcome, dist = solve(child, opponent(YELLOW), memo)
        val = (outcome, dist + 1)                # child_val
        # YELLOW minimizes _rank; remember the COLUMN that achieves it.
        if best_val is None or _rank(val) < _rank(best_val):
            best_col, best_val = col, val
    return best_col                              # None if the board was full


# ---------------------------------------------------------------------------
# Provided — works once the three functions above are implemented.
# ---------------------------------------------------------------------------
_OUTCOME_NAME = {WIN: "red_win", DRAW: "draw", LOSS: "red_loss"}


def grade(board_before: Board, col: int, memo: dict) -> dict:
    """Lookup-based grade of RED's move from `board_before` (the cp_loss analog)."""
    wins = winning_columns(board_before, memo)
    child = copy_board(board_before)
    drop(child, col, RED)
    outcome, dist = solve(child, YELLOW, memo)
    return {
        "winning_columns": wins,        # columns that keep the forced win
        "played_col": col,
        "kept_win": outcome == WIN,     # the binary "correct"
        "result_after_move": _OUTCOME_NAME[outcome],
        "plies_to_end": dist,           # smaller = forces the win faster
    }


def build_memo(start_board: Board, to_move: int = RED) -> dict:
    """Solve from the start position; the recursion fills the whole table."""
    memo: dict = {}
    solve(start_board, to_move, memo)
    return memo


if __name__ == "__main__":
    # One-time precompute + sanity report. Run:
    #   python solver.py            # solve and report
    #   python solver.py solution.pkl   # also pickle the table to disk
    board = load_board("puzzle_config.txt")
    t = time.process_time()
    memo = build_memo(board, RED)
    dt = time.process_time() - t

    root = solve(board, RED, memo)
    print(f"solved in {dt:.1f}s | positions in table: {len(memo):,}")
    print(f"root value (RED to move): outcome={_OUTCOME_NAME[root[0]]}, plies_to_end={root[1]}")
    print(f"winning columns for RED: {winning_columns(board, memo)}")

    if len(sys.argv) > 1:
        import pickle
        with open(sys.argv[1], "wb") as f:
            pickle.dump(memo, f)
        print(f"wrote table to {sys.argv[1]}")
