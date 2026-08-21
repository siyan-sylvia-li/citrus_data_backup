"""Exact analysis of an Othello puzzle: optimal moves, best defense, grading.

The search itself lives in engine.py (`bb_negamax` and friends — bitboard
alpha-beta that plays every line to the end of the game). This module is the
puzzle-level layer on top of it, the same split as
games/connect_four/solver.py: the engine knows the rules, the solver knows what
a "correct answer" is and how wrong a wrong answer was.

Exactness is the point. Our puzzles are endgames with no published solution, so
ground truth has to be computed — and in an endgame it can be, completely.

Value model (always from BLACK's POV, since Black is the participant)
--------------------------------------------------------------------
    value = final_black_discs - final_white_discs   under optimal play
    value > 0 -> Black wins, value == 0 -> draw, value < 0 -> White wins

A move's *cost* is therefore a disc count: playing a move worth +2 when a +8 move
was available is a 6-disc error. That is the natural cp_loss analog for Othello,
and it is what `grade` reports alongside the binary "was it optimal".

Empty squares are NOT awarded to the winner (plain disc difference). That only
changes the margin of games that end early, never who won.

Public API (mirrors games/connect_four/solver.py):
    solve, move_values, best_moves, best_defense, grade, build_memo
"""

from __future__ import annotations

import sys
import time

from engine import (
    BLACK, WHITE, Board, Move,
    ORDER_WEIGHT, exact_move_values, exact_value, empty_count, load_puzzle,
    parse_move, sq_index, to_notation,
)

# Above this many empty squares a full solve gets slow in pure Python (roughly an
# order of magnitude per two empties: ~0.3s at 10, ~5s at 12, ~20s at 14 for a
# full all-moves solve). Warn rather than refuse.
SOLVE_WARN_EMPTIES = 14


def solve(board: Board, to_move: int, memo: dict | None = None) -> int:
    """Exact final disc difference (Black - White) under optimal play.

    `memo` is the shared transposition table; pass the same dict across calls on
    one puzzle and every lookup after the first is nearly free.
    """
    return exact_value(board, to_move, memo)


def move_values(board: Board, to_move: int,
                memo: dict | None = None) -> dict[Move, int]:
    """{move: exact final disc difference (Black - White)} for every legal move."""
    return exact_move_values(board, to_move, memo)


def best_moves(board: Board, to_move: int, memo: dict | None = None) -> list[Move]:
    """Every move that achieves the optimal result for `to_move` (all ties)."""
    values = move_values(board, to_move, memo)
    if not values:
        return []
    best = (max if to_move == BLACK else min)(values.values())
    return sorted(m for m, v in values.items() if v == best)


def best_defense(board: Board, memo: dict | None = None) -> Move | None:
    """WHITE's optimal reply (minimize Black's final disc count), or None to pass.

    Ties break towards the structurally strongest square so the AI takes the
    corner rather than an equal-scoring quiet move — a defense that looks
    sensible to the participant, not merely one that scores the same.
    """
    moves = best_moves(board, WHITE, memo)
    if not moves:
        return None
    return max(moves, key=lambda m: ORDER_WEIGHT[sq_index(m)])


def grade(board_before: Board, move, memo: dict | None = None) -> dict:
    """Exact grade of Black's move from `board_before` (the cp_loss analog).

    Returns both the binary "was it optimal" the study scores on and the
    continuous disc loss, which is the better trajectory signal: a 2-disc slip
    and a 20-disc blunder are not the same mistake.
    """
    memo = {} if memo is None else memo
    played = parse_move(move)
    values = move_values(board_before, BLACK, memo)
    if not values:
        raise ValueError("Black has no legal move in this position")
    best = max(values.values())
    played_value = values.get(played)
    return {
        "best_moves": [to_notation(m) for m in sorted(values) if values[m] == best],
        "optimal_diff": best,                # final Black-White margin with best play
        "played_move": to_notation(played),
        "played_diff": played_value,         # None if the move was illegal
        "disc_loss": None if played_value is None else best - played_value,
        "optimal": played_value == best,     # the binary "correct"
        "still_winning": played_value is not None and played_value > 0,
        "move_values": {to_notation(m): v for m, v in sorted(values.items())},
    }


def build_memo(board: Board, to_move: int = BLACK) -> dict:
    """Solve from the start position; the recursion fills the whole table."""
    memo: dict = {}
    solve(board, to_move, memo)
    return memo


def outcome_name(diff: int) -> str:
    return "black_win" if diff > 0 else "white_win" if diff < 0 else "draw"


if __name__ == "__main__":
    # One-time solve + sanity report. Run:
    #   python solver.py puzzle_config_<id>.txt
    path = sys.argv[1] if len(sys.argv) > 1 else "puzzle_config.txt"
    board, to_move = load_puzzle(path)
    empties = empty_count(board)
    if empties > SOLVE_WARN_EMPTIES:
        print(f"warning: {empties} empty squares — this may take a long while")

    memo: dict = {}
    t = time.process_time()
    values = move_values(board, to_move, memo)
    dt = time.process_time() - t

    root = solve(board, to_move, memo)
    who = "Black" if to_move == BLACK else "White"
    print(f"solved in {dt:.1f}s | empties: {empties} | tt entries: {len(memo):,}")
    print(f"root value ({who} to move): {root:+d} discs -> {outcome_name(root)}")
    print("optimal moves: " + ", ".join(to_notation(m)
                                        for m in best_moves(board, to_move, memo)))
    print("all moves: " + ", ".join(f"{to_notation(m)}={v:+d}"
                                    for m, v in sorted(values.items())))
