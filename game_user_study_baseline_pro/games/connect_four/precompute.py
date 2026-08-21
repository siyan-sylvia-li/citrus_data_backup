"""Precompute the puzzle's winning subtree into solution.json (run once, offline).

Why not solver.solve? That searches the WHOLE game tree (down to ~30 plies),
which is intractable in pure Python from a 12-piece board. We don't need it:

- The forced win is only ~9 plies deep.
- With the retry UX, the participant only ever advances by playing a *winning*
  column, and the AI replies with a single best defense. So the set of positions
  ever reached is a small subtree (Red's winning moves x the AI's one reply).

So we walk just that subtree. At each node we classify moves with a *bounded*
alpha-beta search (depth cap comfortably above the 9-ply win) that is also
distance-aware, so the AI (Yellow) defends as long as possible — forcing the
participant to find the whole line instead of getting a quick freebie.

Output: solution.json mapping engine.state_key(board, to_move) -> entry, where
  Red-to-move nodes carry   {"winning_columns": [...], "plies_to_win": int}
  Yellow-to-move nodes carry {"best_defense": col}
The web layer loads this and does O(1) lookups — no engine at request time.

Run:  python precompute.py            # build + report
      python precompute.py -o solution.json
"""

from __future__ import annotations

import argparse
import json
import math

import engine
from engine import (
    RED, YELLOW, COLS, Board,
    valid_columns, drop, copy_board, winning_move, is_full, opponent, state_key,
    load_board,
)

DEPTH_CAP = 9           # >= the win horizon; deeper is just slower
WIN = 10**6             # mate sentinel; >> any heuristic from engine.evaluate

# Search center columns first — alpha-beta prunes far more when the strongest
# moves are tried early (center control dominates Connect Four).
_ORDER = sorted(range(COLS), key=lambda c: abs(c - COLS // 2))   # [3,2,4,1,5,0,6]


def _ordered_valid(board: Board) -> list[int]:
    valid = set(valid_columns(board))
    return [c for c in _ORDER if c in valid]


def search(board: Board, depth: int, alpha: float, beta: float,
           to_move: int, me: int) -> float:
    """Bounded alpha-beta from `me`'s POV, with mate-distance baked in.

    A win worth `WIN + depth` makes the maximizer prefer *faster* wins (more
    depth left) and the minimizer prefer *slower* losses (defend longer).
    Scores >= WIN mean "forced win for `me` within the horizon".
    """
    if winning_move(board, me):
        return WIN + depth
    if winning_move(board, opponent(me)):
        return -(WIN + depth)
    valid = _ordered_valid(board)
    if not valid:
        return 0.0                         # draw
    if depth == 0:
        return float(engine.evaluate(board, me))

    if to_move == me:                      # maximizing
        value = -math.inf
        for col in valid:
            child = copy_board(board)
            drop(child, col, to_move)
            value = max(value, search(child, depth - 1, alpha, beta, opponent(to_move), me))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:                                  # minimizing
        value = math.inf
        for col in valid:
            child = copy_board(board)
            drop(child, col, to_move)
            value = min(value, search(child, depth - 1, alpha, beta, opponent(to_move), me))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value


def red_winning_columns(board: Board) -> list[int]:
    """OPTIMAL winning columns for RED: the moves that force the FASTEST win.

    We score every move and keep only those achieving the best score (and only
    if that best is a forced win). This makes the realized line the shortest one
    (so "win within 5 moves" is exactly achievable), keeps the precomputed tree
    to the optimal subtree, and avoids the depth-cap edge case.
    """
    scored = []
    for col in valid_columns(board):
        child = copy_board(board)
        drop(child, col, RED)
        scored.append((col, search(child, DEPTH_CAP, -math.inf, math.inf, to_move=YELLOW, me=RED)))
    best = max((s for _, s in scored), default=-math.inf)
    if best < WIN:
        return []                                    # not a forced win here
    return [col for col, s in scored if s == best]   # all moves tying the fastest win


def yellow_best_defense(board: Board) -> int | None:
    """YELLOW's reply that minimizes RED's score (defend longest / delay the win)."""
    best_col, best_score = None, math.inf
    for col in valid_columns(board):
        child = copy_board(board)
        drop(child, col, YELLOW)
        score = search(child, DEPTH_CAP, -math.inf, math.inf, to_move=RED, me=RED)
        if score < best_score:
            best_col, best_score = col, score
    return best_col


def build_solution(start_board: Board) -> dict:
    """Walk the winning subtree from the start (RED to move) and tabulate it."""
    solution: dict[str, dict] = {}

    def visit_red(board: Board) -> None:
        key = state_key(board, RED)
        if key in solution:
            return
        wins = red_winning_columns(board)
        # plies_to_win: from the best line (the smaller the sooner).
        best = search(board, DEPTH_CAP, -math.inf, math.inf, to_move=RED, me=RED)
        plies = (WIN + DEPTH_CAP) - best if best >= WIN else None
        solution[key] = {"winning_columns": wins, "plies_to_win": plies}
        for col in wins:                       # only follow winning moves
            after_red = copy_board(board)
            drop(after_red, col, RED)
            if winning_move(after_red, RED):   # Red already won -> leaf
                continue
            visit_yellow(after_red)

    def visit_yellow(board: Board) -> None:
        key = state_key(board, YELLOW)
        if key in solution:
            return
        defense = yellow_best_defense(board)
        solution[key] = {"best_defense": defense}
        if defense is None:
            return
        after_yellow = copy_board(board)
        drop(after_yellow, defense, YELLOW)
        if winning_move(after_yellow, YELLOW) or is_full(after_yellow):
            return
        visit_red(after_yellow)

    visit_red(start_board)
    return solution


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="solution.json")
    ap.add_argument("--config", default="puzzle_config.txt")
    args = ap.parse_args()

    board = load_board(args.config)
    solution = build_solution(board)

    root = solution[state_key(board, RED)]
    print(f"nodes: {len(solution)}")
    print(f"root winning columns: {root['winning_columns']} (plies_to_win={root['plies_to_win']})")
    with open(args.out, "w") as f:
        json.dump(solution, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
