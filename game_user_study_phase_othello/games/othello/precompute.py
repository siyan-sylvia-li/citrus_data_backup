"""Precompute a puzzle's optimal subtree into solution_<id>.json (run once, offline).

Why precompute at all? The exact solver is fast, but not "inside every web
request" fast, and the AI's reply has to be instant and identical for every
participant. So we walk the subtree the study can actually reach and tabulate it:

- The participant only ever advances by playing, and the AI replies with a single
  best defense, so the reachable set is (Black's optimal moves) x (White's one
  reply) — a small tree even for a 12-empty endgame.
- Off the table (the participant played a suboptimal move) the web layer falls
  back to engine.ai_move, which is itself exact in the endgame.

Output: solution.json mapping engine.state_key(board, to_move) -> entry, where
  Black-to-move nodes carry  {"best_moves": [...], "optimal_diff": int,
                              "move_values": {"d3": +6, ...}, "must_pass": false}
  White-to-move nodes carry  {"best_defense": "c4", "optimal_diff": int}
  Nodes where the side to move has no move carry {"must_pass": true}.
Moves are algebraic ("d3"); the web layer parses them with engine.from_notation.

The table also records the root value under "_meta", so the study copy can say
what "solved" means for this puzzle (e.g. "win by 6").

Run:  python precompute.py --config puzzle_config_e12a.txt -o solution_e12a.json
"""

from __future__ import annotations

import argparse
import json

import solver
from engine import (
    BLACK, WHITE, Board,
    copy_board, apply_move, has_move, is_game_over, empty_count, load_puzzle,
    state_key, to_notation, disc_counts,
)


def build_solution(start_board: Board, to_move: int = BLACK) -> dict:
    """Walk the optimal subtree from the start position and tabulate it."""
    solution: dict[str, dict] = {}
    memo: dict = {}

    def visit(board: Board, side: int) -> None:
        key = state_key(board, side)
        if key in solution or is_game_over(board):
            return

        if not has_move(board, side):                 # forced pass
            solution[key] = {"must_pass": True}
            visit(board, 3 - side)                    # BLACK <-> WHITE
            return

        values = solver.move_values(board, side, memo)
        if side == BLACK:
            best = max(values.values())
            best_moves = sorted(m for m, v in values.items() if v == best)
            solution[key] = {
                "best_moves": [to_notation(m) for m in best_moves],
                "optimal_diff": best,
                "move_values": {to_notation(m): v for m, v in sorted(values.items())},
                "must_pass": False,
            }
            follow = best_moves                       # only follow optimal moves
        else:
            defense = solver.best_defense(board, memo)
            solution[key] = {
                "best_defense": to_notation(defense),
                "optimal_diff": values[defense],
                "must_pass": False,
            }
            follow = [defense]                        # the AI plays exactly one reply

        for move in follow:
            child = copy_board(board)
            apply_move(child, move, side)
            visit(child, 3 - side)

    visit(start_board, to_move)
    return solution


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="puzzle_config.txt")
    ap.add_argument("-o", "--out", default="solution.json")
    args = ap.parse_args()

    board, to_move = load_puzzle(args.config)
    empties = empty_count(board)
    if empties > solver.SOLVE_WARN_EMPTIES:
        print(f"warning: {empties} empty squares — this may take a long while")

    solution = build_solution(board, to_move)
    root = solution[state_key(board, to_move)]
    black, white = disc_counts(board)

    solution["_meta"] = {
        "config": args.config,
        "to_move": "B" if to_move == BLACK else "W",
        "empties": empties,
        "discs": {"black": black, "white": white},
        "optimal_diff": root.get("optimal_diff"),
        "result": solver.outcome_name(root.get("optimal_diff", 0)),
        "root_best_moves": root.get("best_moves"),
    }

    print(f"nodes: {len(solution) - 1}")
    print(f"root: {root.get('best_moves')} -> final margin "
          f"{root.get('optimal_diff'):+d} ({solution['_meta']['result']})")
    with open(args.out, "w") as f:
        json.dump(solution, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
