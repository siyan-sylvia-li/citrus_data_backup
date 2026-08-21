"""Manufacture Othello endgame puzzles and rank them by how discriminating they are.

We have no published solution set for Othello endgames (unlike the Connect Four
phase, which used Zeilberger's book), so the puzzles are grown here and solved
exactly by `solver`. That also means every puzzle ships with its own ground truth.

Method
------
1. Fast bitboard self-play from the standard opening with a weight-greedy policy
   plus noise, stopped when the target number of squares is still empty and Black
   is to move -> a *reachable, legal* endgame position, not a random blob.
2. Solve it exactly (solver.move_values) and keep it only if it discriminates:
   Black must have a real choice, one clearly best move, and a plausible-looking
   alternative that costs real discs.
3. Score each keeper on difficulty proxies (see `metrics`) and print them ranked,
   so a puzzle can be picked deliberately rather than by vibe.
4. Write each as puzzle_config_<tag>.txt + solution_<tag>.json (via
   precompute.build_solution), ready to drop into the study's round list.

Run (from games/othello):
    python generate_puzzles.py --empties 12 --count 4 --attempts 400 --seed 1
    python generate_puzzles.py --empties 10 --min-gap 6 --require-win
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import precompute
import solver
from engine import (
    BLACK, WHITE, CORNERS,
    apply_move, bb_flips, bb_indices, bb_legal, bb_play, board_to_config,
    board_to_text, copy_board, empty_count, from_bits, has_move, index_move,
    is_game_over, legal_moves, new_board, to_bits, to_notation,
)

HERE = Path(__file__).parent


# ---------------------------------------------------------------------------
# Step 1: reachable endgame positions via fast bitboard self-play
# ---------------------------------------------------------------------------
def self_play_to(empties: int, rng: random.Random, noise: float = 0.25):
    """Play a full-speed game and stop at `empties` empty squares, Black to move.

    Returns a Board, or None if the game ended early or stopped on White's turn.
    The policy is "prefer structurally good squares, sometimes don't", which
    produces positions that look like real games — the alternative (uniformly
    random play) leaves absurd blobs that no participant would ever face.
    """
    black, white = to_bits(new_board())
    side = BLACK
    while True:
        own, opp = (black, white) if side == BLACK else (white, black)
        moves = bb_legal(own, opp)
        if not moves:
            if not bb_legal(opp, own):
                return None                       # game over before the target
            side = WHITE if side == BLACK else BLACK
            continue
        if 64 - bin(black | white).count("1") == empties and side == BLACK:
            return from_bits(black, white)

        idx_list = bb_indices(moves)              # already ordered strongest-first
        idx = (rng.choice(idx_list) if rng.random() < noise else idx_list[0])
        new_own, new_opp = bb_play(own, opp, 1 << idx)
        black, white = ((new_opp, new_own) if side == BLACK else (new_own, new_opp))
        side = WHITE if side == BLACK else BLACK


# ---------------------------------------------------------------------------
# Step 2/3: exact evaluation + difficulty metrics
# ---------------------------------------------------------------------------
def greedy_flip_move(board, piece):
    """The move a naive player makes: flip the most discs right now."""
    own, opp = to_bits(board)
    if piece == WHITE:
        own, opp = opp, own
    best, best_n = None, -1
    for idx in bb_indices(bb_legal(own, opp)):
        n = bin(bb_flips(own, opp, 1 << idx)).count("1")
        if n > best_n:
            best, best_n = idx, n
    return None if best is None else index_move(best)


def line_length(board, memo) -> int:
    """How many Black decisions on the optimal line have a *unique* best move.

    A puzzle whose answer is one move and then obvious is easy; one that demands
    a correct choice three times running is the kind of task the study wants.
    """
    b, side, count = copy_board(board), BLACK, 0
    while not is_game_over(b):
        if not has_move(b, side):
            side = WHITE if side == BLACK else BLACK
            continue
        if side == BLACK:
            best = solver.best_moves(b, BLACK, memo)
            if len(best) != 1:
                break
            count += 1
            apply_move(b, best[0], BLACK)
        else:
            apply_move(b, solver.best_defense(b, memo), WHITE)
        side = WHITE if side == BLACK else BLACK
    return count


def metrics(board, memo) -> dict:
    """Everything we rank puzzles on. All values are exact (final disc margins)."""
    values = solver.move_values(board, BLACK, memo)
    ranked = sorted(values.items(), key=lambda kv: -kv[1])
    best_value = ranked[0][1]
    best_moves = [m for m, v in ranked if v == best_value]
    second = next((v for _, v in ranked if v < best_value), None)

    naive = greedy_flip_move(board, BLACK)
    return {
        "empties": empty_count(board),
        "n_legal": len(values),
        "optimal_diff": best_value,
        "best_moves": [to_notation(m) for m in sorted(best_moves)],
        "unique_best": len(best_moves) == 1,
        # Gap to the next-best move: how much a near-miss costs, in discs.
        "gap": None if second is None else best_value - second,
        "worst_diff": ranked[-1][1],
        # Does the obvious "flip the most discs" move throw the win away?
        "naive_move": None if naive is None else to_notation(naive),
        "naive_is_optimal": naive in best_moves,
        "naive_loss": None if naive is None else best_value - values[naive],
        "corner_answer": any(m in CORNERS for m in best_moves),
        "forced_line": line_length(board, memo),
        "move_values": {to_notation(m): v for m, v in sorted(values.items())},
    }


def keep(m: dict, args) -> bool:
    """The filters that make a position usable as a study puzzle."""
    if m["n_legal"] < args.min_choices:
        return False
    if args.require_win and m["optimal_diff"] <= 0:
        return False
    if args.unique and not m["unique_best"]:
        return False
    if m["gap"] is None or m["gap"] < args.min_gap:
        return False
    if args.no_corner_answer and m["corner_answer"]:
        return False
    if m["forced_line"] < args.min_line:
        return False
    if args.trap and m["naive_is_optimal"]:
        return False                       # the greedy move must NOT be the answer
    return True


def difficulty(m: dict) -> float:
    """Rank key: reward puzzles that punish the naive move and stay sharp.

    Deliberately simple and inspectable — the printed table carries the real
    information; this only decides the display order.
    """
    return (2.0 * (m["naive_loss"] or 0)
            + 1.5 * m["forced_line"]
            + 1.0 * m["n_legal"]
            - 0.5 * (m["gap"] or 0))          # a huge gap makes the answer stand out


# ---------------------------------------------------------------------------
# Step 4: write the puzzle + its solution table
# ---------------------------------------------------------------------------
def write_puzzle(board, tag: str, m: dict) -> None:
    config_path = HERE / f"puzzle_config_{tag}.txt"
    config_path.write_text(board_to_config(board, BLACK))

    solution = precompute.build_solution(board, BLACK)
    solution["_meta"] = {
        "config": config_path.name,
        "to_move": "B",
        "empties": m["empties"],
        "optimal_diff": m["optimal_diff"],
        "result": solver.outcome_name(m["optimal_diff"]),
        "root_best_moves": m["best_moves"],
        "metrics": {k: v for k, v in m.items() if k != "move_values"},
    }
    with open(HERE / f"solution_{tag}.json", "w") as f:
        json.dump(solution, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--empties", type=int, default=12, help="empty squares at the puzzle start")
    ap.add_argument("--attempts", type=int, default=300, help="self-play games to try")
    ap.add_argument("--count", type=int, default=4, help="how many puzzles to write")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.25, help="self-play randomness")
    ap.add_argument("--min-choices", type=int, default=4, help="min legal moves for Black")
    ap.add_argument("--min-gap", type=int, default=4, help="min disc gap to the 2nd-best move")
    ap.add_argument("--min-line", type=int, default=2, help="min Black decisions with a unique best move")
    ap.add_argument("--require-win", action="store_true", help="Black must win with best play")
    ap.add_argument("--unique", action="store_true", help="the root must have ONE best move")
    ap.add_argument("--no-corner-answer", action="store_true", help="reject puzzles answered by a corner")
    ap.add_argument("--trap", action="store_true", help="the max-flips move must not be the answer")
    ap.add_argument("--prefix", default=None, help="tag prefix (default e<empties>)")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    prefix = args.prefix or f"e{args.empties}"
    keepers = []
    seen = set()

    for _ in range(args.attempts):
        board = self_play_to(args.empties, rng, args.noise)
        if board is None or not legal_moves(board, BLACK):
            continue
        fingerprint = tuple(tuple(row) for row in board)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        m = metrics(board, {})
        if keep(m, args):
            keepers.append((board, m))

    keepers.sort(key=lambda bm: -difficulty(bm[1]))
    print(f"tried {args.attempts} playouts -> {len(seen)} distinct positions "
          f"-> {len(keepers)} keepers\n")

    for i, (board, m) in enumerate(keepers[:args.count]):
        tag = f"{prefix}{chr(ord('a') + i)}"
        print(f"=== {tag} " + "=" * 40)
        print(board_to_text(board, BLACK))
        print(f"optimal: {m['best_moves']} -> {m['optimal_diff']:+d} "
              f"({solver.outcome_name(m['optimal_diff'])}) | gap {m['gap']} | "
              f"naive {m['naive_move']} costs {m['naive_loss']} | "
              f"forced line {m['forced_line']} | legal {m['n_legal']}")
        print("move values: " + ", ".join(f"{k}={v:+d}" for k, v in m["move_values"].items()))
        if not args.dry_run:
            write_puzzle(board, tag, m)
            print(f"wrote puzzle_config_{tag}.txt + solution_{tag}.json")
        print()


if __name__ == "__main__":
    main()
