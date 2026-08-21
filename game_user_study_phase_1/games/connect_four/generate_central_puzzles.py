"""Manufacture win-in-five Connect Four puzzles whose OPTIMAL FIRST MOVE is central.

Motivation: the study's first puzzle (#15) is a center-control win-in-five; to
test whether participants transfer that strategy, the second (unaided) puzzle
should share the motif — a central opening — rather than the peripheral #6.
Rather than rely on Zeilberger's fixed 15, we search for our own.

Method
------
1. Random self-play (Red first, both sides random legal moves) to a target ply
   count, aborting if anyone connects four early -> a *reachable, legal* board
   with Red to move and no winner yet.
2. Keep it only if Red has a FORCED win in exactly 5 moves (plies_to_win == 9,
   using precompute's distance-aware alpha-beta) AND every optimal first move is
   a central column. This reuses the exact solver the participant plays against.
3. Score each keeper on the same structural metrics as PUZZLE_SELECTION.md and
   rank by z-normalized distance to puzzle 15, so you can pick close matches.
4. Save each as puzzle_config_<tag>.txt + solution_<tag>.json (via
   precompute.build_solution), ready to drop into CF_ROUNDS.

Run (from games/connect_four):
    python generate_central_puzzles.py --count 6 --attempts 20000 --seed 1
    python generate_central_puzzles.py --central 3 4 5   # 1-indexed central band
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

import engine
import precompute
from engine import (
    RED, YELLOW, EMPTY, ROWS, COLS, Board,
    new_board, valid_columns, drop, winning_move, opponent, state_key,
    load_board, board_to_text,
)
from precompute import search, red_winning_columns, build_solution, DEPTH_CAP, WIN

HERE = Path(__file__).parent
_PIECE_TO_FILE = {EMPTY: "*", RED: "R", YELLOW: "Y"}
WIN_IN_FIVE_PLIES = 9          # 5 Red moves interleaved with 4 Yellow replies


def board_to_config(board: Board) -> str:
    """Serialize a Board to puzzle_config format (top row first, ', '-joined)."""
    return "\n".join(
        ", ".join(_PIECE_TO_FILE[board[r][c]] for c in range(COLS))
        for r in range(ROWS - 1, -1, -1)
    ) + "\n"


def random_red_to_move(rng: random.Random, plies: int) -> Board | None:
    """A reachable board after `plies` random legal moves (Red first), or None if
    someone connected four along the way. `plies` even -> Red to move next."""
    board = new_board()
    mover = RED
    for _ in range(plies):
        col = rng.choice(valid_columns(board))
        drop(board, col, mover)
        if winning_move(board, mover):
            return None
        mover = opponent(mover)
    return board


def plies_to_win(board: Board) -> int | None:
    """Fastest forced-win distance (plies) for Red to move, or None if not a
    forced win within the horizon."""
    best = search(board, DEPTH_CAP, -math.inf, math.inf, to_move=RED, me=RED)
    return (WIN + DEPTH_CAP) - best if best >= WIN else None


def metrics(solution: dict, board: Board) -> dict:
    """Structural difficulty metrics on the optimal subtree (mirrors the doc)."""
    red_nodes = [v for v in solution.values() if "winning_columns" in v]
    n = len(red_nodes)
    uniq = sum(1 for v in red_nodes if len(v["winning_columns"]) == 1)
    mean_win_cols = statistics.mean(len(v["winning_columns"]) for v in red_nodes)

    # Principal variation: follow Red's fastest-win col, then Yellow's defense.
    pv_cols, b, forced_run, pv_only, run_open = [], [r[:] for r in board], 0, 0, True
    for _ in range(6):
        entry = solution.get(state_key(b, RED))
        if not entry or not entry.get("winning_columns"):
            break
        wc = entry["winning_columns"]
        pv_cols.append(wc[0] + 1)                 # 1-indexed
        if len(wc) == 1:
            pv_only += 1
            if run_open:
                forced_run += 1
        else:
            run_open = False
        drop(b, wc[0], RED)
        if winning_move(b, RED):
            break
        d = solution.get(state_key(b, YELLOW), {}).get("best_defense")
        if d is None:
            break
        drop(b, d, YELLOW)
    return {
        "first_move": solution[state_key(board, RED)]["winning_columns"],  # 0-indexed
        "pv_cols": pv_cols,
        "forced_run": forced_run,
        "pv_only_moves": pv_only,
        "uniqueness_ratio": round(uniq / n, 3) if n else 0.0,
        "mean_winning_cols": round(mean_win_cols, 3),
        "tree_size": len(solution),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=6, help="how many puzzles to keep")
    ap.add_argument("--attempts", type=int, default=20000, help="max random positions to try")
    ap.add_argument("--plies", type=int, default=12, help="pieces on the starting board (even -> Red to move)")
    ap.add_argument("--central", type=int, nargs="+", default=[3, 4, 5],
                    help="central columns (1-indexed) the optimal FIRST move must fall in")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="c", help="filename tag: puzzle_config_<tag><i>.txt")
    ap.add_argument("--save", action="store_true", help="write config+solution files for keepers")
    args = ap.parse_args()

    central0 = {c - 1 for c in args.central}       # to 0-indexed
    rng = random.Random(args.seed)

    # Reference metrics for puzzle 15 (target of the similarity ranking).
    ref_board = load_board(HERE / "puzzle_config_15.txt")
    ref = metrics(build_solution(ref_board), ref_board)

    keepers, seen = [], set()
    tried = examined = 0
    while len(keepers) < args.count and tried < args.attempts:
        tried += 1
        board = random_red_to_move(rng, args.plies)
        if board is None:
            continue
        key = state_key(board, RED)
        if key in seen:
            continue
        seen.add(key)
        # Cheap gate first: is it a forced win in exactly five?
        if plies_to_win(board) != WIN_IN_FIVE_PLIES:
            continue
        wins = red_winning_columns(board)
        if not wins or not all(c in central0 for c in wins):
            continue
        examined += 1
        sol = build_solution(board)
        keepers.append({"board": board, "solution": sol, "m": metrics(sol, board)})
        print(f"[{len(keepers)}/{args.count}] found after {tried} tries "
              f"(first move cols {[c+1 for c in wins]})")

    if not keepers:
        print(f"No central win-in-five positions found in {tried} tries. "
              f"Try more --attempts or a different --seed/--plies.")
        return

    # Rank by z-normalized distance to puzzle 15 on the shared metric vector.
    feats = ["forced_run", "pv_only_moves", "uniqueness_ratio", "mean_winning_cols", "tree_size"]
    pool = [ref] + [k["m"] for k in keepers]
    stats = {f: (statistics.mean(p[f] for p in pool),
                 statistics.pstdev(p[f] for p in pool) or 1.0) for f in feats}

    def dist(m):
        return math.sqrt(sum(((m[f] - stats[f][0]) / stats[f][1]
                              - (ref[f] - stats[f][0]) / stats[f][1]) ** 2 for f in feats))

    keepers.sort(key=lambda k: dist(k["m"]))

    print(f"\nExamined {examined} central-opening win-in-five positions; keeping {len(keepers)}.")
    print(f"(reference #15: forced_run={ref['forced_run']} pv_only={ref['pv_only_moves']} "
          f"uniq={ref['uniqueness_ratio']} mean_win={ref['mean_winning_cols']} tree={ref['tree_size']})\n")
    hdr = f"{'tag':>5} | {'dist15':>6} | {'first':<9} | {'pv path':<20} | {'frun':>4} | {'uniq':>4} | {'mwc':>4} | {'tree':>4}"
    print(hdr); print("-" * len(hdr))
    for i, k in enumerate(keepers, 1):
        m = k["m"]; tag = f"{args.tag}{i}"
        print(f"{tag:>5} | {dist(m):6.2f} | {str([c+1 for c in m['first_move']]):<9} | "
              f"{str(m['pv_cols']):<20} | {m['forced_run']:>4} | {m['uniqueness_ratio']:>4} | "
              f"{m['mean_winning_cols']:>4} | {m['tree_size']:>4}")

    if args.save:
        print()
        for i, k in enumerate(keepers, 1):
            tag = f"{args.tag}{i}"
            (HERE / f"puzzle_config_{tag}.txt").write_text(board_to_config(k["board"]))
            (HERE / f"solution_{tag}.json").write_text(json.dumps(k["solution"], indent=2))
            # sanity: round-trips to the same board
            assert load_board(HERE / f"puzzle_config_{tag}.txt") == k["board"]
            print(f"wrote puzzle_config_{tag}.txt + solution_{tag}.json")
        print("\nBoards (top row first):")
        for i, k in enumerate(keepers, 1):
            print(f"\n--- {args.tag}{i} ---\n{board_to_text(k['board'])}")


if __name__ == "__main__":
    main()
