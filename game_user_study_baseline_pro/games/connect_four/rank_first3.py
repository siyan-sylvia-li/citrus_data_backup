"""Rank puzzles by the difficulty of their FIRST N (=3) Red moves.

Design decision: the study scores whether the participant plays the first 3
moves optimally (CF_NUM_MOVES=3), and we want puzzles whose *beginning* is
hard. So difficulty should be measured on the first 3 Red moves only -- not the
whole forced-win tree. This re-ranks the 15 Zeilberger win-in-five puzzles by
how close their first-3-move difficulty is to the target (puzzle 15).

Per Red move i (along Red-fastest-win -> Yellow-best-defense, the sequence the
participant actually faces):
  - winning_cols  : how many columns keep the win (1 = only-move = hardest)
  - decoys        : non-winning legal moves that still look non-losing at a
                    shallow (depth-3) search (Guid-Bratko: plausible wrong moves)

Aggregated over the first 3 moves into a difficulty vector, z-normalized across
the 15 puzzles, ranked by Euclidean distance to the target.

Run:  python rank_first3.py [--target 15] [--k 3] [--first-n 3]
"""
from __future__ import annotations

import argparse
import math

import precompute
from engine import RED, YELLOW, state_key, copy_board, drop, winning_move
import rank_similar_puzzles as rsp        # reuse load_positions() + shallow_decoys()


def first_n_profile(board: list[list[int]], n: int) -> dict:
    sol = precompute.build_solution(board)
    cur, moves = copy_board(board), []
    for _ in range(n):
        entry = sol.get(state_key(cur, RED))
        if not entry or not entry.get("winning_columns"):
            break
        wc = entry["winning_columns"]
        moves.append({"winning_cols": len(wc),
                      "decoys": rsp.shallow_decoys(cur, wc),
                      "plies_to_win": entry["plies_to_win"]})
        drop(cur, wc[0], RED)                       # Red takes a fastest-win column
        if winning_move(cur, RED):
            break
        ydef = sol.get(state_key(cur, YELLOW), {}).get("best_defense")
        if ydef is None:
            break
        drop(cur, ydef, YELLOW)
    k = len(moves) or 1
    return {
        "n_moves": len(moves),
        "only_moves": sum(m["winning_cols"] == 1 for m in moves),
        "mean_winning_cols": round(sum(m["winning_cols"] for m in moves) / k, 3),
        "mean_decoys": round(sum(m["decoys"] for m in moves) / k, 3),
        "winning_cols_seq": [m["winning_cols"] for m in moves],
        "decoys_seq": [m["decoys"] for m in moves],
    }


VECTOR_KEYS = ["only_moves", "mean_winning_cols", "mean_decoys"]


def zscore(rows: dict[int, dict]) -> dict[int, dict]:
    out = {n: {} for n in rows}
    for k in VECTOR_KEYS:
        vals = [rows[n][k] for n in rows]
        mu = sum(vals) / len(vals)
        sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5 or 1.0
        for n in rows:
            out[n][k] = (rows[n][k] - mu) / sd
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=15)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--first-n", type=int, default=3)
    args = ap.parse_args()

    positions = rsp.load_positions()
    metrics = {n: first_n_profile(positions[n], args.first_n) for n in sorted(positions)}
    z = zscore(metrics)

    tgt = z[args.target]
    dist = {n: math.sqrt(sum((z[n][k] - tgt[k]) ** 2 for k in VECTOR_KEYS))
            for n in metrics if n != args.target}
    ranked = sorted(dist, key=dist.get)

    print(f"first {args.first_n} moves; distance to puzzle {args.target}\n")
    print(f"{'pz':>3} {'dist':>6}  {'only_moves':>10}  {'mean_win_cols':>13}  "
          f"{'mean_decoys':>11}  {'win_cols/move':>14}  {'decoys/move':>14}")
    for n in [args.target] + ranked:
        d = "  --  " if n == args.target else f"{dist[n]:6.2f}"
        mark = " <-- TARGET" if n == args.target else ""
        m = metrics[n]
        print(f"{n:>3} {d}  {m['only_moves']:>10}  {m['mean_winning_cols']:>13}  "
              f"{m['mean_decoys']:>11}  {str(m['winning_cols_seq']):>14}  "
              f"{str(m['decoys_seq']):>14}{mark}")

    print(f"\nTop {args.k} most similar (first {args.first_n} moves) to puzzle "
          f"{args.target}: " + ", ".join(str(n) for n in ranked[:args.k]))


if __name__ == "__main__":
    main()
