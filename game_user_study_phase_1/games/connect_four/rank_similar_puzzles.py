"""Find win-in-five Connect Four puzzles of similar difficulty to a target.

Source of puzzles: Zeilberger's "Win-In-Five-Moves" problem set
    https://sites.math.rutgers.edu/~zeilberg/C4/ch5/Problems.html
Each S<n> page prints the starting position as a text grid (top row first,
0=empty 1=Red 2=Yellow) -- which is exactly our engine's EMPTY/RED/YELLOW
encoding, just top-to-bottom instead of bottom-to-top.

Pipeline (mirrors DIFFICULTY.md):
  1. scrape the 15 starting positions (cached to zeilberger_positions.json)
  2. rebuild each under STUDY conditions via precompute.build_solution
     (Red plays winning columns; Yellow plays its single best defense) -- so
     difficulty is measured on the same footing the participant experiences
  3. score each puzzle with structural + shallow-search (Guid-Bratko) metrics
  4. z-normalize the metric vectors and rank by Euclidean distance to the target

Run:  python rank_similar_puzzles.py            # target = 15, top 3
      python rank_similar_puzzles.py --target 15 --k 3
"""
from __future__ import annotations

import argparse
import json
import math
import re
import urllib.request
from pathlib import Path

import engine
import precompute
from engine import RED, YELLOW, COLS, state_key, valid_columns, copy_board, drop, winning_move

BASE = "https://sites.math.rutgers.edu/~zeilberg/C4/ch5/"
HERE = Path(__file__).parent
CACHE = HERE / "zeilberger_positions.json"
SHALLOW = 3          # lookahead depth for the "decoy" (Guid-Bratko) metric

_ROW_RE = re.compile(r"^\s*([012])(?:\s*,\s*([012])){6}\s*$")


def fetch_position(n: int) -> list[list[int]]:
    """Return the starting Board (row 0 = bottom) for puzzle n from its S<n> page."""
    text = urllib.request.urlopen(f"{BASE}S{n}").read().decode("utf-8", "replace")
    rows_top_first: list[list[int]] = []
    for line in text.splitlines():
        if _ROW_RE.match(line):
            rows_top_first.append([int(x) for x in re.findall(r"[012]", line)])
            if len(rows_top_first) == 6:
                break
    if len(rows_top_first) != 6:
        raise ValueError(f"puzzle {n}: could not parse a 6x7 grid")
    return list(reversed(rows_top_first))          # file is top-first; row 0 = bottom


def load_positions() -> dict[int, list[list[int]]]:
    if CACHE.exists():
        raw = json.loads(CACHE.read_text())
        return {int(k): v for k, v in raw.items()}
    positions = {n: fetch_position(n) for n in range(1, 16)}
    CACHE.write_text(json.dumps({str(k): v for k, v in positions.items()}, indent=2))
    return positions


def principal_variation(solution: dict, board: list[list[int]]) -> list[dict]:
    """Walk Red-winning-move -> Yellow-best-defense from the root; list Red nodes."""
    red_nodes, cur = [], copy_board(board)
    while True:
        entry = solution.get(state_key(cur, RED))
        if not entry or not entry.get("winning_columns"):
            break
        red_nodes.append(entry)
        drop(cur, entry["winning_columns"][0], RED)   # Red takes a fastest-win column
        if winning_move(cur, RED):
            break
        ydef = solution.get(state_key(cur, YELLOW), {}).get("best_defense")
        if ydef is None:
            break
        drop(cur, ydef, YELLOW)
    return red_nodes


def shallow_decoys(board: list[list[int]], winning_cols: list[int]) -> int:
    """# of non-winning legal Red moves that still look non-losing at shallow sight.

    Guid-Bratko style: a position is hard when several moves are plausible at a
    shallow search but only one (or few) actually win. We count non-winning moves
    whose depth-SHALLOW score is >= 0 (i.e. not obviously refuted yet)."""
    decoys = 0
    for col in valid_columns(board):
        if col in winning_cols:
            continue
        child = copy_board(board)
        drop(child, col, RED)
        if winning_move(child, RED):
            continue
        score = precompute.search(child, SHALLOW, -math.inf, math.inf, to_move=YELLOW, me=RED)
        if score >= 0:
            decoys += 1
    return decoys


def score_puzzle(board: list[list[int]]) -> dict:
    solution = precompute.build_solution(board)
    root = solution[state_key(board, RED)]

    red_entries = [v for v in solution.values() if "winning_columns" in v]
    n_red = len(red_entries)
    uniqueness_ratio = sum(len(v["winning_columns"]) == 1 for v in red_entries) / n_red
    mean_winning_cols = sum(len(v["winning_columns"]) for v in red_entries) / n_red

    pv = principal_variation(solution, board)
    pv_only_moves = sum(len(v["winning_columns"]) == 1 for v in pv)
    # length of the leading run of unique "only-moves" (Pelanek step-dependency)
    forced_run = 0
    for v in pv:
        if len(v["winning_columns"]) == 1:
            forced_run += 1
        else:
            break

    # decoys along the PV Red nodes (reconstruct each PV node's board)
    decoys, cur = [], copy_board(board)
    for entry in pv:
        decoys.append(shallow_decoys(cur, entry["winning_columns"]))
        drop(cur, entry["winning_columns"][0], RED)
        if winning_move(cur, RED):
            break
        ydef = solution.get(state_key(cur, YELLOW), {}).get("best_defense")
        if ydef is None:
            break
        drop(cur, ydef, YELLOW)
    mean_decoys = sum(decoys) / len(decoys) if decoys else 0.0

    return {
        "plies_to_win": root["plies_to_win"],
        "root_winning_cols": len(root["winning_columns"]),
        "tree_size": len(solution),
        "n_red_nodes": n_red,
        "uniqueness_ratio": round(uniqueness_ratio, 3),
        "mean_winning_cols": round(mean_winning_cols, 3),
        "forced_run": forced_run,
        "pv_only_moves": pv_only_moves,
        "mean_decoys": round(mean_decoys, 3),
    }


# Metrics used for the similarity vector, and which direction means "harder".
# (Direction only matters for the human-readable difficulty score, not distance.)
VECTOR_KEYS = ["forced_run", "pv_only_moves", "uniqueness_ratio",
               "mean_winning_cols", "mean_decoys", "tree_size"]


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
    args = ap.parse_args()

    positions = load_positions()
    metrics = {n: score_puzzle(positions[n]) for n in sorted(positions)}
    z = zscore(metrics)

    tgt = z[args.target]
    dist = {n: math.sqrt(sum((z[n][k] - tgt[k]) ** 2 for k in VECTOR_KEYS))
            for n in metrics if n != args.target}
    ranked = sorted(dist, key=dist.get)

    cols = ["forced_run", "pv_only_moves", "uniqueness_ratio", "mean_winning_cols",
            "mean_decoys", "tree_size", "plies_to_win"]
    print(f"{'pz':>3} {'dist':>6}  " + "  ".join(f"{c:>17}" for c in cols))
    for n in [args.target] + ranked:
        d = "  --  " if n == args.target else f"{dist[n]:6.2f}"
        mark = " <-- TARGET" if n == args.target else ""
        print(f"{n:>3} {d}  " + "  ".join(f"{metrics[n][c]:>17}" for c in cols) + mark)

    print(f"\nTop {args.k} most similar to puzzle {args.target}: "
          + ", ".join(str(n) for n in ranked[:args.k]))


if __name__ == "__main__":
    main()
