"""Scan Zeilberger's win-in-four set (ch4) for puzzles with a CENTRAL but
NON-column-4 optimal opening — a strategic match for puzzle 15 (opens on col 4)
that isn't a rote copy of the same first move.

Cheap by design: we only score each position's optimal first move + win length,
and walk the single principal variation (Red fastest-win col -> Yellow best
defense) for its column path. We do NOT expand the whole winning subtree (some
ch4 puzzles have very wide trees), so this stays fast.

Usage:  python scan_win4.py <cache_dir> [--nmax 30]
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

from engine import RED, YELLOW, COLS, valid_columns, copy_board, drop, winning_move, load_board
from precompute import search, yellow_best_defense, DEPTH_CAP, WIN

HERE = Path(__file__).parent
CENTRAL = {3, 4, 5}                       # 1-indexed central band
_ROW = re.compile(r"^\s*[012](?:\s*,\s*[012]){6}\s*$")


def parse_position(text: str):
    rows = []
    for line in text.splitlines():
        if _ROW.match(line):
            rows.append([int(x) for x in re.findall(r"[012]", line)])
            if len(rows) == 6:
                break
    return list(reversed(rows)) if len(rows) == 6 else None


def red_win_and_plies(board):
    """(fastest-win columns, plies_to_win) for Red to move; ([], None) if not a
    forced win within the horizon. Single 7-column search pass."""
    scored = []
    for col in valid_columns(board):
        child = copy_board(board)
        drop(child, col, RED)
        scored.append((col, search(child, DEPTH_CAP, -math.inf, math.inf, to_move=YELLOW, me=RED)))
    best = max((s for _, s in scored), default=-math.inf)
    if best < WIN:
        return [], None
    plies = (WIN + DEPTH_CAP) - best
    return [c for c, s in scored if s == best], plies


def pv_path(board):
    """Column path (1-indexed) along the principal variation."""
    b = copy_board(board); path = []
    for _ in range(6):
        wc, _ = red_win_and_plies(b)
        if not wc:
            break
        path.append(wc[0] + 1)
        drop(b, wc[0], RED)
        if winning_move(b, RED):
            break
        d = yellow_best_defense(b)
        if d is None:
            break
        drop(b, d, YELLOW)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("--nmax", type=int, default=30)
    args = ap.parse_args()
    cache = Path(args.cache)

    print(f"{'P':>3} | {'Rmoves':>6} | {'first':<9} | {'pv path':<22} | central%  note", flush=True)
    print("-" * 66, flush=True)
    results = []
    for n in range(1, args.nmax + 1):
        f = cache / f"S{n}.txt"
        if not f.exists():
            continue
        pos = parse_position(f.read_text(errors="replace"))
        if pos is None:
            continue
        first0, child_plies = red_win_and_plies(pos)     # first0 is 0-indexed
        if not first0:
            print(f"{n:>3} | {'--':>6} | (no forced win found)", flush=True)
            continue
        first = [c + 1 for c in first0]                  # 1-indexed, matches path
        red_moves = (child_plies + 1 + 1) // 2           # root plies = child_plies+1
        path = pv_path(pos)                              # already 1-indexed
        cenf = sum(1 for c in path if c in CENTRAL) / len(path) if path else 0
        good = all(c in CENTRAL for c in first) and 4 not in first
        note = "<-- CENTRAL, not col 4" if good else ("(opens col 4)" if 4 in first else "(peripheral open)")
        results.append((good, -cenf, n, first, red_moves, path))
        print(f"{n:>3} | {red_moves:>6} | {str(first):<9} | {str(path):<22} | {cenf*100:>4.0f}%    {note}", flush=True)

    print("\n=== candidates: central opening (cols 3/5), NOT column 4 (sorted by how central the whole line is) ===", flush=True)
    hits = [r for r in sorted(results) if r[0]]
    for good, negc, n, first, red_moves, path in hits:
        print(f"  P{n}: opens {first}, win-in-{red_moves}, path={path}, central={-negc*100:.0f}%", flush=True)
    if not hits:
        print("  (none — no win-in-four puzzle opens on a central column other than 4)", flush=True)


if __name__ == "__main__":
    main()
