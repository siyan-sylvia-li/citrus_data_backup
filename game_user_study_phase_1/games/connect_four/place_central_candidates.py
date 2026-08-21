"""Rank the central (col 3/5, not col 4) win-in-four candidates by BOTH how
central their optimal line is and how close they are to puzzle 15 in difficulty
(z-normalized over the 15 ch5 puzzles, same scale as PUZZLE_SELECTION.md).

Usage:  python place_central_candidates.py <ch4_cache_dir>
"""
import math
import re
import sys
from pathlib import Path

import rank_similar_puzzles as R
from engine import RED, state_key, drop, winning_move, copy_board

HERE = Path(__file__).parent
CENTRAL = {3, 4, 5}
_ROW = re.compile(r"^\s*[012](?:\s*,\s*[012]){6}\s*$")
CANDIDATES = [4, 6, 17, 22, 24, 25, 28, 29, 30]     # central, not col 4 (from scan)


def parse(fp):
    rows = []
    for line in Path(fp).read_text(errors="replace").splitlines():
        if _ROW.match(line):
            rows.append([int(x) for x in re.findall(r"[012]", line)])
            if len(rows) == 6:
                break
    return list(reversed(rows))


def central_frac(board):
    sol = R.principal_variation.__globals__  # noqa (unused) keep import tidy
    from precompute import build_solution
    s = build_solution(board)
    b = copy_board(board); path = []
    for _ in range(6):
        e = s.get(state_key(b, RED))
        if not e or not e.get("winning_columns"):
            break
        path.append(e["winning_columns"][0] + 1)
        drop(b, e["winning_columns"][0], RED)
        if winning_move(b, RED):
            break
        from engine import YELLOW
        d = s.get(state_key(b, YELLOW), {}).get("best_defense")
        if d is None:
            break
        drop(b, d, YELLOW)
    return sum(1 for c in path if c in CENTRAL) / len(path), path, path[0]


def main():
    cache = Path(sys.argv[1])
    ch5 = R.load_positions()
    m15 = {n: R.score_puzzle(ch5[n]) for n in sorted(ch5)}
    mu, sd = {}, {}
    for k in R.VECTOR_KEYS:
        vals = [m15[n][k] for n in ch5]
        mu[k] = sum(vals) / len(vals)
        sd[k] = (sum((v - mu[k]) ** 2 for v in vals) / len(vals)) ** 0.5 or 1.0

    def dist(metrics):
        return math.sqrt(sum((((metrics[k]-mu[k])/sd[k]) - ((m15[15][k]-mu[k])/sd[k]))**2
                             for k in R.VECTOR_KEYS))

    rows = []
    for n in CANDIDATES:
        board = parse(cache / f"S{n}.txt")
        met = R.score_puzzle(board)
        cf, path, first = central_frac(board)
        rows.append((n, first, cf, path, dist(met), met))

    rows.sort(key=lambda r: (-r[2], r[4]))          # most central, then closest difficulty
    print(f"#15 reference: forced_run=3 decoys=3.2 tree=13 uniq=0.571 (win-in-five)\n")
    print(f"{'P':>4} {'open':>4} {'cen%':>5} {'dist15':>6} {'frun':>4} {'decoy':>5} {'tree':>4} {'uniq':>5}  path")
    for n, first, cf, path, d, met in rows:
        print(f"{n:>4} {first:>4} {cf*100:>4.0f}% {d:>6.2f} {met['forced_run']:>4} "
              f"{met['mean_decoys']:>5} {met['tree_size']:>4} {met['uniqueness_ratio']:>5}  {path}")
    best = min(rows, key=lambda r: r[4])
    print(f"\nClosest difficulty match to #15 among central candidates: P{best[0]} (dist {best[4]:.2f})")


if __name__ == "__main__":
    main()
