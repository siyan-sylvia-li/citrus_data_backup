"""Exact depth-limited mate-distance solver for the Connect Four puzzles.

Why this exists
---------------
The flat round score is "how many of your 3 moves were optimal" -> 4 levels, and
along the line every one of those 3 moves has exactly ONE correct column, so the
metric is three binary events and nothing more. The selectivity rescoring doesn't
help either: its weight is `1 - |best|/|legal|`, and since a forced-win line has
|best| == 1 and |legal| == 7 for ~95% of moves, the weight is a near-constant
0.857 and the score is a rescaling of the flat count (r = 0.94-0.98).

This module grades a move by HOW BADLY it fails instead: the extra plies it costs
against the fastest forced win available at that node. Wrong moves stop being
interchangeable.

Ground truth
------------
The construct is "win within the puzzle's move budget" (p15 = 5 Red moves = 9
plies; w4p6 = 4 Red moves = 7 plies), NOT the full-game value. A full solve says
almost every w4p6 opening eventually wins, which is true and useless here. So we
run a depth-limited mate search: fewest plies in which Red can FORCE a win
against best defence, or None beyond the cap.

Validated to reproduce both stored solutions exactly -- `winning_columns` and
`plies_to_win` (w4p6: [2] / 7 ; p15: [3] / 9).

Board representation
--------------------
Standard Connect Four bitboard: 7 bits per column (6 rows + a sentinel that stops
horizontal/diagonal shifts from wrapping). `pos` is the side-to-move's stones,
`mask` is all stones. Bit index = col * 7 + row, row 0 = bottom.
"""
from __future__ import annotations

from pathlib import Path

W, H = 7, 6
_ORDER = [3, 2, 4, 1, 5, 0, 6]          # centre-first: alpha-beta prunes far more
INF = 10 ** 6

# Puzzle budgets. `nominal` is the forced-win horizon (what the study grades on);
# `cap` is how much deeper we look so that slower wins get a finite distance
# instead of collapsing to "not a win".
PUZZLES = {
    "p15":   {"config": "puzzle_config_15.txt",   "nominal": 9, "cap": 17},
    "pw4p6": {"config": "puzzle_config_w4p6.txt", "nominal": 7, "cap": 15},
}

_CHAR_TO_PIECE = {"*": 0, "R": 1, "Y": 2}       # EMPTY / RED / YELLOW
EMPTY, RED, YELLOW = 0, 1, 2


# --------------------------------------------------------------------------- board
def load_board(path) -> list[list[int]]:
    """Parse puzzle_config.txt -> board[row][col], row 0 = BOTTOM.

    The file is written top row first, so the last line becomes row 0. Same
    convention as engine.load_board; duplicated here to keep this module
    standalone (the two studies ship slightly different engine.py files).
    """
    lines = [l for l in Path(path).read_text().splitlines() if l.strip()]
    rows_top_first = [[_CHAR_TO_PIECE[c] for c in line.split(", ")] for line in lines]
    return list(reversed(rows_top_first))


def to_bitboard(board, to_move: int) -> tuple[int, int]:
    """board[row][col] -> (pos, mask), where pos is `to_move`'s stones."""
    pos = mask = 0
    for r in range(H):
        for c in range(W):
            v = board[r][c]
            if v == EMPTY:
                continue
            bit = 1 << (c * (H + 1) + r)
            mask |= bit
            if v == to_move:
                pos |= bit
    return pos, mask


def _bottom(c: int) -> int:
    return 1 << (c * (H + 1))


def _top_bit(c: int) -> int:
    return 1 << (c * (H + 1) + H - 1)


def can_play(mask: int, c: int) -> bool:
    return (mask & _top_bit(c)) == 0


def legal_moves(mask: int) -> list[int]:
    return [c for c in _ORDER if can_play(mask, c)]


def play(pos: int, mask: int, c: int) -> tuple[int, int]:
    """Drop into column c. Returns (pos, mask) with the side to move SWAPPED.

    Order matters: XOR against the OLD mask so `new_pos` is the opponent's
    stones, THEN add the new stone to the mask. XORing against the new mask
    would hand the freshly played stone to the incoming side to move.
    """
    new_pos = pos ^ mask
    new_mask = mask | (mask + _bottom(c))
    return new_pos, new_mask


def _aligned(p: int) -> bool:
    m = p & (p >> (H + 1))
    if m & (m >> (2 * (H + 1))):
        return True                                  # horizontal
    m = p & (p >> H)
    if m & (m >> (2 * H)):
        return True                                  # diagonal \
    m = p & (p >> (H + 2))
    if m & (m >> (2 * (H + 2))):
        return True                                  # diagonal /
    m = p & (p >> 1)
    return bool(m & (m >> 2))                        # vertical


def is_winning_move(pos: int, mask: int, c: int) -> bool:
    p2, m2 = play(pos, mask, c)
    return _aligned(p2 ^ m2)                         # the mover's stones after the swap


# --------------------------------------------------------------------------- search
def can_win(pos: int, mask: int, red_to_move: bool, plies: int, memo: dict) -> bool:
    """Can RED force a win within `plies` plies? `pos` = side-to-move's stones."""
    if plies <= 0:
        return False
    key = (pos + mask, red_to_move, plies)
    hit = memo.get(key)
    if hit is not None:
        return hit
    moves = legal_moves(mask)
    if not moves:
        memo[key] = False
        return False
    if red_to_move:
        res = any(is_winning_move(pos, mask, c) for c in moves)
        if not res and plies >= 3:                   # need >= 3 more plies for a later win
            for c in moves:
                p2, m2 = play(pos, mask, c)
                if can_win(p2, m2, False, plies - 1, memo):
                    res = True
                    break
    else:
        res = True                                   # Red wins only if EVERY reply still loses
        for c in moves:
            if is_winning_move(pos, mask, c):        # Yellow wins first
                res = False
                break
            p2, m2 = play(pos, mask, c)
            if not can_win(p2, m2, True, plies - 1, memo):
                res = False
                break
    memo[key] = res
    return res


def mate_distance(pos: int, mask: int, red_to_move: bool, cap: int, memo: dict) -> int:
    """Fewest plies in which Red can force a win, or INF if none within `cap`."""
    for d in range(1 if red_to_move else 2, cap + 1):
        if can_win(pos, mask, red_to_move, d, memo):
            return d
    return INF


def distance_after(pos: int, mask: int, c: int, cap: int, memo: dict) -> int:
    """Plies to a forced Red win after Red plays c (1 if it wins outright)."""
    if is_winning_move(pos, mask, c):
        return 1
    p2, m2 = play(pos, mask, c)
    d = mate_distance(p2, m2, False, cap - 1, memo)
    return INF if d >= INF else d + 1


def grade_move(pos: int, mask: int, played: int, cap: int, memo: dict) -> dict:
    """Grade one Red move by mate-distance loss.

    `loss` is the extra plies the move costs versus the best column available at
    this node. Two special cases, both deliberate:

      - No win is live at this node at all (best is INF): every move is equally
        hopeless, so we assign no blame (loss 0, graded False). You can't lose
        what you don't have.
      - The move throws away a live win (played is INF): censored at cap + 2, so
        the penalty is finite and comparable but flagged via `kept_win`.
    """
    per_col = {c: distance_after(pos, mask, c, cap, memo) for c in legal_moves(mask)}
    best, got = min(per_col.values()), per_col[played]
    if best >= INF:
        loss, graded = 0.0, False
    elif got >= INF:
        loss, graded = float(cap + 2 - best), True
    else:
        loss, graded = float(got - best), True
    return {
        "per_col": {c: (None if d >= INF else d) for c, d in sorted(per_col.items())},
        "best_dist": None if best >= INF else best,
        "played_dist": None if got >= INF else got,
        "winning_cols": sorted(c for c, d in per_col.items() if d < INF),
        "kept_win": got < INF,
        "loss": loss,
        "graded": graded,
    }


def find_config(tag: str, *hints) -> Path:
    """Locate puzzle_config_*.txt by searching `hints` then this file's siblings."""
    name = PUZZLES[tag]["config"]
    roots = [Path(h) for h in hints if h] + [Path(__file__).resolve().parent.parent]
    for root in roots:
        for cand in (root / "games" / "connect_four" / name,
                     root / name,
                     root.parent / "games" / "connect_four" / name):
            if cand.exists():
                return cand
    raise FileNotFoundError(f"could not locate {name}; looked under {[str(r) for r in roots]}")
