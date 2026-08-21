"""Heavier correctness checks for the engine's ground truth. `python validate_engine.py`

test_othello.py is the fast suite that runs on every change. This is the slower
"do we actually believe the solutions" pass, aimed at the two ways the shipped
answers could be wrong:

  (1) THE RULES are subtly wrong — a ray direction that wraps at an edge, a flip
      that isn't bracketed correctly. Both of our implementations (nested lists
      and bitboards) were written by the same author from the same understanding,
      so agreeing with each other proves less than it looks. The checks here test
      the rules against facts that don't come from us:
        - perft: the number of distinct move sequences from the standard opening
          at each depth, a published sequence for Othello (see PERFT below).
        - symmetry: Othello's board has 8-fold dihedral symmetry, so a position's
          exact value must be identical under all 8 transforms and its best moves
          must map through the same transform. A bug in ONE of the eight ray
          directions breaks this immediately; it cannot be self-consistent.
        - color swap: swapping both colors and the side to move must negate the
          value exactly.

  (2) THE SEARCH is wrong — alpha-beta cutting a line it shouldn't, or a
      transposition-table entry returned outside the window it was valid for.
      Checked by re-solving positions with a deliberately naive reference search:
      full window, no pruning, no table, no move ordering. Slow and obviously
      correct, which is the point.

Run:
    python validate_engine.py             # symmetry, color swap, reference search, perft to depth 7
    python validate_engine.py --deep      # more positions, perft to depth 9 (minutes)
"""

from __future__ import annotations

import argparse
import random
import time

import engine as E
import solver as S

# Published perft values for Othello/Reversi: the number of distinct legal move
# sequences of each length from the standard opening. Source (leaf-node column):
# https://www.aartbik.com/MISC/reversi.html — the standard move-generator
# conformance test for this game.
#
# CONVENTION, and it matters. Two rules, both only observable from depth 9 —
# the earliest ply at which a pass can happen and at which a game can end (the
# 9-ply wipeout). Depths 1-8 are identical under every convention:
#   1. A forced PASS consumes a ply, exactly like a move.
#   2. A leaf is a node at the target depth OR a game that ended before it, so a
#      finished game keeps being counted at every deeper depth.
# Each rule was found by a mismatch, not assumed. Rule 1: depth 9 gave 3,005,320
# with a free pass, against 3,005,288 published. Rule 2: depth 10 then gave
# 24,571,056 against 24,571,284 — short by exactly the 228 games that end at ply
# 9. Both matched once the convention was right, so the rules themselves were
# never in question. If this ever fails at depth >= 9 again, check the counting
# convention before believing the move generator is broken.
#
# (The engine's own SEARCH does not consume depth on a pass. That is a separate,
# search-efficiency choice; it cannot affect an exact endgame solve, which always
# runs to the end of the game regardless of depth.)
#   depth: nodes
PERFT = {1: 4, 2: 12, 3: 56, 4: 244, 5: 1396, 6: 8200, 7: 55092,
         8: 390216, 9: 3005288}
# Beyond what pure Python can do here in reasonable time, from the same source:
#   10: 24,571,284   11: 212,258,800   12: 1,939,886,636


# ---------------------------------------------------------------------------
# Board transforms (the dihedral group of the square)
# ---------------------------------------------------------------------------
def _transpose(b):
    return [[b[r][c] for r in range(8)] for c in range(8)]


def _flip_rows(b):
    return [row[:] for row in reversed(b)]


def _flip_cols(b):
    return [list(reversed(row)) for row in b]


TRANSFORMS = [
    ("identity", lambda b: [row[:] for row in b], lambda r, c: (r, c)),
    ("flip_v", _flip_rows, lambda r, c: (7 - r, c)),
    ("flip_h", _flip_cols, lambda r, c: (r, 7 - c)),
    ("rot180", lambda b: _flip_cols(_flip_rows(b)), lambda r, c: (7 - r, 7 - c)),
    ("transpose", _transpose, lambda r, c: (c, r)),
    ("anti_transpose", lambda b: _flip_cols(_flip_rows(_transpose(b))),
     lambda r, c: (7 - c, 7 - r)),
    ("rot90", lambda b: _flip_cols(_transpose(b)), lambda r, c: (c, 7 - r)),
    ("rot270", lambda b: _flip_rows(_transpose(b)), lambda r, c: (7 - c, r)),
]


def random_endgame(rng, empties):
    """A reachable position with `empties` squares left and Black to move."""
    while True:
        board, side = E.new_board(), E.BLACK
        while E.empty_count(board) > empties and not E.is_game_over(board):
            moves = E.legal_moves(board, side)
            if not moves:
                side = E.opponent(side)
                continue
            E.apply_move(board, rng.choice(moves), side)
            side = E.opponent(side)
        if not E.is_game_over(board) and E.has_move(board, E.BLACK):
            return board


# ---------------------------------------------------------------------------
# (1) Rules, checked against facts that don't come from us
# ---------------------------------------------------------------------------
def perft(board, side, depth, passed=False) -> int:
    """Leaves of the game tree truncated at `depth`, per the published convention:
    a pass consumes a ply, and a finished game is itself a leaf."""
    moves = E.legal_moves(board, side)
    if not moves:
        if passed:
            return 1                       # game over: a leaf in its own right
        if depth == 0:
            return 1
        return perft(board, E.opponent(side), depth - 1, True)   # the pass is a ply
    if depth == 0:
        return 1
    total = 0
    for move in moves:
        total += perft(E.result_of(board, move, side), E.opponent(side), depth - 1)
    return total


def check_perft(max_depth: int) -> None:
    board = E.new_board()
    for depth in range(1, max_depth + 1):
        t = time.process_time()
        got = perft(board, E.BLACK, depth)
        want = PERFT[depth]
        status = "ok " if got == want else "FAIL"
        print(f"  {status} perft({depth}) = {got:,}"
              + ("" if got == want else f"  expected {want:,}")
              + f"   [{time.process_time() - t:.1f}s]")
        assert got == want, f"perft({depth}): got {got}, published value {want}"


def check_symmetry(boards) -> None:
    """A position's exact value is invariant under all 8 board symmetries, and
    its best moves map through the same transform."""
    for i, board in enumerate(boards):
        base_value = S.solve(board, E.BLACK, {})
        base_best = set(S.best_moves(board, E.BLACK, {}))
        for name, transform, map_sq in TRANSFORMS:
            t_board = transform(board)
            t_value = S.solve(t_board, E.BLACK, {})
            assert t_value == base_value, (
                f"position {i}: value {t_value} under {name}, {base_value} untransformed")
            t_best = set(S.best_moves(t_board, E.BLACK, {}))
            want = {map_sq(r, c) for r, c in base_best}
            assert t_best == want, (
                f"position {i}: best moves {sorted(t_best)} under {name}, expected {sorted(want)}")
    print(f"  ok  value + best moves invariant under all 8 symmetries "
          f"({len(boards)} positions x 8)")


def check_color_swap(boards) -> None:
    """Swapping both colors and the side to move negates the value exactly."""
    for i, board in enumerate(boards):
        value = S.solve(board, E.BLACK, {})
        swapped = S.solve(E.flip_colors(board), E.WHITE, {})
        assert value == -swapped, f"position {i}: {value} vs swapped {swapped}"
        # The best move is the same SQUARE for the mirrored player.
        assert set(S.best_moves(board, E.BLACK, {})) == \
            set(S.best_moves(E.flip_colors(board), E.WHITE, {})), f"position {i}"
    print(f"  ok  color swap negates the value ({len(boards)} positions)")


# ---------------------------------------------------------------------------
# (2) Search, checked against a deliberately naive reference
# ---------------------------------------------------------------------------
def reference_solve(board, side) -> int:
    """Final disc difference (Black - White) with NO pruning, table or ordering.

    Plain exhaustive minimax on the nested-list board — the slowest and most
    obviously correct thing we can write. Nothing here is shared with the
    bitboard search it is checking.
    """
    if E.is_game_over(board):
        black, white = E.disc_counts(board)
        return black - white
    moves = E.legal_moves(board, side)
    if not moves:
        return reference_solve(board, E.opponent(side))
    values = [reference_solve(E.result_of(board, m, side), E.opponent(side))
              for m in moves]
    return max(values) if side == E.BLACK else min(values)


def check_reference_search(boards) -> None:
    for i, board in enumerate(boards):
        fast = S.solve(board, E.BLACK, {})
        slow = reference_solve(board, E.BLACK)
        assert fast == slow, f"position {i}: alpha-beta {fast}, exhaustive {slow}"
        # And every MOVE's value, not just the root: this is what grading uses.
        for move, value in S.move_values(board, E.BLACK, {}).items():
            ref = reference_solve(E.result_of(board, move, E.BLACK), E.WHITE)
            assert value == ref, (f"position {i}, move {E.to_notation(move)}: "
                                  f"table {value}, exhaustive {ref}")
    print(f"  ok  alpha-beta + transposition table == exhaustive minimax "
          f"({len(boards)} positions, every move)")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep", action="store_true", help="more positions, deeper perft")
    args = ap.parse_args()
    rng = random.Random(20260730)

    perft_depth = 9 if args.deep else 7
    n_random = 12 if args.deep else 5
    ref_empties = 10 if args.deep else 8

    print("rules vs. published perft values (standard opening):")
    check_perft(perft_depth)

    # The shipped puzzles themselves, plus random reachable endgames.
    from pathlib import Path
    puzzles = [E.load_board(p) for p in sorted(Path(".").glob("puzzle_config_*.txt"))]
    randoms = [random_endgame(rng, 10) for _ in range(n_random)]

    print(f"\nsymmetry + color swap ({len(puzzles)} shipped puzzles, {len(randoms)} random):")
    check_symmetry(puzzles + randoms)
    check_color_swap(puzzles + randoms)

    print(f"\nsearch vs. exhaustive minimax ({n_random} random positions, "
          f"{ref_empties} empties):")
    check_reference_search([random_endgame(rng, ref_empties) for _ in range(n_random)])

    print("\nall validation checks passed")


if __name__ == "__main__":
    main()
