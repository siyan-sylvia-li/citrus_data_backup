"""Self-checks for the Othello engine, solver, and puzzle tables.

Plain asserts, no pytest needed:  python test_othello.py

What it covers:
  1. Rules — the fast bitboard path agrees with the readable list path, move for
     move and flip for flip, over complete random games.
  2. Search — the exact endgame value equals a full-depth search with the
     independent list-based minimax.
  3. Puzzle loading — every accepted file layout parses to the same board, and
     config round-trips.
  4. Solution tables — every shipped puzzle can be played from its own table the
     way the web layer will play it, passes included, and the game really does
     end on the margin the table promises.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import engine as E
import solver as S

HERE = Path(__file__).parent


def test_bitboard_matches_list_rules(games: int = 25) -> None:
    rng = random.Random(0)
    for _ in range(games):
        board, side = E.new_board(), E.BLACK
        while not E.is_game_over(board):
            moves = E.legal_moves(board, side)
            if not moves:
                side = E.opponent(side)
                continue
            own, opp = E.to_bits(board)
            if side == E.WHITE:
                own, opp = opp, own
            bb_moves = sorted(E.index_move(i) for i in E.bb_indices(E.bb_legal(own, opp)))
            assert bb_moves == sorted(moves), (bb_moves, sorted(moves))

            move = rng.choice(moves)
            bit = 1 << E.sq_index(move)
            bb_flip = sorted(E.index_move(i) for i in E.bb_indices(E.bb_flips(own, opp, bit)))
            assert bb_flip == sorted(E.flips_for(board, move, side))

            new_own, new_opp = E.bb_play(own, opp, bit)
            E.apply_move(board, move, side)
            black, white = E.to_bits(board)
            assert (new_own, new_opp) == ((white, black) if side == E.BLACK else (black, white))
            side = E.opponent(side)
        # A finished game agrees with status() and the disc count.
        black, white = E.disc_counts(board)
        assert E.status(board) == ("black_win" if black > white else
                                   "white_win" if white > black else "draw")
    print(f"ok  rules: bitboard == list over {games} random games")


def test_exact_matches_full_depth_minimax(trials: int = 3, empties: int = 9) -> None:
    rng = random.Random(1)
    for _ in range(trials):
        board, side = E.new_board(), E.BLACK
        while E.empty_count(board) > empties and not E.is_game_over(board):
            moves = E.legal_moves(board, side)
            if not moves:
                side = E.opponent(side)
                continue
            E.apply_move(board, rng.choice(moves), side)
            side = E.opponent(side)
        if E.is_game_over(board) or not E.has_move(board, side):
            continue
        exact = S.solve(board, side, {})
        _, deep = E.minimax(board, 64, float("-inf"), float("inf"), to_move=side, me=E.BLACK)
        assert exact == deep / E.TERMINAL_SCALE, (exact, deep)
    print(f"ok  search: exact endgame value == full-depth minimax ({empties} empties)")


def test_puzzle_formats() -> None:
    # The same position written four ways: the site's 64-char string, the
    # Connect Four-style ", "-separated grid, a bare character grid, and a grid
    # with rulers. All must parse identically.
    flat = "2222200022122200212121222112212021212121222112212012212001111110"
    reference, to_move = E.parse_board(flat)
    assert to_move == E.BLACK                       # defaults to Black to move

    grid = "\n".join(", ".join("*BW"[v] for v in row) for row in reference)
    bare = "\n".join("".join(".XO"[v] for v in row) for row in reference)
    ruled = ("  a b c d e f g h\n"
             + "\n".join(f"{r + 1} " + " ".join(".BW"[v] for v in reference[r])
                         for r in range(8)))
    for text in (grid, bare, ruled):
        assert E.parse_board(text)[0] == reference, text[:40]

    # Side to move: trailing marker on the flat string, or its own line.
    assert E.parse_board(flat + "2")[1] == E.WHITE
    assert E.parse_board(grid + "\nto_move: W")[1] == E.WHITE

    # Config round-trip through a file, and color flipping is an involution.
    path = HERE / "_roundtrip_tmp.txt"
    path.write_text(E.board_to_config(reference, E.WHITE))
    try:
        board, side = E.load_puzzle(path)
        assert board == reference and side == E.WHITE
    finally:
        path.unlink()
    assert E.flip_colors(E.flip_colors(reference)) == reference

    # Notation round-trip over the whole board.
    for r in range(8):
        for c in range(8):
            assert E.from_notation(E.to_notation((r, c))) == (r, c)
    assert E.parse_move("d3") == E.parse_move([2, 3]) == E.parse_move(19)
    print("ok  puzzles: every accepted file layout parses to the same board")


def play_from_table(board, solution: dict) -> tuple[int, int]:
    """Play a puzzle out exactly as the web layer would, using only the table.

    Black plays a tabulated optimal move, White plays its tabulated defense.
    Returns the final (black, white) disc counts.
    """
    board, side = E.copy_board(board), E.BLACK
    while not E.is_game_over(board):
        entry = solution.get(E.state_key(board, side))
        assert entry is not None, f"position missing from table ({side})"
        if entry.get("must_pass"):
            assert not E.has_move(board, side)
            side = E.opponent(side)
            continue
        move = (entry["best_moves"][0] if side == E.BLACK else entry["best_defense"])
        E.apply_move(board, E.from_notation(move), side)
        side = E.opponent(side)
    return E.disc_counts(board)


def test_solution_tables() -> None:
    configs = sorted(HERE.glob("puzzle_config_*.txt"))
    assert configs, "no puzzles found — run import_puzzles.py --write"
    for config in configs:
        tag = config.stem.replace("puzzle_config_", "")
        with open(HERE / f"solution_{tag}.json") as f:
            solution = json.load(f)
        board, to_move = E.load_puzzle(config)
        assert to_move == E.BLACK, f"{tag}: puzzles must be Black to move"

        meta = solution["_meta"]
        root = solution[E.state_key(board, E.BLACK)]

        # The table's root claim matches a fresh exact solve.
        memo: dict = {}
        assert S.solve(board, E.BLACK, memo) == meta["optimal_diff"]
        assert [E.to_notation(m) for m in S.best_moves(board, E.BLACK, memo)] \
            == root["best_moves"]

        # Playing the table out reaches exactly the promised final margin.
        black, white = play_from_table(board, solution)
        assert black - white == meta["optimal_diff"], (tag, black - white)

        # Every tabulated Black move value agrees with the live grader, and the
        # live AI reply agrees with the tabulated defense (the fallback path the
        # web layer uses once the participant leaves the table).
        graded = E.grade_move(board, root["best_moves"][0])
        assert graded["optimal"] and graded["best_moves"] == root["best_moves"]
        assert graded["move_values"] == root["move_values"]

        after = E.copy_board(board)
        E.apply_move(after, E.from_notation(root["best_moves"][0]), E.BLACK)
        if E.has_move(after, E.WHITE):
            table_reply = solution[E.state_key(after, E.WHITE)]["best_defense"]
            assert E.to_notation(E.ai_move(after)) == table_reply, tag
    print(f"ok  tables: {len(configs)} puzzles play out to their promised margin")


def find_pass_position(seed: int = 4):
    """A real (reachable) position where the side to move must pass."""
    rng = random.Random(seed)
    for _ in range(200):
        board, side = E.new_board(), E.BLACK
        while not E.is_game_over(board):
            if E.must_pass(board, side):
                return board, side
            E.apply_move(board, rng.choice(E.legal_moves(board, side)), side)
            side = E.opponent(side)
    raise AssertionError("no pass position found")


def test_pass_handling() -> None:
    """A side with no legal move passes; that is not the end of the game."""
    board, side = find_pass_position()
    assert E.legal_moves(board, side) == []
    assert E.must_pass(board, side)
    assert not E.is_game_over(board)
    assert E.side_to_move(board, side) == E.opponent(side)

    # Search must hand the turn over rather than score the position as finished:
    # a pass node is worth exactly what the position is worth to the other side.
    move, score = E.minimax(board, 3, float("-inf"), float("inf"),
                            to_move=side, me=side)
    handed_over = E.minimax(board, 3, float("-inf"), float("inf"),
                            to_move=E.opponent(side), me=side)[1]
    assert move is None and score == handed_over
    assert score != E.TERMINAL_SCALE * E.disc_diff(board, side) or E.is_game_over(board)
    if side == E.WHITE:                      # the AI passes by returning None
        assert E.ai_move(board) is None
    # The exact solver handles a pass without the caller special-casing it (only
    # worth running when the position is small enough to solve).
    if E.empty_count(board) <= E.EXACT_EMPTIES:
        assert isinstance(S.solve(board, side, {}), int)
    print("ok  passes: a side with no legal move passes and play continues")


if __name__ == "__main__":
    test_bitboard_matches_list_rules()
    test_exact_matches_full_depth_minimax()
    test_puzzle_formats()
    test_pass_handling()
    test_solution_tables()
    print("\nall checks passed")
