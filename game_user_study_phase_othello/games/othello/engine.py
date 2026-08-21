"""Othello (Reversi) engine for the web study.

Pure Python (no numpy) so it runs inside the Flask app and is easy to unit-test.
Same shape as games/connect_four/engine.py — the Flask layer talks to this module
through the same handful of names (load_board, legal_moves, apply_move, minimax,
ai_move, board_to_text, render_image, state_key, status).

Board model
-----------
- `Board` is a list of 8 lists of 8 ints, `board[row][col]`.
- **row 0 is the TOP row**, col 0 is the LEFT column — i.e. the standard Othello
  coordinate system where "d3" is column d (index 3), row 3 (index 2). This is
  the opposite convention from Connect Four's engine (where row 0 is the bottom),
  because Othello notation counts rows downward.
- Cell values are EMPTY / BLACK / WHITE.
- The participant plays BLACK (the side to move in the puzzle); the AI plays
  WHITE. Puzzles are stored with BLACK to move; see `flip_colors` if an imported
  position has White on move.
- A `Board` is plain nested lists, so it is directly JSON-serializable for the
  browser and for storing per-participant session state.

Moves
-----
A move is a `(row, col)` tuple internally and the algebraic string "d3" on the
wire (`to_notation` / `from_notation`). A player with no legal move must pass;
the game ends when neither player can move (usually, but not always, a full
board).

Search
------
`minimax` is depth-limited alpha-beta with the heuristic in `evaluate`. Endgame
puzzles are small enough to search to the very end of the game instead, so
`ai_move`, `best_moves` and `grade_move` all switch to the exact bitboard search
lower down this file once few enough squares are empty (EXACT_EMPTIES). In the
positions the study actually uses that makes the AI's defense unimprovable and
the grading ground truth rather than opinion. Puzzle-level analysis built on that
search — optimal moves, best defense, disc-loss grading — lives in `solver.py`.
"""

from __future__ import annotations

import io
import math
from pathlib import Path

ROWS, COLS = 8, 8

EMPTY, BLACK, WHITE = 0, 1, 2       # cell values
PARTICIPANT, AI = BLACK, WHITE      # role -> which piece

Board = list[list[int]]             # board[row][col], row 0 = top
Move = tuple[int, int]              # (row, col)

# The eight ray directions a move can flip along: (drow, dcol).
DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1),
              (0, -1),           (0, 1),
              (1, -1),  (1, 0),  (1, 1))

# Puzzle files: we accept every spelling of the three cell states we have seen in
# the wild, so positions can be pasted in from other Othello tools unchanged.
#   empty : * . - _ 0
#   black : B X b x 1        (black = the participant, moves first)
#   white : W O o w 2
_CHAR_TO_PIECE = {
    "*": EMPTY, ".": EMPTY, "-": EMPTY, "_": EMPTY, "0": EMPTY,
    "B": BLACK, "X": BLACK, "1": BLACK,
    "W": WHITE, "O": WHITE, "2": WHITE,
}
# How a board renders as text (for the LLM / debugging) and back into a config file.
_PIECE_TO_CHAR = {EMPTY: ".", BLACK: "B", WHITE: "W"}
_PIECE_TO_FILE = {EMPTY: "*", BLACK: "B", WHITE: "W"}

# Exact search kicks in at this many empty squares (see is_endgame). Measured on
# the bitboard search: ~0.1s at 10 empties, ~0.3s typical at 12 (a few seconds
# for the nastiest positions) — fine for a web request, and the imported puzzles
# all start at 9-10 empties.
EXACT_EMPTIES = 12

# Terminal positions dominate any heuristic score, and a bigger disc margin is
# still better than a smaller one, so the AI presses a won endgame.
TERMINAL_SCALE = 10 ** 6


# ============================================================================
# Board construction / mechanical helpers
# ============================================================================
def new_board() -> Board:
    """The standard Othello opening position (Black to move)."""
    board = [[EMPTY] * COLS for _ in range(ROWS)]
    board[3][3] = board[4][4] = WHITE
    board[3][4] = board[4][3] = BLACK
    return board


def empty_board() -> Board:
    """An 8x8 board with no discs (used by puzzle generators/parsers)."""
    return [[EMPTY] * COLS for _ in range(ROWS)]


def copy_board(board: Board) -> Board:
    """A deep copy you can mutate without touching the original."""
    return [row[:] for row in board]


def opponent(piece: int) -> int:
    """BLACK <-> WHITE."""
    return WHITE if piece == BLACK else BLACK


def on_board(row: int, col: int) -> bool:
    return 0 <= row < ROWS and 0 <= col < COLS


def flip_colors(board: Board) -> Board:
    """Swap Black and White. Use to normalize an imported White-to-move puzzle
    into the Black-to-move form the study expects (the position is strategically
    identical, only the colors are relabelled)."""
    return [[EMPTY if v == EMPTY else opponent(v) for v in row] for row in board]


def disc_counts(board: Board) -> tuple[int, int]:
    """(black_discs, white_discs)."""
    black = sum(row.count(BLACK) for row in board)
    white = sum(row.count(WHITE) for row in board)
    return black, white


def disc_diff(board: Board, piece: int) -> int:
    """Disc difference from `piece`'s point of view (the final score of a game)."""
    black, white = disc_counts(board)
    return black - white if piece == BLACK else white - black


def empty_count(board: Board) -> int:
    return sum(row.count(EMPTY) for row in board)


# ============================================================================
# Move notation: (row, col) <-> "d3"
# ============================================================================
def to_notation(move: Move) -> str:
    """(2, 3) -> 'd3'  (column letter a-h, then 1-indexed row from the top)."""
    row, col = move
    return f"{chr(ord('a') + col)}{row + 1}"


def from_notation(text: str) -> Move:
    """'d3' -> (2, 3). Case- and whitespace-insensitive; also accepts '3d'."""
    s = text.strip().lower()
    if len(s) != 2:
        raise ValueError(f"bad move notation: {text!r}")
    if s[0].isdigit():                      # tolerate the reversed "3d" spelling
        s = s[1] + s[0]
    col, row = ord(s[0]) - ord("a"), int(s[1]) - 1
    if not on_board(row, col):
        raise ValueError(f"move off the board: {text!r}")
    return (row, col)


def parse_move(value) -> Move:
    """Coerce whatever the web layer sends into a (row, col) move.

    Accepts "d3", [2, 3], (2, 3), {"row": 2, "col": 3} and the flat index 19.
    Raises ValueError if it cannot be read as a square on the board.
    """
    if isinstance(value, str):
        return from_notation(value)
    if isinstance(value, dict):
        return parse_move((value["row"], value["col"]))
    if isinstance(value, int):
        return (value // COLS, value % COLS)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        row, col = int(value[0]), int(value[1])
        if not on_board(row, col):
            raise ValueError(f"move off the board: {value!r}")
        return (row, col)
    raise ValueError(f"unreadable move: {value!r}")


# ============================================================================
# Rules: legal moves, flipping, game end
# ============================================================================
def flips_for(board: Board, move: Move, piece: int) -> list[Move]:
    """Every disc `piece` would flip by playing `move`. Empty list = illegal move.

    A move is legal iff it lands on an empty square and brackets at least one
    unbroken line of opponent discs between the new disc and another of ours.
    """
    row, col = move
    if not on_board(row, col) or board[row][col] != EMPTY:
        return []
    opp = opponent(piece)
    flipped: list[Move] = []
    for drow, dcol in DIRECTIONS:
        r, c = row + drow, col + dcol
        ray: list[Move] = []
        while on_board(r, c) and board[r][c] == opp:
            ray.append((r, c))
            r, c = r + drow, c + dcol
        # The ray only counts if it is closed off by one of our own discs.
        if ray and on_board(r, c) and board[r][c] == piece:
            flipped.extend(ray)
    return flipped


def is_valid_move(board: Board, move: Move, piece: int) -> bool:
    return bool(flips_for(board, move, piece))


def legal_moves(board: Board, piece: int) -> list[Move]:
    """All squares `piece` may play, in reading order (top-left to bottom-right)."""
    return [(r, c) for r in range(ROWS) for c in range(COLS)
            if flips_for(board, (r, c), piece)]


def has_move(board: Board, piece: int) -> bool:
    """True if `piece` has at least one legal move (cheaper than building the list)."""
    return any(flips_for(board, (r, c), piece)
               for r in range(ROWS) for c in range(COLS))


def apply_move(board: Board, move: Move, piece: int) -> list[Move]:
    """Play `move` for `piece` (mutates board). Returns the discs it flipped.

    Assumes the move is legal — call is_valid_move() first.
    """
    flipped = flips_for(board, move, piece)
    assert flipped, f"illegal move {to_notation(move)} for piece {piece}"
    row, col = move
    board[row][col] = piece
    for r, c in flipped:
        board[r][c] = piece
    return flipped


def result_of(board: Board, move: Move, piece: int) -> Board:
    """A copy of `board` with `move` played — the non-mutating form of apply_move."""
    child = copy_board(board)
    apply_move(child, move, piece)
    return child


def must_pass(board: Board, piece: int) -> bool:
    """True if `piece` has to pass: no move of its own, but the game isn't over."""
    return not has_move(board, piece) and has_move(board, opponent(piece))


def is_game_over(board: Board) -> bool:
    """True when neither side can move (the board need not be full)."""
    return not has_move(board, BLACK) and not has_move(board, WHITE)


def side_to_move(board: Board, nominal: int) -> int | None:
    """Who actually moves next if it is nominally `nominal`'s turn.

    Returns `nominal`, or the opponent when `nominal` must pass, or None if the
    game is over. Lets the web layer handle passes without duplicating the rules.
    """
    if has_move(board, nominal):
        return nominal
    if has_move(board, opponent(nominal)):
        return opponent(nominal)
    return None


def status(board: Board) -> str:
    """One of: 'black_win', 'white_win', 'draw', 'ongoing'."""
    if not is_game_over(board):
        return "ongoing"
    black, white = disc_counts(board)
    if black > white:
        return "black_win"
    if white > black:
        return "white_win"
    return "draw"


def state_key(board: Board, to_move: int) -> str:
    """JSON-friendly key for a (position, side-to-move): 64 cell digits + side.

    Used as the lookup key in the precomputed solution table (solver/precompute).
    """
    flat = "".join(str(board[r][c]) for r in range(ROWS) for c in range(COLS))
    return f"{flat}:{to_move}"


# ============================================================================
# Puzzle loading
# ============================================================================
def parse_board(text: str) -> tuple[Board, int]:
    """Parse a puzzle position out of text. Returns (board, to_move).

    Three layouts are accepted, so positions can be pasted in from other tools:

    1. Eight rows, top row first, cells separated by ", " (the Connect Four
       config style)::

           *, *, *, *, *, *, *, *
           ...

    2. Eight rows of eight bare characters (the common Othello text dump)::

           --------
           ---OX---
           ...

    3. One 64-character line, optionally followed by the side to move::

           ---------------OX--------OXX------XXX---------------------------- X

    Blank lines and `#` comments are ignored. Column-letter / row-number rulers
    (a b c d e f g h down the side or across the top) are stripped. The side to
    move may also be given on its own line as `to_move: B` (or W/X/O); it
    defaults to BLACK, which is what the study's puzzles use.
    """
    to_move = None
    rows: list[list[int]] = []
    flat: list[int] = []

    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith(("to_move", "to move", "turn", "side")):
            marker = low.split(":", 1)[-1].strip()[:1]
            to_move = _CHAR_TO_PIECE.get(marker.upper(), BLACK) or BLACK
            continue
        if low.replace(" ", "").startswith("abcdefgh"):     # column ruler
            continue

        # Strip a leading row number ("3 ..OXX...") and any separators, then read
        # the line as cell characters.
        body = line
        if body[0].isdigit() and len(body) > 1 and not body[1].isdigit():
            body = body[1:]
        # Works for both layouts: ", "-separated single-char cells and a bare run
        # of characters. Anything that isn't a cell character (rulers, stray
        # punctuation) is dropped.
        chars = "".join(body.replace(",", " ").replace("|", " ").split())
        chars = "".join(ch for ch in chars if ch.upper() in _CHAR_TO_PIECE)

        if len(chars) == COLS:
            rows.append([_CHAR_TO_PIECE[ch.upper()] for ch in chars])
        elif len(chars) in (ROWS * COLS, ROWS * COLS + 1):
            # Single-line board, possibly with the side-to-move glued on the end.
            if len(chars) == ROWS * COLS + 1:
                to_move = _CHAR_TO_PIECE[chars[-1].upper()] or BLACK
                chars = chars[:-1]
            flat = [_CHAR_TO_PIECE[ch.upper()] for ch in chars]
        elif chars:
            raise ValueError(f"cannot read {len(chars)} cells as a board row: {raw!r}")

    if flat:
        rows = [flat[r * COLS:(r + 1) * COLS] for r in range(ROWS)]
    if len(rows) != ROWS:
        raise ValueError(f"expected {ROWS} board rows, got {len(rows)}")
    return rows, (BLACK if to_move is None else to_move)


def load_puzzle(path: str | Path) -> tuple[Board, int]:
    """Read a puzzle_config file. Returns (board, to_move)."""
    return parse_board(Path(path).read_text())


def load_board(path: str | Path) -> Board:
    """Read a puzzle_config file and return just the Board.

    Mirrors connect_four.engine.load_board so the Flask layer can stay
    game-agnostic; the study's puzzles are always Black (the participant) to
    move. Use `load_puzzle` when you need the side to move as well.
    """
    return load_puzzle(path)[0]


def board_to_config(board: Board, to_move: int = BLACK) -> str:
    """Serialize a Board back to puzzle_config format (top row first, ', '-joined)."""
    body = "\n".join(", ".join(_PIECE_TO_FILE[board[r][c]] for c in range(COLS))
                     for r in range(ROWS))
    return f"{body}\nto_move: {_PIECE_TO_FILE[to_move]}\n"


# ============================================================================
# Evaluation + search
# ============================================================================
# Classic positional weights: corners are permanent, the squares next to them
# (C- and X-squares) hand the corner away, edges are worth holding.
SQUARE_WEIGHTS = (
    (120, -20,  20,   5,   5,  20, -20, 120),
    (-20, -40,  -5,  -5,  -5,  -5, -40, -20),
    ( 20,  -5,  15,   3,   3,  15,  -5,  20),
    (  5,  -5,   3,   3,   3,   3,  -5,   5),
    (  5,  -5,   3,   3,   3,   3,  -5,   5),
    ( 20,  -5,  15,   3,   3,  15,  -5,  20),
    (-20, -40,  -5,  -5,  -5,  -5, -40, -20),
    (120, -20,  20,   5,   5,  20, -20, 120),
)
CORNERS = ((0, 0), (0, COLS - 1), (ROWS - 1, 0), (ROWS - 1, COLS - 1))


def _frontier_discs(board: Board, piece: int) -> int:
    """Discs of `piece` adjacent to an empty square — they are the flippable ones."""
    count = 0
    for r in range(ROWS):
        for c in range(COLS):
            if board[r][c] != piece:
                continue
            if any(on_board(r + dr, c + dc) and board[r + dr][c + dc] == EMPTY
                   for dr, dc in DIRECTIONS):
                count += 1
    return count


def evaluate(board: Board, piece: int) -> int:
    """Heuristic score of `board` from `piece`'s point of view (higher = better).

    Used by minimax when it hits the depth limit. Three classic Othello terms:
    position (corners good, corner-adjacent squares bad), mobility (how many
    moves each side has), and frontier discs (fewer is better — they are what the
    opponent can flip). Disc count only starts to matter near the end of the
    game, which is why it is weighted by how full the board is.
    """
    opp = opponent(piece)

    positional = 0
    for r in range(ROWS):
        row, weights = board[r], SQUARE_WEIGHTS[r]
        for c in range(COLS):
            v = row[c]
            if v == piece:
                positional += weights[c]
            elif v == opp:
                positional -= weights[c]

    my_moves = len(legal_moves(board, piece))
    opp_moves = len(legal_moves(board, opp))
    mobility = 0
    if my_moves + opp_moves:
        mobility = 10 * (my_moves - opp_moves)
    if my_moves == 0 and opp_moves > 0:
        mobility -= 50                       # being forced to pass is genuinely bad

    frontier = -5 * (_frontier_discs(board, piece) - _frontier_discs(board, opp))

    # Disc difference is noise in the opening and everything at the end.
    filled = ROWS * COLS - empty_count(board)
    discs = disc_diff(board, piece) * max(0, filled - 40) // 2

    return positional + mobility + frontier + discs


def _terminal_score(board: Board, me: int) -> float:
    """Score of a finished game from `me`'s POV: sign decides, margin breaks ties."""
    return TERMINAL_SCALE * disc_diff(board, me)


def _ordered_moves(board: Board, moves: list[Move]) -> list[Move]:
    """Try structurally strong squares first — alpha-beta prunes far more that way."""
    return sorted(moves, key=lambda m: -SQUARE_WEIGHTS[m[0]][m[1]])


def minimax(
    board: Board,
    depth: int,
    alpha: float,
    beta: float,
    to_move: int,
    me: int,
) -> tuple[Move | None, float]:
    """Alpha-beta minimax. Returns (best_move, score) scored from `me`'s POV.

    Parameters
    ----------
    to_move : whose turn it is in `board` (BLACK or WHITE)
    me      : the side we're scoring for (scores are high when `me` is winning)

    Side-agnostic (like the Connect Four engine) so it can score either color.
    A player with no legal move passes: we recurse with the turn handed over and
    the depth *unchanged*, which is safe because two passes in a row means the
    game is over and the terminal test above fires first.
    """
    if is_game_over(board):
        return (None, _terminal_score(board, me))

    moves = legal_moves(board, to_move)
    if not moves:                          # forced pass, not a terminal node
        _, score = minimax(board, depth, alpha, beta, opponent(to_move), me)
        return (None, score)
    if depth <= 0:
        return (None, float(evaluate(board, me)))

    maximizing = to_move == me
    value = -math.inf if maximizing else math.inf
    best_move = moves[0]
    for move in _ordered_moves(board, moves):
        child = result_of(board, move, to_move)
        _, score = minimax(child, depth - 1, alpha, beta, opponent(to_move), me)
        if maximizing:
            if score > value:
                value, best_move = score, move
            alpha = max(alpha, value)
        else:
            if score < value:
                value, best_move = score, move
            beta = min(beta, value)
        if alpha >= beta:                  # alpha-beta cutoff
            break
    return (best_move, value)


# ============================================================================
# Exact endgame search (bitboards)
# ============================================================================
# Endgame puzzles are small enough to search to the very end of the game, which
# turns "the engine's opinion" into ground truth: the value of a move is the
# final disc difference it leads to under optimal play by both sides. The list
# representation above is too slow for that, so the exact search runs on
# bitboards — two 64-bit ints, one per color, bit `row * 8 + col` set for a disc
# on that square (bit 0 = a1 = top-left). Same rules, ~50x faster.
#
# solver.py builds the puzzle-level API (grading, precomputed tables) on top of
# these primitives; the functions here are what makes ai_move exact in the
# endgame without the web layer needing anything but this module.

FULL = (1 << 64) - 1
_NOT_A_FILE = FULL & ~sum(1 << (r * 8) for r in range(8))           # no column a
_NOT_H_FILE = FULL & ~sum(1 << (r * 8 + 7) for r in range(8))       # no column h

# (shift, mask) per ray direction. Positive shift = towards higher indices
# (rightwards / downwards); the mask kills wraparound at the board edge.
_BB_DIRS = (
    (1, _NOT_A_FILE),    # right       (row, col+1)
    (-1, _NOT_H_FILE),   # left        (row, col-1)
    (8, FULL),           # down        (row+1, col)
    (-8, FULL),          # up          (row-1, col)
    (9, _NOT_A_FILE),    # down-right
    (7, _NOT_H_FILE),    # down-left
    (-7, _NOT_A_FILE),   # up-right
    (-9, _NOT_H_FILE),   # up-left
)

# Move-ordering priority, flattened from SQUARE_WEIGHTS: corners first, X/C
# squares last. Ordering matters enormously for alpha-beta — a strong first move
# prunes most of its siblings.
ORDER_WEIGHT = tuple(SQUARE_WEIGHTS[i // 8][i % 8] for i in range(64))

# Transposition-table entry flags.
_EXACT_FLAG, _LOWER_FLAG, _UPPER_FLAG = 0, 1, 2


def to_bits(board: Board) -> tuple[int, int]:
    """Nested-list Board -> (black_bits, white_bits)."""
    black = white = 0
    for r in range(ROWS):
        row = board[r]
        for c in range(COLS):
            v = row[c]
            if v == BLACK:
                black |= 1 << (r * 8 + c)
            elif v == WHITE:
                white |= 1 << (r * 8 + c)
    return black, white


def from_bits(black: int, white: int) -> Board:
    """(black_bits, white_bits) -> nested-list Board."""
    board = empty_board()
    for i in range(64):
        bit = 1 << i
        if black & bit:
            board[i // 8][i % 8] = BLACK
        elif white & bit:
            board[i // 8][i % 8] = WHITE
    return board


def sq_index(move: Move) -> int:
    """(row, col) -> bit index."""
    return move[0] * 8 + move[1]


def index_move(index: int) -> Move:
    """Bit index -> (row, col)."""
    return (index // 8, index % 8)


def _shift(x: int, s: int, mask: int) -> int:
    return ((x << s) & mask) if s > 0 else ((x >> -s) & mask)


def bb_legal(own: int, opp: int) -> int:
    """Bitmask of squares the side holding `own` may play.

    Along each ray: walk a contiguous run of opponent discs starting next to one
    of ours; the empty square just past that run is a legal move. Six shifts
    cover the longest possible run (a line of 8 can flip at most 6).
    """
    empty = ~(own | opp) & FULL
    moves = 0
    for s, mask in _BB_DIRS:
        t = _shift(own, s, mask) & opp
        t |= _shift(t, s, mask) & opp
        t |= _shift(t, s, mask) & opp
        t |= _shift(t, s, mask) & opp
        t |= _shift(t, s, mask) & opp
        t |= _shift(t, s, mask) & opp
        moves |= _shift(t, s, mask) & empty
    return moves


def bb_flips(own: int, opp: int, move_bit: int) -> int:
    """Bitmask of opponent discs flipped by playing `move_bit`."""
    flipped = 0
    for s, mask in _BB_DIRS:
        t = _shift(move_bit, s, mask) & opp
        t |= _shift(t, s, mask) & opp
        t |= _shift(t, s, mask) & opp
        t |= _shift(t, s, mask) & opp
        t |= _shift(t, s, mask) & opp
        t |= _shift(t, s, mask) & opp
        if _shift(t, s, mask) & own:        # the run is closed by our own disc
            flipped |= t
    return flipped


def bb_play(own: int, opp: int, move_bit: int) -> tuple[int, int]:
    """Play `move_bit`; returns the position from the OPPONENT's point of view
    ((new_opp, new_own)) — the form the negamax recursion wants."""
    flipped = bb_flips(own, opp, move_bit)
    return (opp ^ flipped, own | flipped | move_bit)


def bb_indices(mask: int) -> list[int]:
    """Set bit indices of `mask`, strongest square first (for move ordering)."""
    idx = []
    while mask:
        low = mask & -mask
        idx.append(low.bit_length() - 1)
        mask ^= low
    idx.sort(key=lambda i: -ORDER_WEIGHT[i])
    return idx


def bb_negamax(own: int, opp: int, alpha: int, beta: int, passed: bool, tt: dict) -> int:
    """Exact final disc difference from the POV of the side holding `own`.

    Alpha-beta with a transposition table. Because the search always runs to the
    end of the game, a position's value never depends on the depth it was found
    at, so TT entries stay valid for the whole session — but an entry may be a
    bound rather than an exact value, which is what the flag records.
    """
    moves = bb_legal(own, opp)
    if not moves:
        if passed:                                   # two passes -> game over
            return bin(own).count("1") - bin(opp).count("1")
        return -bb_negamax(opp, own, -beta, -alpha, True, tt)

    key = (own, opp)
    entry = tt.get(key)
    if entry is not None:
        value, flag = entry
        if flag == _EXACT_FLAG:
            return value
        if flag == _LOWER_FLAG:
            if value >= beta:
                return value
            alpha = max(alpha, value)
        else:                                        # _UPPER_FLAG
            if value <= alpha:
                return value
            beta = min(beta, value)

    alpha0 = alpha
    best = -65                                       # worse than any real margin
    for idx in bb_indices(moves):
        child_own, child_opp = bb_play(own, opp, 1 << idx)
        score = -bb_negamax(child_own, child_opp, -beta, -alpha, False, tt)
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:                            # cutoff
            break

    tt[key] = (best, _LOWER_FLAG if best >= beta else
               _UPPER_FLAG if best <= alpha0 else _EXACT_FLAG)
    return best


def exact_value(board: Board, to_move: int, tt: dict | None = None) -> int:
    """Exact final disc difference (Black - White) under optimal play.

    Pass the same `tt` dict across calls on one puzzle and every lookup after the
    first is nearly free.
    """
    tt = {} if tt is None else tt
    black, white = to_bits(board)
    own, opp = (black, white) if to_move == BLACK else (white, black)
    value = bb_negamax(own, opp, -65, 65, False, tt)
    return value if to_move == BLACK else -value


def exact_move_values(board: Board, to_move: int,
                      tt: dict | None = None) -> dict[Move, int]:
    """{move: exact final disc difference (Black - White)} for every legal move.

    Each move gets a full window (no cross-move pruning) so the values are exact
    and therefore comparable — that is what lets us count *every* move tied at
    the top as optimal, and say how many discs a suboptimal move actually cost.
    """
    tt = {} if tt is None else tt
    black, white = to_bits(board)
    own, opp = (black, white) if to_move == BLACK else (white, black)
    values: dict[Move, int] = {}
    for idx in bb_indices(bb_legal(own, opp)):
        child_own, child_opp = bb_play(own, opp, 1 << idx)
        # child_own is the opponent's position, so negate for `to_move`'s view.
        score = -bb_negamax(child_own, child_opp, -65, 65, False, tt)
        values[index_move(idx)] = score if to_move == BLACK else -score
    return values


def is_endgame(board: Board) -> bool:
    """True when the position is small enough to search exactly (see EXACT_EMPTIES)."""
    return empty_count(board) <= EXACT_EMPTIES


# ============================================================================
# What the web layer calls: best moves, the AI's reply, grading
# ============================================================================
def best_moves(board: Board, to_move: int, depth: int = 4) -> list[Move]:
    """Every move tied for best for `to_move`.

    Exact (final disc difference) in the endgame, heuristic otherwise. Ties are
    real ties — each move is scored with a full window — which matters when we
    grade the participant: playing *a* best move counts, not just the one the
    search happened to pick first.
    """
    moves = legal_moves(board, to_move)
    if not moves:
        return []
    if is_endgame(board):
        values = exact_move_values(board, to_move)
        best = (max if to_move == BLACK else min)(values.values())
        return sorted(m for m, v in values.items() if v == best)
    scored = []
    for move in moves:
        child = result_of(board, move, to_move)
        _, score = minimax(child, depth - 1, -math.inf, math.inf,
                           to_move=opponent(to_move), me=to_move)
        scored.append((move, score))
    best_score = max(score for _, score in scored)
    return sorted(move for move, score in scored if score == best_score)


def ai_move(board: Board, depth: int = 4) -> Move | None:
    """The AI's chosen move (it plays WHITE). None if it has to pass / game over.

    In the endgame — every position the study uses — this is exact: the AI plays
    the reply that minimizes Black's final disc count, so its defense cannot be
    improved on and the participant has to find the whole line. Ties are broken
    towards the structurally strongest square, so the AI grabs the corner instead
    of an equal-scoring quiet move (a defense that reads as sensible, not weird).
    """
    if not has_move(board, AI):
        return None
    if is_endgame(board):
        values = exact_move_values(board, AI)
        best = min(values.values())                  # White minimizes Black's margin
        tied = [m for m, v in values.items() if v == best]
        return max(tied, key=lambda m: ORDER_WEIGHT[sq_index(m)])
    move, _ = minimax(board, depth, float("-inf"), float("inf"), to_move=AI, me=AI)
    return move


def grade_move(board_before: Board, move, depth: int = 4) -> dict:
    """Continuous quality score for the participant's move (the cp_loss analog).

    From BLACK's POV: compare the score of the best available move with the move
    actually played. In an endgame position (`exact` True) the scores are final
    disc differences, so `score_loss` reads directly as "this move cost N discs".

    For a puzzle with a precomputed table, prefer the O(1) table lookup (see
    precompute.py); this is the live fallback once play leaves the table.
    """
    played = parse_move(move)
    exact = is_endgame(board_before)
    if exact:
        scored = exact_move_values(board_before, BLACK)
    else:
        scored = {}
        for candidate in legal_moves(board_before, BLACK):
            child = result_of(board_before, candidate, BLACK)
            _, score = minimax(child, depth - 1, -math.inf, math.inf,
                               to_move=WHITE, me=BLACK)
            scored[candidate] = score
    if not scored:
        raise ValueError("Black has no legal move in this position")
    best_score = max(scored.values())
    played_score = scored.get(played)
    return {
        "best_moves": [to_notation(m) for m in sorted(scored) if scored[m] == best_score],
        "best_score": best_score,
        "played_move": to_notation(played),
        "played_score": played_score,          # None if the move was illegal
        "score_loss": None if played_score is None else best_score - played_score,
        "optimal": played_score == best_score,
        "still_winning": played_score is not None and played_score > 0,
        "exact": exact,          # True = scores are final disc differences
        "move_values": {to_notation(m): v for m, v in sorted(scored.items())},
    }


# ============================================================================
# Rendering (text for the LLM prompt, PNG for a vision LLM / debugging)
# ============================================================================
def board_to_text(board: Board, to_move: int | None = None) -> str:
    """Render the board for the LLM system prompt (and debugging).

    Columns are labeled a-h and rows 1-8 from the top, matching both the
    participant's screen and standard Othello notation, so the assistant and the
    participant name squares the same way::

          a b c d e f g h
        1 . . . . . . . .
        2 . . . . . . . .
        3 . . . B . . . .
        4 . . . B B . . .
        5 . . . W B . . .
        6 . . . . . . . .
        7 . . . . . . . .
        8 . . . . . . . .

    Pass `to_move` to append the disc count and that side's legal moves.
    """
    header = "  " + " ".join(chr(ord("a") + c) for c in range(COLS))
    rows = [f"{r + 1} " + " ".join(_PIECE_TO_CHAR[board[r][c]] for c in range(COLS))
            for r in range(ROWS)]
    out = header + "\n" + "\n".join(rows)
    if to_move is not None:
        black, white = disc_counts(board)
        moves = legal_moves(board, to_move)
        who = "Black (B)" if to_move == BLACK else "White (W)"
        out += f"\n\nBlack: {black} discs, White: {white} discs."
        out += (f"\n{who} to move. Legal moves: "
                + (", ".join(to_notation(m) for m in moves) if moves else "none (must pass)"))
    return out


def board_to_prompt(
    board: Board,
    to_move: int = BLACK,
    *,
    legal: bool = True,
    flips: bool = False,
    annotate: bool = False,
    lists: bool = True,
) -> str:
    """Render the position for an LLM being *tested* on the puzzle.

    `board_to_text` is the in-study rendering, kept minimal because the study's
    assistant also receives the board as an image. This one is for offline model
    evaluation, where the text is all the model gets, so it states everything a
    reader would otherwise have to infer: which way the rows run, what each
    square is called, and what is currently playable.

    The flags exist because they change WHAT IS BEING MEASURED, and the right
    setting depends on the question:

    legal=True (default)
        Lists the legal moves. Without it you are mostly measuring whether the
        model can compute Othello legality from a grid — most cannot, so the
        answers come back illegal and the endgame question is never reached.
        Turn it off only if legality is the thing you want to test.
    flips=False (default)
        With flips=True each legal move is shown with the discs it would turn
        and the resulting count. That is a large hint: it removes the lookahead
        the puzzle is built around. Useful as an upper-bound condition.
    annotate=False
        Labels corners / X-squares / C-squares. Names the very concept these
        puzzles turn on, so leave it off unless you are probing whether the
        model can use the vocabulary when handed it.
    """
    me = "Black (B)" if to_move == BLACK else "White (W)"
    opp = "White (W)" if to_move == BLACK else "Black (B)"
    black, white = disc_counts(board)
    out = [f"Othello endgame. You are {me}; {opp} is your opponent.",
           "",
           "Board — row 1 is the TOP row, column a is the LEFTMOST column.",
           "'B' is a black disc, 'W' a white disc, '.' an empty square.",
           ""]
    out.append("    " + " ".join(chr(ord("a") + c) for c in range(COLS)))
    for r in range(ROWS):
        out.append(f"  {r + 1} " + " ".join(_PIECE_TO_CHAR[board[r][c]] for c in range(COLS)))
    out.append("")

    if lists:
        # The same position as explicit coordinates. Redundant on purpose: models
        # that mis-index the grid often read a coordinate list correctly.
        def squares(value):
            return [to_notation((r, c)) for r in range(ROWS) for c in range(COLS)
                    if board[r][c] == value] or ["none"]
        out += [f"Black discs ({black}): " + ", ".join(squares(BLACK)),
                f"White discs ({white}): " + ", ".join(squares(WHITE)),
                f"Empty squares ({empty_count(board)}): " + ", ".join(squares(EMPTY)),
                ""]

    if annotate:
        named = {"corners": CORNERS,
                 "X-squares (diagonally adjacent to a corner)": tuple(
                     (r, c) for r in (1, ROWS - 2) for c in (1, COLS - 2)),
                 "C-squares (edge squares beside a corner)": (
                     (0, 1), (1, 0), (0, 6), (1, 7), (6, 0), (7, 1), (7, 6), (6, 7))}
        for label, squares_ in named.items():
            out.append(f"{label}: " + ", ".join(to_notation(m) for m in squares_))
        out.append("")

    out += ["Rules that matter here:",
            "- A move is legal only if the disc you place traps at least one unbroken",
            "  line of opponent discs (in any of the 8 directions) between it and",
            "  another of your own discs. Every trapped disc flips to your colour.",
            "- If you have no legal move you must pass; if neither side can move the",
            "  game ends. Whoever has more discs at the end wins.",
            "- Discs flip back and forth all game, so being behind now means nothing.",
            "  Only the final count decides it.",
            ""]

    moves = legal_moves(board, to_move)
    if legal:
        if not moves:
            out.append("You have no legal move and must pass.")
        elif flips:
            out.append("Your legal moves, with what each one would flip:")
            for m in moves:
                turned = flips_for(board, m, to_move)
                after = result_of(board, m, to_move)
                ab, aw = disc_counts(after)
                out.append(f"  {to_notation(m)} — flips {len(turned)} "
                           f"({', '.join(to_notation(f) for f in turned)}) "
                           f"→ Black {ab}, White {aw}")
        else:
            out.append("Your legal moves: " + ", ".join(to_notation(m) for m in moves))
        out.append("")

    out += ["Play the move that leaves you with the most discs once the board is",
            "finished, assuming your opponent plays as well as possible.",
            "Answer with a single square, e.g. d3.",
            "DO NOT ATTEMPT TO SIMULATE THE FULL GAMEPLAY. YOU HAVE LIMITED TIME."]
    return "\n".join(out)


def board_to_json(board: Board, to_move: int = BLACK) -> dict:
    """The position as structured data, for programmatic prompting or grading."""
    black, white = disc_counts(board)
    return {
        "rows": ["".join(_PIECE_TO_CHAR[v] for v in row) for row in board],
        "row_1_is": "top", "column_a_is": "left",
        "black_squares": [to_notation((r, c)) for r in range(ROWS) for c in range(COLS)
                          if board[r][c] == BLACK],
        "white_squares": [to_notation((r, c)) for r in range(ROWS) for c in range(COLS)
                          if board[r][c] == WHITE],
        "empty_squares": [to_notation((r, c)) for r in range(ROWS) for c in range(COLS)
                          if board[r][c] == EMPTY],
        "black_count": black, "white_count": white,
        "to_move": "black" if to_move == BLACK else "white",
        "legal_moves": [to_notation(m) for m in legal_moves(board, to_move)],
    }


def parse_llm_move(text: str, board: Board | None = None,
                   piece: int = BLACK) -> Move | None:
    """Pull a square out of free-form model output, or None if there isn't one.

    Models wrap answers in prose and markup ("**H4**", "I'd play h-4."), so a
    strict parser scores presentation rather than play. When `board` is given,
    prefer a square that is actually legal — that stops a stray coordinate
    mentioned in the reasoning ("the a8 corner is White's") from being read as
    the answer ahead of the real one.
    """
    import re

    found = [m for m in re.finditer(r"\b([a-hA-H])\s*[-–—]?\s*([1-8])\b", text)]
    if not found:
        return None
    squares = []
    for m in found:
        try:
            squares.append(from_notation(m.group(1) + m.group(2)))
        except ValueError:
            continue
    if not squares:
        return None
    if board is not None:
        legal = [s for s in squares if is_valid_move(board, s, piece)]
        if legal:
            return legal[-1]        # last legal square mentioned = the conclusion
    return squares[-1]


def render_image(board: Board, legal_for: int | None = None) -> bytes:
    """Render the board as a PNG (bytes) for a vision LLM.

    Black discs = the participant, white discs = the AI, green felt otherwise.
    Columns are labeled a-h across the top and rows 1-8 down the left side, to
    match the participant's screen so both share one coordinate system. Pass
    `legal_for` to mark that side's legal moves with a faint dot. PIL is imported
    lazily so the engine still loads where Pillow isn't present.
    """
    from PIL import Image, ImageDraw, ImageFont

    cell, pad, label = 60, 12, 26
    width = pad * 2 + label + COLS * cell
    height = pad * 2 + label + ROWS * cell
    img = Image.new("RGB", (width, height), (240, 240, 240))
    d = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    left, top = pad + label, pad + label

    for c in range(COLS):                                  # column letters (above)
        d.text((left + c * cell + cell // 2, pad + label // 2),
               chr(ord("a") + c), fill=(50, 50, 50), font=font, anchor="mm")
    for r in range(ROWS):                                  # row numbers (left)
        d.text((pad + label // 2, top + r * cell + cell // 2),
               str(r + 1), fill=(50, 50, 50), font=font, anchor="mm")

    d.rectangle([left, top, left + COLS * cell, top + ROWS * cell], fill=(27, 122, 62))
    for i in range(ROWS + 1):                              # grid lines
        d.line([(left, top + i * cell), (left + COLS * cell, top + i * cell)],
               fill=(15, 80, 40), width=2)
        d.line([(left + i * cell, top), (left + i * cell, top + ROWS * cell)],
               fill=(15, 80, 40), width=2)

    margin = 6
    fill = {BLACK: (20, 20, 20), WHITE: (250, 250, 250)}
    for r in range(ROWS):
        for c in range(COLS):
            if board[r][c] == EMPTY:
                continue
            x0, y0 = left + c * cell + margin, top + r * cell + margin
            d.ellipse([x0, y0, x0 + cell - 2 * margin, y0 + cell - 2 * margin],
                      fill=fill[board[r][c]], outline=(15, 80, 40))

    if legal_for is not None:
        for r, c in legal_moves(board, legal_for):
            cx, cy = left + c * cell + cell // 2, top + r * cell + cell // 2
            d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=(120, 190, 140))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


if __name__ == "__main__":       # quick smoke check: python engine.py [puzzle_config.txt]
    import sys

    if len(sys.argv) > 1:
        board, to_move = load_puzzle(sys.argv[1])
    else:
        board, to_move = new_board(), BLACK
    print(board_to_text(board, to_move))
    print("\nempties:", empty_count(board), "| status:", status(board))
    print("best for side to move:",
          [to_notation(m) for m in best_moves(board, to_move)])
