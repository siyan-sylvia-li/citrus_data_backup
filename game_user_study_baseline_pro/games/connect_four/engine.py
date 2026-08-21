"""Connect Four engine for the web study.

Pure Python (no pygame, no numpy) so it runs inside the Flask app and is easy to
unit-test. Adapted in spirit from Keith Galli's Connect4-Python — the original
numpy version is in `connect_four_ai.py` in this folder; keep it open as a
reference while you fill in the TODOs.

Board model
-----------
- `Board` is a list of ROWS lists of COLS ints.
- **row 0 is the BOTTOM row** (matches Galli's code: pieces fall to low indices).
- Cell values are EMPTY / RED / YELLOW.
- The participant plays RED (the side to move in the puzzle); the AI plays YELLOW.
- A `Board` is plain nested lists, so it's directly JSON-serializable for the
  browser and for storing per-participant state.

What you implement (search for "TODO(you)"):
  load_board, winning_move, is_full, evaluate, minimax, ai_move,
  board_to_text, grade_move
The mechanical helpers above those are provided so you can focus on the logic.
"""

from __future__ import annotations

import io
import math
from pathlib import Path

ROWS, COLS = 6, 7
CONNECT = 4                      # pieces in a row needed to win

EMPTY, RED, YELLOW = 0, 1, 2     # cell values
PARTICIPANT, AI = RED, YELLOW    # role -> which piece

# puzzle_config.txt characters: R = red (participant), B = the AI, * = empty.
_CHAR_TO_PIECE = {"*": EMPTY, "R": RED, "Y": YELLOW}
# How a board renders as text (for the LLM / debugging).
_PIECE_TO_CHAR = {EMPTY: ".", RED: "R", YELLOW: "Y"}

Board = list[list[int]]          # board[row][col], row 0 = bottom


# ============================================================================
# Provided helpers — mechanical board operations. Read them; you'll use them.
# ============================================================================
def new_board() -> Board:
    """An empty 6x7 board."""
    return [[EMPTY] * COLS for _ in range(ROWS)]


def copy_board(board: Board) -> Board:
    """A deep copy you can mutate without touching the original."""
    return [row[:] for row in board]


def is_valid_column(board: Board, col: int) -> bool:
    """Can a piece still be dropped in this column? (top cell empty)"""
    return 0 <= col < COLS and board[ROWS - 1][col] == EMPTY


def get_next_open_row(board: Board, col: int) -> int | None:
    """Lowest empty row in a column, or None if the column is full."""
    for r in range(ROWS):
        if board[r][col] == EMPTY:
            return r
    return None


def valid_columns(board: Board) -> list[int]:
    """All columns a piece can currently be dropped into."""
    return [c for c in range(COLS) if is_valid_column(board, c)]


def drop(board: Board, col: int, piece: int) -> int:
    """Drop `piece` into `col` (mutates board). Returns the row it landed on.

    Assumes the column is valid — call is_valid_column() first.
    """
    row = get_next_open_row(board, col)
    assert row is not None, f"column {col} is full"
    board[row][col] = piece
    return row


def opponent(piece: int) -> int:
    """RED <-> YELLOW."""
    return YELLOW if piece == RED else RED


def state_key(board: Board, to_move: int) -> str:
    """JSON-friendly key for a (position, side-to-move): 42 cell digits + side.

    Used as the lookup key in the precomputed solution table.
    """
    flat = "".join(str(board[r][c]) for r in range(ROWS) for c in range(COLS))
    return f"{flat}:{to_move}"


# ============================================================================
# TODO(you) — core game logic. Adapt from connect_four_ai.py where noted.
# ============================================================================
def load_board(path: str | Path) -> Board:
    """Parse puzzle_config.txt into a Board.

    The file has ROWS lines, **top row first**, with cells separated by ", ",
    e.g.  "*, R, R, B, B, R, B".  Map each character with _CHAR_TO_PIECE.

    Gotcha: the file is top-to-bottom but our Board is bottom-to-top (row 0 =
    bottom), so the LAST line of the file is board[0].

    Hints: Path(path).read_text().splitlines(); line.split(", ");
           build the rows then reverse them.
    """
    lines = Path(path).read_text().splitlines()
    rows_top_first = [[_CHAR_TO_PIECE[c] for c in line.split(", ")]
                      for line in lines if line.strip()]
    # File is top-to-bottom but row 0 is the bottom -> reverse, and wrap in
    # list() because reversed() returns a one-shot iterator, not a list.
    return list(reversed(rows_top_first))


def winning_move(board: Board, piece: int) -> bool:
    """True if `piece` has CONNECT-in-a-row anywhere.

    Check all four directions: horizontal, vertical, and both diagonals.
    Adapt winning_move() from connect_four_ai.py — but it uses numpy indexing
    (board[r][c]); here board is a plain list of lists, indexed the same way.
    """
    # Check horizontal locations for win
    for c in range(COLS - 3):
        for r in range(ROWS):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True

	# Check vertical locations for win
    for c in range(COLS):
        for r in range(ROWS - 3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True

	# Check positively sloped diaganols
    for c in range(COLS - 3):
        for r in range(ROWS-3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True

	# Check negatively sloped diaganols
    for c in range(COLS - 3):
        for r in range(3, ROWS):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True
    return False


def is_full(board: Board) -> bool:
    """True if there are no legal moves left (a draw if nobody has won).

    Hint: one line using valid_columns().
    """
    return len(valid_columns(board)) == 0


def evaluate(board: Board, piece: int) -> int:
    """Heuristic score of `board` from `piece`'s point of view (higher = better).

    Used by minimax when it hits the depth limit (non-terminal positions).
    Adapt score_position() + evaluate_window() from connect_four_ai.py. The idea:
    slide a length-CONNECT window over every row/column/diagonal and reward
    windows with 2-3 of your pieces (and penalize the opponent's threats).
    """
    def evaluate_window(window, piece):
        score = 0
        opp_piece = opponent(piece)

        if window.count(piece) == 4:
            score += 100
        elif window.count(piece) == 3 and window.count(EMPTY) == 1:
            score += 5
        elif window.count(piece) == 2 and window.count(EMPTY) == 2:
            score += 2

        if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
            score -= 4

        return score
    
    score = 0

    ## Score center column
    center_array = [board[r][COLS // 2] for r in range(ROWS)]
    center_count = center_array.count(piece)
    score += center_count * 3

    ## Score Horizontal
    for r in range(ROWS):
        row_array = board[r]
        for c in range(COLS - 3):
            window = row_array[c:c + CONNECT]
            score += evaluate_window(window, piece)

    ## Score Vertical
    for c in range(COLS):
        col_array = [board[r][c] for r in range(ROWS)]
        for r in range(ROWS - 3):
            window = col_array[r:r + CONNECT]
            score += evaluate_window(window, piece)

    ## Score posiive sloped diagonal
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            window = [board[r+i][c+i] for i in range(CONNECT)]
            score += evaluate_window(window, piece)

    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            window = [board[r+3-i][c+i] for i in range(CONNECT)]
            score += evaluate_window(window, piece)

    return score


def minimax(
    board: Board,
    depth: int,
    alpha: float,
    beta: float,
    to_move: int,
    me: int,
) -> tuple[int | None, float]:
    """Alpha-beta minimax. Return (best_col, score) scored from `me`'s POV.

    Parameters
    ----------
    to_move : whose turn it is in `board` (RED or YELLOW)
    me      : the side we're scoring for (scores are high when `me` is winning)

    Adapt minimax() from connect_four_ai.py, but make it side-agnostic via the
    `to_move`/`me` params instead of a maximizingPlayer bool, so you can score
    forced wins for either color. Terminal values:
        `me` wins      -> +1e14
        opponent wins  -> -1e14
        board full     ->  0
    At depth 0 return (None, evaluate(board, me)).
    Maximize when to_move == me, otherwise minimize; recurse with
    to_move=opponent(to_move).
    """
    # Terminal: someone has connected four, or the board is full.
    if winning_move(board, me):
        return (None, 1e14)
    if winning_move(board, opponent(me)):
        return (None, -1e14)
    valid_locations = valid_columns(board)
    if not valid_locations:               # board full -> draw
        return (None, 0.0)
    if depth == 0:                        # depth cutoff -> heuristic
        return (None, float(evaluate(board, me)))

    maximizing = to_move == me
    value = -math.inf if maximizing else math.inf
    best_col = valid_locations[0]
    for col in valid_locations:
        child = copy_board(board)
        drop(child, col, to_move)
        # After to_move plays, it's the other side's turn; `me` is unchanged.
        _, score = minimax(child, depth - 1, alpha, beta, opponent(to_move), me)
        if maximizing:
            if score > value:
                value, best_col = score, col
            alpha = max(alpha, value)
        else:
            if score < value:
                value, best_col = score, col
            beta = min(beta, value)
        if alpha >= beta:                 # alpha-beta cutoff
            break
    return (best_col, value)


# ============================================================================
# TODO(you) — glue the engine exposes to the web layer.
# ============================================================================
def ai_move(board: Board, depth: int = 5) -> int | None:
    """The AI's chosen column (it plays YELLOW). None if the board is full.

    Hint: call minimax(board, depth, -inf, +inf, to_move=AI, me=AI) and take [0].
    """
    best_col, _ = minimax(board, depth, float("-inf"), float("inf"), to_move=AI, me=AI)
    return best_col


def board_to_text(board: Board) -> str:
    """Render the board for the LLM system prompt (and debugging).

    Top row first, columns labeled 0..6, using _PIECE_TO_CHAR. Example:

        1 2 3 4 5 6 7
        . . . . . . .
        . . . . . . .
        . . . . . . .
        . . . . . R .
        . R Y Y R Y .
        . R R Y Y R Y

    Columns are shown 1-indexed (internal index + 1) to match the human UI.
    Remember row 0 is the bottom, so iterate rows high -> low.
    """
    # Columns are 0-indexed internally, but shown 1-indexed to match the human
    # UI and everyday convention (so the LLM and the participant speak the same
    # numbers). The mapping is display = internal + 1.
    header = " ".join(str(c + 1) for c in range(COLS))
    rows = [" ".join(_PIECE_TO_CHAR[board[r][c]] for c in range(COLS))
            for r in range(ROWS - 1, -1, -1)]   # top row first
    return header + "\n" + "\n".join(rows)


def render_image(board: Board) -> bytes:
    """Render the board as a PNG (bytes) for a vision LLM.

    Red discs = the participant ("R"), yellow = the AI ("Y"), empty = light holes.
    Columns are labeled 1-7 above the board and rows 1-6 (1 = bottom) down the
    left side, to match the participant's screen so both share one coordinate
    system. PIL is imported lazily so the engine still loads where Pillow isn't
    present.
    """
    from PIL import Image, ImageDraw, ImageFont

    cell, pad, label_h, margin = 64, 14, 26, 7
    row_label_w = 26                               # gutter on the left for row numbers
    width = row_label_w + COLS * cell + 2 * pad
    height = ROWS * cell + 2 * pad + label_h
    img = Image.new("RGB", (width, height), (240, 240, 240))
    d = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    board_left = pad + row_label_w
    top = pad + label_h

    for c in range(COLS):                          # 1-indexed column labels (above)
        cx = board_left + c * cell + cell // 2
        d.text((cx, label_h // 2 + 2), str(c + 1), fill=(50, 50, 50), font=font, anchor="mm")

    for r in range(ROWS):                          # 1-indexed row labels (left), 1 = bottom
        vis = ROWS - 1 - r
        cy = top + vis * cell + cell // 2
        d.text((pad + row_label_w // 2, cy), str(r + 1), fill=(50, 50, 50), font=font, anchor="mm")

    d.rounded_rectangle([board_left, top, board_left + COLS * cell, top + ROWS * cell],
                        radius=12, fill=(21, 101, 192))

    fill = {EMPTY: (245, 245, 245), RED: (211, 47, 47), YELLOW: (251, 192, 45)}
    for r in range(ROWS):                          # r = 0 is the bottom row
        vis = ROWS - 1 - r                         # 0 = top row visually
        for c in range(COLS):
            x0 = board_left + c * cell + margin
            y0 = top + vis * cell + margin
            d.ellipse([x0, y0, x0 + cell - 2 * margin, y0 + cell - 2 * margin],
                      fill=fill[board[r][c]])

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def grade_move(board_before: Board, col: int, depth: int = 6) -> dict:
    """Continuous quality score for the participant's move (chess-cp_loss analog).

    From RED's POV: compare the minimax score of the *best* available move
    against the move actually played, both evaluated after the AI's optimal
    reply (i.e. search from `to_move=AI, me=RED` once you've dropped RED's piece).

    Suggested return shape:
        {
          "best_col": int, "best_score": float,
          "played_col": int, "played_score": float,
          "score_loss": float,           # best_score - played_score, >= 0
          "still_winning": bool,         # played_score still a forced win?
        }
    Use a large threshold (e.g. score >= 1e9) to decide "still_winning".

    NOTE: for this fixed puzzle we grade via the exact precompute instead —
    see solver.grade(). This live-minimax version is left as an optional stub.
    """
    raise NotImplementedError("TODO(you): grade the participant's move (or use solver.grade)")


# ============================================================================
# Provided — status reporting (used by the Flask layer). Works once the
# functions above are implemented.
# ============================================================================
def status(board: Board) -> str:
    """One of: 'red_win', 'yellow_win', 'draw', 'ongoing'."""
    if winning_move(board, RED):
        return "red_win"
    if winning_move(board, YELLOW):
        return "yellow_win"
    if is_full(board):
        return "draw"
    return "ongoing"
