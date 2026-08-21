"""Classify what STRATEGY each puzzle's solution requires, not just how hard it is.

Two puzzles can share every difficulty number — same empties, same gap, same
punishment for the greedy move — and still test different ideas. That matters
for the study's transfer design: round 2 is meant to test whether the insight
from the AI-assisted round 1 carries over, which it cannot do if the two puzzles
reward different insights.

The discriminating feature is the SQUARE TYPE of the moves on the optimal line:

    corner    a1 a8 h1 h8            permanent, can never be flipped back
    C-square  a2 b1 a7 b8 g1 h2 …    edge-adjacent to a corner
    X-square  b2 b7 g2 g7            diagonally adjacent to a corner — the square
                                     every primer says never to play, because it
                                     usually hands the corner over
    edge      the rest of row/col 1 and 8
    interior  everything else

A puzzle answered by a corner rewards the beginner heuristic. A puzzle answered
by an X-square requires overriding it. Those are different tasks, however similar
their difficulty metrics look.

Run (from games/othello):
    python motif_analysis.py
"""

from __future__ import annotations

from pathlib import Path

import engine as E
import solver as S

CORNERS = {(0, 0), (0, 7), (7, 0), (7, 7)}
X_SQUARES = {(1, 1), (1, 6), (6, 1), (6, 6)}
C_SQUARES = {(0, 1), (1, 0), (0, 6), (1, 7), (6, 0), (7, 1), (7, 6), (6, 7)}


def square_class(move) -> str:
    if move in CORNERS:
        return "corner"
    if move in X_SQUARES:
        return "X-square"
    if move in C_SQUARES:
        return "C-square"
    if move[0] in (0, 7) or move[1] in (0, 7):
        return "edge"
    return "interior"


def analyse(tag: str) -> dict:
    """Walk the optimal line and record what kind of play it demands."""
    board = E.load_board(f"puzzle_config_{tag}.txt")
    memo: dict = {}
    values = S.move_values(board, E.BLACK, memo)
    first = S.best_moves(board, E.BLACK, memo)[0]
    # The move a naive player makes: flip the most discs available right now.
    greedy = max(E.legal_moves(board, E.BLACK),
                 key=lambda m: len(E.flips_for(board, m, E.BLACK)))

    b, side = E.copy_board(board), E.BLACK
    classes, line, margins = [], [], []
    scored = forced = white_passes = corners = 0
    while not E.is_game_over(b):
        if not E.has_move(b, side):
            if side == E.WHITE:
                white_passes += 1
            side = E.opponent(side)
            continue
        if side == E.BLACK:
            options = len(E.legal_moves(b, E.BLACK))
            move = S.best_moves(b, E.BLACK, memo)[0]
            if options == 1:
                forced += 1                      # not a decision; auto-played in the app
            else:
                scored += 1
                classes.append(square_class(move))
            if move in CORNERS:
                corners += 1
            line.append(E.to_notation(move) + ("*" if options == 1 else ""))
            E.apply_move(b, move, E.BLACK)
        else:
            E.apply_move(b, S.best_defense(b, memo), E.WHITE)
        black, white = E.disc_counts(b)
        margins.append(black - white)
        side = E.opponent(side)

    black, white = E.disc_counts(b)
    return {
        "tag": tag,
        "answer": E.to_notation(first),
        "class": square_class(first),
        "classes": classes,
        "line": " ".join(line),
        "scored": scored,
        "forced": forced,
        "corners": corners,
        "white_passes": white_passes,
        "greedy_cost": max(values.values()) - values[greedy],
        "worst_margin": min(margins),
        "final_margin": black - white,
    }


def main() -> None:
    tags = sorted(p.stem.replace("puzzle_config_", "")
                  for p in Path(".").glob("puzzle_config_*.txt"))
    rows = [analyse(t) for t in tags]

    print(f"{'puzzle':<12} {'ans':>4} {'class':>9} {'dec':>4} {'grdy':>5} {'crnr':>5} "
          f"{'pass':>5} {'worst':>6} {'final':>6}  line by class")
    for r in rows:
        print(f"{r['tag']:<12} {r['answer']:>4} {r['class']:>9} "
              f"{r['scored']:>2}+{r['forced']:<1} {r['greedy_cost']:>5} {r['corners']:>5} "
              f"{r['white_passes']:>5} {r['worst_margin']:>+6d} {r['final_margin']:>+6d}  "
              + " -> ".join(r["classes"]))

    print("\nGrouped by the type of square the answer is:")
    for klass in ("X-square", "corner", "C-square", "edge", "interior"):
        same = [r for r in rows if r["class"] == klass]
        if not same:
            continue
        print(f"  {klass:<9} " + ", ".join(f"{r['tag']}({r['answer']}, greedy {r['greedy_cost']})"
                                          for r in same))
    print("\nA transfer pair should share the answer's square class AND punish the "
          "\ngreedy move similarly; see README.md for the pair the study uses.")


if __name__ == "__main__":
    main()
