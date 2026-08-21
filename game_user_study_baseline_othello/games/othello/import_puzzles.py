"""Import real Othello endgame puzzles, solve them exactly, and rank them.

Source: othelloclub.com's daily endgame puzzle archive
(https://othelloclub.com/en/puzzle_ep.php), whose task — "win from here" — is
exactly the task we set participants. Its positions are 64-character strings
(0 empty / 1 black / 2 white, row by row from the top) plus a side to move;
othelloclub_puzzles.txt holds a fetched batch in `DATE|TURN|BOARD` form.

Those puzzles ship with no solution, which is fine: we compute ground truth
ourselves with the exact endgame solver, so every imported puzzle arrives with
its optimal move(s), the exact final margin, and the cost in discs of every
alternative.

Normalization: puzzles that are White to move are color-flipped so the
participant always plays Black. The position is strategically identical — only
the labels change — and it keeps one convention across the study.

Run (from games/othello):
    python import_puzzles.py --dry-run                       # solve + rank the batch
    python import_puzzles.py --max-empties 12 --write        # emit config + solution
    python import_puzzles.py --board <64 chars> --turn 1 --tag mine --write
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import generate_puzzles
import precompute
import solver
from engine import (
    BLACK,
    board_to_config, board_to_text, empty_count, flip_colors, has_move,
    parse_board,
)

HERE = Path(__file__).parent
SOURCE_FILE = HERE / "othelloclub_puzzles.txt"


def read_source(path: Path) -> list[tuple[str, int, str]]:
    """Parse `DATE|TURN|BOARD` lines into (tag_id, turn, board_string)."""
    out = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        date, turn, board = (part.strip() for part in line.split("|"))
        out.append((date, int(turn), board))
    return out


def from_url(url: str) -> tuple[str, int, str]:
    """Pull (date, turn, board) straight out of an othelloclub board.php URL."""
    q = parse_qs(urlparse(url).query)
    return (q.get("date", ["url"])[0], int(q.get("turn", ["1"])[0]), q["board"][0])


def normalize(board_string: str, turn: int):
    """(64-char string, turn) -> a Board with BLACK to move.

    White-to-move positions are color-flipped; `flipped` says whether that
    happened, so the imported file can record it.
    """
    board, _ = parse_board(board_string)
    flipped = turn == 2
    return (flip_colors(board) if flipped else board), flipped


def report_line(tag: str, m: dict, seconds: float) -> str:
    verdict = solver.outcome_name(m["optimal_diff"])
    return (f"{tag:<12} empties={m['empties']:>2}  legal={m['n_legal']:>2}  "
            f"best={','.join(m['best_moves']):<12} {m['optimal_diff']:+3d} "
            f"({verdict:<10}) gap={str(m['gap']):>4}  "
            f"naive={m['naive_move']} costs {str(m['naive_loss']):>3}  "
            f"line={m['forced_line']}  [{seconds:.1f}s]")


def write_puzzle(board, tag: str, m: dict, source: dict) -> None:
    config_path = HERE / f"puzzle_config_{tag}.txt"
    header = (f"# {source.get('site', 'imported')} puzzle {source.get('date', tag)}"
              f"{' (color-flipped: originally White to move)' if source.get('flipped') else ''}\n")
    config_path.write_text(header + board_to_config(board, BLACK))

    solution = precompute.build_solution(board, BLACK)
    solution["_meta"] = {
        "config": config_path.name,
        "to_move": "B",
        "source": source,
        "empties": m["empties"],
        "optimal_diff": m["optimal_diff"],
        "result": solver.outcome_name(m["optimal_diff"]),
        "root_best_moves": m["best_moves"],
        "metrics": {k: v for k, v in m.items() if k != "move_values"},
    }
    with open(HERE / f"solution_{tag}.json", "w") as f:
        json.dump(solution, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(SOURCE_FILE), help="DATE|TURN|BOARD file")
    ap.add_argument("--board", help="a single 64-char board string (instead of --source)")
    ap.add_argument("--turn", type=int, default=1, help="side to move for --board (1=black, 2=white)")
    ap.add_argument("--url", help="an othelloclub board.php URL to import")
    ap.add_argument("--tag", help="id for a single imported puzzle (default: its date)")
    ap.add_argument("--prefix", default="oc", help="tag prefix for batch imports")
    ap.add_argument("--max-empties", type=int, default=13,
                    help="skip positions with more empty squares (solve time explodes)")
    ap.add_argument("--require-win", action="store_true",
                    help="keep only puzzles Black actually wins with best play")
    ap.add_argument("--write", action="store_true", help="write config + solution files")
    ap.add_argument("--dry-run", action="store_true", help="report only (the default)")
    args = ap.parse_args()

    if args.url:
        entries = [from_url(args.url)]
    elif args.board:
        entries = [(args.tag or "custom", args.turn, args.board)]
    else:
        entries = read_source(Path(args.source))

    print(f"{len(entries)} puzzle(s)\n")
    kept = []
    for date, turn, board_string in entries:
        board, flipped = normalize(board_string, turn)
        tag = args.tag if (args.tag and len(entries) == 1) else f"{args.prefix}{date}"
        empties = empty_count(board)
        if empties > args.max_empties:
            print(f"{tag:<12} empties={empties:>2}  skipped (over --max-empties)")
            continue
        if not has_move(board, BLACK):
            print(f"{tag:<12} skipped: Black has no legal move")
            continue

        t = time.process_time()
        m = generate_puzzles.metrics(board, {})
        seconds = time.process_time() - t
        print(report_line(tag, m, seconds))

        if args.require_win and m["optimal_diff"] <= 0:
            continue
        kept.append((tag, board, m, {"site": "othelloclub.com", "date": date,
                                     "original_turn": "black" if turn == 1 else "white",
                                     "flipped": flipped, "board_string": board_string}))

    if not args.write:
        print(f"\n{len(kept)} puzzle(s) kept — rerun with --write to emit files")
        return

    print()
    for tag, board, m, source in kept:
        write_puzzle(board, tag, m, source)
        print(f"wrote puzzle_config_{tag}.txt + solution_{tag}.json")
    if kept:
        print("\n" + board_to_text(kept[0][1], BLACK))


if __name__ == "__main__":
    main()
