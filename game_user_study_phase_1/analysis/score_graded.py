"""Write a graded (difficulty-aware) per-move score for each participant.

Companion to the scheme-1 selectivity rescoring in data_analysis_2group.ipynb.
Same contract: writes a NEW file per participant and never touches
cf_score_<round>.json. Skips anyone already scored unless --overwrite, so it is
cheap to re-run when new recordings land.

Output (one per round):
    cf_score_graded_p15.json
    cf_score_graded_pw4p6.json

The metric is mate-distance loss -- how many extra plies a move costs against
the fastest forced win available at that node. See cf_mate_solver for why this
beats both the flat 0-3 count and the selectivity rescoring, and for the two
special cases (no live win -> no blame; win thrown away -> censored at cap + 2).

Usage:
    python analysis/score_graded.py                          # ./recordings-download
    python analysis/score_graded.py <recordings_dir> [...]   # one or more dirs
    python analysis/score_graded.py --overwrite
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cf_mate_solver as S                                   # noqa: E402

HEX = re.compile(r"^[0-9a-f]{24}$")          # real prolific ids; drops test accounts
ROUNDS = ["p15", "pw4p6"]


def score_attempt(tag: str, moves: list[dict], config: Path, memo: dict) -> dict | None:
    """Replay one attempt's actual game and grade every Red move."""
    cap = S.PUZZLES[tag]["cap"]
    board = S.load_board(config)
    pos, mask = S.to_bitboard(board, S.RED)
    per_move, total, n_graded = [], 0.0, 0
    for m in sorted(moves, key=lambda r: r["move_number"]):
        col = m["col"]
        if not S.can_play(mask, col):                        # log inconsistent with replay
            return None
        g = S.grade_move(pos, mask, col, cap, memo)
        total += g["loss"]
        n_graded += bool(g["graded"])
        per_move.append({
            "move_number": m["move_number"], "col": col,
            "best_dist": g["best_dist"], "played_dist": g["played_dist"],
            "n_winning_cols": len(g["winning_cols"]), "winning_cols": g["winning_cols"],
            "kept_win": g["kept_win"], "graded": g["graded"], "loss": g["loss"],
        })
        pos, mask = S.play(pos, mask, col)                   # Red moves
        ai = m.get("ai_col")
        if ai is not None and S.can_play(mask, ai):
            pos, mask = S.play(pos, mask, ai)                # Yellow replies
    return {"graded_ply_loss": total, "n_graded_moves": n_graded,
            "mean_ply_loss": (total / n_graded if n_graded else 0.0),
            "per_move": per_move}


def score_participant(pdir: Path, tag: str, config: Path, memo: dict) -> dict | None:
    mf = pdir / f"moves_{tag}.jsonl"
    if not mf.exists():
        return None
    moves = [json.loads(l) for l in mf.read_text().splitlines() if l.strip()]
    if not moves:
        return None
    by_attempt: dict[int, list[dict]] = {}
    for m in moves:
        by_attempt.setdefault(m.get("attempt", 1), []).append(m)

    attempts = {}
    for a, mv in sorted(by_attempt.items()):
        s = score_attempt(tag, mv, config, memo)
        if s is not None:
            attempts[a] = s
    if 1 not in attempts:
        return None

    # Attempt 1 is primary: it is the cold, unaided-by-retry run, and it is what
    # the transfer analysis uses. Later attempts are recorded but not headlined.
    primary = attempts[1]
    flat = None
    sf = pdir / f"cf_score_{tag}.json"
    if sf.exists():
        try:
            flat = json.load(open(sf)).get("score")
        except Exception:
            pass
    return {
        "puzzle": S.PUZZLES[tag]["config"].replace("puzzle_config_", "").replace(".txt", ""),
        "round_tag": tag,
        "method": ("mate_distance: per-move loss = extra plies to a forced Red win vs the "
                   "best column at that node; no live win -> no blame; win thrown away -> "
                   "censored at cap + 2"),
        "nominal_plies_to_win": S.PUZZLES[tag]["nominal"],
        "search_cap_plies": S.PUZZLES[tag]["cap"],
        "graded_ply_loss": primary["graded_ply_loss"],
        "mean_ply_loss": primary["mean_ply_loss"],
        "n_graded_moves": primary["n_graded_moves"],
        "per_move": primary["per_move"],
        "attempts_scored": sorted(attempts),
        "attempt_ply_loss": {str(a): s["graded_ply_loss"] for a, s in sorted(attempts.items())},
        "flat_score": flat,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recordings", nargs="*", default=["recordings-download"],
                    help="one or more recordings-download directories")
    ap.add_argument("--overwrite", action="store_true", help="rescore participants already scored")
    ap.add_argument("--include-tester", action="store_true", help="keep non-24-hex ids")
    args = ap.parse_args()

    memo: dict[str, dict] = {t: {} for t in ROUNDS}
    for rec in args.recordings:
        root = Path(rec).resolve()
        if not root.is_dir():
            print(f"!! not a directory: {root}")
            continue
        configs = {t: S.find_config(t, root.parent) for t in ROUNDS}
        written = skipped = failed = 0
        for pdir in sorted(p for p in root.iterdir() if p.is_dir()):
            if not args.include_tester and not HEX.match(pdir.name):
                continue
            for tag in ROUNDS:
                out = pdir / f"cf_score_graded_{tag}.json"
                if out.exists() and not args.overwrite:
                    skipped += 1
                    continue
                res = score_participant(pdir, tag, configs[tag], memo[tag])
                if res is None:
                    failed += 1
                    continue
                out.write_text(json.dumps(res, indent=2))
                written += 1
        print(f"{root}: wrote {written}, skipped {skipped}, no-data {failed}")
    print(f"memo sizes: " + ", ".join(f"{t}={len(memo[t]):,}" for t in ROUNDS))


if __name__ == "__main__":
    main()
