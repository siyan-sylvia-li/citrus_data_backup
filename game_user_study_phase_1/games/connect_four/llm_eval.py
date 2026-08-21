"""Prompt an LLM with a study puzzle and score its answer. Connect Four edition.

Mirror of games/othello/llm_eval.py in the Othello phase, same API:

    prompt_for(puzzle)          -> the prompt text
    score_answer(puzzle, text)  -> what the model said and what it cost
    PuzzleSession(puzzle)       -> play it out move by move (messages=True for chat)
    play_puzzle(puzzle, ask)    -> run a whole puzzle against a model
    evaluate_models(...)        -> run several models concurrently

One real difference from Othello, worth knowing before you read any numbers.
Othello endgames are solved exactly to the last disc, so every move there has a
ground-truth cost in discs. Connect Four from a 12-piece board is not exhaustible
in pure Python, so ground truth here is exact only WITHIN the win horizon:
`precompute.search` (distance-aware alpha-beta, depth-capped at DEPTH_CAP=9)
decides forced wins exactly, and once a position is off any forced-win line the
"best" column is the depth-limited engine's opinion rather than proof. Runs
report `exact` per decision so you can filter to the provable ones.

Positions on a puzzle's precomputed line cost nothing to score: solution_<id>.json
already holds the winning columns and the distance to the win, so the search only
runs once a model has left that line.

The scoring vocabulary follows from that:

    optimal      played a column that achieves the best available result
    kept_win     the forced win survived this move (the study's stricter metric)
    plies_to_win how long the win now takes — a move can keep the win and still
                 delay it, which is the closest thing here to Othello's disc loss

Run standalone:

    python llm_eval.py 15                    # print the prompt
    python llm_eval.py 15 --answer "col 4"   # grade an answer
    python llm_eval.py 15 --landing          # easier variant (see engine)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import threading
from pathlib import Path

import engine as E
import precompute as P

HERE = Path(__file__).parent
# The puzzles the phase-1 study used, in the order participants met them.
STUDY_PUZZLES = ["15", "w4p6"]
FIRST_N = 3            # the study scored this many moves per puzzle


def load(puzzle: str) -> E.Board:
    """Accept a puzzle id or a path to a config file."""
    path = Path(puzzle)
    if not path.exists():
        path = HERE / f"puzzle_config_{puzzle}.txt"
    return E.load_board(path)


def solution_for(puzzle: str) -> dict:
    """The precomputed optimal subtree, if this puzzle has one."""
    import json
    path = HERE / f"solution_{puzzle}.json"
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def prompt_for(puzzle: str, **kwargs) -> str:
    return E.board_to_prompt(load(puzzle), E.RED, **kwargs)


def _red_engine_best(board: E.Board) -> list[int]:
    """Engine's best column(s) once no forced win remains — opinion, not proof.

    Same fallback the study's app uses to score moves off the winning line, so a
    model and a participant are judged against the same standard.
    """
    scored = []
    for c in E.valid_columns(board):
        child = E.copy_board(board)
        E.drop(child, c, E.RED)
        if E.winning_move(child, E.RED):
            scored.append((c, math.inf))
            continue
        _, sc = E.minimax(child, 5, -math.inf, math.inf, to_move=E.YELLOW, me=E.RED)
        scored.append((c, sc))
    if not scored:
        return []
    best = max(s for _, s in scored)
    return [c for c, s in scored if s == best]


# Depth-9 alpha-beta is expensive and the SAME positions recur constantly —
# every model starts from the puzzle root, and runs share long prefixes. Caching
# by state key turns a CPU-bound batch into a network-bound one, which is what
# makes the thread pool in evaluate_models worth anything (threads cannot
# parallelise the search itself; the GIL serialises it).
_ANALYSE_CACHE: dict[str, dict] = {}
_SEARCH_CACHE: dict[str, float] = {}


def search_after(board: E.Board) -> float:
    """Red's distance-aware score with Yellow to move, cached by position."""
    key = E.state_key(board, E.YELLOW)
    if key not in _SEARCH_CACHE:
        _SEARCH_CACHE[key] = P.search(board, P.DEPTH_CAP, -math.inf, math.inf,
                                      to_move=E.YELLOW, me=E.RED)
    return _SEARCH_CACHE[key]


def analyse(board: E.Board, solution: dict | None = None) -> dict:
    """Ground truth for Red in this position, as far as it can be established.

    Checks the puzzle's PRECOMPUTED table first: while play is on the optimal
    line — most of a good model's run — the winning columns and the distance to
    the win are already tabulated, so no search happens at all. The depth-9
    search only covers positions the table never reached, i.e. after a
    deviation. Same order of preference app.py uses to score participants.
    """
    key = E.state_key(board, E.RED)
    if solution:
        entry = solution.get(key)
        if entry and entry.get("winning_columns"):
            return {"best_columns": entry["winning_columns"], "exact": True,
                    "plies_to_win": entry.get("plies_to_win"), "from_table": True}
    if key in _ANALYSE_CACHE:
        return _ANALYSE_CACHE[key]
    result = _analyse_uncached(board)
    _ANALYSE_CACHE[key] = result
    return result


def _analyse_uncached(board: E.Board) -> dict:
    winning = P.red_winning_columns(board)           # exact within DEPTH_CAP
    if winning:
        score = P.search(board, P.DEPTH_CAP, -math.inf, math.inf,
                         to_move=E.RED, me=E.RED)
        return {"best_columns": winning, "exact": True,
                "plies_to_win": (P.WIN + P.DEPTH_CAP) - score, "from_table": False}
    return {"best_columns": _red_engine_best(board), "exact": False,
            "plies_to_win": None, "from_table": False}


def score_answer(puzzle: str, text: str) -> dict:
    """Grade free-form model output against the position's best column(s)."""
    board = load(puzzle)
    truth = analyse(board, solution_for(puzzle))
    best_1 = [c + 1 for c in truth["best_columns"]]

    col = E.parse_llm_move(text, board)
    if col is None:
        return {"puzzle": puzzle, "unparsed": True, "illegal": False,
                "answer": None, "best_columns": best_1, "optimal": False,
                "exact": truth["exact"]}
    if not E.is_valid_column(board, col):
        return {"puzzle": puzzle, "unparsed": False, "illegal": True,
                "answer": col + 1, "best_columns": best_1, "optimal": False,
                "exact": truth["exact"]}

    child = E.copy_board(board)
    E.drop(child, col, E.RED)
    # A tabulated winning column keeps the win by definition, so only a
    # deviation needs the confirming search — and only that costs anything.
    if col in truth["best_columns"] and truth["exact"]:
        after, kept = None, True
    else:
        after = search_after(child)
        kept = after >= P.WIN
    return {
        "puzzle": puzzle, "unparsed": False, "illegal": False,
        "answer": col + 1, "best_columns": best_1,
        "optimal": col in truth["best_columns"],
        "kept_win": kept,
        "plies_to_win_before": truth["plies_to_win"],
        "plies_to_win_after": (None if after is None else
                               (P.WIN + P.DEPTH_CAP) - after + 1
                               if after >= P.WIN else None),
        "exact": truth["exact"],
        # Chance level, so a score has something to beat.
        "chance_optimal": len(truth["best_columns"]) / len(E.valid_columns(board)),
    }


class PuzzleSession:
    """Play a puzzle move by move, exactly as a participant does.

    Yellow defends with the puzzle's precomputed best defense while the game is
    on that line, and with the engine once it leaves — the same opponent the
    study's participants face. A position with only one playable column is
    dropped automatically rather than asked for, since it is not a decision.

        s = PuzzleSession("15")
        while not s.done:
            r = s.play(model(s.prompt()))
        print(s.summary())
    """

    def __init__(self, puzzle: str, *, messages: bool = False,
                 system: str | None = None, **prompt_kwargs):
        """`messages=True` keeps the exchange as a chat instead of re-prompting
        from scratch each move. The position is Markov — history carries no
        information the board lacks — so this is a treatment, not a fix: it adds
        continuity of reasoning and the risk of anchoring on an earlier mistake.
        It matches how the study's assistant is deployed; stateless is the
        cleaner per-decision measurement.
        """
        self.puzzle = puzzle
        self.board = load(puzzle)
        self.solution = solution_for(puzzle)
        self.prompt_kwargs = prompt_kwargs
        self.start = analyse(self.board, self.solution)
        self.decisions: list[dict] = []
        self.auto_moves: list[int] = []
        self.use_messages = messages
        self.messages: list[dict] = []
        # A rejected answer does not advance the board, so a caller looping on
        # `while not s.done` would spin forever against a model that keeps
        # answering illegally. Fail loudly instead.
        self.failures = 0
        self.max_failures = 8
        self._autoplay()
        if messages:
            if system:
                self.messages.append({"role": "system", "content": system})
            self.messages.append({"role": "user", "content": self.prompt()})

    # ---- state ----------------------------------------------------------
    @property
    def done(self) -> bool:
        return (E.winning_move(self.board, E.RED)
                or E.winning_move(self.board, E.YELLOW)
                or E.is_full(self.board))

    def prompt(self, **overrides) -> str:
        return E.board_to_prompt(self.board, E.RED,
                                 **{**self.prompt_kwargs, **overrides})

    def board_text(self) -> str:
        return E.board_to_text(self.board)

    def _update_text(self, played, reply, forced) -> str:
        """The next chat turn: only what changed since the model last looked."""
        lines = [f"You dropped in column {played + 1}."]
        if reply is not None:
            lines.append(f"Yellow answered in column {reply + 1}.")
        for c in forced:
            lines.append(f"Only column {c + 1} was playable, so it was dropped for you.")
        lines += ["", E.grid_text(self.board), ""]
        if self.done:
            lines.append({"red_win": "You won — four in a row.",
                          "yellow_win": "Yellow got four in a row. You lost.",
                          "draw": "The board is full. It's a draw."}[E.status(self.board)])
        else:
            lines.append("Playable columns: " + ", ".join(
                str(c + 1) for c in E.valid_columns(self.board)))
            lines.append("Answer with a single column number, 1-7.")
        return "\n".join(lines)

    # ---- playing --------------------------------------------------------
    def play(self, answer) -> dict:
        """Play the model's answer and advance the game.

        An illegal or unreadable answer does NOT advance the board — it comes
        back with `illegal`/`unparsed` set so the caller can re-ask. Silently
        substituting a legal column would score the harness, not the model.
        """
        if self.done:
            raise RuntimeError("puzzle is already finished")
        if self.failures >= self.max_failures:
            raise RuntimeError(
                f"{self.failures} answers in a row were illegal or unreadable; "
                "giving up on this puzzle")

        if isinstance(answer, int):
            col = answer
        else:
            text = answer_text(answer)
            if self.use_messages:
                self.messages.append({"role": "assistant", "content": text})
            col = E.parse_llm_move(text, self.board)

        playable = [c + 1 for c in E.valid_columns(self.board)]
        if col is None or not E.is_valid_column(self.board, col):
            problem = ("I could not find a column number in that answer."
                       if col is None else f"Column {col + 1} is full.")
            if self.use_messages:
                self.messages.append({"role": "user", "content":
                                      f"{problem} Playable columns: "
                                      f"{', '.join(map(str, playable))}. "
                                      "Answer with a single column number."})
            self.failures += 1
            return {"ok": False, "unparsed": col is None, "illegal": col is not None,
                    "answer": None if col is None else col + 1,
                    "playable_columns": playable, "board_text": self.board_text()}

        # Score against the ground truth BEFORE dropping, like the study does.
        truth = analyse(self.board, self.solution)
        on_line = col in truth["best_columns"] and truth["exact"]
        E.drop(self.board, col, E.RED)
        kept = (True if on_line else
                bool(E.winning_move(self.board, E.RED)
                     or search_after(self.board) >= P.WIN))
        record = {
            "column": col + 1,
            "optimal": col in truth["best_columns"],
            "kept_win": kept,
            "best_columns": [c + 1 for c in truth["best_columns"]],
            "exact": truth["exact"],
            "plies_to_win_before": truth["plies_to_win"],
        }
        self.failures = 0
        self.decisions.append(record)

        reply = self._reply()
        forced = self._autoplay()
        if self.use_messages:
            self.messages.append({"role": "user",
                                  "content": self._update_text(col, reply, forced)})
        return {
            "ok": True, "unparsed": False, "illegal": False, **record,
            "opponent_column": None if reply is None else reply + 1,
            "auto_played": [c + 1 for c in forced],
            "status": E.status(self.board),
            "done": self.done,
            "board_text": self.board_text(),
        }

    def _reply(self) -> int | None:
        """Yellow's answer: the tabulated defense on the line, else the engine."""
        if self.done:
            return None
        col = self.solution.get(E.state_key(self.board, E.YELLOW), {}).get("best_defense")
        if col is None or not E.is_valid_column(self.board, col):
            col = E.ai_move(self.board)
        if col is None:
            return None
        E.drop(self.board, col, E.YELLOW)
        return col

    def _autoplay(self) -> list[int]:
        """Drop for the model when only one column is playable — not a decision."""
        auto = []
        while not self.done:
            cols = E.valid_columns(self.board)
            if len(cols) != 1:
                break
            E.drop(self.board, cols[0], E.RED)
            auto.append(cols[0])
            self._reply()
        self.auto_moves += auto
        return auto

    # ---- result ---------------------------------------------------------
    def summary(self) -> dict:
        opt = [d["optimal"] for d in self.decisions]
        return {
            "puzzle": self.puzzle,
            "decisions": len(self.decisions),
            "optimal_moves": sum(opt),
            "optimal_first_n": sum(opt[:FIRST_N]),
            "decisions_first_n": len(opt[:FIRST_N]),
            "kept_win_moves": sum(d["kept_win"] for d in self.decisions),
            "opener_optimal": opt[0] if opt else None,
            "exact_decisions": sum(d["exact"] for d in self.decisions),
            "result": E.status(self.board) if self.done else None,
            "won": E.winning_move(self.board, E.RED) if self.done else None,
            # The puzzle is a forced win in this many plies; did they take it?
            "plies_to_win_at_start": self.start["plies_to_win"],
            "completed": self.done,
            "columns": [str(d["column"]) + ("" if d["optimal"] else "!")
                        for d in self.decisions],
            "auto_played": [c + 1 for c in self.auto_moves],
            "chat_turns": len(self.messages) if self.use_messages else None,
        }


def answer_text(response) -> str:
    """Pull the model's text out of whatever the SDK handed back.

    dspy returns a list (of strings, or dicts with "text" plus separate
    reasoning); other clients return objects or strings. Stringifying the whole
    response would let a number in hidden reasoning or metadata be read as the
    answer, so unwrap deliberately.
    """
    if isinstance(response, (list, tuple)):
        if not response:
            return ""
        response = response[0]
    if isinstance(response, dict):
        for key in ("text", "content", "message", "output_text"):
            if key in response:
                return str(response[key])
        return str(response)
    for attr in ("text", "content", "output_text"):
        if hasattr(response, attr) and isinstance(getattr(response, attr), str):
            return getattr(response, attr)
    return str(response)


def play_puzzle(puzzle: str, ask, *, max_retries: int = 2, verbose: bool = False,
                **session_kwargs) -> dict:
    """Run a whole puzzle against a model. `ask(payload) -> response`.

    Illegal or unreadable answers are re-asked up to `max_retries` times with the
    problem stated, then the puzzle is abandoned and recorded as `aborted` — a
    model that cannot produce a legal column has failed differently from one that
    plays badly, and the two should not be pooled.
    """
    s = PuzzleSession(puzzle, **session_kwargs)
    retries = 0
    while not s.done:
        payload = s.messages if s.use_messages else s.prompt()
        result = None
        for _ in range(max_retries + 1):
            result = s.play(ask(payload))
            if result["ok"]:
                break
            retries += 1
            if s.use_messages:
                payload = s.messages
            else:
                problem = ("I could not find a column number in that answer."
                           if result["unparsed"] else
                           f"Column {result['answer']} is full. Playable: "
                           + ", ".join(map(str, result["playable_columns"])))
                payload = s.prompt() + f"\n\n{problem} Answer with one column number."
        if not result["ok"]:
            out = s.summary()
            out.update(aborted="illegal_move", retries=retries)
            return out
        if verbose:
            print(f"  col {result['column']}"
                  f"{'' if result['optimal'] else ' (not best)'} -> {result['status']}")
    out = s.summary()
    out.update(aborted=None, retries=retries)
    return out


def evaluate_models(models, puzzles=None, *, ask=None, repeats: int = 1,
                    max_workers: int = 8, messages: bool = True,
                    max_retries: int = 2, progress: bool = True,
                    **session_kwargs) -> list[dict]:
    """Run several models over several puzzles concurrently. One row per run.

    Threads, because the calls are network-bound — but only ACROSS runs. Within a
    puzzle the moves are strictly sequential, so a single run is never split.
    A model that raises is recorded with its exception and does not take the
    batch down with it.
    """
    puzzles = list(puzzles or STUDY_PUZZLES)
    if isinstance(models, dict):
        callables = dict(models)
    else:
        if ask is None:
            raise ValueError("pass ask=callable(model_name, payload) with a list of names")
        callables = {name: (lambda payload, _n=name: ask(_n, payload)) for name in models}

    jobs = [(n, p, r) for n in callables for p in puzzles for r in range(repeats)]
    lock, done_count = threading.Lock(), 0

    def run(job):
        name, puzzle, rep = job
        row = {"model": name, "puzzle": puzzle, "repeat": rep, "error": None}
        try:
            row.update(play_puzzle(puzzle, callables[name], max_retries=max_retries,
                                   messages=messages, **session_kwargs))
        except Exception as exc:                       # noqa: BLE001 - report, don't crash
            row["error"] = f"{type(exc).__name__}: {exc}"
        return row

    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            with lock:
                rows.append(row)
                done_count += 1
                if progress:
                    # Print the exception text, not just "ERROR" — with a long
                    # model list the failures are the informative part (a bad
                    # model name and a rate limit need different responses).
                    status = (f"ERROR {row['error']}" if row["error"]
                              else row.get("aborted") or
                              f"{row.get('optimal_moves')}/{row.get('decisions')} optimal, "
                              f"{row.get('result')}")
                    print(f"[{done_count}/{len(jobs)}] {row['model']} "
                          f"{row['puzzle']}: {status}", flush=True)
    failed = [r for r in rows if r["error"]]
    if progress and failed:
        print(f"\n{len(failed)} of {len(jobs)} runs failed:")
        for r in failed:
            print(f"  {r['model']} ({r['puzzle']}): {r['error']}")
    order = {n: i for i, n in enumerate(callables)}
    rows.sort(key=lambda r: (order[r["model"]], puzzles.index(r["puzzle"]), r["repeat"]))
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    """Collapse per-run rows to one row per model, for ranking or a Pareto plot."""
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    out = []
    for name, runs in by_model.items():
        ok = [r for r in runs if not r["error"] and not r.get("aborted")]
        decisions = sum(r["decisions"] for r in ok)
        out.append({
            "model": name,
            "runs": len(runs),
            "failed": sum(1 for r in runs if r["error"]),
            "aborted": sum(1 for r in runs if r.get("aborted")),
            "wins": sum(1 for r in ok if r.get("won")),
            "optimal_moves": sum(r["optimal_moves"] for r in ok),
            "decisions": decisions,
            "optimal_rate": (sum(r["optimal_moves"] for r in ok) / decisions
                             if decisions else None),
            "opener_optimal": sum(1 for r in ok if r.get("opener_optimal")),
            "kept_win_moves": sum(r["kept_win_moves"] for r in ok),
            "retries": sum(r.get("retries", 0) for r in runs),
        })
    out.sort(key=lambda r: (r["wins"] is None, -(r["wins"] or 0),
                            -(r["optimal_rate"] or 0)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("puzzle", nargs="?", default="15", help="puzzle id or config path")
    ap.add_argument("--all", action="store_true", help="every puzzle the study used")
    ap.add_argument("--answer", help="model output to grade instead of printing a prompt")
    ap.add_argument("--landing", action="store_true", help="show where each drop lands")
    ap.add_argument("--no-legal", action="store_true", help="withhold the playable columns")
    ap.add_argument("--json", action="store_true", help="structured representation instead")
    args = ap.parse_args()

    for puzzle in (STUDY_PUZZLES if args.all else [args.puzzle]):
        if args.answer:
            r = score_answer(puzzle, args.answer)
            verdict = ("could not parse a column" if r["unparsed"] else
                       f"column {r['answer']} is FULL" if r["illegal"] else
                       f"column {r['answer']} is optimal" if r["optimal"] else
                       f"column {r['answer']} is not best")
            extra = "" if r.get("kept_win") is None else \
                f", forced win {'kept' if r['kept_win'] else 'LOST'}"
            print(f"{puzzle}: {verdict}{extra}  "
                  f"(best: {', '.join(map(str, r['best_columns']))}"
                  f"{'' if r['exact'] else ', engine opinion only'})")
        elif args.json:
            import json
            print(json.dumps(E.board_to_json(load(puzzle), E.RED), indent=2))
        else:
            if args.all:
                print(f"\n{'=' * 70}\n{puzzle}\n{'=' * 70}")
            print(prompt_for(puzzle, legal=not args.no_legal, landing=args.landing))


if __name__ == "__main__":
    main()
