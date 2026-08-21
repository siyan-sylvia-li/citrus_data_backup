"""Prompt an LLM with a study puzzle and score its answer exactly.

The study's puzzles come with exhaustively-computed ground truth, which makes
them unusually clean as a model eval: there is a unique best move, every
alternative has an exact cost in discs, and "how wrong was it" is a number
rather than a judgement.

Two pieces, so you can drive the model however you like (dspy, the raw SDK,
whatever) and still grade the same way:

    prompt_for(puzzle)         -> the prompt text
    score_answer(puzzle, text) -> what the model said and what it cost

Scoring reports disc loss, not just right/wrong. A model that plays the
second-best move in oc20260727 gives up 32 discs and loses a won game; one that
plays it in bg20260726 gives up 8 and still might win. Binary correctness hides
that difference.

Run standalone to see a prompt and the grading of a hypothetical answer:

    python llm_eval.py oc20260727                  # print the prompt
    python llm_eval.py oc20260727 --answer "h4"    # grade an answer
    python llm_eval.py oc20260727 --flips          # easier variant (see engine)
    python llm_eval.py --all --answer b7           # every study puzzle at once
"""

from __future__ import annotations

import argparse
import concurrent.futures
import threading
from pathlib import Path

import engine as E
import solver as S

HERE = Path(__file__).parent
# The puzzles the study actually uses, in the order participants meet them.
STUDY_PUZZLES = ["oc20260727", "b220260706", "bg20260726"]


def load(puzzle: str):
    """Accept a puzzle id, a path to a config, or a raw 64-char board string."""
    if len(puzzle) >= 64:
        board, to_move = E.parse_board(puzzle)
        return board, to_move
    path = Path(puzzle)
    if not path.exists():
        path = HERE / f"puzzle_config_{puzzle}.txt"
    return E.load_puzzle(path)


def answer_text(response) -> str:
    """Pull the model's text out of whatever the SDK handed back.

    dspy returns a list (of strings, or of dicts with "text" plus separate
    reasoning); other clients return objects or plain strings. Stringifying the
    whole response would let a coordinate mentioned in hidden reasoning or in
    metadata be read as the answer, so unwrap deliberately.
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


def solution_for(puzzle: str) -> dict:
    """The precomputed optimal subtree, if this puzzle has one.

    Mirrors connect_four/llm_eval.solution_for. The study's assistant consults
    this table to know the optimal move before answering, so anything simulating
    that assistant needs it too.
    """
    import json
    try:
        return json.loads((HERE / f"solution_{puzzle}.json").read_text())
    except FileNotFoundError:
        return {}


def prompt_for(puzzle: str, **kwargs) -> str:
    board, to_move = load(puzzle)
    return E.board_to_prompt(board, to_move, **kwargs)


def score_answer(puzzle: str, text: str) -> dict:
    """Grade free-form model output against the exact solution.

    `disc_loss` is the headline: how many discs worse than optimal the model's
    move is, under best play by both sides afterwards. 0 means it found the
    answer. `illegal` and `unparsed` are separate failure modes and worth
    reporting separately — they mean different things about a model.
    """
    board, to_move = load(puzzle)
    memo: dict = {}
    values = S.move_values(board, to_move, memo)
    best_value = (max if to_move == E.BLACK else min)(values.values())
    best = [E.to_notation(m) for m in S.best_moves(board, to_move, memo)]

    move = E.parse_llm_move(text, board, to_move)
    if move is None:
        return {"puzzle": puzzle, "unparsed": True, "illegal": False,
                "answer": None, "best_moves": best, "optimal": False,
                "disc_loss": None, "optimal_margin": best_value}
    played = E.to_notation(move)
    if move not in values:
        return {"puzzle": puzzle, "unparsed": False, "illegal": True,
                "answer": played, "best_moves": best, "optimal": False,
                "disc_loss": None, "optimal_margin": best_value}
    return {
        "puzzle": puzzle, "unparsed": False, "illegal": False,
        "answer": played, "best_moves": best,
        "optimal": played in best,
        "disc_loss": abs(best_value - values[move]),
        "move_value": values[move],
        "optimal_margin": best_value,
        # Chance level for this puzzle, so a score has something to beat.
        "chance_optimal": len(best) / len(values),
    }


class PuzzleSession:
    """Play a puzzle move by move, exactly as a participant does.

    White defends optimally (the same exact search the study's AI uses), a
    position where you have only one legal square is auto-played rather than
    asked for, and passes are handled — so a model is scored on the same
    decisions, against the same opponent, as a human participant.

        s = PuzzleSession("oc20260727")
        while not s.done:
            r = s.play(model(s.prompt()))
            print(r["board_text"])
        print(s.summary())
    """

    def __init__(self, puzzle: str, *, messages: bool = False,
                 system: str | None = None, **prompt_kwargs):
        """`messages=True` keeps the exchange as a chat instead of re-prompting
        from scratch each move. The position is Markov, so history carries no
        information the board lacks — what it adds is continuity of reasoning
        (and the risk of anchoring on an earlier mistake). It is the condition
        that matches how the study's assistant is actually deployed; stateless is
        the cleaner per-decision measurement. Run both and compare.

        In chat mode `self.messages` always ends with the turn the model should
        answer next, so the loop is:  s.play(model(s.messages))
        """
        self.puzzle = puzzle
        self.board, self.to_move = load(puzzle)
        self.memo: dict = {}
        self.prompt_kwargs = prompt_kwargs
        self.optimal_margin = S.solve(self.board, self.to_move, self.memo)
        self.decisions: list[dict] = []
        self.auto_moves: list[str] = []
        self.use_messages = messages
        self.messages: list[dict] = []
        # A move that is never accepted does not advance the board, so a caller
        # looping on `while not s.done` would spin forever against a model that
        # keeps answering illegally. Fail loudly instead.
        self.failures = 0
        self.max_failures = 8
        self._autoplay()                      # a puzzle could open on a forced move
        if messages:
            if system:
                self.messages.append({"role": "system", "content": system})
            self.messages.append({"role": "user", "content": self.prompt()})

    # ---- state ----------------------------------------------------------
    @property
    def done(self) -> bool:
        return E.is_game_over(self.board)

    def prompt(self, **overrides) -> str:
        return E.board_to_prompt(self.board, self.to_move,
                                 **{**self.prompt_kwargs, **overrides})

    def board_text(self) -> str:
        return E.board_to_text(self.board, self.to_move)

    def _update_text(self, played, flipped, replies, forced, you_passed, opp_passed) -> str:
        """The next chat turn: only what changed since the model last looked.

        Restating the rules every turn would waste the context and, worse, make
        the history redundant — the point of chat mode is that the model can
        carry its plan forward, not that it re-reads the primer six times.
        """
        opp = "White" if self.to_move == E.BLACK else "Black"
        lines = [f"You played {played}, flipping {len(flipped)} disc(s)"
                 + (f" ({', '.join(flipped)})." if flipped else ".")]
        for mv, turned in replies:
            lines.append(f"{opp} replied {mv}, flipping {len(turned)} disc(s)"
                         + (f" ({', '.join(turned)})." if turned else "."))
        if opp_passed:
            lines.append(f"{opp} had no legal move and passed.")
        if you_passed:
            lines.append("You had no legal move, so your turn was skipped.")
        if forced:
            lines.append(f"Only one square was legal for you, so {', '.join(forced)} "
                         "was played automatically.")
        black, white = E.disc_counts(self.board)
        lines += ["", "    " + " ".join(chr(ord("a") + c) for c in range(E.COLS))]
        for r in range(E.ROWS):
            lines.append(f"  {r + 1} " + " ".join(
                {E.EMPTY: ".", E.BLACK: "B", E.WHITE: "W"}[self.board[r][c]]
                for c in range(E.COLS)))
        lines += ["", f"Black {black}, White {white}. "
                      f"{E.empty_count(self.board)} square(s) still empty."]
        if self.done:
            margin = black - white if self.to_move == E.BLACK else white - black
            lines.append("The game is over — you "
                         + ("won." if margin > 0 else "drew." if margin == 0 else "lost."))
        else:
            lines.append("Your legal moves: " + ", ".join(
                E.to_notation(m) for m in E.legal_moves(self.board, self.to_move)))
            lines.append("Answer with a single square.")
        return "\n".join(lines)

    # ---- playing --------------------------------------------------------
    def play(self, answer) -> dict:
        """Play the model's answer (free text or a move) and advance the game.

        An illegal or unreadable answer does NOT advance the board — it comes
        back with `illegal`/`unparsed` set so the caller can re-ask. That is a
        deliberate choice: silently substituting a legal move would score the
        harness rather than the model.
        """
        if self.done:
            raise RuntimeError("puzzle is already finished")
        if self.failures >= self.max_failures:
            raise RuntimeError(
                f"{self.failures} answers in a row were illegal or unreadable; "
                "giving up on this puzzle")
        if isinstance(answer, tuple):
            move = answer
        else:
            text = answer_text(answer)
            if self.use_messages:
                self.messages.append({"role": "assistant", "content": text})
            move = E.parse_llm_move(text, self.board, self.to_move)
        legal_now = [E.to_notation(m) for m in E.legal_moves(self.board, self.to_move)]
        if move is None or not E.is_valid_move(self.board, move, self.to_move):
            problem = ("I could not find a square in that answer."
                       if move is None else
                       f"{E.to_notation(move)} is not a legal move here.")
            if self.use_messages:
                # The correction goes into the history, which is the realistic
                # condition — a deployed assistant would see its own bad turn.
                self.messages.append({"role": "user", "content":
                                      f"{problem} Legal moves: {', '.join(legal_now)}. "
                                      "Answer with a single legal square."})
            self.failures += 1
            return {"ok": False, "unparsed": move is None, "illegal": move is not None,
                    "answer": None if move is None else E.to_notation(move),
                    "legal_moves": legal_now, "board_text": self.board_text()}

        # Score against the exact solution BEFORE playing, like the study does.
        values = S.move_values(self.board, self.to_move, self.memo)
        best = (max if self.to_move == E.BLACK else min)(values.values())
        played = E.to_notation(move)
        record = {
            "move": played,
            "optimal": played in [E.to_notation(m) for m in
                                  S.best_moves(self.board, self.to_move, self.memo)],
            "disc_loss": abs(best - values[move]),
            "best_moves": [E.to_notation(m) for m in
                           S.best_moves(self.board, self.to_move, self.memo)],
        }
        self.failures = 0
        flipped = [E.to_notation(f) for f in E.apply_move(self.board, move, self.to_move)]
        self.decisions.append(record)

        replies, black_passed, white_passed = self._reply()
        forced = self._autoplay()
        black, white_n = E.disc_counts(self.board)
        if self.use_messages:
            self.messages.append({"role": "user", "content": self._update_text(
                played, flipped, replies, forced, black_passed, white_passed)})
        return {
            "ok": True, "unparsed": False, "illegal": False,
            **record,
            "flipped": flipped,
            "opponent_moves": [mv for mv, _ in replies],   # >1 iff you had to pass
            "auto_played": forced,            # squares played for you (only one legal)
            "you_passed": black_passed, "opponent_passed": white_passed,
            "black_discs": black, "white_discs": white_n,
            "empties": E.empty_count(self.board),
            "done": self.done,
            "board_text": self.board_text(),
        }

    def _reply(self) -> tuple[list[tuple[str, list[str]]], bool, bool]:
        """Opponent's reply, repeated while you have no legal move.

        Returns [(square, discs it flipped), ...] so chat mode can narrate it.
        """
        opp = E.opponent(self.to_move)
        moves, you_passed, opp_passed = [], False, False
        while not E.is_game_over(self.board):
            if E.has_move(self.board, opp):
                mv = S.best_defense(self.board, self.memo) if opp == E.WHITE \
                    else S.best_moves(self.board, opp, self.memo)[0]
                turned = [E.to_notation(f) for f in E.apply_move(self.board, mv, opp)]
                moves.append((E.to_notation(mv), turned))
            else:
                opp_passed = True
            if E.has_move(self.board, self.to_move):
                break
            you_passed = True
        return moves, you_passed, opp_passed

    def _autoplay(self) -> list[str]:
        """Play out positions with a single legal square — not decisions."""
        auto = []
        while not self.done:
            moves = E.legal_moves(self.board, self.to_move)
            if len(moves) != 1:
                break
            E.apply_move(self.board, moves[0], self.to_move)
            auto.append(E.to_notation(moves[0]))
            self._reply()
        self.auto_moves += auto
        return auto

    # ---- result ---------------------------------------------------------
    def summary(self) -> dict:
        """The same measures the study reports for a participant."""
        black, white = E.disc_counts(self.board)
        margin = black - white if self.to_move == E.BLACK else white - black
        losses = sum(d["disc_loss"] for d in self.decisions)
        return {
            "puzzle": self.puzzle,
            "decisions": len(self.decisions),
            "optimal_moves": sum(d["optimal"] for d in self.decisions),
            "opener_optimal": self.decisions[0]["optimal"] if self.decisions else None,
            # The headline: discs short of perfect play. Equals the sum of the
            # per-move errors, so the outcome and the process agree by construction.
            "discs_lost": self.optimal_margin - margin if self.done else losses,
            "final_margin": margin if self.done else None,
            "won": margin > 0 if self.done else None,
            "optimal_margin": self.optimal_margin,
            "completed": self.done,
            "moves": [d["move"] + ("" if d["optimal"] else "!") for d in self.decisions],
            "auto_played": self.auto_moves,
            "chat_turns": len(self.messages) if self.use_messages else None,
        }


def play_puzzle(puzzle: str, ask, *, max_retries: int = 2, verbose: bool = False,
                **prompt_kwargs) -> dict:
    """Run a whole puzzle against a model. `ask(prompt) -> str`.

    Illegal or unreadable answers are re-asked up to `max_retries` times with the
    problem stated, then the puzzle is abandoned — recorded as `aborted`, not
    quietly scored, since a model that cannot produce a legal move has failed in
    a different way from one that plays badly.
    """
    s = PuzzleSession(puzzle, **prompt_kwargs)
    retries = 0
    while not s.done:
        # Chat mode: the pending turn (and any correction) is already in
        # s.messages. Stateless: rebuild the prompt, appending the problem.
        payload = s.messages if s.use_messages else s.prompt()
        result = None
        for attempt in range(max_retries + 1):
            result = s.play(ask(payload))
            if result["ok"]:
                break
            retries += 1
            if s.use_messages:
                payload = s.messages
            else:
                problem = ("I could not find a square in that answer."
                           if result["unparsed"] else
                           f"{result['answer']} is not a legal move. Legal moves: "
                           + ", ".join(result["legal_moves"]))
                payload = s.prompt() + f"\n\n{problem} Answer with one legal square."
        if not result["ok"]:
            out = s.summary()
            out.update(aborted="illegal_move", retries=retries)
            return out
        if verbose:
            print(f"  {result['move']}{'' if result['optimal'] else ' (-%d)' % result['disc_loss']}"
                  f" -> {result['black_discs']}-{result['white_discs']}")
    out = s.summary()
    out.update(aborted=None, retries=retries)
    return out


def evaluate_models(models, puzzles=None, *, ask=None, repeats: int = 1,
                    max_workers: int = 8, messages: bool = True,
                    max_retries: int = 2, progress: bool = True,
                    **prompt_kwargs) -> list[dict]:
    """Run several models over several puzzles concurrently. One row per run.

    The calls are network-bound, so threads are the right tool — but only ACROSS
    runs. Within a puzzle the moves are strictly sequential (each depends on the
    board the last one produced), so a single run is never parallelised.

    models
        {"name": callable(payload) -> response}, or a list of names together
        with `ask=callable(name, payload) -> response`. `payload` is a message
        list when messages=True, otherwise a prompt string.
    repeats
        Runs per (model, puzzle). Only meaningful above temperature 0.

    A model that raises — rate limit, context overflow, provider error — is
    recorded with its exception and does NOT take the batch down with it. With a
    long model list that is the difference between a partial result and none.
    """
    puzzles = list(puzzles or STUDY_PUZZLES)
    if isinstance(models, dict):
        callables = dict(models)
    else:
        if ask is None:
            raise ValueError("pass ask=callable(model_name, payload) with a list of names")
        callables = {name: (lambda payload, _n=name: ask(_n, payload)) for name in models}

    jobs = [(name, puzzle, rep)
            for name in callables for puzzle in puzzles for rep in range(repeats)]
    lock = threading.Lock()
    done_count = 0

    def run(job):
        name, puzzle, rep = job
        row = {"model": name, "puzzle": puzzle, "repeat": rep, "error": None}
        try:
            row.update(play_puzzle(puzzle, callables[name], max_retries=max_retries,
                                   messages=messages, **prompt_kwargs))
        except Exception as exc:                      # noqa: BLE001 - report, don't crash
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
                              f"{row.get('discs_lost')} discs lost")
                    print(f"[{done_count}/{len(jobs)}] {row['model']} "
                          f"{row['puzzle']}: {status}", flush=True)
    # Deterministic order regardless of completion order.
    failed = [r for r in rows if r["error"]]
    if progress and failed:
        print(f"\n{len(failed)} of {len(jobs)} runs failed:")
        for r in failed:
            print(f"  {r['model']} ({r['puzzle']}): {r['error']}")
    order = {name: i for i, name in enumerate(callables)}
    rows.sort(key=lambda r: (order[r["model"]], puzzles.index(r["puzzle"]), r["repeat"]))
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    """Collapse per-run rows to one row per model, for ranking or a Pareto plot.

    `discs_lost` is the measure to rank on: it is on the same scale as the human
    participants' score, lower is better, and it averages sensibly across
    puzzles of different sizes in a way win/lose does not.
    """
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    out = []
    for name, runs in by_model.items():
        ok = [r for r in runs if not r["error"] and not r.get("aborted")]
        out.append({
            "model": name,
            "runs": len(runs),
            "failed": sum(1 for r in runs if r["error"]),
            "aborted": sum(1 for r in runs if r.get("aborted")),
            "discs_lost": (sum(r["discs_lost"] for r in ok) / len(ok)) if ok else None,
            "wins": sum(1 for r in ok if r.get("won")),
            "opener_optimal": sum(1 for r in ok if r.get("opener_optimal")),
            "optimal_moves": sum(r["optimal_moves"] for r in ok),
            "decisions": sum(r["decisions"] for r in ok),
            "retries": sum(r.get("retries", 0) for r in runs),
        })
    out.sort(key=lambda r: (r["discs_lost"] is None, r["discs_lost"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("puzzle", nargs="?", default="oc20260727",
                    help="puzzle id, config path, or a 64-char board string")
    ap.add_argument("--all", action="store_true", help="every puzzle the study uses")
    ap.add_argument("--answer", help="model output to grade instead of printing a prompt")
    ap.add_argument("--flips", action="store_true", help="show what each legal move flips")
    ap.add_argument("--no-legal", action="store_true", help="withhold the legal-move list")
    ap.add_argument("--annotate", action="store_true", help="label corners / X- / C-squares")
    ap.add_argument("--json", action="store_true", help="structured representation instead")
    args = ap.parse_args()

    puzzles = STUDY_PUZZLES if args.all else [args.puzzle]
    for puzzle in puzzles:
        if args.answer:
            r = score_answer(puzzle, args.answer)
            verdict = ("could not parse a square" if r["unparsed"] else
                       f"{r['answer']} is ILLEGAL" if r["illegal"] else
                       f"{r['answer']} is optimal" if r["optimal"] else
                       f"{r['answer']} costs {r['disc_loss']} discs")
            print(f"{puzzle}: {verdict}  (best: {', '.join(r['best_moves'])})")
        elif args.json:
            import json
            board, to_move = load(puzzle)
            print(json.dumps(E.board_to_json(board, to_move), indent=2))
        else:
            if len(puzzles) > 1:
                print(f"\n{'=' * 70}\n{puzzle}\n{'=' * 70}")
            print(prompt_for(puzzle, legal=not args.no_legal, flips=args.flips,
                             annotate=args.annotate))


if __name__ == "__main__":
    main()
