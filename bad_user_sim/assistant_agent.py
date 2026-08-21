"""The in-interface AI assistant, wired the way the real studies wire it.

Fidelity matters: if this assistant is weaker or blinder than the deployed one,
every conclusion the simulation reaches about AI-assisted play describes a
different assistant. Mirrored from `/api/chat` in each study's app.py, point for
point, for BOTH games:

  1. system prompt read from that study's own games/<game>/system_prompt.txt —
     the file itself, not a copy
  2. persisted history is user/assistant TEXT only, exactly what the study
     rebuilds from conversation.jsonl
  3. per turn, appended fresh and NOT persisted:
       a. the current board as a PNG, captioned the way that study captions it
       b. a COACHING AID system message naming the optimal move(s) — from the
          precomputed table while on the optimal line, from the engine once off
  4. reasoning_effort per study (Othello medium, Connect Four low)
  5. if the image cannot be rendered, fall back to the text board

(3) is both the easiest part to omit and the one that matters most: the study's
assistant is never guessing at the position. It is told the answer and
instructed not to hand it over unless asked. Without the coaching aid this is a
substantially worse coach, and the simulated participant would look
correspondingly worse for reasons that have nothing to do with the participant.

The captions and coaching aids are copied verbatim from each app.py, which
remains the source of truth. `check_matches_study()` re-reads them and reports
drift; run it after editing study copy.
"""

from __future__ import annotations

import base64
import importlib.util
import sys
from pathlib import Path

import dspy
import dotenv

from constants import game_config, ASSISTANT_MODES

dotenv.load_dotenv()


# ---------------------------------------------------------------------------
# Loading a game's modules
# ---------------------------------------------------------------------------
# Both games ship modules called `engine` and `llm_eval`, and each imports its
# siblings by bare name. A plain `sys.path.insert` therefore lets you load one
# game per process and silently gives you the WRONG engine for the second. This
# loads each game under its own namespace, binding the bare names only while
# that game's modules execute, so both can be live at once.
_LOADED: dict[str, dict] = {}


def load_game_modules(game: str) -> dict:
    cfg = game_config(game)
    key = cfg["key"]
    if key in _LOADED:
        return _LOADED[key]

    mods: dict[str, object] = {}
    siblings = ("engine", "solver", "precompute", "llm_eval")   # dependency order
    for name in siblings:
        path = cfg["game_dir"] / f"{name}.py"
        if not path.exists():                 # othello has no precompute dep, etc.
            continue
        spec = importlib.util.spec_from_file_location(f"{key}__{name}", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"{key}__{name}"] = mod
        saved = {n: sys.modules.get(n) for n in siblings}
        try:
            for n, m in mods.items():         # `import engine` -> THIS game's engine
                sys.modules[n] = m
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
        finally:
            for n, old in saved.items():
                if old is None:
                    sys.modules.pop(n, None)
                else:
                    sys.modules[n] = old
        mods[name] = mod

    _LOADED[key] = {"cfg": cfg, **mods}
    return _LOADED[key]


# ---------------------------------------------------------------------------
# Per-game wiring: caption + coaching aid, verbatim from each study's app.py
# ---------------------------------------------------------------------------
class _OthelloWiring:
    """game_user_study_phase_othello/app.py, /api/chat."""

    def __init__(self, mods):
        self.E, self.S = mods["engine"], mods["solver"]

    def caption(self, board, **_) -> str:
        E = self.E
        legal = ", ".join(E.to_notation(m) for m in E.legal_moves(board, E.BLACK))
        return ("Current Othello board (you help Black; White is the AI opponent). "
                "Columns are a-h left to right, rows 1-8 top to bottom; the small "
                "pale dots mark Black's legal moves. The game is played to the end; "
                f"{E.empty_count(board)} square(s) are still empty. Black may play: {legal}.")

    def render(self, board) -> bytes:
        return self.E.render_image(board, legal_for=self.E.BLACK)

    def board_text(self, board) -> str:
        return self.E.board_to_text(board, self.E.BLACK)

    def coaching_aid(self, board, solution, memo) -> str | None:
        E, S = self.E, self.S
        key = E.state_key(board, E.BLACK)
        entry = solution.get(key) or {}
        best = entry.get("best_moves") or [
            E.to_notation(m) for m in S.best_moves(board, E.BLACK, memo)]
        if not best:
            return None
        best_str = ", ".join(best)
        lead = ("COACHING AID — the move(s) that keep Black's best result from the current "
                f"position are {best_str}."
                if key in solution else
                "COACHING AID — the participant has left the optimal line, but the strongest "
                f"available move(s) here are {best_str}.")
        return (f"{lead} Make sure you never steer the participant incorrectly.\n"
                "When it helps, reference specific squares by their coordinates (columns a-h "
                "left to right, rows 1-8 top to bottom) to point out a corner, an edge, or a "
                "disc that would be flipped back. Keep it brief (a couple of sentences or a "
                "short list).\n"
                "Do NOT tell them they made a mistake. Reason carefully about the board image "
                "and think step by step. DO NOT PROVIDE THE ANSWER UNLESS EXPLICITLY ASKED TO "
                "DO SO.")

    probes = {
        "caption": "Current Othello board (you help Black; White is the AI opponent). ",
        "coaching-aid lead": "COACHING AID — the move(s) that keep Black's best result",
        "off-line lead": "COACHING AID — the participant has left the optimal line",
        "no-answer rule": "DO NOT PROVIDE THE ANSWER UNLESS EXPLICITLY ASKED TO ",
    }


class _ConnectFourWiring:
    """game_user_study_phase_1/app.py, /api/chat."""

    def __init__(self, mods):
        self.E, self.L = mods["engine"], mods["llm_eval"]

    def caption(self, board, moves_left=None, **_) -> str:
        # The study quotes a remaining-move count because participants got a
        # fixed quota of 3. The sim plays to the end, so if no quota is supplied
        # we say how many columns are still playable instead of inventing one.
        if moves_left is None:
            return ("Current Connect Four board (you help Red \"R\"; Yellow \"Y\" is the "
                    "AI opponent). The participant is playing this puzzle to the end:")
        return ("Current Connect Four board (you help Red \"R\"; Yellow \"Y\" is the "
                f"AI opponent). The participant has {moves_left} move(s) left:")

    def render(self, board) -> bytes:
        return self.E.render_image(board)

    def board_text(self, board) -> str:
        return self.E.board_to_text(board)

    def coaching_aid(self, board, solution, memo) -> str | None:
        E = self.E
        cols = (solution.get(E.state_key(board, E.RED), {}) or {}).get("winning_columns", [])
        if cols:
            cols_1 = ", ".join(str(c + 1) for c in cols)
            return ("COACHING AID — the move(s) "
                    f"that keep Red's forced win from the current position are column {cols_1}. "
                    "Make sure you never steer the participant incorrectly.\n"
                    "When it helps, reference specific squares by "
                    "column (1-7) and row to point out a relevant run, threat, or diagonal. "
                    "Keep it brief (a couple of sentences or a short "
                    "list).\n"
                    "Reason carefully about the board image and think step by step. DO NOT "
                    "PROVIDE THE ANSWER UNLESS EXPLICITLYL ASKED TO DO SO.")
        best_cols = self.L._red_engine_best(board)
        if not best_cols:
            return None
        best_1 = ", ".join(str(c + 1) for c in best_cols)
        return ("COACHING AID — the participant is no longer on a forced-win line, so there is no "
                f"guaranteed win, but the strongest available move(s) here are column {best_1}. "
                "Help them keep playing well: point out the key threat or block, reference specific "
                "squares by column (1-7) and row, and keep it brief.\n"
                "Do NOT tell them they made a mistake or that the win is lost. Reason carefully about "
                "the board image and think step by step. DO NOT PROVIDE THE ANSWER UNLESS EXPLICITLY ASKED TO DO SO.")

    probes = {
        "caption": 'Current Connect Four board (you help Red \\"R\\"; Yellow \\"Y\\" is the ',
        "coaching-aid lead": "that keep Red's forced win from the current position are column ",
        "off-line lead": "COACHING AID — the participant is no longer on a forced-win line",
        "no-answer rule": "DO NOT PROVIDE THE ANSWER UNLESS EXPLICITLY ASKED TO DO SO.",
    }


_WIRING = {"othello": _OthelloWiring, "connect_four": _ConnectFourWiring}


class AssistantAgent:
    """The study's coach, over a live PuzzleSession, for either game.

    `puzzle` is the same PuzzleSession the simulated participant is playing, so
    the assistant always sees the CURRENT board; a snapshot would let the two
    drift apart mid-game.

    Use a different model than the student to avoid self-collusion.
    """

    def __init__(self, model_name, puzzle=None, *, game="othello", preset="default",
                 system_prompt_path=None, reasoning_effort=None, **lm_kwargs):
        """`lm_kwargs` go straight to dspy.LM — e.g. max_tokens for a coach model
        that reasons at length, which the studies never needed because they run a
        model whose default output budget is already generous."""
        mods = load_game_modules(game)
        self.cfg = mods["cfg"]
        self.game = self.cfg["key"]
        self.E = mods["engine"]
        self.L = mods["llm_eval"]
        self.wiring = _WIRING[self.game](mods)

        # reasoning_effort is an OpenAI-family parameter. litellm rejects the whole
        # request when a provider does not accept it -- together_ai raises
        # UnsupportedParamsError before a single token is sent, which killed every
        # run of a Together-hosted coach at its first call. Send it only where it is
        # understood, so swapping the coach model is possible at all.
        kw = dict(lm_kwargs)
        effort = reasoning_effort or self.cfg["reasoning_effort"]
        if effort and model_name.split("/", 1)[0] in ("openai", "azure", "anthropic"):
            kw["reasoning_effort"] = effort
        # A reasoning-heavy open model needs room: its chain of thought shares the
        # output budget with the reply, and a small cap returns an EMPTY reply rather
        # than an error (observed on gemma-4 and Kimi at 64 tokens).
        # 4096 was not enough: with the full system prompt, the board image and the
        # coaching aid, Kimi-K2.6 spent the whole budget on chain-of-thought and
        # returned an EMPTY reply on 25 consecutive turns. The same model answers
        # fine in isolation, so this is budget exhaustion, not extraction.
        kw.setdefault("max_tokens", 16384)
        self.lm = dspy.LM(model_name, **kw)
        prompt_path = Path(system_prompt_path or self.cfg["system_prompt"])
        self.system = prompt_path.read_text(encoding="utf-8").strip()
        self.puzzle = puzzle
        if preset not in ASSISTANT_MODES:
            raise ValueError(f"unknown assistant preset {preset!r}; "
                             f"expected one of {sorted(ASSISTANT_MODES)}")
        self.preset = preset
        self.solution = self.L.solution_for(puzzle.puzzle) if puzzle is not None else {}
        self.conversation_history = [{"role": "system", "content": self.system}]
        self.used_image = None          # per turn; False means the text fallback

    # ---- the per-turn context the study attaches -------------------------
    def _board_message(self, board, moves_left=None) -> dict:
        caption = self.wiring.caption(board, moves_left=moves_left)
        try:
            data_url = ("data:image/png;base64,"
                        + base64.b64encode(self.wiring.render(board)).decode())
            self.used_image = True
            return {"role": "user", "content": [
                {"type": "text", "text": caption},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]}
        except Exception:                       # Pillow missing / render error
            self.used_image = False
            return {"role": "system",
                    "content": caption + "\n" + self.wiring.board_text(board)}

    def _coaching_aid(self, board) -> str | None:
        memo = getattr(self.puzzle, "memo", {}) if self.puzzle is not None else {}
        return self.wiring.coaching_aid(board, self.solution, memo)

    def build_messages(self, question: str, moves_left=None) -> list[dict]:
        """Exactly what the study sends: history, question, board, coaching aid."""
        messages = self.conversation_history + [{"role": "user", "content": question}]
        board = self.puzzle.board if self.puzzle is not None else None
        if board is not None:
            messages = messages + [self._board_message(board, moves_left)]
            hint = self._coaching_aid(board)
            if hint:
                messages = messages + [{"role": "system", "content": hint}]

        # LAST, and outside the board branch. Last because recency is what makes a
        # style instruction stick when it overrides an earlier line in the system
        # prompt; outside because how to answer has nothing to do with whether a
        # board was attached, and nesting it there silently dropped the manipulation
        # on any turn without one.
        mode = ASSISTANT_MODES[self.preset]
        if mode:
            messages = messages + [{"role": "system", "content": mode}]
        return messages

    # ---- asking -----------------------------------------------------------
    def answer(self, question: str, moves_left=None) -> str:
        messages = self.build_messages(question, moves_left)
        board = self.puzzle.board if self.puzzle is not None else None
        try:
            raw = self.lm(messages=messages)
        except Exception as exc:
            # Sim-only concession the studies do not need: they always run a
            # vision model, whereas a simulation may point this at one that
            # cannot take images. Retry once with the text board so a
            # non-vision model degrades instead of dying.
            if board is None or self.used_image is not True:
                raise
            if not any(w in str(exc).lower() for w in ("image", "vision", "multimodal")):
                raise
            self.used_image = False
            messages = [m for m in messages if not isinstance(m.get("content"), list)]
            messages = messages + [{"role": "system", "content":
                                    self.wiring.caption(board, moves_left=moves_left)
                                    + "\n" + self.wiring.board_text(board)}]
            hint = self._coaching_aid(board)
            if hint:
                messages = messages + [{"role": "system", "content": hint}]
            raw = self.lm(messages=messages)

        reply = self.L.answer_text(raw)
        # Sim-only fallback. Several Together-hosted reasoning models return
        # {"text": "", "reasoning_content": "<the whole reply>"}; the study's
        # answer_text finds the empty "text" key and returns it, so the coach says
        # nothing at all -- Kimi-K2.6 produced 25 consecutive empty replies that way.
        # The study's caution about reading reasoning is right in general (a square
        # named in hidden chain-of-thought must not be mistaken for the answer), so
        # this only applies when there is no text to prefer, and the study file is
        # left untouched.
        if not reply.strip():
            blob = raw[0] if isinstance(raw, (list, tuple)) and raw else raw
            if isinstance(blob, dict):
                reply = str(blob.get("reasoning_content") or "").strip()
        # History carries the TEXT only. The board and the coaching aid are
        # re-attached fresh each turn, exactly as the study does when it
        # rebuilds the thread from conversation.jsonl.
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": reply})
        return reply


def check_matches_study(game=None) -> list[str]:
    """Report wording that has drifted from a study's app.py. [] means in sync.

    The captions and coaching aids are duplicated here because app.py cannot be
    imported without standing up Flask, an OpenAI client and the judge panel.
    Duplication rots silently, so this makes the rot visible.
    """
    games = [game] if game else list(_WIRING)
    problems = []
    for g in games:
        cfg = game_config(g)
        wiring = _WIRING[cfg["key"]]
        try:
            src = Path(cfg["app"]).read_text(encoding="utf-8")
        except FileNotFoundError:
            problems.append(f"{cfg['label']}: cannot read {cfg['app']}")
            continue
        for name, probe in wiring.probes.items():
            if probe not in src:
                problems.append(f"{cfg['label']} {name}: no longer in app.py — study copy changed")
        effort = f'reasoning_effort="{cfg["reasoning_effort"]}"'
        if effort not in src:
            problems.append(f"{cfg['label']} reasoning effort: {effort} not in app.py")
    return problems


if __name__ == "__main__":
    drift = check_matches_study()
    print("both games in sync with their study apps" if not drift
          else "DRIFT:\n  " + "\n  ".join(drift))
