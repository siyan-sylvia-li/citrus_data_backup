"""Simulate a study participant playing the puzzle, with the study's assistant.

The participant is a pydantic-ai agent given a persona and two tools — play a
move, ask the assistant — over the SAME PuzzleSession the real study uses, with
the SAME opponent and the SAME coach (see assistant_agent.py). What the agent is
told mirrors what a participant sees on screen, and nothing more:

  - the board, the legal moves, the disc/piece counts
  - what its move flipped and how the opponent answered
  - NEVER whether a move was good. The study withholds that deliberately; a
    simulated participant that learns "that was optimal" mid-game is playing a
    different task, and its trace cannot be compared with a human's.

Set GAME below; everything else follows from it (module paths, puzzle, prompts).
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

import dotenv
import httpx
from pydantic_ai import Agent, RunContext, ModelRetry

try:                                  # UsageLimits moved to the top level in recent versions
    from pydantic_ai import UsageLimits
except ImportError:
    from pydantic_ai.usage import UsageLimits

from constants import (USER_SYSTEM_PROMPT, EDU_LABELS, OCC_LABELS, AIF_LABELS,
                       DEFAULT_QUESTION_BUDGET, STREAM_ONLY_MODELS, game_config,
                       bare_model_name, model_max_tokens, STUDENT_MODES,
                       ASSISTANT_MODES, ACT_GATES, GATE_MODEL,
                       GATE_MAX_STREAK)
from assistant_agent import AssistantAgent, load_game_modules
from argparse import ArgumentParser

dotenv.load_dotenv()

# ---- which game ---------------------------------------------------------------
# "Othello" or "Connect Four". Read from the environment because the three
# module-level lines below (game modules, tool instructions) resolve at IMPORT
# time — earlier than argparse runs — so a --game flag would have to be threaded
# through all of them. SIM_GAME="Connect Four" ./run_agent_sim.sh runs the CF sweep.
GAME = os.environ.get("SIM_GAME", "Othello")

# How the consult tool is FRAMED to the student. "default" is the original wording,
# kept exactly so every arm already collected stays comparable.
#
# "shared_context" fixes three things the original says that push the simulated
# student away from how people actually ask (measured against 621 human turns):
#   1. "ask when something genuinely confuses you" sets a high bar. Humans ask
#      "was c8 good?" -- checking, not confusion. The bar pushes toward large,
#      packaged questions.
#   2. "limited question budget" makes each turn scarce, so the rational move is to
#      maximise information per message. That is the board restatement (74-81% of
#      simulated turns name a square, vs 11% of human ones) and the 26-word median
#      against a human 7.
#   3. Nothing tells the student the assistant SEES the board and remembers the
#      thread, so it re-specifies state defensively, as if talking to a stateless
#      API rather than someone sitting with it.
# Under --questions_required the scarcity framing is not merely unhelpful but false:
# the student is required to use every consultation.
TOOL_FRAMING = os.environ.get("SIM_TOOL_FRAMING", "default")

CFG = game_config(GAME)
MODS = load_game_modules(GAME)        # this game's engine/solver/llm_eval, isolated
llm_eval = MODS["llm_eval"]
PuzzleSession = llm_eval.PuzzleSession

TOOL_INSTRUCTIONS = {
    "connect_four": """

    # Tools Available
    You are a study participant with about 8 minutes to finish a Connect Four end game puzzle. You are equipped with tools:
    - play_move(move): Drop a Red disc into a column, 1-7. It falls to the lowest empty row. Your goal is four red discs in a row (horizontal, vertical, or diagonal).
    - view_board(): Look at the board again. Free, and does not use a question.
    - consult_assistant(question): {CONSULT_LINE}

    Dropping a disc ONLY counts if you call play_move — writing a column number in your reply does nothing.

    Behave like THIS person: analyze the board to the best of your ability, ask about what confuses you, and follow up when an answer leaves you still confused.
    Play the puzzle until the end.""",
    "othello": """

    # Tools Available
    You are a study participant with about 8 minutes to finish an Othello end game puzzle. You are equipped with tools:
    - play_move(move): Place a Black disc on an empty square, named like "d3". Any discs you trap get flipped to your color. Your goal is to have the most discs when the board is finished.
    - view_board(): Look at the board again. Free, and does not use a question.
    - consult_assistant(question): {CONSULT_LINE}

    Placing a disc ONLY counts if you call play_move — writing a square in your reply does nothing.

    Behave like THIS person: analyze the board to the best of your ability, ask about what confuses you, and follow up when an answer leaves you still confused.
    Play the puzzle until the end.""",
}[CFG["key"]]

CONSULT_LINE = {
    "default": "ask when something genuinely confuses you (limited question budget).",
    "shared_context": (
        "talk to the assistant. It is looking at the SAME board you are and remembers "
        "everything already said, so you never need to restate the position, list your "
        "legal moves, or recap what happened — just say the thing you want to say. Use "
        "it whenever you would naturally: to check an idea, to get a second opinion, to "
        "say what you are seeing, or when you are stuck. It does not have to be a "
        "question."),
}[TOOL_FRAMING]
TOOL_INSTRUCTIONS = TOOL_INSTRUCTIONS.replace("{CONSULT_LINE}", CONSULT_LINE)


@dataclass
class GameDeps:
    """Per-run state. The Agent is created once; everything that varies per student
    lives here and is reached from tools via ctx.deps."""
    profile: dict
    puzzle: PuzzleSession
    assistant: AssistantAgent
    question_budget: int
    # Round 2 has no assistant. Without this the sim would hand the unaided
    # round a coach, and the transfer comparison the study exists to make
    # would be meaningless.
    assistant_enabled: bool = True
    # Which behaviour manipulation the student is under (constants.STUDENT_MODES).
    student_mode: str = "default"
    # Consultations the run must make before it may end (0 = the study's own
    # behaviour, ask as much or as little as you like up to the budget). Set equal to
    # the budget to fix the count: a manipulation that changes how questions are
    # PHRASED must not also change how MANY there are, or the outcome comparison
    # cannot tell the two apart -- which is exactly what happened to the `solution`
    # arm, where turns went 2.8 -> 5.2 in all 10 cells alongside the margin gain.
    questions_required: int = 0
    # Act gate name (constants.ACT_GATES) or None. Enforced in consult_assistant.
    gate: str | None = None
    gate_blocked: int = 0          # rejections, so the cost of the gate is recorded
    gate_errors: int = 0           # classifier failures, which FAIL OPEN (see below)
    gate_passthrough: int = 0      # let through at the streak cap -- leaked manipulation
    gate_streak: int = 0           # consecutive rejections on the current consultation
    questions_asked: int = 0
    trace: list = field(default_factory=list)   # ordered tool calls, for behaviour analysis
    moves: list = field(default_factory=list)   # one entry per scored decision
    # Serialises consult_assistant. pydantic-ai runs sync tools in a threadpool, so a
    # model that emits several consult_assistant calls in ONE response had them all
    # read questions_asked before any of them incremented it -- a check-then-act race
    # that let gpt-oss-120b ask 17 and 18 questions against a budget of 10. The same
    # window also let two answers interleave appends to the assistant's single
    # conversation_history. A real participant asks one question at a time.
    lock: threading.Lock = field(default_factory=threading.Lock)


# ---- the agent: created ONCE, no model bound (model is passed per run) ----------
# retries has to cover output-validator rejections too: a weak model may try to
# end the run with a text answer several times before it uses the tool.
student_agent = Agent(deps_type=GameDeps, retries=12)


@student_agent.output_validator
def must_finish_the_puzzle(ctx: RunContext[GameDeps], output: str) -> str:
    """Refuse to end the run while the board is unfinished.

    pydantic-ai ends a run as soon as the model replies with text, and a weak
    model will happily answer "b6" in prose instead of calling play_move — which
    looks like a completed run but produces zero decisions and no data. Sending
    it back with the current board turns that into another turn.
    """
    if not ctx.deps.puzzle.done:
        raise ModelRetry(
            "You have not finished the puzzle yet. Naming a square in your reply "
            "does NOT play it — you must call the play_move tool. Keep playing "
            "until the board is finished.\n\n" + board_view(ctx.deps.puzzle))
    short = ctx.deps.questions_required - ctx.deps.questions_asked
    if ctx.deps.assistant_enabled and short > 0:
        raise ModelRetry(
            f"Before finishing, you still need to consult the AI assistant "
            f"{short} more time(s) — use the consult_assistant tool. Ask about "
            f"whatever you are least sure of in this position.")
    return output


_GATE_CLF = None
_GATE_LOCK = threading.Lock()


def _gate_classifier():
    """One dspy classifier on GATE_MODEL, built once and shared across threads.

    Single seat, not the 3-model panel: the panel exists to make OFFLINE labels
    robust by majority vote, but this runs inside the loop on every consultation, so
    a third of the cost matters and a tie-break has nowhere to go.
    """
    global _GATE_CLF
    with _GATE_LOCK:
        if _GATE_CLF is None:
            import dspy
            from dialogue_act_annotation import DialogueActClassifierSignature
            clf = dspy.ChainOfThought(DialogueActClassifierSignature)
            clf.set_lm(dspy.LM(GATE_MODEL, max_tokens=1024, timeout=60))
            _GATE_CLF = clf
    return _GATE_CLF


def _violates(question, gate):
    """(blocked, errored). Fails OPEN on a classifier error.

    Failing open lets a violating message through rather than stalling the run on a
    provider hiccup — but it silently weakens the manipulation, so errors are counted
    into the summary instead of being swallowed.
    """
    import re
    if gate.get("act") or gate.get("require"):
        try:
            from dialogue_act_annotation import TAXONOMY
            acts = set(_gate_classifier()(utterance=question, taxonomy=TAXONOMY).dialogue_acts)
        except Exception as e:
            _ = e
            return False, True
        req = gate.get("require")
        if req:
            # Requirement gate: block when the target act is MISSING. `all` demands
            # every listed act in the same message; otherwise any one satisfies it.
            hit = all(r in acts for r in req) if gate.get("all") else any(r in acts for r in req)
            return not hit, False
        return gate["act"] in acts, False
    return (bool(re.search(gate["block"], question, re.I))
            and not re.search(gate["allow"], question, re.I)), False


def split_modes(spec):
    """"a,b" -> ["a", "b"]; a bare name -> [name]. Empty parts dropped."""
    return [m.strip() for m in str(spec).split(",") if m.strip()]


@student_agent.system_prompt
def persona_prompt(ctx: RunContext[GameDeps]) -> str:
    p = ctx.deps.profile
    sys_prompt = USER_SYSTEM_PROMPT.format(
        persona=p["persona"],
        game=CFG["label"],
        age=str(p["age"]),
        education=EDU_LABELS.get(p["education"], p["education"]),
        occupation=OCC_LABELS.get(p["occupation"], p["occupation"]),
        ai_frequency=AIF_LABELS.get(p["ai_frequency"], p["ai_frequency"]),
    ) + TOOL_INSTRUCTIONS
    # From deps, not from the module-level `args`: `args` only exists when this file
    # is run as a script, so reading it here raised NameError for any other caller
    # (a notebook, a test, anything importing the module). deps is also where every
    # other per-run setting already lives.
    # Modes COMPOSE: "human_style,no_concept_q" appends both fragments, in order.
    # Before this, combining two behaviours meant a third mode duplicating the first
    # one's text -- which is how human_style_think came to carry a verbatim copy of
    # human_style's fourteen examples.
    return sys_prompt + "".join(STUDENT_MODES[m] for m in split_modes(ctx.deps.student_mode))


def board_view(puzzle) -> str:
    """What the participant can see: position, counts, and what is playable."""
    return puzzle.prompt(legal=True)


@student_agent.tool
def view_board(ctx: RunContext[GameDeps]) -> str:
    """Look at the current board again. Does not use up a question."""
    ctx.deps.trace.append({"tool": "view_board"})
    return board_view(ctx.deps.puzzle)


@student_agent.tool
def play_move(ctx: RunContext[GameDeps], move: str) -> str:
    """Play one move. Othello takes a square like "d3"; Connect Four a column 1-7."""
    puzzle = ctx.deps.puzzle
    if puzzle.done:
        return "The game is already over. Stop playing."

    result = puzzle.play(move)
    if not result["ok"]:
        # Retry rather than burn the move: the real interface simply refuses an
        # illegal click, so a participant never "spends" a turn on one.
        options = result.get("legal_moves") or result.get("playable_columns") or []
        raise ModelRetry(
            ("I could not read a move from that." if result["unparsed"]
             else f"{result['answer']} is not legal here.")
            + " You can play: " + ", ".join(str(o) for o in options))

    ctx.deps.trace.append({"tool": "play_move", "move": result.get("move")
                           or result.get("column")})
    ctx.deps.moves.append({**result, "ts": datetime.now(timezone.utc).isoformat()})

    # Report only what the screen would show. NOT `optimal`, NOT `disc_loss`,
    # NOT `best_moves` — the study never tells participants how good a move was,
    # and leaking it here would make the simulated trace incomparable.
    lines = []
    if CFG["key"] == "othello":
        lines.append(f"You played {result['move']}, flipping {len(result['flipped'])} disc(s)"
                     + (f" ({', '.join(result['flipped'])})." if result["flipped"] else "."))
        for mv in result["opponent_moves"]:
            lines.append(f"White replied {mv}.")
        if result["opponent_passed"]:
            lines.append("White had no legal move and passed.")
        if result["you_passed"]:
            lines.append("You had no legal move, so your turn was skipped.")
        if result["auto_played"]:
            lines.append(f"Only one square was legal, so {', '.join(result['auto_played'])} "
                         "was played for you.")
        lines.append(f"Black {result['black_discs']}, White {result['white_discs']}. "
                     f"{result['empties']} square(s) still empty.")
    else:
        lines.append(f"You dropped in column {result['column']}.")
        if result.get("opponent_column"):
            lines.append(f"Yellow answered in column {result['opponent_column']}.")
        for c in result.get("auto_played", []):
            lines.append(f"Only column {c} was playable, so it was dropped for you.")
        lines.append({"red_win": "You got four in a row — you won.",
                      "yellow_win": "Yellow got four in a row. You lost.",
                      "draw": "The board is full. It's a draw.",
                      "ongoing": ""}.get(result.get("status"), ""))

    lines += ["", board_view(puzzle) if not puzzle.done else "The game is over."]
    return "\n".join(l for l in lines if l != "")


@student_agent.tool
def consult_assistant(ctx: RunContext[GameDeps], question: str) -> str:
    """Send a message to the in-interface AI assistant."""
    if not ctx.deps.assistant_enabled:
        return ("There is no AI assistant for this puzzle — you are on your own. "
                "Work it out and play.")
    # Gate BEFORE the budget check and before the lock: a rejected message never
    # reached the assistant, so it must not consume a consultation. Otherwise the
    # gated arm would end up with fewer real turns than its baseline and the
    # comparison would be confounded by volume again.
    gate = ACT_GATES.get(ctx.deps.gate or "")
    if gate:
        bad, errored = _violates(question or "", gate)
        if errored:
            ctx.deps.gate_errors += 1
        if bad and ctx.deps.gate_streak >= GATE_MAX_STREAK:
            # Cap reached: let it through rather than spend another of the agent's 12
            # retries. Counted separately so the leak shows up in the summary.
            ctx.deps.gate_passthrough += 1
            bad = False
        if bad:
            ctx.deps.gate_blocked += 1
            ctx.deps.gate_streak += 1
            raise ModelRetry(gate["retry"])
        ctx.deps.gate_streak = 0

    with ctx.deps.lock:
        if ctx.deps.questions_asked >= ctx.deps.question_budget:
            return ("You are running low on time; keep playing rather than asking the "
                    "AI assistant more questions.")
        response = ctx.deps.assistant.answer(question)
        ctx.deps.questions_asked += 1
        ctx.deps.trace.append({"tool": "consult_assistant", "question": question})
    return response


def assistant_trace_to_rows(history, start=None):
    """[GIVEN] Convert AssistantAgent.conversation_history -> conversation.jsonl rows
    (human schema). Timestamps are synthetic & monotonic."""
    turns = [m for m in history if m.get("role") in ("user", "assistant")]
    ts = start or datetime.now(timezone.utc)
    rows, i = [], 0
    while i + 1 < len(turns):
        if turns[i]["role"] == "user" and turns[i + 1]["role"] == "assistant":
            user_ts = ts
            asst_ts = ts + timedelta(seconds=5)
            rows.append({
                "user": turns[i]["content"],
                "user_ts": user_ts.isoformat(),
                "assistant": turns[i + 1]["content"],
                "assistant_started_ts": user_ts.isoformat(),
                "assistant_ts": asst_ts.isoformat(),
            })
            ts = asst_ts + timedelta(seconds=10)
            i += 2
        else:
            i += 1
    return rows


def moves_to_study_rows(moves, puzzle_id, game_key, optimal_margin=None):
    """Shape the move log like a real participant's moves_p<puzzle>.jsonl.

    Same keys as the study writes, so score_from_logs.py and the rest of the
    analysis can read a simulated run without special-casing it.
    """
    rows = []
    running_loss = 0
    for i, m in enumerate(moves, 1):
        row = {"game": game_key, "puzzle": puzzle_id, "round": 0, "attempt": 1,
               "decision_number": i, "optimal": m["optimal"], "ts": m["ts"],
               "best_moves": m.get("best_moves") or m.get("best_columns"),
               "game_over": m.get("done")}
        if game_key == "othello":
            # kept_win is recoverable without another search: per-move errors
            # telescope, so the position's value after decision k is exactly
            # optimal_margin minus the losses so far.
            running_loss += m["disc_loss"] or 0
            row.update(move=m["move"], disc_loss=m["disc_loss"],
                       kept_win=(optimal_margin - running_loss) > 0
                       if optimal_margin is not None else None,
                       ai_moves=m["opponent_moves"],
                       forced_moves=m["auto_played"],
                       black_passed=m["you_passed"], white_passed=m["opponent_passed"],
                       black_discs=m["black_discs"], white_discs=m["white_discs"])
        else:
            row.update(move=str(m["column"]), col=m["column"] - 1,
                       kept_win=m.get("kept_win"),
                       ai_col=(m.get("opponent_column") or 0) - 1
                       if m.get("opponent_column") else None)
        rows.append(row)
    return rows


# The provider's default output cap is small — small enough that a reasoning
# model can exhaust it on chain-of-thought before emitting anything at all
# ("Model token limit (provider default) exceeded before any response was
# generated"). Kimi in particular wants a lot of room. Lower it for a
# small-context model, which will otherwise reject a budget bigger than its
# whole window. The per-model seeds live in constants.MODEL_MAX_TOKENS; a model
# that still complains is shrunk further at runtime by _adapt_to_provider().


async def _discard_events(ctx, stream):
    """Event handler that keeps nothing.

    Its only job is to exist: pydantic-ai issues STREAMING model requests
    whenever a run has an event_stream_handler, and still drives the graph
    (every tool call included) to completion — unlike run_stream_sync, which
    stops at the model's first text output. That makes this the whole fix for a
    provider that answers 400 'This model only supports streaming'.
    """
    async for _ in stream:
        pass


# Provider complaints we know how to answer, matched against the lower-cased
# error text. Substrings, because the wording differs per model and per gateway.
_BUDGET_COMPLAINTS = ("max_new_tokens", "max_tokens", "context length",
                      "context_length", "input validation error",
                      "too many tokens", "reduce the length")
# Floor for the shrink ladder. 4096 was too low for a reasoning-heavy student to
# emit any reply, so hitting the floor meant losing the cell rather than degrading.
_MIN_MAX_TOKENS = 16_384
_RATE_LIMIT_SIGNS = ("429", "rate limit", "rate_limit", "too many requests",
                     "quota")
_STREAMING_COMPLAINTS = ("only supports streaming", "streaming_required",
                         'set "stream": true')


# ---- building the student's model ---------------------------------------------
# Two reasons this is not just a "provider:name" string handed to pydantic-ai:
#
# 1. RATE LIMITS. Together's limits are dynamic — "they shift with live model
#    capacity and your traffic shape" — so a sweep of parallel students trips a
#    429 sooner or later. A 429 says "later", not "no": retrying the ONE request
#    (honouring Retry-After) keeps the game going, whereas letting it surface
#    kills a run mid-puzzle and loses the sweep entry. Spacing runs out in the
#    shell reduces the collisions but cannot remove them, because the limit
#    moves.
# 2. TOOLS + REASONING ON OPENAI. gpt-5.6 refuses the combination on Chat
#    Completions: 400 "Function tools with reasoning_effort are not supported
#    ... use /v1/responses or set reasoning_effort to 'none'". Every student run
#    is nothing BUT tool calls, and a participant that cannot think is not the
#    participant we are simulating, so the Responses API is the only option that
#    keeps both. pydantic-ai is heading the same way — in v2 `openai:` resolves
#    to Responses by default.
RETRY_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})

# A model given 100k output tokens can legitimately spend minutes on one reply.
HTTP_TIMEOUT = httpx.Timeout(900.0, connect=30.0)


def _retrying_http_client(wrapped: httpx.AsyncBaseTransport | None = None) -> httpx.AsyncClient:
    """An httpx client that waits out transient rejections instead of raising.

    `wrapped` replaces the underlying transport; it exists so the retry
    behaviour can be tested without a provider that will 429 on demand.
    """
    from tenacity import retry_if_exception_type, stop_after_attempt, wait_exponential
    from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after

    def raise_only_if_retryable(response: httpx.Response) -> None:
        # Anything else — a 400 about max_tokens, a 401 — must reach us
        # unretried: those need a different call, not a later one.
        if response.status_code in RETRY_STATUSES:
            response.raise_for_status()

    def say_we_are_waiting(state) -> None:
        """Announce a back-off.

        Without this a rate-limited sweep is indistinguishable from a slow one:
        the waiting happens inside the transport, so the run just sits there
        looking like a model that thinks for a long time. Knowing which it is
        decides whether to lower JOBS or leave it alone.
        """
        exc = state.outcome.exception() if state.outcome else None
        code = getattr(getattr(exc, "response", None), "status_code", "?")
        print(f"    (HTTP {code}: waiting {state.next_action.sleep:.0f}s, "
              f"attempt {state.attempt_number})", flush=True)

    return httpx.AsyncClient(
        timeout=HTTP_TIMEOUT,
        transport=AsyncTenacityTransport(
            config=RetryConfig(
                retry=retry_if_exception_type(httpx.HTTPStatusError),
                # Together's 429 body says "retry starting from ~2s
                # (X-RateLimit-Reset header)" and asks for exponential back-off;
                # wait_retry_after uses the header when there is one.
                wait=wait_retry_after(
                    fallback_strategy=wait_exponential(multiplier=2, min=2, max=120),
                    max_wait=300,
                ),
                stop=stop_after_attempt(8),
                reraise=True,
                before_sleep=say_we_are_waiting,
            ),
            validate_response=raise_only_if_retryable,
            wrapped=wrapped,
        ),
    )


def build_student_model(model_str: str):
    """"provider:name" -> a pydantic-ai Model that retries transient rejections.

    Prefixes: `together:`, `openai:` (Responses API), `anthropic:`, and
    `openai-chat:` as an escape hatch for a model that needs Chat Completions.
    """
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.together import TogetherProvider

    if ":" not in model_str:
        raise ValueError(f"model {model_str!r} needs a provider prefix, "
                         "e.g. together:zai-org/GLM-5.2 or openai:gpt-5.6-sol")
    provider_name, name = model_str.split(":", 1)
    client = _retrying_http_client()

    if provider_name == "together":
        return OpenAIChatModel(name, provider=TogetherProvider(http_client=client))
    if provider_name == "openai":
        return OpenAIResponsesModel(name, provider=OpenAIProvider(http_client=client))
    if provider_name == "openai-chat":
        return OpenAIChatModel(name, provider=OpenAIProvider(http_client=client))
    if provider_name == "anthropic":
        return AnthropicModel(name, provider=AnthropicProvider(http_client=client))
    raise ValueError(f"unknown provider {provider_name!r} in {model_str!r}")


class StudentSimulator:
    """Builds per-run deps, drives the agent, writes traces."""

    def __init__(self, student_model, assistant_model, output_dir, profile, assistant_preset,
                 puzzle=None, question_budget=None, request_limit=40,
                 model_settings=None, assistant_lm_kwargs=None, student_mode="default",
                 questions_required=0, assistant_off=False, gate=None):
        self.student_model = student_model      # e.g. "together:moonshotai/Kimi-K2.6"
        self.model_name = bare_model_name(student_model)
        # Built here, not per attempt: the retrying client is what absorbs a 429,
        # and rebuilding it each attempt would throw away its connection pool.
        self.model = build_student_model(student_model)
        self.output_dir = Path(output_dir)
        self.request_limit = request_limit
        # Per-run overrides win; anything unset falls back to what this model is
        # known to accept.
        self.model_settings = dict({"max_tokens": model_max_tokens(student_model)},
                                   **(model_settings or {}))
        # Whether to issue streaming requests. Starts from what we already know
        # about this model and is turned on for good the moment one refuses.
        self.stream = self.model_name in STREAM_ONLY_MODELS
        self.assistant_model = assistant_model
        self.assistant_lm_kwargs = assistant_lm_kwargs or {}
        self.assistant_preset = assistant_preset
        unknown = [m for m in split_modes(student_mode) if m not in STUDENT_MODES]
        if unknown:
            raise ValueError(f"unknown student_mode component(s) {unknown}; "
                             f"expected from {sorted(STUDENT_MODES)} "
                             f"(combine with commas)")
        self.student_mode = student_mode
        # A run with no assistant cannot satisfy a consultation floor; leaving it set
        # would make the output validator demand questions that are impossible to ask
        # and burn every retry before failing.
        self.questions_required = 0 if assistant_off else questions_required
        if gate and gate not in ACT_GATES:
            raise ValueError(f"unknown gate {gate!r}; expected one of {sorted(ACT_GATES)}")
        self.gate = gate
        # One puzzle, or every round the study runs. Passing `puzzle` keeps the
        # old single-puzzle behaviour.
        self.plan = ([{"puzzles": [puzzle], "ai": True}] if puzzle else CFG["rounds"])
        # The no-assistance control: the assisted round runs WITHOUT a coach, so the
        # solo score measures what this student reaches unaided. Without it, "assistant
        # style does not affect transfer" cannot be told apart from "any competent
        # assistant is equally sufficient" or "the assistant is irrelevant". New dicts,
        # never a mutation of CFG["rounds"] -- that object is shared across the process.
        self.assistant_off = assistant_off
        if assistant_off:
            self.plan = [dict(r, ai=False) for r in self.plan]
        self.profile = profile
        self.base_question_budget = (question_budget if question_budget is not None
                                     else DEFAULT_QUESTION_BUDGET)
        self.deps = None
        self.result = None
        self.runs = []                 # one summary per puzzle played

    def _start_puzzle(self, puzzle_id, ai_enabled):
        """Fresh board, and for an assisted round a fresh assistant thread.

        The study clears the conversation at the start of each round, so the
        coach does not remember the previous puzzle — but the PARTICIPANT does,
        which is what run() preserves through message_history.
        """
        session = PuzzleSession(puzzle_id)
        assistant = (AssistantAgent(self.assistant_model, session, game=GAME, preset=self.assistant_preset,
                                    **self.assistant_lm_kwargs) if ai_enabled else None)
        if assistant is None:
            # Never consulted (the tool refuses first); present only so the
            # trace writer has something with an empty history to read.
            assistant = AssistantAgent.__new__(AssistantAgent)
            assistant.conversation_history = []
        self.deps = GameDeps(
            profile=self.profile,
            puzzle=session,
            assistant=assistant,
            question_budget=self.base_question_budget if ai_enabled else 0,
            assistant_enabled=ai_enabled,
            student_mode=self.student_mode,
            questions_required=(self.questions_required if ai_enabled else 0),
            gate=self.gate,
        )
        return session

    def run(self):
        """Play every round in order, carrying the participant's memory forward.

        message_history is the transfer mechanism: the same person, having just
        worked through round 1 with the coach, meets round 2 without it. A fresh
        agent per round would measure nothing about transfer.
        """
        history = None
        for round_index, rnd in enumerate(self.plan):
            for puzzle_id in rnd["puzzles"]:
                self.result = self._play_puzzle(puzzle_id, rnd["ai"], history)
                history = self.result.all_messages()
                self.runs.append(self.save_traces(round_index, puzzle_id))
        return self.runs

    def _play_puzzle(self, puzzle_id, ai_enabled, history, attempts=6):
        """Play one puzzle, re-attempting only when the PROVIDER told us how to call it.

        Every attempt restarts from a fresh board (_start_puzzle), because a run
        that died mid-game leaves a half-played position and a half-written
        trace; scoring those together would invent a participant who played the
        same endgame twice. Earlier rounds are untouched — `history` still
        carries the same person's memory forward.
        """
        for _ in range(attempts):
            self._start_puzzle(puzzle_id, ai_enabled)
            intro = ("You are looking at the puzzle now. Play it to the end."
                     if ai_enabled else
                     "Here is the next puzzle. There is NO AI assistant for this "
                     "one — you are on your own. Play it to the end.")
            try:
                return student_agent.run_sync(
                    user_prompt=intro + "\n\n" + board_view(self.deps.puzzle),
                    deps=self.deps,
                    model=self.model,
                    model_settings=self.model_settings,
                    message_history=history,
                    usage_limits=UsageLimits(request_limit=self.request_limit),
                    event_stream_handler=_discard_events if self.stream else None,
                )
            except Exception as exc:
                if not self._adapt_to_provider(exc):
                    raise
        raise RuntimeError(f"{self.student_model}: still rejected after {attempts} "
                           f"attempts at {puzzle_id}")

    def _adapt_to_provider(self, exc) -> bool:
        """Change how we call this model based on what it just complained about.

        True if something changed and the attempt is worth repeating. Both
        complaints are call-convention problems, not model failures: without
        this a sweep entry dies on a 400 we could simply have complied with.
        """
        msg = str(exc).lower()

        # THE DOOM LOOP. pydantic-ai's own complaint when a reply runs past the
        # configured budget is "Model token limit (N) exceeded ... Increase the
        # `max_tokens` model setting" -- which contains "max_tokens" and so matched
        # _BUDGET_COMPLAINTS below. The handler then QUARTERED the budget. The model
        # needed more room and we gave it less, three times, down to the 4096 floor
        # where Kimi cannot finish a reply at all. That is what cost 3/20 baseline
        # cells their unassisted puzzles: the budget is sticky on self.model_settings,
        # so one downshift during the assisted puzzle poisons every puzzle after it.
        #
        # These two look alike and mean opposite things:
        #   provider 422 "inputs + max_new_tokens must be <= N"  -> request too big, shrink
        #   pydantic-ai "Model token limit (N) exceeded"         -> reply too big, do NOT shrink
        if "token limit" in msg and "exceeded" in msg:
            print(f"    ({self.model_name}: reply exceeded max_tokens "
                  f"{self.model_settings.get('max_tokens')} -- not shrinking)", flush=True)
            return False

        # A 429 is congestion, not a call-convention problem, and Together's
        # rate-limit text mentions tokens too.
        if any(s in msg for s in _RATE_LIMIT_SIGNS):
            return False

        if any(s in msg for s in _STREAMING_COMPLAINTS) and not self.stream:
            self.stream = True
            print(f"    ({self.model_name}: streaming-only, switching)", flush=True)
            return True

        if any(s in msg for s in _BUDGET_COMPLAINTS):
            # Quarter it: the provider says what the ceiling is only sometimes,
            # and a couple of downshifts land under it faster than stepping.
            current = self.model_settings.get("max_tokens") or 0
            if current > _MIN_MAX_TOKENS:
                self.model_settings["max_tokens"] = max(_MIN_MAX_TOKENS, current // 4)
                # Log WHAT asked for the downshift, not just that one happened --
                # the doom loop above was invisible for want of this one line.
                print(f"    ({self.model_name}: max_tokens {current} -> "
                      f"{self.model_settings['max_tokens']} | {str(exc)[:120]})", flush=True)
                return True

        return False

    def save_traces(self, round_index=0, puzzle_id=None):
        """Write this puzzle's trace, assistant thread, moves and score.

        Every file is suffixed with the puzzle id, matching how the study keeps
        a participant's rounds apart on disk.
        """
        puzzle_id = puzzle_id or self.deps.puzzle.puzzle
        self.output_dir.mkdir(parents=True, exist_ok=True)
        json.dump(self.deps.trace,
                  open(self.output_dir / f"game_trace_p{puzzle_id}.json", "w"), indent=2)
        if self.result is not None:
            open(self.output_dir / f"student_messages_p{puzzle_id}.json", "wb").write(
                self.result.all_messages_json())

        rows = assistant_trace_to_rows(
            getattr(self.deps.assistant, "conversation_history", []))
        with open(self.output_dir / f"conversation_p{puzzle_id}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        study_rows = moves_to_study_rows(self.deps.moves, puzzle_id, CFG["key"],
                                         getattr(self.deps.puzzle, "optimal_margin", None))
        with open(self.output_dir / f"moves_p{puzzle_id}.jsonl", "w") as f:
            for r in study_rows:
                f.write(json.dumps(r) + "\n")

        summary = {**self.deps.puzzle.summary(),
                   "game": CFG["key"],
                   "round": round_index,
                   "ai_available": self.deps.assistant_enabled,
                   "model_settings": self.model_settings,
                   "questions_asked": self.deps.questions_asked,
                   "question_budget": self.deps.question_budget,
                   "profile": self.profile,
                   "student_model": self.student_model,
                   # The behaviour condition. Without these a summary cannot say which
                   # arm it belongs to, and the run directory name is the only record.
                   "student_mode": self.student_mode,
                   "questions_required": self.questions_required,
                   "assistant_off": self.assistant_off,
                   "assistant_model": self.assistant_model,
                   "tool_framing": TOOL_FRAMING,
                   "gate": self.gate,
                   "gate_blocked": self.deps.gate_blocked if self.deps else 0,
                   "gate_errors": self.deps.gate_errors if self.deps else 0,
                   "gate_passthrough": self.deps.gate_passthrough if self.deps else 0,
                   "gate_model": GATE_MODEL if self.gate else None,
                   "assistant_preset": self.assistant_preset,
                   # Recorded because both are negotiated at runtime: a summary
                   # without them cannot explain why two runs of the same model
                   # were called differently.
                   "streamed": self.stream}
        json.dump(summary, open(self.output_dir / f"summary_p{puzzle_id}.json", "w"),
                  indent=2)
        return summary


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--preset", type=int)
    parser.add_argument("--student_model", type=str)
    # Choices come from the mode tables, so a new manipulation is added in
    # constants.py alone; a hardcoded list here silently rejects it instead.
    # Free-form so modes can be combined: --student_mode human_style,no_concept_q
    parser.add_argument("--student_mode", type=str, default="default",
                        help="one mode, or several comma-separated: "
                             + ",".join(sorted(STUDENT_MODES)))
    parser.add_argument("--asst_mode", type=str, default="default",
                        choices=sorted(ASSISTANT_MODES))
    # Fixes the number of consultations so arms differ only in HOW questions are
    # phrased. Also becomes the budget, so the count is exact rather than a ceiling.
    parser.add_argument("--questions_required", type=int, default=0)
    # The coach itself as an experimental factor. dspy.LM naming ("provider/model"),
    # which differs from the student's "provider:org/model" -- they go to different
    # client libraries.
    parser.add_argument("--assistant_model", type=str, default="openai/gpt-5.5")
    parser.add_argument("--gate", type=str, default=None, choices=sorted(ACT_GATES),
                        help="ENFORCE an act knockout by rejecting violating messages")
    parser.add_argument("--assistant_off", action="store_true",
                        help="no-assistance control: run the assisted round without a coach")

    args = parser.parse_args()
    # Gate on the file existing rather than a hardcoded [1..5]: the persona pool is
    # extended by build_preset_personas.py, so any range baked in here goes stale and
    # silently blocks new presets.
    preset_f = Path(f"preset_personas/preset_{args.preset}.json")
    if not preset_f.exists():
        available = sorted(int(p.stem.split("_")[1])
                           for p in Path("preset_personas").glob("preset_*.json"))
        raise SystemExit(f"no such preset: {args.preset}. Available: {available}")

    profile = json.load(open(preset_f))
    safe_file = "preset_" + str(args.preset) + args.student_model.replace(":", "_").replace("/", "_")
    import sys
    # One output root per game, so a run directory stays "one persona x model on
    # ONE game" — the unit the 2-group analysis treats as a participant. Othello
    # keeps the original path so already-collected runs are still found.
    run_root = ("test_agentic_run" if CFG["key"] == "othello"
                else f"test_agentic_run_{CFG['key']}")
    # A behaviour condition gets its OWN root. Without this every condition writes to
    # preset_<n><model>/ — so a `--student_mode think` run lands on the default run's
    # directory, the resume marker below sees a summary already there and exits 0, and
    # the manipulation silently never runs. The default condition keeps the original
    # path so everything already collected is still found.
    cond = "_".join(t for t in (args.student_mode.replace(",", "+"), args.asst_mode)
                    if t != "default")
    if args.questions_required and not args.assistant_off:
        cond = (cond + "_" if cond else "") + f"q{args.questions_required}"
    if args.gate:
        cond = (cond + "_" if cond else "") + f"gate-{args.gate}"
    if args.assistant_off:
        cond = (cond + "_" if cond else "") + "noai"
    # A different coach is a different condition; without this every assistant model
    # writes into the same run directory and the resume marker skips all but the first.
    if TOOL_FRAMING != "default":
        cond = (cond + "_" if cond else "") + TOOL_FRAMING
    if args.assistant_model != "openai/gpt-5.5":
        tag = args.assistant_model.replace("/", "_").replace(":", "_")
        cond = (cond + "_" if cond else "") + f"asst-{tag}"
    if cond:
        run_root = f"{run_root}__{cond}"
    # Resume marker: the first puzzle of the final (no-assistant) round, which is
    # only written once the assisted round is already done. Derived from the game
    # config rather than hardcoded, so it points at w4p6 for Connect Four.
    done_marker = CFG["rounds"][-1]["puzzles"][0]
    if os.path.exists(f"{run_root}/{safe_file}/summary_p{done_marker}.json"):
        sys.exit(0)
    sim = StudentSimulator(
        student_model=args.student_model,
        assistant_model=args.assistant_model,
        output_dir=f"{run_root}/{safe_file}",
        profile=profile,
        question_budget=(args.questions_required if args.questions_required else 10),
        questions_required=args.questions_required,
        assistant_off=args.assistant_off,
        gate=args.gate,
        assistant_preset=args.asst_mode,
        student_mode=args.student_mode,
        # max_tokens is left unset on purpose: it is seeded per model from
        # constants.MODEL_MAX_TOKENS and shrunk further if the provider still
        # refuses. Passing it here would override both.
    )
    runs = sim.run()                     # every round the study runs, in order
    for r in runs:
        print(f"round {r['round']} {r['puzzle']:<12} ai={str(r['ai_available']):<5} "
              f"questions={r['questions_asked']} "
              f"{r['optimal_moves']}/{r['decisions']} optimal  won={r['won']}")
