import base64
import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

import markdown
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    stream_with_context,
    url_for,
)

import openai
import dotenv
import random

dotenv.load_dotenv()

OPENAI_CLIENT = openai.OpenAI()
CHAT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.5")
PROLIFIC_CODE = os.environ.get("PROLIFIC_CODE", "PLACEHOLDER")
# Three screen-out paths/codes: #1 skilled Othello players (screened on the
# eligibility page), #2 low-quality prompts (screened after the prompt task),
# #3 in-game gate — a confident, correct unaided first move (screened mid-game).
SCREENOUT_SKILL_CODE = os.environ.get("SCREENOUT_SKILL_CODE", "PLACEHOLDER")
SCREENOUT_PROMPT_CODE = os.environ.get("SCREENOUT_PROMPT_CODE", "PLACEHOLDER")
SCREENOUT_INGAME_CODE = os.environ.get("SCREENOUT_INGAME_CODE", "PLACEHOLDER")

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "citrus-blast")

BASE_DIR = Path(__file__).parent
GAMES_DIR = BASE_DIR / "games"
# Where participant data is written. In production this is a GCS bucket mounted
# via Cloud Run's volume mount (gcsfuse) so data survives restarts/redeploys.
# Locally it defaults to ./recordings.
RECORDINGS_DIR = Path(os.environ.get("RECORDINGS_DIR", BASE_DIR / "recordings"))

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# This deployment is Othello-only (the Connect Four / chess assets were removed).
DEFAULT_GAME = "othello"

# ---- Per-game engines -------------------------------------------------------
# Each game's logic lives in games/<game>/engine.py and is loaded here as a
# module, keeping app.py game-agnostic.
import importlib.util as _ilu  # noqa: E402


def _load_local_module(name: str, path: Path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_OTH_DIR = GAMES_DIR / "othello"
oth_engine = _load_local_module("oth_engine", _OTH_DIR / "engine.py")

# Per-round countdown timers (seconds), env-configurable. Round 1 gets 8 min,
# round 2 gets 3 min — the puzzles run to completion, which is 6 and 5 scored
# decisions respectively (see OTH_ROUNDS).
#
# Set from the first batch's timings. Engaged participants spend ~4 MINUTES on
# their opening decision (orientation: reading the board, learning the rules,
# first questions to the assistant) and then 30-65s each afterwards, so round 1
# is dominated by a fixed startup cost rather than by per-move thinking. Two of
# five used 87% and 95% of the old 7-minute budget and one of them still timed
# out mid-line. Round 2 needs none of that startup — the longest anyone spent on
# it was 76s — so 3 minutes still leaves well over twice the observed maximum.
OTH_TIME_LIMIT_SECONDS = int(os.environ.get("OTH_TIME_LIMIT_SECONDS", "480"))       # round 1
# Each transfer puzzle gets its own clock, so a slow first one cannot eat the
# second. 90s x 2 keeps the block inside the 3 minutes round 2 used to have; the
# longest anyone spent on a single transfer puzzle was 76s.
OTH_TIME_LIMIT_TRANSFER = int(os.environ.get("OTH_TIME_LIMIT_TRANSFER", "90"))     # each transfer puzzle

# Two-round within-subjects design: every participant plays these puzzles IN
# ORDER. Each round names a puzzle (the puzzle_config_<id>.txt / solution_<id>.json
# pair), whether the in-game AI chat assistant is available ("ai"), and whether
# the external AI-assessment survey is shown after that round's NASA-TLX.
#
# "gate": the round STARTS with the assistant off. The participant makes an
# unaided first move and rates their confidence; if that move was optimal AND
# confidence is high they are screened out. Otherwise the assistant is unlocked:
# a wrong first move resets the board for a clean attempt, while a correct first
# move keeps its progress and continues. See /api/gate.
#
# Both puzzles are othelloclub.com endgames, imported and solved exactly by
# games/othello (see its README for the ranked table and the motif analysis).
# They are MOTIF-MATCHED so round 2 tests transfer of the specific insight from
# round 1 rather than general endgame play. Both answers are X-SQUARES — the
# square diagonally adjacent to a corner that every primer says never to play:
#   round 1  oc20260727  b7 (+18), 10 empties, 6 scored decisions, greedy costs 40
#   round 2  b220260706  g7 (+12),  5 empties, 3 scored decisions, greedy costs 16
#
# Round 2 is deliberately a SMALL puzzle (from othelloclub's beginner archive).
# The first two batches showed round-2 engagement collapsing to a median of 15
# seconds — and it tracked round-1 OUTCOME exactly: the two participants who won
# round 1 gave round 2 41s and 76s, the six who lost gave it 5-27s, with no
# overlap. One of them had spent 428s on round 1 first, so it is demoralisation,
# not laziness. A 10-empty round 2 cannot be solved in the attention people
# actually give it, and it produced 0 wins in 8 attempts — an outcome with no
# variance measures nothing. Five empties can be read in 15 seconds.
#
# Small does not mean guessable: random legal play wins this one only 4.8% of the
# time (most beginner puzzles are 15-30%, which is why this one was chosen out of
# 98 candidates). Its root offers TWO X-squares, g7 (+12) and g2 (-4), so "play
# next to the corner" is a coin flip and the concept has to be applied, not
# recited. b7 — round 1's answer — is not playable, so coordinate memory can
# neither help nor mislead.
#
# NOTE the rounds now differ in size (6 decisions vs 3), so compare per-decision
# RATE and the win outcome, never raw counts. See score_from_logs.py.
# Alternates kept on disk (games/othello holds only the puzzles in use plus
# these; anything else regenerates from the archive files with one import
# command -- see that folder's README):
#  b220260507 - the third puzzle that passed the transfer-candidate filter, if
#               you ever want a three-puzzle block;
#  oc20260713 - same motif, 5 decisions, but 10 empties and 0 wins in 8 tries;
#  oc20260725 - a corner grab, tests general endgame play rather than transfer.
OTH_ROUNDS = [
    {"puzzle": "oc20260727", "ai": True,  "ai_survey": True,  "gate": True,
     "time_limit": OTH_TIME_LIMIT_SECONDS,
     # "intro": a heads-up popup shown once before the round starts (title + HTML body).
     # DRAFT wording — review; note it hints at the unaided-first-move structure.
     "intro": {"title": "Puzzle 1",
               "body": "An <strong>AI assistant</strong> will be available to help you for this round. This is your <strong>only chance</strong> to use it in this study.<br>You MUST send the assistant <strong>5+ messages while you play</strong>. You will not be able to finish the round otherwise."}},
    # Rounds 2-3: the transfer BLOCK — two small puzzles back to back, no
    # assistant, no survey in between ("survey": False suppresses it, see
    # /round-done). Two puzzles rather than one because round 2 is the primary
    # outcome and a single 5-empty puzzle yields only ~6 distinct discs-lost
    # values; two of them roughly doubles the resolution and stops one unlucky
    # puzzle from deciding a participant's score, while still fitting in the
    # 3 minutes the old single round-2 puzzle had. Both answers are X-squares at
    # two different coordinates (g7, b2) and neither is b7, so round-1 coordinate
    # memory is worth nothing and the concept has to generalise.
    {"puzzle": "b220260706", "ai": False, "ai_survey": False, "survey": False,
     "time_limit": OTH_TIME_LIMIT_TRANSFER,
     "intro": {"title": "Puzzles 2 and 3",
               "body": "Two more puzzles, back to back. There is <strong>no AI assistant</strong> "
                       "for either, and you will <strong>not</strong> need to fill out the AI "
                       "assistant questionnaire afterward."}},
    {"puzzle": "bg20260726", "ai": False, "ai_survey": False,
     "time_limit": OTH_TIME_LIMIT_TRANSFER,
     "intro": {"title": "Puzzle 3", "body": "Last puzzle — same rules."}},
]


def oth_puzzle_file(puzzle: str) -> Path:
    return _OTH_DIR / f"puzzle_config_{puzzle}.txt"


# Each puzzle's precomputed solution table, loaded once and keyed by puzzle id.
# The board and its solution are always drawn from the same id so they can never
# drift apart.
OTH_SOLUTIONS = {}
for _r in OTH_ROUNDS:
    with open(_OTH_DIR / f"solution_{_r['puzzle']}.json") as _f:
        OTH_SOLUTIONS[_r["puzzle"]] = json.load(_f)

# The participant plays each puzzle TO COMPLETION — until neither side can move.
# Othello's payoff is entirely terminal (on round 1's optimal line Black is 20-40
# down on discs after three moves and finishes 41-23), so a truncated game shows
# the participant a position that reads as lost when they have played perfectly.
# Playing it out also makes the outcome objective: they won, or they didn't.
#
# The round therefore has no move quota. Its length is a property of the puzzle:
# 5-7 Black decisions for the imported set. We never tell them mid-game whether a
# move was correct; the live disc count is the only feedback, and the score lands
# on the results screen afterward.
#
# OTH_FIRST_N is only the SUBSCORE window: how many of the participant's opening
# decisions are also scored separately, so this phase stays comparable with the
# Connect Four phase (which scored exactly 3 moves).
OTH_FIRST_N = int(os.environ.get("OTH_FIRST_N", "3"))

# The participant is Black (moves first in every puzzle); the AI is White.
BLACK, WHITE = oth_engine.BLACK, oth_engine.WHITE


# ---- Othello glue -----------------------------------------------------------
# Two rules-level differences from the Connect Four version drive everything
# below: a move is a SQUARE ("d3"), not a column, and either side can be forced
# to PASS. The client is never told whether a move was good, only where discs
# now are and where it may play next.
def oth_legal_moves(board, piece=None) -> list[str]:
    """Legal moves for `piece` (default: the participant) in "d3" notation."""
    moves = oth_engine.legal_moves(board, BLACK if piece is None else piece)
    return [oth_engine.to_notation(m) for m in moves]


def oth_grade(board, played: str, solution: dict) -> tuple[list[str], dict]:
    """(best moves, {move: exact value}) for Black in this position.

    Straight from the precomputed table while the participant is on the optimal
    line; once they leave it the engine grades live, which is still exact at
    endgame sizes. One call serves both the score and the disc-loss record, so
    the two can never disagree.
    """
    entry = solution.get(oth_engine.state_key(board, BLACK)) or {}
    if entry.get("best_moves"):
        return entry["best_moves"], entry.get("move_values", {})
    graded = oth_engine.grade_move(board, played)
    return graded["best_moves"], graded["move_values"]


def oth_black_best(board, solution) -> list[str]:
    """Black's best move(s) here — the table's if we're on it, else the engine's.

    This is the set a move is scored against AND the set the assistant is
    coached toward, so following good advice always earns credit.
    """
    entry = solution.get(oth_engine.state_key(board, BLACK)) or {}
    if entry.get("best_moves"):
        return entry["best_moves"]
    return [oth_engine.to_notation(m) for m in oth_engine.best_moves(board, BLACK)]


def oth_ai_turns(board, solution) -> tuple[list[str], bool, bool]:
    """Play White's reply, handling passes on both sides. Mutates `board`.

    Returns (moves White played, did Black have to pass, did White have to pass).
    White keeps moving while Black has no legal reply, so the board handed back
    to the participant is always one they can actually play in (or a finished
    game). On the precomputed line White plays the table's best defense;
    off it, the engine's — exact either way at these sizes.
    """
    ai_moves, black_passed, white_passed = [], False, False
    while not oth_engine.is_game_over(board):
        if oth_engine.has_move(board, WHITE):
            entry = solution.get(oth_engine.state_key(board, WHITE)) or {}
            move = entry.get("best_defense")
            move = (oth_engine.from_notation(move) if move else None)
            if move is None or not oth_engine.is_valid_move(board, move, WHITE):
                move = oth_engine.ai_move(board)      # off the table -> live engine
            oth_engine.apply_move(board, move, WHITE)
            ai_moves.append(oth_engine.to_notation(move))
        else:
            white_passed = True
        if oth_engine.has_move(board, BLACK):
            break
        black_passed = True                           # Black passes; White goes again
    return ai_moves, black_passed, white_passed


def oth_autoplay_forced(board, solution) -> list[str]:
    """Play out any position where Black has exactly ONE legal square. Mutates `board`.

    A single legal move is not a decision, so we neither ask for it nor score it —
    asking would hand out a free point and waste the participant's clock. Each
    puzzle ends on one of these (the last empty square), and White answers each
    one as usual. Returns the squares played automatically.
    """
    auto: list[str] = []
    while not oth_engine.is_game_over(board):
        moves = oth_engine.legal_moves(board, BLACK)
        if len(moves) != 1:
            break
        oth_engine.apply_move(board, moves[0], BLACK)
        auto.append(oth_engine.to_notation(moves[0]))
        oth_ai_turns(board, solution)
    return auto


def oth_board_state(board) -> dict:
    """Everything the client needs to draw the position (and nothing more)."""
    black, white = oth_engine.disc_counts(board)
    return {
        "board": board,
        "legal_moves": oth_legal_moves(board),
        "black_discs": black,
        "white_discs": white,
        "empties": oth_engine.empty_count(board),
        "game_over": oth_engine.is_game_over(board),
        "status": oth_engine.status(board),
    }


def oth_finish_round(end_reason: str, completed: bool) -> dict:
    """Close out the current round and write its score summary.

    Called from BOTH endings: the game finishing, and the clock running out. A
    timeout is a real ending — the round is over and the decisions they did make
    still get scored — so it must mark the round finished like any other. It
    previously did not, which left `oth_finished` False: /performance bounced the
    participant to /study, /study bounced them to /post-survey, and they looped
    on the TLX instead of reaching the next round. One participant in the first
    batch was lost that way.

    `completed` False means the board never finished, so there is no final
    result; the per-decision scores are still exact.
    """
    rnd = current_round()
    board = session.get("oth_board")
    scores = oth_round_scores(session.get("oth_optimal_flags", []),
                              session.get("oth_kept_flags", []))
    attempt = session.get("oth_attempt", 1)
    solved = completed and scores["score"] == scores["decisions"] and scores["decisions"] > 0

    attempts = session.get("oth_attempts", [])
    attempts.append({**scores, "attempt": attempt, "solved": solved,
                     "completed": completed, "end_reason": end_reason})
    session["oth_attempts"] = attempts
    best = max(attempts, key=lambda a: (a["score"], a["score_first_n"]))
    session["oth_best"] = best
    session["oth_finished"] = True

    black, white = oth_engine.disc_counts(board) if board else (0, 0)
    summary = {
        "puzzle": rnd["puzzle"], "round": current_round_index(), "ai": rnd["ai"],
        "score": best["score"], "decisions": best["decisions"],
        "score_first_n": best["score_first_n"],
        "decisions_first_n": best["decisions_first_n"],
        "kept_win_score": best["kept_win_score"],
        "kept_win_first_n": best["kept_win_first_n"],
        "first_n": OTH_FIRST_N,
        "solved": bool(best.get("solved")),
        "attempts": attempts, "attempts_used": attempt,
        "end_reason": end_reason,
        # completed False -> the clock ended it, so these are the position at the
        # buzzer, NOT a final result. Don't read final_margin as an outcome.
        "completed": completed,
        "final_black_discs": black, "final_white_discs": white,
        "final_margin": black - white,
        "result": oth_engine.status(board) if (board and completed) else "unfinished",
        "game_over": completed,
        "forced_moves": session.get("oth_forced_moves", []),
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    safe_pid = session.get("safe_pid", "anon")
    with open(round_path(participant_dir(safe_pid), "oth_score", "json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def oth_round_scores(optimal_flags: list[bool], kept_flags: list[bool]) -> dict:
    """Both scorings of one attempt, from its per-decision flags.

    `score` is over every scored decision in the puzzle; `first_n` is over the
    opening OTH_FIRST_N of them, which is the measure comparable with the
    Connect Four phase. `kept_win` counts decisions after which Black was still
    winning — the stricter metric, on the same two windows.
    """
    n = OTH_FIRST_N
    return {
        "score": sum(optimal_flags),
        "decisions": len(optimal_flags),
        "score_first_n": sum(optimal_flags[:n]),
        "decisions_first_n": len(optimal_flags[:n]),
        "kept_win_score": sum(kept_flags),
        "kept_win_first_n": sum(kept_flags[:n]),
    }

# Confidence scale for the gated round's after-first-move popup (1-5). A rating
# in CONFIDENCE_HIGH, together with an optimal first move, screens the participant out.
CONFIDENCE_OPTIONS = [
    ("1", "Not at all confident"),
    ("2", "Slightly confident"),
    ("3", "Moderately confident"),
    ("4", "Very confident"),
    ("5", "Extremely confident"),
]
CONFIDENCE_VALUES = {v for v, _ in CONFIDENCE_OPTIONS}
CONFIDENCE_HIGH = {"4", "5"}


def current_round_index() -> int:
    return session.get("oth_round", 0)


def current_round() -> dict:
    return OTH_ROUNDS[current_round_index()]


def ai_enabled() -> bool:
    """Whether the AI chat assistant is available to the participant RIGHT NOW.
    A gated round keeps it off until the confidence gate is passed
    (session['ai_unlocked']); otherwise it is the round's static 'ai' flag."""
    rnd = current_round()
    if rnd.get("gate") and not session.get("ai_unlocked"):
        return False
    return rnd["ai"]


def round_path(pdir: Path, base: str, ext: str) -> Path:
    """Per-round participant data file, suffixed with the current puzzle id so
    the two rounds never overwrite each other (e.g. moves_p15.jsonl)."""
    return pdir / f"{base}_p{current_round()['puzzle']}.{ext}"


# ---- Minimum engagement with the assistant ----------------------------------
# A participant who never talks to the AI tells us nothing about AI-assisted
# play, so the assisted round requires at least this many completed exchanges
# before they may leave it. Only completed exchanges count (a failed request
# writes no line), and the requirement never applies where there is no
# assistant to talk to — a no-AI round, or before the gate unlocks it.
#
# It gates the FINISH BUTTON, not the game: they can keep chatting after the
# board is finished (the round-1 intro invites exactly that — asking questions
# to prepare for puzzle 2). A timeout always ends the round regardless, so the
# requirement can never trap someone whose clock has run out.
OTH_MIN_AI_TURNS = int(os.environ.get("OTH_MIN_AI_TURNS", "5"))


def ai_turns_taken() -> int:
    """Completed assistant exchanges in the current round (one per logged line)."""
    path = round_path(participant_dir(session.get("safe_pid", "anon")),
                      "conversation", "jsonl")
    try:
        with open(path) as f:
            return sum(1 for line in f if line.strip())
    except FileNotFoundError:
        return 0


def ai_turn_requirement() -> dict:
    """How many assistant exchanges are required here, and how many they've had."""
    required = OTH_MIN_AI_TURNS if ai_enabled() else 0
    taken = ai_turns_taken() if required else 0
    return {
        "required": required,
        "taken": taken,
        "remaining": max(0, required - taken),
        "met": taken >= required,
    }

# ---- Stage 1: prompt-creation pre-filter ------------------------------------
# Participants first write a prompt for a fixed scenario; a multi-model judge
# panel (defined in prompt_filter.py: JudgeSuite) each scores it 1-4 on a
# holistic rubric, and the MEDIAN of those scores must clear JUDGE_PASS_THRESHOLD
# for the participant to continue into the game. Everything (the prompt, every
# model's score, the mean, the median, the decision) is logged to prompt_task.json.
JUDGE_PASS_THRESHOLD = float(os.environ.get("JUDGE_PASS_THRESHOLD", "3.0"))
# Which panel statistic decides eligibility. Phase 2 recruits NON-adopters, so a
# participant is KEPT when the statistic falls BELOW the threshold -- the opposite
# direction from the phase-1 screener.
#
# "mean" is the default here, and the change from phase 1's "median" is deliberate.
# On phase 1's 145 fully-judged prompts the two rules differ on 41 participants, and
# every one of those is the same pattern: scores (2,3,3), median 3.0, mean 2.67 -- one
# judge calls the prompt "developing", two call it "proficient". Those prompts
# typically name the cookbook and the three-year horizon but omit the
# traditional-vs-self-publishing decision and every constraint, which the rubric calls
# incomplete coverage. Treating them as adopters both mis-labels them and shrinks the
# eligible pool from 40% to 12%, roughly tripling the screening cost.
JUDGE_PASS_RULE = os.environ.get("JUDGE_PASS_RULE", "mean")   # "mean" | "median"

# Skip the prompt task entirely. Set on a SEPARATE deployment used to re-recruit the 17
# participants phase 1 screened out at the prompt task: phase 1 kept median >= 3 and
# dropped the rest, phase 2 keeps mean < 3, so those 17 are already known to qualify and
# re-testing them would only add attrition. None of them reached the Othello game in
# phase 1 -- they were screened out before the primer -- so there is no task exposure.
#
# Their phase-1 verdict is copied into prompt_task.json with source="phase_1_carryover",
# so the analysis funnel reads the same shape as for fresh recruits.
SKIP_PROMPT_TASK = os.environ.get("SKIP_PROMPT_TASK", "0") == "1"
JUDGE_MIN_PROMPT_CHARS = int(os.environ.get("JUDGE_MIN_PROMPT_CHARS", "40"))  # reject near-empty

# ---- Scaffolded arm: contrasting-cases intervention --------------------------
# Real Connect Four messages from phase-1 participants, shown verbatim. Group A asks
# the assistant for the answer; Group B narrates its own reasoning, names specific
# candidate moves, and asks how a move already played turned out.
#
# The two columns are LENGTH-MATCHED on purpose (A: 100 words / 10 messages, B: 96 / 9)
# and A's third participant is the single longest message in the table. Without that,
# the obvious reading of the contrast is "Group B writes more" -- which the phase-1 data
# says is not the mechanism: message length predicts transfer only through thinking
# aloud, and drops to nothing once thinking aloud is in the model (b=+0.044, p=.85).
# Variants are named and recorded per participant, because the material will be
# iterated during collection and "which table did this person see" must be recoverable
# from the data rather than from memory. Every message is VERBATIM from a phase-1
# Connect Four participant; variants differ only in which real conversations appear and
# in what order, never in invented copy.
#
# v0_live  -- the table the first participants saw.
# v2_think_pair -- rows 1 and 2 lead with a participant narrating their own reasoning
#   ("I was thinking of putting one in 3-3 / it would prevent yellow..."). Chosen after
#   a simulated-reader pretest (intervention_pretest.py, 18 readers/variant): naming of
#   the think-aloud dimension rose 78% -> 100% and of request type 83% -> 94%, with the
#   correct forced choice at 94%. Those rates are a CEILING, not a prediction -- the two
#   real participants on v0 named request type 2/2 and think-aloud 0/2 -- so the pretest
#   ranks variants, it does not estimate what humans will notice.
_A_TERSE = ["What do you think would be the best move right now?", "And now?", "Now?"]
_A_REPEAT = ["Which move would be the best one and why", "What about next move?",
             "what about 6", "so which one would be the best move?", "What about now?",
             "Give me the best move"]
_A_LONG = ["Analyze the current Connect Four board carefully. I am Red and need to "
           "choose the best first three columns to keep Red on track to force a win. "
           "Please identify the strongest move now, explain the key threat it creates, "
           "and give the best follow-up columns after the opponent responds. Use "
           "1-indexed column numbers."]
_B_CANDIDATE = ["I am about to move into column 3. will this work",
                "what would you suggest",
                "so just block up column 4 here. what's best to win with 2 moves left",
                "ok what next", "any risks to column 6"]
_B_PLAN = ["Where is the biggest threat from Red?",
           "But if I let Red place in that location, I can put a disc on top of it in 5, "
           "then 2 and 3 gets me a win."]
_B_RETRO = ["Was my previous move effective?",
            "That worked well to block the immediate threat - it felt like the only "
            "option, am I right?"]
_B_THINK1 = ["I was thinking of putting one in 3-3",
             "it would prevent yellow and open up one opportuhity for me",
             "what is the best next move"]
_B_THINK2 = ["I was considering column 3, is that a good option?", "so column 4 first?",
             "is it better to focus on blocking or building my own win?"]

# --- v9 blocks: Group B attaches a REASON to each move ------------------------
# _A_OUTSOURCE replaces the long elaborate prompt used in v0/v2. Participants' own
# observations showed that one backfiring -- three of four WRONG forced choices cited
# Group A's detail as a virtue ("quite in-depth instructions and are very detailed",
# "a full single prompt to get the AI to lay out a plan"). This one is equally long but
# ends "Reply with only the column number", so the outsourcing is unmistakable.
_A_OUTSOURCE = [
    "Analyze the current board and tell me the best sequence of my next three moves "
    "for Red. Please list only the column numbers in order and briefly explain why "
    "they are optimal.",
    "Given Yellow's last move, what is now the best second move for Red? Reply with "
    "only the column number.",
]
# Both are PREFIXES of real conversations, not curated selections.
_B_REASON_MID = [
    "I\u2019m considering row 3 because I think that gives me a chance to win "
    "horizontally or diagonally. Is that the best move?",
    "Thanks! Should I do column 3 now? It gives me a chance to win diagonally going "
    "from the bottom right up too",
]
_B_REASON_SHORT = [
    "If I add to column 2, will that reduce my possibilities rather than improve my "
    "chances?",
    "How about column 5, then?",
    "how about 4, blocking yellow's progress and adding to the diagonal?",
]

# Both variants keep Group A LONGER than Group B (v0: 100w/10 msgs vs 96w/9;
# v2: 100w/10 vs 86w/8). That direction matters: if B looked longer, the obvious
# reading of the contrast would be "write more", and phase-1 says length predicts
# transfer only through thinking aloud, dropping to nothing once it is controlled
# (b=+0.044, p=.85).
INTERVENTION_VARIANTS = {
    "v0_live": [(_A_REPEAT, _B_CANDIDATE), (_A_TERSE, _B_PLAN), (_A_LONG, _B_RETRO)],
    "v2_think_pair": [(_A_TERSE, _B_THINK1), (_A_REPEAT, _B_THINK2), (_A_LONG, _B_PLAN)],
    # v9: two rows, and Group B gives a REASON rather than only naming a candidate.
    # v0/v2 made candidate-naming the repeated feature and showed a justification once;
    # that is what participants copied -- scaffolded think-aloud turns carried a stated
    # reason in ~14% of cases against vanilla's ~25%, and the coached act did not
    # predict transfer while the spontaneous one did. Here 4 of 5 Group B messages
    # attach a consequence to the move. Group A stays LONGER (83 words vs 76) so
    # verbosity is not available as the perceived difference.
    "v9_two_rows": [(_A_OUTSOURCE, _B_REASON_MID), (_A_REPEAT, _B_REASON_SHORT)],
}
INTERVENTION_VERSION = os.environ.get("INTERVENTION_VERSION", "v0_live")
if INTERVENTION_VERSION not in INTERVENTION_VARIANTS:
    raise SystemExit(f"unknown INTERVENTION_VERSION {INTERVENTION_VERSION!r}; "
                     f"expected one of {sorted(INTERVENTION_VARIANTS)}")
INTERVENTION_PAIRS = INTERVENTION_VARIANTS[INTERVENTION_VERSION]

# The observation gate scores EFFORT, not correctness -- see intervention_filter.py.
# Scaffolded participants are selected for scoring BELOW the prompt-task threshold, so
# screening them on whether they identified the intended dimension would confound low
# effort with the low prompt-construction ability that defines the sample.
EFFORT_PASS_THRESHOLD = float(os.environ.get("EFFORT_PASS_THRESHOLD", "2.0"))
EFFORT_MIN_CHARS = int(os.environ.get("EFFORT_MIN_CHARS", "50"))
EFFORT_FAIL_OPEN = os.environ.get("EFFORT_FAIL_OPEN", "1") == "1"
SCREENOUT_EFFORT_CODE = os.environ.get("SCREENOUT_EFFORT_CODE", SCREENOUT_PROMPT_CODE)
# If EVERY judge errors out (or the panel is unavailable), let the participant
# through rather than strand them.
JUDGE_FAIL_OPEN = os.environ.get("JUDGE_FAIL_OPEN", "true").strip().lower() in (
    "1", "true", "yes", "on",
)

# Scenario shown to the participant (HTML; rendered with | safe).
# Text mirrors the original Google-form screener for consistency.
PROMPT_TASK_SCENARIO = (
    "<p><strong>Fictional Scenario:</strong></p>"
    "<ul>"
    "<li>Julie posts cooking videos to 1.2M followers who love her quick, healthy recipes.</li>"
    "<li>Over the next three years she wants to grow online and publish a cookbook "
    "(torn on traditional vs. self-publishing).</li>"
    "<li>She works four days weekly, limited time and budget.</li>"
    "</ul>"
    "<p>The goal for the AI agent here is to <strong>devise a concrete three-year plan</strong>.</p>"
)

# NOTE: the judge rubric now lives in prompt_filter.py (JUDGE_RUBRIC), the single
# source of truth used by the JudgeSuite panel.

# Likert scale, reused for the eligibility skill question and the post-study AI items.
LIKERT_OPTIONS = [
    ("1", "Strongly disagree"),
    ("2", "Disagree"),
    ("3", "Neutral"),
    ("4", "Agree"),
    ("5", "Strongly agree"),
]
LIKERT_VALUES = {v for v, _ in LIKERT_OPTIONS}

# Eligibility skill-rating prompt.
SKILL_PROMPTS = {
    "othello": "I consider myself a skilled Othello (Reversi) player.",
}
# Skill question accepts the Likert values PLUS an explicit "don't know the game"
# option, which does NOT screen the participant out (a non-player is eligible).
SKILL_DONT_KNOW = ("dont_know", "I do not know this game")
SKILL_VALUES = LIKERT_VALUES | {SKILL_DONT_KNOW[0]}

# How often the participant uses generative AI (collected on the eligibility
# page as a demographic; does NOT affect eligibility).
GENAI_USAGE_OPTIONS = [
    ("never", "Never"),
    ("less_than_monthly", "Less than monthly"),
    ("about_monthly", "About monthly"),
    ("few_times_month", "A few times a month"),
    ("weekly", "Weekly"),
    ("few_times_week", "A few times a week"),
    ("daily", "Daily"),
    ("several_times_day", "Several times a day"),
]
GENAI_USAGE_VALUES = {v for v, _ in GENAI_USAGE_OPTIONS}

# ---- Post-consent demographics (age + education + occupation) ---------------
# Accepted age range for participants.
AGE_MIN, AGE_MAX = 18, 120

EDUCATION_OPTIONS = [
    ("less_than_hs", "Less than high school (Grades 1-8 or no formal schooling)"),
    ("hs_incomplete", "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)"),
    ("hs_graduate", "High school graduate (Grade 12 with diploma or GED certificate)"),
    ("some_college", "Some college, no degree (includes some community college)"),
    ("associate", "Two-year associate degree from a college or university"),
    ("bachelors", "Four year college or university degree/Bachelor's degree (e.g., BS, BA, AB)"),
    ("some_postgrad", "Some postgraduate or professional schooling, no postgraduate degree "
                      "(e.g. some graduate school)"),
    ("postgrad", "Postgraduate or professional degree, including master's, doctorate, medical "
                 "or law degree (e.g., MA, MS, PhD, JD, graduate school)"),
]
EDUCATION_VALUES = {v for v, _ in EDUCATION_OPTIONS}

OCCUPATION_OPTIONS = [
    ("hospitality", "Hospitality or service"),
    ("healthcare", "Health care and social assistance"),
    ("manufacturing", "Manufacturing, mining or construction"),
    ("retail", "Retail and trade"),
    ("education", "Education"),
    ("finance", "Banking, finance, accounting, real estate or insurance"),
    ("transportation", "Transportation"),
    ("government", "Government, public administration or military"),
    ("it", "Information/Technology"),
    ("agriculture", "Agriculture, forestry, fishing and hunting"),
    ("professional", "Professional, scientific and technical services"),
    ("arts", "Arts, entertainment and recreation"),
    ("other", "Other"),
]
OCCUPATION_VALUES = {v for v, _ in OCCUPATION_OPTIONS}

# ---- Post-study survey ------------------------------------------------------
# Page 1: NASA-TLX workload scale. Each item is rated 1 (Low) .. 7 (High).
NASA_TLX_ITEMS = [
    ("mental_demand", "Mental Demand",
     "How much mental and perceptual activity was required (e.g., thinking, "
     "deciding, calculating, remembering, looking, searching, etc)? Was the task "
     "easy or demanding, simple or complex, exacting or forgiving?"),
    ("temporal_demand", "Temporal Demand",
     "How much time pressure did you feel due to the rate or pace at which the "
     "task occurred? Was the pace slow and leisurely or rapid and frantic?"),
    ("performance", "Performance",
     "How successful do you think you were in accomplishing the goals of the task? "
     "How satisfied were you with your performance in accomplishing these goals?"),
    ("effort", "Effort",
     "How hard did you have to work (mentally and physically) to accomplish your "
     "level of performance?"),
    ("frustration", "Frustration",
     "How discouraged, stressed, irritated, and annoyed versus gratified, relaxed, "
     "content, and complacent did you feel during the task?"),
]
TLX_MIN, TLX_MAX = 1, 7
TLX_VALUES = {str(i) for i in range(TLX_MIN, TLX_MAX + 1)}

# Page 2: the AI-assistant assessment is now collected via an external survey
# form; participants are sent to this link, then return to continue the study.
POST_SURVEY_FORM_URL = os.environ.get("POST_SURVEY_FORM_URL", "")

# Set PROLIFIC_COMPLETION_URL to your study's completion link, e.g.
# https://app.prolific.com/submissions/complete?cc=XXXXXXXX
PROLIFIC_COMPLETION_URL = os.environ.get(
    "PROLIFIC_COMPLETION_URL",
    f"https://app.prolific.com/submissions/complete?cc={PROLIFIC_CODE}",
)

PROLIFIC_SCREENOUT_SKILL_URL = os.environ.get(
    "PROLIFIC_SCREENOUT_SKILL_URL",
    f"https://app.prolific.com/submissions/complete?cc={SCREENOUT_SKILL_CODE}",
)
PROLIFIC_SCREENOUT_PROMPT_URL = os.environ.get(
    "PROLIFIC_SCREENOUT_PROMPT_URL",
    f"https://app.prolific.com/submissions/complete?cc={SCREENOUT_PROMPT_CODE}",
)
PROLIFIC_SCREENOUT_INGAME_URL = os.environ.get(
    "PROLIFIC_SCREENOUT_INGAME_URL",
    f"https://app.prolific.com/submissions/complete?cc={SCREENOUT_INGAME_CODE}",
)

def _no_cache(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp


@app.route("/debug/oth-board.png")
def debug_oth_board():
    """Dev aid: view the exact Othello image the LLM is fed. Shows the live
    session board if there is one, else the starting position."""
    board = session.get("oth_board") or oth_engine.load_board(oth_puzzle_file(current_round()["puzzle"]))
    return _no_cache(Response(oth_engine.render_image(board, legal_for=BLACK),
                              mimetype="image/png"))


@app.route("/debug/oth-state")
def debug_oth_state():
    """Raw server-side Othello state, to confirm what the session holds."""
    board = session.get("oth_board")
    solution = OTH_SOLUTIONS[current_round()["puzzle"]]
    return _no_cache(jsonify({
        "has_session_board": board is not None,
        "decisions_made": len(session.get("oth_optimal_flags", [])),
        "score": sum(session.get("oth_optimal_flags", [])),
        "finished": session.get("oth_finished", False),
        "forced_moves": session.get("oth_forced_moves", []),
        # Answers, for debugging only — never sent to the participant's client.
        "best_moves": oth_black_best(board, solution) if board else None,
        "on_solution_table": bool(board) and
            oth_engine.state_key(board, BLACK) in solution,
        **(oth_board_state(board) if board else {"board": None}),
    }))


@app.route("/debug/oth-reset")
def debug_oth_reset():
    """Reset the session board + move state to the current puzzle's starting position."""
    rnd = current_round()
    board = oth_engine.load_board(oth_puzzle_file(rnd["puzzle"]))
    forced = oth_autoplay_forced(board, OTH_SOLUTIONS[rnd["puzzle"]])
    session["oth_board"] = board
    session["oth_optimal_flags"] = []
    session["oth_kept_flags"] = []
    session["oth_forced_moves"] = forced
    session["oth_started"] = True
    session["oth_finished"] = False
    return _no_cache(jsonify({"ok": True, **oth_board_state(board)}))


@app.route("/debug/reset-session")
def debug_reset_session():
    """Dev aid: wipe the entire session and start the flow over at consent
    (handy for re-testing without restarting the server or clearing cookies)."""
    session.clear()
    return _no_cache(redirect(url_for("consent")))


def require_participant():
    if not session.get("prolific_id"):
        return redirect(url_for("eligibility"))
    return None


def participant_dir(safe_pid: str) -> Path:
    p = RECORDINGS_DIR / safe_pid
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---- Stage 1: prompt judging ------------------------------------------------
# A single multi-model judge panel (Qwen + Llama + GPT, defined in
# prompt_filter.py) is built once at startup and reused for every prompt.
try:
    from prompt_filter import JudgeSuite
    _JUDGE_SUITE = JudgeSuite()
except Exception as _judge_init_err:        # dspy missing / LM construction failed
    _JUDGE_SUITE = None
    app.logger.warning("JudgeSuite unavailable: %s", _judge_init_err)

try:
    from intervention_filter import EffortSuite
    _EFFORT_SUITE = EffortSuite()
except Exception as _effort_init_err:                      # pragma: no cover
    _EFFORT_SUITE = None
    app.logger.warning("EffortSuite unavailable: %s", _effort_init_err)


def judge_observation(text: str, task: str = None) -> dict:
    """Effort verdict on the scaffolded arm's free-text observation.

    Same shape and median rule as judge_prompt, but `passed` means "engaged enough to
    keep", not "wrote a good answer". If the whole panel fails, EFFORT_FAIL_OPEN decides
    -- keeping the participant, because a model outage is not evidence of low effort.
    """
    from intervention_filter import TASK_SCAFFOLDED
    task = task or TASK_SCAFFOLDED
    verdict = {"judges": [], "n_judges": 0, "mean_score": None, "median_score": None,
               "threshold": EFFORT_PASS_THRESHOLD, "passed": EFFORT_FAIL_OPEN}
    if _EFFORT_SUITE is None:
        verdict["error"] = "effort suite unavailable"
        return verdict
    try:
        mean_score, scores = _EFFORT_SUITE(response=text, task=task)
    except Exception as e:
        app.logger.warning("effort judging failed: %s", e)
        verdict["error"] = str(e)
        return verdict
    valid = [x for x in scores.values() if x is not None]
    verdict["judges"] = [{"model": n, "score": x} for n, x in scores.items()]
    verdict["n_judges"] = len(valid)
    verdict["mean_score"] = round(mean_score, 3) if mean_score is not None else None
    verdict["median_score"] = statistics.median(valid) if valid else None
    verdict["passed"] = (bool(verdict["median_score"] >= EFFORT_PASS_THRESHOLD)
                         if valid else EFFORT_FAIL_OPEN)
    return verdict



_PHASE1_VERDICTS = {}
if SKIP_PROMPT_TASK:
    try:
        _PHASE1_VERDICTS = json.load(open(BASE_DIR / "returning_participants.json"))
        app.logger.info("loaded %d phase-1 verdicts", len(_PHASE1_VERDICTS))
    except Exception as e:                                  # pragma: no cover
        app.logger.warning("returning_participants.json unavailable: %s", e)


def _phase1_verdict(prolific_id: str) -> dict:
    """The participant's phase-1 prompt verdict, or a stub if they are not on the list.

    A stub rather than a rejection: recruitment is controlled by the Prolific allowlist,
    so anyone reaching this deployment is meant to be here. Someone off-list is a
    recruitment slip, not a participant to screen out mid-session -- it is recorded and
    sorted out in analysis.
    """
    v = dict(_PHASE1_VERDICTS.get(prolific_id) or {})
    v.setdefault("source", "phase_1_carryover_unmatched")
    v["passed"] = True
    v["rule"] = "carried over from phase 1"
    v["matched_phase1"] = prolific_id in _PHASE1_VERDICTS
    return v


def judge_prompt(prompt_text: str) -> dict:
    """Score a prompt with the multi-model judge panel.

    Pass/fail is decided on the MEDIAN of the per-model 1-4 scores (robust to a
    single dissenting model and stable against the quantized-mean boundary).
    Mean is still recorded for reference. Models that error are dropped; if ALL
    fail (or the panel is unavailable), JUDGE_FAIL_OPEN decides.
    """
    verdict = {
        "judges": [],
        "n_judges": 0,
        "mean_score": None,
        "median_score": None,
        "threshold": JUDGE_PASS_THRESHOLD,
        "passed": JUDGE_FAIL_OPEN,
    }
    if _JUDGE_SUITE is None:
        verdict["error"] = "judge suite unavailable"
        return verdict
    try:
        mean_score, scores = _JUDGE_SUITE(prompt=prompt_text)
    except Exception as e:
        app.logger.warning("judging failed: %s", e)
        verdict["error"] = str(e)
        return verdict

    valid = [s for s in scores.values() if s is not None]
    verdict["judges"] = [{"model": name, "score": s} for name, s in scores.items()]
    verdict["n_judges"] = len(valid)
    verdict["mean_score"] = round(mean_score, 3) if mean_score is not None else None
    verdict["median_score"] = statistics.median(valid) if valid else None
    # Both statistics are always recorded, so a rule change mid-collection stays
    # auditable and either rule can be re-applied to already-collected participants.
    verdict["rule"] = JUDGE_PASS_RULE
    decisive = (verdict["mean_score"] if JUDGE_PASS_RULE == "mean"
                else verdict["median_score"])
    verdict["decisive_score"] = decisive
    verdict["passed"] = (
        bool(decisive < JUDGE_PASS_THRESHOLD) if valid else JUDGE_FAIL_OPEN
    )
    return verdict


def load_system_prompt(game_type: str) -> str:
    path = GAMES_DIR / game_type / "system_prompt.txt"
    try:
        # The board is sent as an image per-message in the chat route, so the
        # system prompt carries no board text.
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return DEFAULT_SYSTEM_PROMPT


def _solution_hint():
    """Coaching aid for the assistant: the optimal move(s) from the CURRENT
    position, in board notation ("d3").

    On the precomputed optimal line these come from the solution table; once the
    participant deviates the engine supplies the best remaining move(s) — the
    SAME set their move is scored against, so following the advice always earns
    credit and the assistant never goes silent after a mistake. Coaches toward
    the move without revealing it. Returns None only when there is no board.

    DRAFT wording — this is study copy; review it before running participants.
    """
    board = session.get("oth_board")
    if not board:
        return None
    solution = OTH_SOLUTIONS[current_round()["puzzle"]]
    on_table = oth_engine.state_key(board, BLACK) in solution
    best = oth_black_best(board, solution)
    if not best:
        return None
    best_str = ", ".join(best)
    lead = ("COACHING AID — the move(s) that keep Black's best result from the current "
            f"position are {best_str}."
            if on_table else
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



def _eligibility_kwargs(prefill, error):
    game = session.get("game", DEFAULT_GAME)
    return {
        "prefill": prefill,
        "error": error,
        "game": game,
        "likert_options": LIKERT_OPTIONS,
        "skill_prompt": SKILL_PROMPTS.get(game, "I am a skilled player of this game."),
        "skill_dont_know": SKILL_DONT_KNOW,
    }


@app.route("/eligibility", methods=["GET", "POST"])
def eligibility():
    """Eligibility (after consent): Prolific ID + skill rating. Skilled Othello
    players (skill 4-5) are screened out HERE (screen-out code #1). The skill
    question also offers "I do not know this game", which is eligible.
    (GenAI usage is now asked on the post-study survey.)"""
    if not session.get("consented"):
        return redirect(url_for("consent"))

    if request.method == "POST":
        prolific_id = (request.form.get("prolific_id") or "").strip()
        skill_rating = (request.form.get("skill_rating") or "").strip()

        prefill = {
            "prolific_id": prolific_id,
            "skill_rating": skill_rating,
            "study_id": request.form.get("study_id", ""),
            "session_id": request.form.get("session_id", ""),
        }

        def _err(msg):
            return render_template("eligibility.html", **_eligibility_kwargs(prefill, msg))

        if not prolific_id:
            return _err("Please enter your Prolific ID to continue.")
        if skill_rating not in SKILL_VALUES:
            return _err("Please rate how skilled a player you are.")

        # Determine the experimental condition
        # session["condition"] = random.choice(["vanilla", "scaffolded"])
        session["condition"] = "scaffolded"

        safe_pid = "".join(c for c in prolific_id if c.isalnum() or c in "-_") or "anon"
        session["prolific_id"] = prolific_id
        session["safe_pid"] = safe_pid
        session["study_id"] = prefill["study_id"]
        session["session_id"] = prefill["session_id"]
        demographics = {
            "skill_rating": skill_rating,          # "1".."5" or "dont_know"
        }
        session["demographics"] = demographics

        # Persist a first record now so screened-out participants are still logged.
        pdir = participant_dir(safe_pid)
        with open(pdir / "demographics.json", "w") as f:
            json.dump(
                {
                    "prolific_id": prolific_id,
                    "game": DEFAULT_GAME,
                    "rounds": OTH_ROUNDS,
                    **demographics,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                },
                f,
                indent=2,
            )

        # Screen-out #1: skilled Othello players (skill 4-5) do not continue.
        # "dont_know" and 1-3 are eligible.
        if skill_rating in ("4", "5"):
            session["screened"] = "skill"
            return redirect(url_for("screened_thank_you_skill"))

        if SKIP_PROMPT_TASK:
            # Pre-qualified cohort: eligibility is already established from phase 1.
            session["prompt_passed"] = True
            carry = _phase1_verdict(prolific_id)
            with open(participant_dir(safe_pid) / "prompt_task.json", "w") as f:
                json.dump(carry, f, indent=2)
            return redirect(url_for("intervention" if session["condition"] == "scaffolded"
                                    else "filler"))

        return redirect(url_for("prompt_task"))

    prefill = {
        "prolific_id": request.args.get("PROLIFIC_PID", ""),
        "skill_rating": "",
        "study_id": request.args.get("STUDY_ID", ""),
        "session_id": request.args.get("SESSION_ID", ""),
    }
    return render_template("eligibility.html", **_eligibility_kwargs(prefill, None))


@app.route("/", methods=["GET", "POST"])
def consent():
    """Entry point. Consent comes FIRST — no participant data (not even the
    Prolific ID) is collected before consent. On agreement, continue to the
    eligibility page."""
    session["game"] = DEFAULT_GAME
    topic_readable = "Othello"
    if request.method == "POST":
        if not request.form.get("consent"):
            return render_template(
                "consent.html",
                error="Please confirm that you consent to participate before continuing.",
                topic=topic_readable,
            )
        session["consented"] = True
        session["consented_at"] = datetime.now(timezone.utc).isoformat()
        return redirect(url_for("eligibility"))
    return render_template("consent.html", error=None, topic=topic_readable)


@app.route("/demographics", methods=["GET", "POST"])
def demographics():
    """Final demographics (after the post-survey): age + education + occupation."""
    if not session.get("prolific_id"):
        return redirect(url_for("eligibility"))
    if not session.get("consented"):
        return redirect(url_for("consent"))
    if not session.get("prompt_passed"):     # can't reach the end without clearing the filter
        return redirect(url_for("prompt_task"))

    if request.method == "POST":
        age_raw = (request.form.get("age") or "").strip()
        education = (request.form.get("education") or "").strip()
        occupation = (request.form.get("occupation") or "").strip()
        occupation_other = (request.form.get("occupation_other") or "").strip()
        genai_usage = (request.form.get("genai_usage") or "").strip()

        prefill = {
            "age": age_raw,
            "education": education,
            "occupation": occupation,
            "occupation_other": occupation_other,
            "genai_usage": genai_usage,
        }

        def _err(msg):
            return render_template(
                "demographics.html",
                prefill=prefill,
                error=msg,
                education_options=EDUCATION_OPTIONS,
                occupation_options=OCCUPATION_OPTIONS,
                genai_usage_options=GENAI_USAGE_OPTIONS,
                age_min=AGE_MIN,
                age_max=AGE_MAX,
            )

        age = int(age_raw) if age_raw.isdigit() else None
        if age is None or not (AGE_MIN <= age <= AGE_MAX):
            return _err(f"Please enter a valid age between {AGE_MIN} and {AGE_MAX}.")
        if education not in EDUCATION_VALUES:
            return _err("Please select your highest level of education.")
        if occupation not in OCCUPATION_VALUES:
            return _err("Please select your industry or field.")
        if genai_usage not in GENAI_USAGE_VALUES:
            return _err("Please tell us how often you use generative AI.")

        demo = session.get("demographics", {})
        demo["age"] = age
        demo["education"] = education
        demo["occupation"] = occupation
        if occupation == "other":
            demo["occupation_other"] = occupation_other
        demo["genai_usage"] = genai_usage
        session["demographics"] = demo

        # Rewrite demographics.json with the full set (page 1 + this page).
        safe_pid = session.get("safe_pid", "anon")
        with open(participant_dir(safe_pid) / "demographics.json", "w") as f:
            json.dump(
                {
                    "prolific_id": session.get("prolific_id"),
                    "game": session.get("game", DEFAULT_GAME),
                    "rounds": OTH_ROUNDS,
                    **demo,
                    "consented_at": session.get("consented_at"),
                    "submitted_at": datetime.now(timezone.utc).isoformat(),
                },
                f,
                indent=2,
            )
        return redirect(url_for("thank_you"))

    prefill = {"age": "", "education": "", "occupation": "", "occupation_other": "", "genai_usage": ""}
    return render_template(
        "demographics.html",
        prefill=prefill,
        error=None,
        education_options=EDUCATION_OPTIONS,
        occupation_options=OCCUPATION_OPTIONS,
        genai_usage_options=GENAI_USAGE_OPTIONS,
        age_min=AGE_MIN,
        age_max=AGE_MAX,
    )

@app.route("/prompt-task", methods=["GET", "POST"])
def prompt_task():
    """Stage 1: the participant writes a prompt for a fixed scenario; a panel of
    LLM judges scores it 1-4 and the average decides whether they continue into
    the game (pass) or are screened out (fail)."""
    gate = require_participant()
    if gate:
        return gate
    if not session.get("consented"):         # consent precedes the prompt task
        return redirect(url_for("consent"))

    if request.method == "POST":
        created = (request.form.get("created_prompt") or "").strip()
        if len(created) < JUDGE_MIN_PROMPT_CHARS:
            return render_template(
                "prompt_task.html",
                scenario=PROMPT_TASK_SCENARIO,
                prefill=created,
                error=f"Please write a more complete prompt (at least {JUDGE_MIN_PROMPT_CHARS} characters).",
            )

        verdict = judge_prompt(created)
        safe_pid = session.get("safe_pid", "anon")
        with open(participant_dir(safe_pid) / "prompt_task.json", "w") as f:
            json.dump(
                {
                    "created_prompt": created,
                    **verdict,
                    "submitted_at": datetime.now(timezone.utc).isoformat(),
                },
                f,
                indent=2,
            )
        session["prompt_passed"] = verdict["passed"]
        if verdict["passed"]:
            if session["condition"] == "scaffolded":
                return redirect(url_for("intervention"))
            return redirect(url_for("filler"))
        # Screen-out #2: low-quality prompt.
        session["screened"] = "prompt"
        return redirect(url_for("screened_thank_you_prompt"))

    return render_template(
        "prompt_task.html", scenario=PROMPT_TASK_SCENARIO, prefill="", error=None
    )


@app.route("/filler", methods=["GET", "POST"])
def filler():
    """Vanilla arm's matched task. Exists so the two arms are screened IDENTICALLY.

    Without it the scaffolded arm clears two filters (prompt task + effort) and vanilla
    clears one, so any effect is confounded with having removed the low-effort
    participants from one arm only -- and on Prolific those slots are refilled, so the
    samples genuinely differ rather than merely shrinking.

    The content is deliberately inert: it revisits the warm-up scenario, which touches
    neither Othello strategy nor how to talk to an assistant. Anything about either
    would make this a competing intervention rather than a control.
    """
    gate = require_participant()
    if gate:
        return gate
    if session.get("condition") != "vanilla":
        return redirect(url_for("primer"))
    if not session.get("prompt_passed"):
        return redirect(url_for("prompt_task"))

    if request.method == "POST":
        from intervention_filter import TASK_VANILLA
        reflection = (request.form.get("reflection") or "").strip()
        if len(reflection) < EFFORT_MIN_CHARS:
            return render_template("filler.html", scenario=PROMPT_TASK_SCENARIO,
                                   min_chars=EFFORT_MIN_CHARS, prefill=reflection,
                                   error=f"Please write at least {EFFORT_MIN_CHARS} "
                                         f"characters.")
        verdict = judge_observation(reflection, task=TASK_VANILLA)
        safe_pid = session.get("safe_pid", "anon")
        with open(participant_dir(safe_pid) / "filler.json", "w") as f:
            json.dump({"reflection": reflection, "effort": verdict,
                       "submitted_at": datetime.now(timezone.utc).isoformat()},
                      f, indent=2)
        if not verdict["passed"]:
            session["screened"] = "effort"
            return redirect(url_for("screened_thank_you_effort"))
        session["filler_done"] = True
        return redirect(url_for("primer"))

    return render_template("filler.html", scenario=PROMPT_TASK_SCENARIO,
                           min_chars=EFFORT_MIN_CHARS, prefill="", error=None)


@app.route("/intervention", methods=["GET", "POST"])
def intervention():
    """Scaffolded arm only: contrasting cases, free-text observation, forced choice.

    Order matters. The participant writes what they noticed BEFORE being asked which
    group did better, so the observation is their own reading rather than a
    rationalisation of the answer they just picked.
    """
    gate = require_participant()
    if gate:
        return gate
    if session.get("condition") != "scaffolded":
        return redirect(url_for("primer"))
    if not session.get("prompt_passed"):
        return redirect(url_for("prompt_task"))

    if request.method == "POST":
        observation = (request.form.get("observation") or "").strip()
        choice = (request.form.get("choice") or "").strip().upper()

        def _err(msg):
            return render_template("intervention.html", pairs=INTERVENTION_PAIRS,
                                   min_chars=EFFORT_MIN_CHARS, prefill=observation,
                                   error=msg)

        if len(observation) < EFFORT_MIN_CHARS:
            return _err(f"Please write at least {EFFORT_MIN_CHARS} characters about how "
                        f"the two groups differ.")
        if choice not in ("A", "B"):
            return _err("Please choose which group you think plays better.")

        verdict = judge_observation(observation)
        safe_pid = session.get("safe_pid", "anon")
        with open(participant_dir(safe_pid) / "intervention.json", "w") as f:
            json.dump({"variant": INTERVENTION_VERSION,
                       "observation": observation,
                       "choice": choice,
                       "choice_correct": choice == "B",
                       "effort": verdict,
                       "submitted_at": datetime.now(timezone.utc).isoformat()},
                      f, indent=2)

        if not verdict["passed"]:
            # Screen-out #4: no genuine engagement with the intervention. Distinct from
            # the prompt screen-out, which is about ability; this one is about effort.
            session["screened"] = "effort"
            return redirect(url_for("screened_thank_you_effort"))

        session["intervention_done"] = True
        session["intervention_choice"] = choice
        return render_template("intervention_feedback.html", correct=choice == "B")

    return render_template("intervention.html", pairs=INTERVENTION_PAIRS,
                           min_chars=EFFORT_MIN_CHARS, prefill="", error=None)


@app.route("/intervention/continue", methods=["POST"])
def intervention_continue():
    """Leaves the feedback page. Separate route so the correction cannot be skipped
    by navigating straight to /primer -- primer checks intervention_done."""
    gate = require_participant()
    if gate:
        return gate
    if session.get("condition") == "scaffolded" and not session.get("intervention_done"):
        return redirect(url_for("intervention"))
    if session.get("condition") == "vanilla" and not session.get("filler_done"):
        return redirect(url_for("filler"))
    return redirect(url_for("primer"))


@app.route("/screened-effort", methods=["GET", "POST"])
def screened_thank_you_effort():
    """Screen-out #4: did not engage with the intervention."""
    return render_template("thank_you.html",
                           prolific_url=f"https://app.prolific.com/submissions/"
                                        f"complete?cc={SCREENOUT_EFFORT_CODE}")


@app.route("/primer")
def primer():
    gate = require_participant()
    if gate:
        return gate
    # Scaffolded participants must complete the intervention first; otherwise the arm
    # is only nominal and /primer is reachable by typing the URL.
    if session.get("condition") == "scaffolded" and not session.get("intervention_done"):
        return redirect(url_for("intervention"))
    if session.get("condition") == "vanilla" and not session.get("filler_done"):
        return redirect(url_for("filler"))
    # Must have cleared the Stage 1 prompt filter to reach the game.
    if not session.get("prompt_passed"):
        return redirect(url_for("prompt_task"))
    # Standardized rules primer shown before the timed game. The timer only
    # starts on /study, so reading this does not count against the participant.
    return render_template("othello_primer.html", game=session.get("game", DEFAULT_GAME))


@app.route("/study")
def study():
    gate = require_participant()
    if gate:
        return gate
    if not session.get("prompt_passed"):     # cannot skip the Stage 1 filter
        return redirect(url_for("prompt_task"))
    game = session.get("game", DEFAULT_GAME)
    rnd = current_round()

    safe_pid = session.get("safe_pid", "anon")

    # The round is over -> can't replay it by refreshing or navigating back.
    # Exception: a finished game with the assistant-turn requirement still
    # outstanding keeps rendering (board locked) so they can finish talking to
    # the assistant. post_survey sends them back here until it's met.
    if session.get("oth_timed_out"):
        return redirect(url_for("round_done"))
    if session.get("oth_finished") and ai_turn_requirement()["met"]:
        return redirect(url_for("round_done"))

    if not session.get("oth_started"):
        # First entry to THIS round only: fresh board + fresh logs. Refreshing
        # afterward will NOT re-initialize, so a refresh can't reset progress or
        # stack up moves. (Round state is cleared when advancing to the next puzzle.)
        board = oth_engine.load_board(oth_puzzle_file(rnd["puzzle"]))
        # A puzzle could in principle open with only one legal square; that is
        # not a decision, so play it before the participant ever sees the board.
        session["oth_forced_moves"] = oth_autoplay_forced(board, OTH_SOLUTIONS[rnd["puzzle"]])
        session["oth_board"] = board
        # One flag per SCORED decision (forced moves are excluded from both).
        session["oth_optimal_flags"] = []
        session["oth_kept_flags"] = []
        session["oth_started"] = True
        # Two-tries state (rounds with "attempts" > 1). Best attempt is kept.
        session["oth_attempt"] = 1
        session["oth_best"] = None
        session["oth_attempts"] = []
        session["oth_attempt_done"] = False
        pdir = participant_dir(safe_pid)
        round_path(pdir, "moves", "jsonl").unlink(missing_ok=True)
        round_path(pdir, "conversation", "jsonl").unlink(missing_ok=True)
    else:
        board = session.get("oth_board")          # resume current state on refresh

    decisions_made = len(session.get("oth_optimal_flags", []))
    ai_now = ai_enabled()
    # gate_active: this is the gated round and the gate hasn't been passed yet,
    # so the confidence popup may still fire (after the first move) — render its
    # markup + JS. gate_pending: the first move was already made (e.g. the
    # participant refreshed mid-gate), so show the popup immediately on load.
    gate_active = bool(rnd.get("gate")) and not session.get("ai_unlocked")
    gate_pending = gate_active and bool(session.get("oth_gate_asked"))
    # Intro popup for a multi-attempt round (round 2): shown once on fresh entry,
    # before any move — a heads-up that there's no AI and no AI survey after.
    show_round_intro = (bool(rnd.get("intro"))
                        and decisions_made == 0
                        and not session.get("oth_finished"))
    black_discs, white_discs = oth_engine.disc_counts(board)
    # The solution table stays server-side; the client only gets the board and
    # the squares it may play — never anything about which of them is good.
    return render_template(
        "othello.html",
        game=game,
        board=board,                 # board[row][col], row 0 = TOP
        rows=oth_engine.ROWS,
        cols=oth_engine.COLS,
        legal_moves=oth_legal_moves(board),
        black_discs=black_discs,
        white_discs=white_discs,
        empties=oth_engine.empty_count(board),
        game_over=oth_engine.is_game_over(board),
        time_limit_seconds=rnd["time_limit"],
        ai_assistant_enabled=ai_now,     # visible/usable right now
        ai_turns_taken=ai_turns_taken() if rnd["ai"] else 0,
        # Round-level, not live: in the gated round the assistant is still locked
        # at render time, and the client only enforces this once it unlocks —
        # which is exactly when the server starts enforcing it too.
        min_ai_turns=(OTH_MIN_AI_TURNS if rnd["ai"] else 0),
        render_chat=(ai_now or gate_active),  # include chat DOM+JS (hidden until unlocked)
        gate_active=gate_active,
        gate_pending=gate_pending,
        confidence_options=CONFIDENCE_OPTIONS,
        show_round_intro=show_round_intro,
        round_intro=rnd.get("intro"),
    )


@app.route("/api/oth-move", methods=["POST"])
def oth_move():
    """Othello: the participant plays a square (Black).

    The puzzle is played TO COMPLETION: every legal move is played, the AI
    (White) replies, and the round ends when neither side can move. The
    participant is NEVER told whether a move was optimal — the live disc count is
    the only feedback. We silently record each decision's optimality; the score
    is how many of their decisions were optimal. On the precomputed line the AI
    plays its tabulated best defense; once the participant deviates, the AI falls
    back to the engine — which is itself exact at endgame sizes, so its defense
    stays unimprovable either way.

    Two Othello-specific wrinkles: a move is a square ("d3"), and either side can
    be forced to pass. White keeps replying until Black has a legal move again,
    and a position where Black has only ONE legal square is auto-played rather
    than asked for — it is not a decision, so it is neither scored nor timed.
    """
    if not session.get("prolific_id"):
        return jsonify({"error": "not authorized"}), 401

    rnd = current_round()
    solution = OTH_SOLUTIONS[rnd["puzzle"]]
    board = session.get("oth_board")
    optimal_flags = list(session.get("oth_optimal_flags", []))
    kept_flags = list(session.get("oth_kept_flags", []))
    if board is None:
        return jsonify({"error": "no active game"}), 400
    if session.get("oth_finished") or oth_engine.is_game_over(board):
        return jsonify({"ok": False, "reason": "game_over", "done": True})
    # Gated round: once the unaided first move is made, no further moves until the
    # confidence gate is resolved (either screen-out or board-reset + AI unlock).
    if rnd.get("gate") and session.get("oth_gate_asked") and not session.get("ai_unlocked"):
        return jsonify({"ok": False, "reason": "awaiting_gate"})

    data = request.get_json(silent=True) or {}
    try:
        move = oth_engine.parse_move(data.get("move"))
    except (ValueError, KeyError, TypeError):
        move = None
    if move is None or not oth_engine.is_valid_move(board, move, BLACK):
        # Not a legal square — no move is spent. This is a rules message, not a
        # quality one: the client shows legal squares, so it should never happen.
        return jsonify({"ok": False, "reason": "illegal_move",
                        "legal_moves": oth_legal_moves(board)})
    played = oth_engine.to_notation(move)

    # Score the move BEFORE playing it.
    #   best_moves = the exact optimal move(s) from THIS position (table or engine)
    #   values     = every legal move's exact final disc margin, from Black's POV
    # optimal credits the best move from wherever they are; kept_win is the
    # stricter "are they still winning" flag, and disc_loss is the continuous
    # cost — all three recorded per move for trajectory analysis.
    best_moves, values = oth_grade(board, played, solution)
    optimal = played in best_moves
    played_value = values.get(played)
    best_value = max(values.values()) if values else None
    disc_loss = (None if played_value is None or best_value is None
                 else best_value - played_value)
    kept_win = bool(played_value is not None and played_value > 0)

    oth_engine.apply_move(board, move, BLACK)      # the disc ALWAYS lands
    optimal_flags.append(optimal)
    kept_flags.append(kept_win)
    decisions_made = len(optimal_flags)

    # White replies (and keeps replying while Black has to pass), then any
    # single-legal-square positions are played out for the participant.
    ai_moves, black_passed, white_passed = oth_ai_turns(board, solution)
    forced_moves = oth_autoplay_forced(board, solution)
    session["oth_forced_moves"] = session.get("oth_forced_moves", []) + forced_moves

    # Gated round: the first decision is an unaided probe. Record its optimality
    # and ask for confidence before any further play (the client shows a popup).
    gate_needed = (rnd.get("gate") and decisions_made == 1
                   and not session.get("ai_unlocked")
                   and not session.get("oth_gate_asked"))
    if gate_needed:
        session["oth_gate_optimal_first"] = optimal
        session["oth_gate_asked"] = True

    session["oth_board"] = board
    session["oth_optimal_flags"] = optimal_flags
    session["oth_kept_flags"] = kept_flags

    # Attempt / two-tries handling. A round may allow multiple attempts
    # ("attempts" in OTH_ROUNDS); if this attempt ends unsolved and retries
    # remain, the round is NOT finished — the client offers one more try.
    attempt = session.get("oth_attempt", 1)
    max_attempts = rnd.get("attempts", 1)
    # The attempt ends when the GAME does — there is no move quota any more.
    game_over = oth_engine.is_game_over(board)
    attempt_done = game_over
    scores = oth_round_scores(optimal_flags, kept_flags)
    # "Solved" = played the whole puzzle without a single suboptimal decision.
    # Scored against the decisions actually offered, which varies with how the
    # participant played (passes shift), so it can't be a fixed denominator.
    solved = attempt_done and scores["score"] == scores["decisions"]
    retry_available = False
    round_done = False
    if attempt_done:
        if attempt < max_attempts and not solved:
            # Another try remains: bank this attempt, don't close the round.
            attempts = session.get("oth_attempts", [])
            attempts.append({**scores, "attempt": attempt, "solved": solved,
                             "completed": True, "end_reason": "game_over"})
            session["oth_attempts"] = attempts
            session["oth_best"] = max(attempts, key=lambda a: (a["score"], a["score_first_n"]))
            retry_available = True
            session["oth_attempt_done"] = True       # this attempt over, awaiting retry
        else:
            round_done = True

    safe_pid = session.get("safe_pid", "anon")
    pdir = participant_dir(safe_pid)
    black_discs, white_discs = oth_engine.disc_counts(board)
    record = {
        "game": DEFAULT_GAME,
        "puzzle": rnd["puzzle"],
        "round": current_round_index(),
        "attempt": attempt,
        "decision_number": decisions_made,   # scored decisions only (forced moves excluded)
        "move": played,                   # algebraic square, e.g. "d3"
        "optimal": optimal,               # best move from the current position (option-2 scoring)
        "kept_win": kept_win,             # still winning after this move (stricter metric)
        "disc_loss": disc_loss,           # discs given up vs. the best move (the cp_loss analog)
        "played_value": played_value,     # exact final margin this move leads to
        "best_value": best_value,         # exact final margin the best move leads to
        "best_moves": best_moves,         # the set 'optimal' is scored against
        "ai_moves": ai_moves,             # White's reply/replies (more than one iff Black passed)
        "black_passed": black_passed,     # Black had no legal move and had to pass
        "white_passed": white_passed,     # White had no legal move and had to pass
        "forced_moves": forced_moves,     # squares auto-played next (only one was legal)
        "black_discs": black_discs,
        "white_discs": white_discs,
        "game_over": game_over,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    with open(round_path(pdir, "moves", "jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")
    if round_done:
        oth_finish_round("game_over", completed=True)

    # NOTE: response deliberately omits per-move optimality so the client can
    # never reveal correctness to the participant. "gate" asks the client to
    # collect confidence before continuing (carries no correctness signal).
    # "retry_available" DOES imply this attempt wasn't a full solve (the study
    # intentionally offers a retry only on a non-solved attempt).
    return jsonify({
        "ok": True,
        **oth_board_state(board),
        "played": played,
        "ai_moves": ai_moves,             # squares White answered with, in order
        "forced_moves": forced_moves,     # squares played for them (only one was legal)
        "black_passed": black_passed,     # the client explains why its turn was skipped
        "white_passed": white_passed,
        "decisions_made": decisions_made,
        "done": attempt_done,
        "retry_available": retry_available,
        "gate": bool(gate_needed),
    })


@app.route("/api/oth-retry", methods=["POST"])
def oth_retry():
    """Start the next attempt of a multi-try round (round 2): reset the board to
    the puzzle start. Allowed only when the current attempt has ended unsolved
    and retries remain. Prior attempt's moves stay logged (each carries its
    'attempt' number); the round score is the best across attempts."""
    if not session.get("prolific_id"):
        return jsonify({"error": "not authorized"}), 401
    rnd = current_round()
    attempt = session.get("oth_attempt", 1)
    if session.get("oth_finished"):
        return jsonify({"ok": False, "reason": "round_over"})
    if not session.get("oth_attempt_done"):
        return jsonify({"ok": False, "reason": "attempt_not_done"}), 400
    if attempt >= rnd.get("attempts", 1):
        return jsonify({"ok": False, "reason": "no_retries_left"}), 400

    board = oth_engine.load_board(oth_puzzle_file(rnd["puzzle"]))
    session["oth_forced_moves"] = oth_autoplay_forced(board, OTH_SOLUTIONS[rnd["puzzle"]])
    session["oth_board"] = board
    session["oth_optimal_flags"] = []
    session["oth_kept_flags"] = []
    session["oth_attempt"] = attempt + 1
    session["oth_attempt_done"] = False
    return jsonify({
        "ok": True,
        **oth_board_state(board),
        "decisions_made": 0,
        "attempt": attempt + 1,
    })


@app.route("/api/gate", methods=["POST"])
def oth_gate():
    """Resolve the gated round's confidence gate (submitted after the unaided
    first move). Optimal first move + high confidence -> screen out. Otherwise
    unlock the AI: a WRONG first move resets the board for a clean AI-assisted
    attempt; a CORRECT first move keeps its progress and continues from there.
    The probe move + confidence are logged either way."""
    if not session.get("prolific_id"):
        return jsonify({"error": "not authorized"}), 401
    rnd = current_round()
    if not rnd.get("gate"):
        return jsonify({"error": "no gate for this round"}), 400
    if session.get("ai_unlocked"):                 # already resolved (idempotent)
        return jsonify({"ok": True, "unlocked": True})
    if not session.get("oth_gate_asked"):           # first move not made yet
        return jsonify({"ok": False, "reason": "gate_not_ready"}), 400

    data = request.get_json(silent=True) or {}
    conf = str(data.get("confidence", "")).strip()
    if conf not in CONFIDENCE_VALUES:
        return jsonify({"ok": False, "reason": "invalid_confidence"}), 400

    optimal_first = bool(session.get("oth_gate_optimal_first"))
    high_conf = conf in CONFIDENCE_HIGH
    screenout = optimal_first and high_conf

    safe_pid = session.get("safe_pid", "anon")
    pdir = participant_dir(safe_pid)
    with open(round_path(pdir, "gate", "json"), "w") as f:
        json.dump(
            {
                "puzzle": rnd["puzzle"],
                "round": current_round_index(),
                "first_move_optimal": optimal_first,
                "confidence": int(conf),
                "high_confidence": high_conf,
                "screened_out": screenout,
                "submitted_at": datetime.now(timezone.utc).isoformat(),
            },
            f, indent=2,
        )

    if screenout:
        session["screened"] = "ingame"
        return jsonify({"ok": True, "screenout": True,
                        "redirect": url_for("screened_thank_you_ingame")})

    # Survived the gate: unlock the assistant.
    session["ai_unlocked"] = True
    if optimal_first:
        # Correct first move: it counts — keep their progress and simply continue
        # from the current position with the assistant now available.
        board = session.get("oth_board")
        decisions_made = len(session.get("oth_optimal_flags", []))
    else:
        # Wrong first move: reset to a clean, fully AI-assisted attempt. The
        # unaided probe move does NOT count, so clear it and start fresh.
        board = oth_engine.load_board(oth_puzzle_file(rnd["puzzle"]))
        session["oth_board"] = board
        session["oth_forced_moves"] = oth_autoplay_forced(board, OTH_SOLUTIONS[rnd["puzzle"]])
        session["oth_optimal_flags"] = []
        session["oth_kept_flags"] = []
        session["oth_finished"] = False
        decisions_made = 0
        round_path(pdir, "moves", "jsonl").unlink(missing_ok=True)
    return jsonify({
        "ok": True,
        "unlocked": True,
        **oth_board_state(board),
        "decisions_made": decisions_made,
    })


@app.route("/api/game", methods=["POST"])
def submit_game():
    """Final, consolidated game submission written when the participant finishes."""
    if not session.get("prolific_id"):
        return jsonify({"error": "not authorized"}), 401

    data = request.get_json(silent=True) or {}
    end_reason = data.get("end_reason")        # "manual" or "timeout"
    turns = ai_turn_requirement()

    # Leaving early is only blocked when they chose to leave. A timeout ends the
    # round no matter what — and clears the requirement for the pages after it,
    # so nobody is stranded between the game and the survey.
    if end_reason != "timeout" and not turns["met"]:
        return jsonify({"ok": False, "reason": "need_more_ai_turns", **turns})
    if end_reason == "timeout":
        session["oth_timed_out"] = True
        # Close the round out properly: score what they managed and mark it
        # finished, so /performance shows results and they continue to the next
        # round instead of looping back onto the survey.
        if not session.get("oth_finished"):
            oth_finish_round("timeout", completed=False)

    record = {
        "game": session.get("game", DEFAULT_GAME),
        "puzzle": current_round()["puzzle"],
        "round": current_round_index(),
        "moves": data.get("moves") or [],
        "final_fen": data.get("final_fen"),
        "end_reason": end_reason,
        "ai_turns": turns["taken"],            # completed exchanges with the assistant
        "ai_turns_required": turns["required"],
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    safe_pid = session.get("safe_pid", "anon")
    with open(round_path(participant_dir(safe_pid), "game_submission", "json"), "w") as f:
        json.dump(record, f, indent=2)
    return jsonify({"ok": True})


@app.route("/api/focus-event", methods=["POST"])
def focus_event():
    if not session.get("prolific_id"):
        return jsonify({"error": "not authorized"}), 401
    data = request.get_json(silent=True) or {}
    event = {
        "type": data.get("type"),
        "reason": data.get("reason"),
        "game": session.get("game", DEFAULT_GAME),
        "puzzle": current_round()["puzzle"],
        "round": current_round_index(),
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    safe_pid = session.get("safe_pid", "anon")
    pdir = participant_dir(safe_pid)
    with open(round_path(pdir, "focus_events", "jsonl"), "a") as f:
        f.write(json.dumps(event) + "\n")
    return jsonify({"ok": True})

def _render_tlx(answers, error):
    return render_template(
        "post_survey_tlx.html",
        tlx_items=NASA_TLX_ITEMS,
        tlx_ticks=list(range(TLX_MIN, TLX_MAX + 1)),
        answers=answers,
        error=error,
    )


def _render_ai_form(error=None):
    safe_pid = session.get("safe_pid", "anon")
    curr_post_survey = POST_SURVEY_FORM_URL.replace("<PROLIFIC_ID>", safe_pid)
    return render_template("post_survey.html", form_url=curr_post_survey, error=error)


def _save_post_survey(ai_assessment=None):
    """Persist this round's NASA-TLX (and whether the AI was assessed externally)."""
    rnd = current_round()
    tlx = session.get("tlx_answers", {})
    safe_pid = session.get("safe_pid", "anon")
    with open(round_path(participant_dir(safe_pid), "post_survey", "json"), "w") as f:
        json.dump(
            {
                "game": session.get("game", DEFAULT_GAME),
                "puzzle": rnd["puzzle"],
                "round": current_round_index(),
                "nasa_tlx": {key: int(val) for key, val in tlx.items()},
                "ai_assessment": ai_assessment,
                "submitted_at": datetime.now(timezone.utc).isoformat(),
            },
            f,
            indent=2,
        )
    session.pop("tlx_answers", None)


@app.route("/post-survey", methods=["GET", "POST"])
def post_survey():
    gate = require_participant()
    if gate:
        return gate
    # Backstop for the finish-button check: navigating straight here doesn't skip
    # the assistant-turn requirement. A timed-out round is exempt (see submit_game).
    if not session.get("oth_timed_out") and not ai_turn_requirement()["met"]:
        return redirect(url_for("study"))

    if request.method == "POST":
        page = request.form.get("page")

        # Page 1: NASA-TLX -> validate, stash in session.
        if page == "tlx":
            answers = {key: (request.form.get(key) or "").strip() for key, _, _ in NASA_TLX_ITEMS}
            if any(val not in TLX_VALUES for val in answers.values()):
                return _render_tlx(answers, "Please rate every scale before continuing.")
            session["tlx_answers"] = answers
            # Only the AI-enabled round(s) get the external AI-assessment form.
            if current_round()["ai_survey"]:
                return _render_ai_form()
            _save_post_survey()
            return redirect(url_for("performance"))

        # Page 2: external AI-assessment form -> persist the TLX, then continue.
        if page == "external":
            _save_post_survey(ai_assessment="external_form")
            return redirect(url_for("performance"))

    # GET -> start at page 1.
    return _render_tlx({key: "" for key, _, _ in NASA_TLX_ITEMS}, None)


def oth_puzzle_result() -> dict:
    """This puzzle's result, for the block results screen: scores plus who won.

    Whether they WON has to come from the board, not from the score dict — a
    participant can win without playing perfectly, and on a short puzzle they can
    also play every scored decision optimally and still be handed a loss by an
    auto-played forced move.
    """
    board = session.get("oth_board")
    black, white = oth_engine.disc_counts(board) if board else (0, 0)
    best = session.get("oth_best") or oth_round_scores([], [])
    return {
        "puzzle": current_round()["puzzle"], **best,
        "black_discs": black, "white_discs": white,
        "won_it": bool(best.get("completed", True)) and black > white,
    }


def advance_round() -> bool:
    """Move to the next round, clearing this one's game state. False if none left."""
    idx = current_round_index()
    if idx + 1 >= len(OTH_ROUNDS):
        return False
    session["oth_round"] = idx + 1
    for k in ("oth_board", "oth_optimal_flags", "oth_kept_flags", "oth_started",
              "oth_finished", "tlx_answers", "oth_forced_moves", "oth_timed_out",
              "ai_unlocked", "oth_gate_asked", "oth_gate_optimal_first",
              "oth_attempt", "oth_best", "oth_attempts", "oth_attempt_done"):
        session.pop(k, None)
    return True


@app.route("/round-done")
def round_done():
    """Where the board sends the participant when a round ends.

    Most rounds go to their survey. A round marked "survey": False is part of a
    multi-puzzle block (the transfer block), so we go straight on to the next
    puzzle — one survey covers the block, which keeps the participant's time
    down and stops the TLX being asked about a 30-second puzzle.
    """
    gate = require_participant()
    if gate:
        return gate
    if not session.get("oth_finished"):
        return redirect(url_for("study"))
    if current_round().get("survey", True):
        return redirect(url_for("post_survey"))
    # Bank this puzzle's result so the block's results screen can show them all.
    session["oth_block"] = session.get("oth_block", []) + [oth_puzzle_result()]
    advance_round()
    return redirect(url_for("study"))


@app.route("/performance", methods=["GET", "POST"])
def performance():
    """Per-puzzle results screen: shows the optimal-move score, then advances to
    the next puzzle (if any) or to the final demographics."""
    gate = require_participant()
    if gate:
        return gate
    idx = current_round_index()

    if request.method == "POST":
        session.pop("oth_block", None)          # the block's results have been shown
        if advance_round():
            return redirect(url_for("study"))
        return redirect(url_for("demographics"))

    # Must have finished the round to see its results.
    if not session.get("oth_finished"):
        return redirect(url_for("study"))

    best = session.get("oth_best") or oth_round_scores([], [])
    board = session.get("oth_board")
    black_discs, white_discs = oth_engine.disc_counts(board) if board else (0, 0)
    # A multi-puzzle block reports every puzzle in it, not just the last one.
    block = session.get("oth_block", []) + [oth_puzzle_result()]
    return render_template(
        "performance.html",
        block=block if len(block) > 1 else None,
        score=best["score"],
        decisions=best["decisions"],
        score_first_n=best["score_first_n"],
        decisions_first_n=best["decisions_first_n"],
        first_n=OTH_FIRST_N,
        # The game itself: the objective outcome the score is a process measure of.
        # A round the clock ended has no outcome — the template must not announce
        # a win or a loss for a board that never finished.
        completed=bool(best.get("completed", True)),
        black_discs=black_discs,
        white_discs=white_discs,
        won=black_discs > white_discs,
        drew=black_discs == white_discs,
        round_num=idx + 1,
        total_rounds=len(OTH_ROUNDS),
        is_last=(idx + 1 >= len(OTH_ROUNDS)),
    )


@app.route("/screened-skill", methods=["GET", "POST"])
def screened_thank_you_skill():
    """Screen-out #1: skilled Othello players."""
    return render_template("thank_you.html", prolific_url=PROLIFIC_SCREENOUT_SKILL_URL)


@app.route("/screened-prompt", methods=["GET", "POST"])
def screened_thank_you_prompt():
    """Screen-out #2: low-quality prompt."""
    return render_template("thank_you.html", prolific_url=PROLIFIC_SCREENOUT_PROMPT_URL)


@app.route("/screened-ingame", methods=["GET", "POST"])
def screened_thank_you_ingame():
    """Screen-out #3: confident, correct unaided first move (in-game gate)."""
    return render_template("thank_you.html", prolific_url=PROLIFIC_SCREENOUT_INGAME_URL)


@app.route("/thank-you", methods=["GET", "POST"])
def thank_you():
    return render_template("thank_you.html", prolific_url=PROLIFIC_COMPLETION_URL)


def parse_conversation_history(convo_lines):
    parsed_convo_lines = [json.loads(l) for l in convo_lines]
    system_prompt = load_system_prompt(session.get("game", DEFAULT_GAME))
    convo_history = [{"role": "system", "content": system_prompt}]
    for l_json in parsed_convo_lines:
        convo_history.append({"role": "user", "content": l_json["user"]})
        convo_history.append({"role": "assistant", "content": l_json["assistant"]})
    return convo_history


@app.route("/api/chat", methods=["POST"])
def chat():
    if not ai_enabled():                     # assistant off (no-AI round, or gate not yet passed)
        return jsonify({"error": "assistant disabled"}), 403
    if not session.get("prolific_id"):
        return jsonify({"error": "not authorized"}), 401
    data = request.get_json(silent=True) or {}
    safe_pid = session.get("safe_pid", "anon")
    convo_path = round_path(participant_dir(safe_pid), "conversation", "jsonl")

    try:
        with open(convo_path) as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []
    conversation_history = parse_conversation_history(lines)

    user_message = (data.get("message") or "").strip()
    user_ts = datetime.now(timezone.utc).isoformat()
    messages = conversation_history + [{"role": "user", "content": user_message}]
    board = session.get("oth_board")
    empties = oth_engine.empty_count(board) if board else 0
    legal = ", ".join(oth_legal_moves(board)) if board else ""
    caption = ("Current Othello board (you help Black; White is the AI opponent). "
               "Columns are a-h left to right, rows 1-8 top to bottom; the small "
               "pale dots mark Black's legal moves. The game is played to the end; "
               f"{empties} square(s) are still empty. Black may play: {legal}.")
    try:
        # The legal moves are marked on the image AND listed in the caption: a
        # coach that mis-reads which squares are playable is worse than useless.
        png = oth_engine.render_image(board, legal_for=BLACK)
        data_url = "data:image/png;base64," + base64.b64encode(png).decode()
        messages = messages + [{"role": "user", "content": [
            {"type": "text", "text": caption},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]}]
    except Exception as e:  # Pillow missing / render error -> fall back to text
        app.logger.warning("board image render failed, using text: %s", e)
        board_text = oth_engine.board_to_text(board, BLACK) if board else "No move yet."
        messages = messages + [{"role": "system", "content": caption + "\n" + board_text}]

    hint = _solution_hint()
    if hint:
        messages = messages + [{"role": "system", "content": hint}]

    def generate():
        pieces: list[str] = []
        assistant_started_ts = None
        try:
            stream = OPENAI_CLIENT.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                reasoning_effort="medium",
                stream=True,
            )
            for event in stream:
                if not event.choices:
                    continue
                delta = event.choices[0].delta.content
                if delta:
                    if assistant_started_ts is None:
                        assistant_started_ts = datetime.now(timezone.utc).isoformat()
                    pieces.append(delta)
                    yield json.dumps({"delta": delta}) + "\n"
        except Exception as e:
            yield json.dumps({"error": str(e)}) + "\n"
            return
        reply = "".join(pieces)
        reply_html = markdown.markdown(reply, extensions=["extra", "sane_lists"])
        assistant_ts = datetime.now(timezone.utc).isoformat()
        with open(convo_path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "user": user_message,
                        "user_ts": user_ts,
                        "assistant": reply,
                        "assistant_started_ts": assistant_started_ts,
                        "assistant_ts": assistant_ts,
                    }
                )
                + "\n"
            )
        yield json.dumps({"done": True, "html": reply_html,
                          "ai_turns": ai_turns_taken()}) + "\n"

    return Response(
        stream_with_context(generate()),
        mimetype="application/x-ndjson",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )

if __name__ == "__main__":
    # Local dev server only -- production runs under gunicorn (see Dockerfile), which
    # never reaches this block.
    #
    # http.server calls socket.getfqdn() while binding, which on macOS goes out to an
    # mDNS reverse lookup. If the machine's ComputerName contains a non-ASCII character
    # -- a curly apostrophe in "Siyan's MacBook Pro", for instance -- the lookup can
    # return bytes that Python then fails to decode, and the server dies at startup with
    # UnicodeDecodeError before serving anything. The hostname is cosmetic here, so
    # short-circuit the lookup rather than requiring people to rename their Mac.
    import socket as _socket

    _socket.getfqdn = lambda name="": "localhost"
    app.run(host="0.0.0.0", port=5001)