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

dotenv.load_dotenv()

OPENAI_CLIENT = openai.OpenAI()
CHAT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.5")
PROLIFIC_CODE = os.environ.get("PROLIFIC_CODE", "PLACEHOLDER")
# Three screen-out paths/codes: #1 skilled Connect 4 players (screened on the
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

# This deployment is Connect Four-only (chess assets were removed).
DEFAULT_GAME = "connect_four"

# ---- Per-game engines -------------------------------------------------------
# Each game's logic lives in games/<game>/engine.py and is loaded here as a
# module, keeping app.py game-agnostic.
import importlib.util as _ilu  # noqa: E402


def _load_local_module(name: str, path: Path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_CF_DIR = GAMES_DIR / "connect_four"
cf_engine = _load_local_module("cf_engine", _CF_DIR / "engine.py")

# Per-round countdown timers (seconds), env-configurable. Puzzle 15 gets 6 min,
# puzzle 6 gets 3 min. Referenced by CF_ROUNDS below.
CF_TIME_LIMIT_SECONDS = int(os.environ.get("CF_TIME_LIMIT_SECONDS", "360"))       # round 1 / puzzle 15
CF_TIME_LIMIT_SECONDS_2 = int(os.environ.get("CF_TIME_LIMIT_SECONDS_2", "180"))   # round 2 / puzzle 6

# Two-round within-subjects design: every participant plays these puzzles IN
# ORDER. Each round names a puzzle (the puzzle_config_N.txt / solution_N.json
# pair), whether the in-game AI chat assistant is available ("ai"), and whether
# the external AI-assessment survey is shown after that round's NASA-TLX.
#
# "gate": the round STARTS with the assistant off. The participant makes an
# unaided first move and rates their confidence; if that move was optimal AND
# confidence is high they are screened out. Otherwise the assistant is unlocked:
# a wrong first move resets the board for a clean attempt, while a correct first
# move keeps its progress and continues. See /api/gate.
CF_ROUNDS = [
    {"puzzle": "15",    "ai": True,  "ai_survey": True,  "gate": True,
     "time_limit": CF_TIME_LIMIT_SECONDS,
     # "intro": a heads-up popup shown once before the round starts (title + HTML body).
     # DRAFT wording — review; note it hints at the unaided-first-move structure.
     "intro": {"title": "Puzzle 1",
               "body": "An <strong>AI assistant</strong> will be available to help you for this round. This is your <strong>only chance</strong> to use it in this study. You may ask your questions about Connect Four to help your second puzzle."}},
    # Round 2: transfer test — win-in-four, opens col 3 (central like #15's col 4,
    # but a different move so it isn't rote), optimal line 3->1->5->3. From
    # Zeilberger ch4 P6; solved under our defense model (solution_w4p6.json).
    # Single attempt (no retry).
    {"puzzle": "w4p6", "ai": False, "ai_survey": False,
     "time_limit": CF_TIME_LIMIT_SECONDS_2,
     "intro": {"title": "Puzzle 2",
               "body": "There is <strong>no AI assistant</strong> here, and you will "
                       "<strong>not</strong> need to fill out the AI assistant questionnaire afterward."}},
]


def cf_puzzle_file(puzzle: str) -> Path:
    return _CF_DIR / f"puzzle_config_{puzzle}.txt"


# Each puzzle's precomputed solution table, loaded once and keyed by puzzle id.
# The board and its solution are always drawn from the same id so they can never
# drift apart.
CF_SOLUTIONS = {}
for _r in CF_ROUNDS:
    with open(_CF_DIR / f"solution_{_r['puzzle']}.json") as _f:
        CF_SOLUTIONS[_r["puzzle"]] = json.load(_f)

# The participant makes exactly this many moves per puzzle, then the round ends.
# We never tell them mid-game whether a move was correct; the per-puzzle score
# (shown on the results screen afterward) is how many of their moves were optimal.
CF_NUM_MOVES = int(os.environ.get("CF_NUM_MOVES", "3"))

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
    return session.get("cf_round", 0)


def current_round() -> dict:
    return CF_ROUNDS[current_round_index()]


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

# ---- Stage 1: prompt-creation pre-filter ------------------------------------
# Participants first write a prompt for a fixed scenario; a multi-model judge
# panel (defined in prompt_filter.py: JudgeSuite) each scores it 1-4 on a
# holistic rubric, and the MEDIAN of those scores must clear JUDGE_PASS_THRESHOLD
# for the participant to continue into the game. Everything (the prompt, every
# model's score, the mean, the median, the decision) is logged to prompt_task.json.
JUDGE_PASS_THRESHOLD = float(os.environ.get("JUDGE_PASS_THRESHOLD", "3.0"))   # median score needed
JUDGE_MIN_PROMPT_CHARS = int(os.environ.get("JUDGE_MIN_PROMPT_CHARS", "40"))  # reject near-empty
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
    "connect_four": "I consider myself a skilled Connect 4 player.",
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


@app.route("/debug/cf-board.png")
def debug_cf_board():
    """Dev aid: view the exact Connect Four image the LLM is fed. Shows the live
    session board if there is one, else the starting position."""
    board = session.get("cf_board") or cf_engine.load_board(cf_puzzle_file(current_round()["puzzle"]))
    return _no_cache(Response(cf_engine.render_image(board), mimetype="image/png"))


@app.route("/debug/cf-state")
def debug_cf_state():
    """Raw server-side Connect Four state, to confirm what the session holds."""
    board = session.get("cf_board")
    return _no_cache(jsonify({
        "has_session_board": board is not None,
        "moves_made": session.get("cf_moves_made"),
        "moves_left": CF_NUM_MOVES - session.get("cf_moves_made", 0),
        "score": session.get("cf_score"),
        "finished": session.get("cf_finished", False),
        "piece_count": sum(c != cf_engine.EMPTY for row in (board or []) for c in row),
        "board_bottom_row_first": board,
    }))


@app.route("/debug/cf-reset")
def debug_cf_reset():
    """Reset the session board + move state to the current puzzle's starting position."""
    session["cf_board"] = cf_engine.load_board(cf_puzzle_file(current_round()["puzzle"]))
    session["cf_moves_made"] = 0
    session["cf_score"] = 0
    session["cf_started"] = True
    session["cf_finished"] = False
    return _no_cache(jsonify({"ok": True, "moves_left": CF_NUM_MOVES}))


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
    verdict["passed"] = (
        bool(verdict["median_score"] >= JUDGE_PASS_THRESHOLD) if valid else JUDGE_FAIL_OPEN
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


def _red_engine_best_cols(board):
    """Engine's best RED column(s) from the CURRENT position — used OFF the
    forced-win line (where no move preserves a win). Returns every column tying
    the best minimax score, so a participant is credited for playing any
    best-available move. Shared by scoring (cf_move) and the assistant hint so
    the two never disagree — following good advice always earns credit."""
    scored = []
    for c in cf_engine.valid_columns(board):
        child = cf_engine.copy_board(board)
        cf_engine.drop(child, c, cf_engine.RED)
        if cf_engine.winning_move(child, cf_engine.RED):
            scored.append((c, float("inf")))
            continue
        _, sc = cf_engine.minimax(child, 5, float("-inf"), float("inf"),
                                  to_move=cf_engine.YELLOW, me=cf_engine.RED)
        scored.append((c, sc))
    if not scored:
        return []
    best = max(s for _, s in scored)
    return [c for c, s in scored if s == best]


def red_best_cols(board, solution):
    """RED's best column(s) from the current position: the exact winning_columns
    while on the forced-win line, else the engine's best move(s). This is the
    'optimal' set a move is scored against (option-2 scoring)."""
    wc = solution.get(cf_engine.state_key(board, cf_engine.RED), {}).get("winning_columns", [])
    return wc if wc else _red_engine_best_cols(board)


def _solution_hint():
    """Coaching aid for the assistant. On the forced-win line: the optimal
    move(s) for the CURRENT position. Off the line (participant deviated): the
    engine's best available move(s), so the assistant stays helpful after a
    mistake instead of going silent. Coaches toward the move without revealing
    it. Returns None only when there is no board.
    """
    board = session.get("cf_board")
    if not board:
        return None
    solution = CF_SOLUTIONS[current_round()["puzzle"]]
    cols = solution.get(cf_engine.state_key(board, cf_engine.RED), {}).get("winning_columns", [])
    if cols:
        cols_1 = ", ".join(str(c + 1) for c in cols)
        return ("COACHING AID — the move(s) "
                f"that keep Red's forced win from the current position are column {cols_1}. "
                "Make sure you never steer the participant incorrectly.\n"
                "When it helps, reference specific squares by "
                "column (1-7) and row to point out a relevant run, threat, or diagonal. "
                "Keep it brief (a couple of sentences or a short "
                "list).\n"
                "Reason carefully about the board image and think step by step. DO NOT PROVIDE THE ANSWER UNLESS EXPLICITLYL ASKED TO DO SO.")
    # Off the forced-win line (the participant has deviated): no guaranteed win
    # remains, but keep the assistant useful by coaching toward the engine's best
    # available move(s) — the SAME set the move is scored against, so following
    # this advice earns credit.
    best_cols = _red_engine_best_cols(board)
    if not best_cols:
        return None
    best_1 = ", ".join(str(c + 1) for c in best_cols)
    return ("COACHING AID — the participant is no longer on a forced-win line, so there is no "
            f"guaranteed win, but the strongest available move(s) here are column {best_1}. "
            "Help them keep playing well: point out the key threat or block, reference specific "
            "squares by column (1-7) and row, and keep it brief.\n"
            "Do NOT tell them they made a mistake or that the win is lost. Reason carefully about "
            "the board image and think step by step. DO NOT PROVIDE THE ANSWER UNLESS EXPLICITLY ASKED TO DO SO.")



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
    """Eligibility (after consent): Prolific ID + skill rating. Skilled Connect 4
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
                    "game": "connect_four",
                    "rounds": CF_ROUNDS,
                    **demographics,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                },
                f,
                indent=2,
            )

        # Screen-out #1: skilled Connect 4 players (skill 4-5) do not continue.
        # "dont_know" and 1-3 are eligible.
        if skill_rating in ("4", "5"):
            session["screened"] = "skill"
            return redirect(url_for("screened_thank_you_skill"))

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
    topic_readable = "Connect Four"
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
                    "rounds": CF_ROUNDS,
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
            return redirect(url_for("primer"))
        # Screen-out #2: low-quality prompt.
        session["screened"] = "prompt"
        return redirect(url_for("screened_thank_you_prompt"))

    return render_template(
        "prompt_task.html", scenario=PROMPT_TASK_SCENARIO, prefill="", error=None
    )


@app.route("/primer")
def primer():
    gate = require_participant()
    if gate:
        return gate
    # Must have cleared the Stage 1 prompt filter to reach the game.
    if not session.get("prompt_passed"):
        return redirect(url_for("prompt_task"))
    # Standardized rules primer shown before the timed game. The timer only
    # starts on /study, so reading this does not count against the participant.
    return render_template("connect_four_primer.html", game=session.get("game", DEFAULT_GAME))


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

    # Already made all the moves this round -> can't replay by refreshing/returning.
    if session.get("cf_finished"):
        return redirect(url_for("post_survey"))

    if not session.get("cf_started"):
        # First entry to THIS round only: fresh board + fresh logs. Refreshing
        # afterward will NOT re-initialize, so a refresh can't reset progress or
        # stack up moves. (Round state is cleared when advancing to the next puzzle.)
        board = cf_engine.load_board(cf_puzzle_file(rnd["puzzle"]))
        session["cf_board"] = board
        session["cf_moves_made"] = 0
        session["cf_score"] = 0
        session["cf_started"] = True
        # Two-tries state (rounds with "attempts" > 1). Best score across attempts.
        session["cf_attempt"] = 1
        session["cf_best_score"] = 0
        session["cf_attempt_scores"] = []
        session["cf_attempt_wl_scores"] = []
        session["cf_wl_score"] = 0
        session["cf_attempt_done"] = False
        pdir = participant_dir(safe_pid)
        round_path(pdir, "moves", "jsonl").unlink(missing_ok=True)
        round_path(pdir, "conversation", "jsonl").unlink(missing_ok=True)
    else:
        board = session.get("cf_board")          # resume current state on refresh

    moves_made = session.get("cf_moves_made", 0)
    ai_now = ai_enabled()
    # gate_active: this is the gated round and the gate hasn't been passed yet,
    # so the confidence popup may still fire (after the first move) — render its
    # markup + JS. gate_pending: the first move was already made (e.g. the
    # participant refreshed mid-gate), so show the popup immediately on load.
    gate_active = bool(rnd.get("gate")) and not session.get("ai_unlocked")
    gate_pending = gate_active and bool(session.get("cf_gate_asked"))
    # Intro popup for a multi-attempt round (round 2): shown once on fresh entry,
    # before any move — a heads-up that there's no AI and no AI survey after.
    show_round_intro = (bool(rnd.get("intro"))
                        and moves_made == 0
                        and not session.get("cf_finished"))
    # The solution table stays server-side; the client only gets the board.
    return render_template(
        "connect_four.html",
        game=game,
        board=board,                 # board[row][col], row 0 = bottom
        rows=cf_engine.ROWS,
        cols=cf_engine.COLS,
        num_moves=CF_NUM_MOVES,
        moves_made=moves_made,
        moves_left=CF_NUM_MOVES - moves_made,
        time_limit_seconds=rnd["time_limit"],
        ai_assistant_enabled=ai_now,     # visible/usable right now
        render_chat=(ai_now or gate_active),  # include chat DOM+JS (hidden until unlocked)
        gate_active=gate_active,
        gate_pending=gate_pending,
        confidence_options=CONFIDENCE_OPTIONS,
        show_round_intro=show_round_intro,
        round_intro=rnd.get("intro"),
    )


@app.route("/api/cf-move", methods=["POST"])
def cf_move():
    """Connect Four: the participant drops in a column (RED).

    The participant makes a fixed number of moves (CF_NUM_MOVES). EVERY move
    drops a disc and the AI (Yellow) replies, so the board always advances and
    the participant is NEVER told whether a move was optimal. We silently record
    each move's optimality; the score is how many of their moves were optimal.
    On the precomputed forced-win line the AI plays its best defense; once the
    participant deviates, the AI falls back to the engine's general minimax.
    """
    if not session.get("prolific_id"):
        return jsonify({"error": "not authorized"}), 401

    rnd = current_round()
    solution = CF_SOLUTIONS[rnd["puzzle"]]
    board = session.get("cf_board")
    moves_made = session.get("cf_moves_made", 0)
    score = session.get("cf_score", 0)            # option-2: best-move-from-position count
    wl_score = session.get("cf_wl_score", 0)      # strict: moves that kept the forced win
    if board is None:
        return jsonify({"error": "no active game"}), 400
    if session.get("cf_finished") or moves_made >= CF_NUM_MOVES:
        return jsonify({"ok": False, "reason": "game_over", "done": True})
    # Gated round: once the unaided first move is made, no further moves until the
    # confidence gate is resolved (either screen-out or board-reset + AI unlock).
    if rnd.get("gate") and session.get("cf_gate_asked") and not session.get("ai_unlocked"):
        return jsonify({"ok": False, "reason": "awaiting_gate"})

    data = request.get_json(silent=True) or {}
    col = data.get("col")                       # internal 0-indexed column
    if col is None or not cf_engine.is_valid_column(board, col):
        return jsonify({"ok": False, "reason": "invalid_column"})   # full/invalid; no move spent

    # Score the move BEFORE dropping it.
    #   winning_cols  = moves that KEEP Red's forced win (empty once off the line)
    #   best_cols     = winning_cols on the line, else the engine's best move(s)
    # optimal (option-2) credits the best move from wherever they are; on_winning_line
    # is the stricter "did it keep the forced win" flag, recorded for trajectory analysis.
    winning_cols = solution.get(
        cf_engine.state_key(board, cf_engine.RED), {}
    ).get("winning_columns", [])
    best_cols = winning_cols if winning_cols else _red_engine_best_cols(board)
    on_winning_line = col in winning_cols
    optimal = col in best_cols

    cf_engine.drop(board, col, cf_engine.RED)   # the disc ALWAYS drops
    moves_made += 1
    if optimal:
        score += 1
    if on_winning_line:
        wl_score += 1

    # AI (Yellow) replies unless Red just won or the board is full.
    ai_col = None
    if not cf_engine.winning_move(board, cf_engine.RED) and not cf_engine.is_full(board):
        ai_col = solution.get(
            cf_engine.state_key(board, cf_engine.YELLOW), {}
        ).get("best_defense")
        if ai_col is None or not cf_engine.is_valid_column(board, ai_col):
            ai_col = cf_engine.ai_move(board)   # off the precomputed line -> general AI
        if ai_col is not None:
            cf_engine.drop(board, ai_col, cf_engine.YELLOW)

    # Gated round: the first move is an unaided probe. Record its optimality and
    # ask for confidence before any further play (the client shows a popup).
    gate_needed = (rnd.get("gate") and moves_made == 1
                   and not session.get("ai_unlocked")
                   and not session.get("cf_gate_asked"))
    if gate_needed:
        session["cf_gate_optimal_first"] = optimal
        session["cf_gate_asked"] = True

    session["cf_board"] = board
    session["cf_moves_made"] = moves_made
    session["cf_score"] = score
    session["cf_wl_score"] = wl_score

    # Attempt / two-tries handling. A round may allow multiple attempts
    # ("attempts" in CF_ROUNDS); if this attempt ends unsolved and retries
    # remain, the round is NOT finished — the client offers one more try.
    attempt = session.get("cf_attempt", 1)
    max_attempts = rnd.get("attempts", 1)
    attempt_done = moves_made >= CF_NUM_MOVES
    solved = attempt_done and score == CF_NUM_MOVES
    retry_available = False
    round_done = False
    if attempt_done:
        best = max(session.get("cf_best_score", 0), score)
        session["cf_best_score"] = best
        scores = session.get("cf_attempt_scores", [])
        scores.append(score)
        session["cf_attempt_scores"] = scores
        wl_scores = session.get("cf_attempt_wl_scores", [])
        wl_scores.append(wl_score)
        session["cf_attempt_wl_scores"] = wl_scores
        if attempt < max_attempts and not solved:
            retry_available = True
            session["cf_attempt_done"] = True       # this attempt over, awaiting retry
        else:
            round_done = True
            session["cf_finished"] = True
            session["cf_score"] = best              # downstream (performance) shows best

    safe_pid = session.get("safe_pid", "anon")
    pdir = participant_dir(safe_pid)
    record = {
        "game": "connect_four",
        "puzzle": rnd["puzzle"],
        "round": current_round_index(),
        "attempt": attempt,
        "move_number": moves_made,
        "col": col,
        "optimal": optimal,               # best move from the current position (option-2 scoring)
        "on_winning_line": on_winning_line,   # kept Red's forced win (stricter metric)
        "winning_cols": winning_cols,     # forced-win-preserving cols ([] once off the line)
        "best_cols": best_cols,           # the set 'optimal' is scored against
        "ai_col": ai_col,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    with open(round_path(pdir, "moves", "jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")
    if round_done:
        wl_list = session.get("cf_attempt_wl_scores", [])
        with open(round_path(pdir, "cf_score", "json"), "w") as f:
            json.dump(
                {"puzzle": rnd["puzzle"], "round": current_round_index(),
                 "ai": rnd["ai"],
                 # option-2 (best-move) score is primary; winning-line score is the
                 # stricter "found the forced win" metric. Both are best-of-attempts.
                 "score": session["cf_best_score"],
                 "winning_line_score": max(wl_list) if wl_list else 0,
                 "attempt_scores": session.get("cf_attempt_scores", []),
                 "attempt_winning_line_scores": wl_list,
                 "attempts_used": attempt, "num_moves": CF_NUM_MOVES,
                 "submitted_at": datetime.now(timezone.utc).isoformat()},
                f, indent=2,
            )

    # NOTE: response deliberately omits per-move optimality so the client can
    # never reveal correctness to the participant. "gate" asks the client to
    # collect confidence before continuing (carries no correctness signal).
    # "retry_available" DOES imply this attempt wasn't a full solve (the study
    # intentionally offers a retry only on a non-solved attempt).
    return jsonify({
        "ok": True,
        "board": board,
        "ai_col": ai_col,
        "moves_made": moves_made,
        "num_moves": CF_NUM_MOVES,
        "moves_left": CF_NUM_MOVES - moves_made,
        "done": attempt_done,
        "retry_available": retry_available,
        "gate": bool(gate_needed),
    })


@app.route("/api/cf-retry", methods=["POST"])
def cf_retry():
    """Start the next attempt of a multi-try round (round 2): reset the board to
    the puzzle start. Allowed only when the current attempt has ended unsolved
    and retries remain. Prior attempt's moves stay logged (each carries its
    'attempt' number); the round score is the best across attempts."""
    if not session.get("prolific_id"):
        return jsonify({"error": "not authorized"}), 401
    rnd = current_round()
    attempt = session.get("cf_attempt", 1)
    if session.get("cf_finished"):
        return jsonify({"ok": False, "reason": "round_over"})
    if not session.get("cf_attempt_done"):
        return jsonify({"ok": False, "reason": "attempt_not_done"}), 400
    if attempt >= rnd.get("attempts", 1):
        return jsonify({"ok": False, "reason": "no_retries_left"}), 400

    board = cf_engine.load_board(cf_puzzle_file(rnd["puzzle"]))
    session["cf_board"] = board
    session["cf_moves_made"] = 0
    session["cf_score"] = 0
    session["cf_wl_score"] = 0
    session["cf_attempt"] = attempt + 1
    session["cf_attempt_done"] = False
    return jsonify({
        "ok": True,
        "board": board,
        "moves_made": 0,
        "num_moves": CF_NUM_MOVES,
        "moves_left": CF_NUM_MOVES,
        "attempt": attempt + 1,
    })


@app.route("/api/gate", methods=["POST"])
def cf_gate():
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
    if not session.get("cf_gate_asked"):           # first move not made yet
        return jsonify({"ok": False, "reason": "gate_not_ready"}), 400

    data = request.get_json(silent=True) or {}
    conf = str(data.get("confidence", "")).strip()
    if conf not in CONFIDENCE_VALUES:
        return jsonify({"ok": False, "reason": "invalid_confidence"}), 400

    optimal_first = bool(session.get("cf_gate_optimal_first"))
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
        board = session.get("cf_board")
        moves_made = session.get("cf_moves_made", 0)
    else:
        # Wrong first move: reset to a clean, fully AI-assisted attempt. The
        # unaided probe move does NOT count, so clear it and start fresh.
        board = cf_engine.load_board(cf_puzzle_file(rnd["puzzle"]))
        session["cf_board"] = board
        session["cf_moves_made"] = 0
        session["cf_score"] = 0
        session["cf_finished"] = False
        moves_made = 0
        round_path(pdir, "moves", "jsonl").unlink(missing_ok=True)
    return jsonify({
        "ok": True,
        "unlocked": True,
        "board": board,
        "moves_made": moves_made,
        "num_moves": CF_NUM_MOVES,
        "moves_left": CF_NUM_MOVES - moves_made,
    })


@app.route("/api/game", methods=["POST"])
def submit_game():
    """Final, consolidated game submission written when the participant finishes."""
    if not session.get("prolific_id"):
        return jsonify({"error": "not authorized"}), 401

    data = request.get_json(silent=True) or {}
    record = {
        "game": session.get("game", DEFAULT_GAME),
        "puzzle": current_round()["puzzle"],
        "round": current_round_index(),
        "moves": data.get("moves") or [],
        "final_fen": data.get("final_fen"),
        "end_reason": data.get("end_reason"),  # "manual" or "timeout"
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


@app.route("/performance", methods=["GET", "POST"])
def performance():
    """Per-puzzle results screen: shows the optimal-move score, then advances to
    the next puzzle (if any) or to the final demographics."""
    gate = require_participant()
    if gate:
        return gate
    idx = current_round_index()

    if request.method == "POST":
        if idx + 1 < len(CF_ROUNDS):
            # Advance to the next round: bump the index and clear this round's
            # game state so /study re-initializes for the next puzzle.
            session["cf_round"] = idx + 1
            for k in ("cf_board", "cf_moves_made", "cf_score", "cf_started",
                      "cf_finished", "tlx_answers",
                      "ai_unlocked", "cf_gate_asked", "cf_gate_optimal_first",
                      "cf_attempt", "cf_best_score", "cf_attempt_scores", "cf_attempt_done",
                      "cf_wl_score", "cf_attempt_wl_scores"):
                session.pop(k, None)
            return redirect(url_for("study"))
        return redirect(url_for("demographics"))

    # Must have finished the round to see its results.
    if not session.get("cf_finished"):
        return redirect(url_for("study"))

    return render_template(
        "performance.html",
        score=session.get("cf_score", 0),
        num_moves=CF_NUM_MOVES,
        round_num=idx + 1,
        total_rounds=len(CF_ROUNDS),
        is_last=(idx + 1 >= len(CF_ROUNDS)),
    )


@app.route("/screened-skill", methods=["GET", "POST"])
def screened_thank_you_skill():
    """Screen-out #1: skilled Connect 4 players."""
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
    board = session.get("cf_board")
    moves_left = CF_NUM_MOVES - session.get("cf_moves_made", 0)
    caption = ("Current Connect Four board (you help Red \"R\"; Yellow \"Y\" is the "
               f"AI opponent). The participant has {moves_left} move(s) left:")
    try:
        png = cf_engine.render_image(board)
        data_url = "data:image/png;base64," + base64.b64encode(png).decode()
        messages = messages + [{"role": "user", "content": [
            {"type": "text", "text": caption},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]}]
    except Exception as e:  # Pillow missing / render error -> fall back to text
        app.logger.warning("board image render failed, using text: %s", e)
        board_text = cf_engine.board_to_text(board) if board else "No move yet."
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
                reasoning_effort="low",
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
        yield json.dumps({"done": True, "html": reply_html}) + "\n"

    return Response(
        stream_with_context(generate()),
        mimetype="application/x-ndjson",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)