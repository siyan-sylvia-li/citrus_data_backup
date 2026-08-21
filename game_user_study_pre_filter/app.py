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
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
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
# Two screen-out paths/codes: #1 for skilled Connect 4 players (screened on
# page 1), #2 for low-quality prompts (screened after the prompt task).
SCREENOUT_SKILL_CODE = os.environ.get("SCREENOUT_SKILL_CODE", "PLACEHOLDER")
SCREENOUT_PROMPT_CODE = os.environ.get("SCREENOUT_PROMPT_CODE", "PLACEHOLDER")

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "citrus-blast")

BASE_DIR = Path(__file__).parent
GAMES_DIR = BASE_DIR / "games"
# Where participant data is written. In production this is a GCS bucket mounted
# via Cloud Run's volume mount (gcsfuse) so data survives restarts/redeploys.
# Locally it defaults to ./recordings.
RECORDINGS_DIR = Path(os.environ.get("RECORDINGS_DIR", BASE_DIR / "recordings"))

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# Which game this deployment runs: "chess" or "connect_four".
# This deployment is Connect Four-only (chess assets were removed).
DEFAULT_GAME = os.environ.get("GAME", "connect_four")

# Participant filtering criteria
PARTICIPANT_FILTER_PHASE = os.environ.get("PARTICIPANT_FILTER", 1)

# ---- Per-game engines -------------------------------------------------------
# Each game's logic lives in games/<game>/engine.py and is loaded here as a
# module, keeping app.py game-agnostic.
import importlib.util as _ilu  # noqa: E402


def _load_local_module(name: str, path: Path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_CHESS_DIR = GAMES_DIR / "chess"
try:
    chess_engine = _load_local_module("chess_engine", _CHESS_DIR / "engine.py")
except FileNotFoundError:
    chess_engine = None          # chess removed in this (Connect Four-only) deployment

_CF_DIR = GAMES_DIR / "connect_four"
cf_engine = _load_local_module("cf_engine", _CF_DIR / "engine.py")
with open(_CF_DIR / "solution.json") as _f:
    CF_SOLUTION = json.load(_f)
# Total move budget for Connect Four. The forced win is 5 optimal moves; the
# extra slack tolerates a few wrong attempts. EVERY attempt (right or wrong)
# spends one, so probing all columns runs the budget out.
CF_MOVE_BUDGET = int(os.environ.get("CF_MOVE_BUDGET", "8"))
# The participant makes exactly this many moves, then the game ends. We never
# tell them whether a move was correct; we silently score how many were optimal.
CF_NUM_MOVES = int(os.environ.get("CF_NUM_MOVES", "3"))
# Countdown timer for the game (seconds); shown on the game screen.
CF_TIME_LIMIT_SECONDS = int(os.environ.get("CF_TIME_LIMIT_SECONDS", "360"))  # 6 minutes

# Study condition: when on, the assistant is told the optimal move for the
# current position and coaches toward it (without revealing it outright) — a
# reliably-correct "good AI" condition. Off (default) = the AI reasons unaided.
AI_KNOWS_SOLUTION = os.environ.get("AI_KNOWS_SOLUTION", "false").strip().lower() in (
    "1", "true", "yes", "on",
)

# ---- Stage 1: prompt-creation pre-filter ------------------------------------
# Participants first write a prompt for a fixed scenario; a multi-model judge
# panel (defined in prompt_filter.py: JudgeSuite) each scores it 1-4 on a
# holistic rubric, and the AVERAGE must clear JUDGE_PASS_THRESHOLD for the
# participant to continue into the game. Everything (the prompt, every model's
# score, the mean, the decision) is logged to prompt_task.json.
JUDGE_PASS_THRESHOLD = float(os.environ.get("JUDGE_PASS_THRESHOLD", "3.0"))   # mean score needed
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

# Likert scale, reused for the pre-study game-familiarity question.
LIKERT_OPTIONS = [
    ("1", "Strongly disagree"),
    ("2", "Disagree"),
    ("3", "Neutral"),
    ("4", "Agree"),
    ("5", "Strongly agree"),
]
LIKERT_VALUES = {v for v, _ in LIKERT_OPTIONS}

# Friendly display label per game.
GAME_LABEL = {"chess": "chess", "connect_four": "Connect 4"}

# Pre-survey prompts, phrased per game.
GAME_FAMILIARITY_PROMPT = "I am familiar with the rules of {game}."
SKILL_PROMPTS = {
    "chess": "I am a skilled chess player.",
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
    ("physical_demand", "Physical Demand",
     "How much physical activity was required (e.g., pushing, pulling, turning, "
     "controlling, activating, etc.)? Was the task easy or demanding, slow or "
     "brisk, slack or strenuous, restful or laborious?"),
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

# Page 2: AI-assistant Likert items (rated on LIKERT_OPTIONS, 1..5).
AI_LIKERT_ITEMS = [
    ("ai_consistent", "The AI assistant's responses were consistent with the intentions of my questions."),
    ("ai_easy_to_use", "The AI assistant was easy to use.")
]

# Optional open-ended feedback at the end of the post-study survey.
AI_FEEDBACK_PROMPT = (
    "Any additional feedback you would like to share about the AI assistant or the task itself?"
)

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

def _no_cache(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp


@app.route("/debug/cf-board.png")
def debug_cf_board():
    """Dev aid: view the exact Connect Four image the LLM is fed. Shows the live
    session board if there is one, else the starting position."""
    board = session.get("cf_board") or cf_engine.load_board(_CF_DIR / "puzzle_config.txt")
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
    """Reset the session board + move state to the puzzle's starting position."""
    session["cf_board"] = cf_engine.load_board(_CF_DIR / "puzzle_config.txt")
    session["cf_moves_made"] = 0
    session["cf_score"] = 0
    session["cf_started"] = True
    session["cf_finished"] = False
    return _no_cache(jsonify({"ok": True, "moves_left": CF_NUM_MOVES}))


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
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return DEFAULT_SYSTEM_PROMPT
    if game_type == "connect_four":
        # The board is sent as an image per-message in the chat route, so the
        # system prompt carries no board text.
        return text
    # Chess: inject the fixed puzzle FEN.
    return text.replace("STARTING_FEN", chess_engine.load_puzzle(_CHESS_DIR)["start_fen"])


def _solution_hint():
    """Coaching aid for the AI_KNOWS_SOLUTION condition: the optimal move(s) for
    the CURRENT position, with an instruction to coach toward them without
    revealing. Returns None when the flag is off or no solution is known.
    """
    if not AI_KNOWS_SOLUTION:
        return None
    game = session.get("game", DEFAULT_GAME)
    if game == "connect_four":
        board = session.get("cf_board")
        cols = (CF_SOLUTION.get(cf_engine.state_key(board, cf_engine.RED), {})
                .get("winning_columns", [])) if board else []
        if not cols:
            return None
        cols_1 = ", ".join(str(c + 1) for c in cols)
        return ("PRIVATE COACHING AID — the participant must NOT learn this: the move(s) "
                f"that keep Red's forced win from the current position are column {cols_1}. "
                "Use this ONLY to keep your own hints accurate so you never steer them "
                "wrong.\n"
                "Be concrete but SELECTIVE. When it helps, reference specific squares by "
                "column (1-7) and row to point out a relevant run, threat, or diagonal. "
                "Keep it brief (a couple of sentences or a short "
                "list). The goal is a focused nudge toward what to look at, not an "
                "exhaustive board inventory.\n"
                "The ONE thing you must hold back is the answer itself. Helping them SEE the board "
                "clearly is good; making the final choice FOR them is not. Reason carefully about the board image and think step by step.")
    return None


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
    game = session.get("game", DEFAULT_GAME)

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
                    "game": game,
                    "ai_knows_solution": AI_KNOWS_SOLUTION,
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
    topic_readable = "Connect Four" if session["game"] == "connect_four" else "Chess"
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
                    "ai_knows_solution": AI_KNOWS_SOLUTION,
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
    if session.get("game", "chess") == "chess":
        return render_template("chess_primer.html", game=session.get("game", DEFAULT_GAME))
    elif session.get("game", "chess") == "connect_four":
        return render_template("connect_four_primer.html", game=session.get("game", DEFAULT_GAME))
    


@app.route("/study")
def study():
    gate = require_participant()
    if gate:
        return gate
    if not session.get("prompt_passed"):     # cannot skip the Stage 1 filter
        return redirect(url_for("prompt_task"))
    game = session.get("game", DEFAULT_GAME)

    safe_pid = session.get("safe_pid", "anon")

    if game == "connect_four":
        # Already made all the moves -> can't replay by refreshing/returning.
        if session.get("cf_finished"):
            return redirect(url_for("post_survey"))

        if not session.get("cf_started"):
            # First entry only: fresh board + fresh logs. Refreshing afterward
            # will NOT re-initialize, so a refresh can't reset progress or stack
            # up moves.
            board = cf_engine.load_board(_CF_DIR / "puzzle_config.txt")
            session["cf_board"] = board
            session["cf_moves_made"] = 0
            session["cf_score"] = 0
            session["cf_started"] = True
            participant_dir(safe_pid).joinpath("moves.jsonl").unlink(missing_ok=True)
            participant_dir(safe_pid).joinpath("conversation.jsonl").unlink(missing_ok=True)
        else:
            board = session.get("cf_board")          # resume current state on refresh

        moves_made = session.get("cf_moves_made", 0)
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
            time_limit_seconds=CF_TIME_LIMIT_SECONDS,
        )

    puzzle = chess_engine.load_puzzle(_CHESS_DIR)
    # Note: the solution is intentionally NOT passed to the template so the
    # answer never reaches the browser; scoring happens in /api/move.
    return render_template(
        "chess.html",
        game=game,
        start_fen=puzzle["start_fen"],
        orientation=puzzle.get("orientation", "white"),
        puzzle_type=puzzle.get("type", "single"),
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

    board = session.get("cf_board")
    moves_made = session.get("cf_moves_made", 0)
    score = session.get("cf_score", 0)
    if board is None:
        return jsonify({"error": "no active game"}), 400
    if session.get("cf_finished") or moves_made >= CF_NUM_MOVES:
        return jsonify({"ok": False, "reason": "game_over", "done": True})

    data = request.get_json(silent=True) or {}
    col = data.get("col")                       # internal 0-indexed column
    if col is None or not cf_engine.is_valid_column(board, col):
        return jsonify({"ok": False, "reason": "invalid_column"})   # full/invalid; no move spent

    # Score against the forced-win line BEFORE moving (optimal = keeps the win).
    optimal_cols = CF_SOLUTION.get(
        cf_engine.state_key(board, cf_engine.RED), {}
    ).get("winning_columns", [])
    optimal = col in optimal_cols

    cf_engine.drop(board, col, cf_engine.RED)   # the disc ALWAYS drops
    moves_made += 1
    if optimal:
        score += 1

    # AI (Yellow) replies unless Red just won or the board is full.
    ai_col = None
    if not cf_engine.winning_move(board, cf_engine.RED) and not cf_engine.is_full(board):
        ai_col = CF_SOLUTION.get(
            cf_engine.state_key(board, cf_engine.YELLOW), {}
        ).get("best_defense")
        if ai_col is None or not cf_engine.is_valid_column(board, ai_col):
            ai_col = cf_engine.ai_move(board)   # off the precomputed line -> general AI
        if ai_col is not None:
            cf_engine.drop(board, ai_col, cf_engine.YELLOW)

    done = moves_made >= CF_NUM_MOVES
    session["cf_board"] = board
    session["cf_moves_made"] = moves_made
    session["cf_score"] = score
    if done:
        session["cf_finished"] = True

    safe_pid = session.get("safe_pid", "anon")
    pdir = participant_dir(safe_pid)
    record = {
        "game": "connect_four",
        "move_number": moves_made,
        "col": col,
        "optimal": optimal,
        "optimal_cols": optimal_cols,
        "ai_col": ai_col,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    with open(pdir / "moves.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    if done:
        with open(pdir / "cf_score.json", "w") as f:
            json.dump(
                {"score": score, "num_moves": CF_NUM_MOVES,
                 "submitted_at": datetime.now(timezone.utc).isoformat()},
                f, indent=2,
            )

    # NOTE: response deliberately omits per-move optimality so the client can
    # never reveal correctness to the participant.
    return jsonify({
        "ok": True,
        "board": board,
        "ai_col": ai_col,
        "moves_made": moves_made,
        "num_moves": CF_NUM_MOVES,
        "moves_left": CF_NUM_MOVES - moves_made,
        "done": done,
    })


@app.route("/api/move", methods=["POST"])
def submit_move():
    """Chess: score one participant move (delegated to the chess engine)."""
    if not session.get("prolific_id"):
        return jsonify({"error": "not authorized"}), 401

    data = request.get_json(silent=True) or {}
    game = session.get("game", DEFAULT_GAME)
    puzzle = chess_engine.load_puzzle(_CHESS_DIR)
    frm, to, san = data.get("from"), data.get("to"), data.get("san")
    ply = int(data.get("ply") or 1)

    resp = chess_engine.evaluate_move(puzzle, ply, frm, to, san)

    record = {
        "game": game,
        "puzzle_source": puzzle.get("source"),
        "ply": ply,
        "from": frm,
        "to": to,
        "san": san,
        "fen_after": data.get("fen_after"),
        "correct": resp.get("correct"),
        "solved": resp.get("solved"),
        "eval": resp.get("eval"),
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    session["current_fen_state"] = record["fen_after"]
    safe_pid = session.get("safe_pid", "anon")
    with open(participant_dir(safe_pid) / "moves.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")

    return jsonify(resp)


@app.route("/api/game", methods=["POST"])
def submit_game():
    """Final, consolidated game submission written when the participant finishes."""
    if not session.get("prolific_id"):
        return jsonify({"error": "not authorized"}), 401

    data = request.get_json(silent=True) or {}
    record = {
        "game": session.get("game", DEFAULT_GAME),
        "moves": data.get("moves") or [],
        "final_fen": data.get("final_fen"),
        "end_reason": data.get("end_reason"),  # "manual" or "timeout"
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    safe_pid = session.get("safe_pid", "anon")
    with open(participant_dir(safe_pid) / "game_submission.json", "w") as f:
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
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    safe_pid = session.get("safe_pid", "anon")
    pdir = participant_dir(safe_pid)
    with open(pdir / "focus_events.jsonl", "a") as f:
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


def _render_ai_likert(answers, error, feedback=""):
    return render_template(
        "post_survey.html",
        items=AI_LIKERT_ITEMS,
        likert_options=LIKERT_OPTIONS,
        answers=answers,
        feedback=feedback,
        feedback_prompt=AI_FEEDBACK_PROMPT,
        error=error,
    )


@app.route("/post-survey", methods=["GET", "POST"])
def post_survey():
    gate = require_participant()
    if gate:
        return gate

    if request.method == "POST":
        page = request.form.get("page")

        # Page 1: NASA-TLX -> validate, stash in session, advance to page 2.
        if page == "tlx":
            answers = {key: (request.form.get(key) or "").strip() for key, _, _ in NASA_TLX_ITEMS}
            if any(val not in TLX_VALUES for val in answers.values()):
                return _render_tlx(answers, "Please rate every scale before continuing.")
            session["tlx_answers"] = answers
            return _render_ai_likert({key: "" for key, _ in AI_LIKERT_ITEMS}, None)

        # Page 2: AI Likert -> validate, persist both pages, finish.
        if page == "ai_likert":
            answers = {key: (request.form.get(key) or "").strip() for key, _ in AI_LIKERT_ITEMS}
            feedback = (request.form.get("additional_feedback") or "").strip()
            if any(val not in LIKERT_VALUES for val in answers.values()):
                return _render_ai_likert(answers, "Please answer every question before continuing.", feedback)
            tlx = session.get("tlx_answers", {})
            safe_pid = session.get("safe_pid", "anon")
            with open(participant_dir(safe_pid) / "post_survey.json", "w") as f:
                json.dump(
                    {
                        "game": session.get("game", DEFAULT_GAME),
                        "nasa_tlx": {key: int(val) for key, val in tlx.items()},
                        "ai_likert": {key: int(val) for key, val in answers.items()},
                        "additional_feedback": feedback,
                        "submitted_at": datetime.now(timezone.utc).isoformat(),
                    },
                    f,
                    indent=2,
                )
            session.pop("tlx_answers", None)
            # Demographics (age + education + occupation) come last, after surveys.
            return redirect(url_for("demographics"))

    # GET -> start at page 1.
    return _render_tlx({key: "" for key, _, _ in NASA_TLX_ITEMS}, None)


@app.route("/screened-skill", methods=["GET", "POST"])
def screened_thank_you_skill():
    """Screen-out #1: skilled Connect 4 players."""
    return render_template("thank_you.html", prolific_url=PROLIFIC_SCREENOUT_SKILL_URL)


@app.route("/screened-prompt", methods=["GET", "POST"])
def screened_thank_you_prompt():
    """Screen-out #2: low-quality prompt."""
    return render_template("thank_you.html", prolific_url=PROLIFIC_SCREENOUT_PROMPT_URL)


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
    if not session.get("prolific_id"):
        return jsonify({"error": "not authorized"}), 401
    data = request.get_json(silent=True) or {}
    safe_pid = session.get("safe_pid", "anon")
    convo_path = participant_dir(safe_pid) / "conversation.jsonl"

    try:
        with open(convo_path) as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []
    conversation_history = parse_conversation_history(lines)

    user_message = (data.get("message") or "").strip()
    user_ts = datetime.now(timezone.utc).isoformat()
    messages = conversation_history + [{"role": "user", "content": user_message}]
    if session["game"] == "chess":
        messages = messages + [{"role": "system", "content": "Current chessboard FEN state: " + session.get("current_fen_state", "No move has been made")}]
    elif session["game"] == "connect_four":
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

    hint = _solution_hint()           # only set in the AI_KNOWS_SOLUTION condition
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