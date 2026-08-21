from pathlib import Path

# ------- Where the real studies live -------- #
# The simulated assistant reads each study's OWN files — system prompt, engine,
# solution tables — rather than copies, so the sim and the study cannot drift
# apart on anything except the two prompt strings duplicated in
# assistant_agent.py (which check_matches_study() watches).
ROOT = Path(__file__).resolve().parent.parent

GAMES = {
    "othello": {
        "label": "Othello",
        "study_dir": ROOT / "game_user_study_phase_othello",
        "game_dir": ROOT / "game_user_study_phase_othello" / "games" / "othello",
        "puzzle": "oc20260727",          # the assisted round
        # The study's rounds, in order (app.py OTH_ROUNDS). Round 2 is a BLOCK of
        # two puzzles played back to back with no assistant and no survey between.
        "rounds": [
            {"puzzles": ["oc20260727"], "ai": True},
            {"puzzles": ["b220260706", "bg20260726"], "ai": False},
        ],
        "reasoning_effort": "medium",    # matches that study's /api/chat
    },
    "connect_four": {
        "label": "Connect Four",
        "study_dir": ROOT / "game_user_study_phase_1",
        "game_dir": ROOT / "game_user_study_phase_1" / "games" / "connect_four",
        "puzzle": "15",
        "rounds": [                      # app.py CF_ROUNDS in phase 1
            {"puzzles": ["15"], "ai": True},
            {"puzzles": ["w4p6"], "ai": False},
        ],
        "reasoning_effort": "low",       # phase 1 used low
    },
}


def game_config(game: str) -> dict:
    """Accept "Othello", "othello", "Connect Four", "connect_four"."""
    key = game.strip().lower().replace(" ", "_").replace("-", "_")
    if key not in GAMES:
        raise ValueError(f"unknown game {game!r}; expected one of {list(GAMES)}")
    cfg = dict(GAMES[key], key=key)
    cfg["app"] = cfg["study_dir"] / "app.py"
    cfg["system_prompt"] = cfg["game_dir"] / "system_prompt.txt"
    return cfg


# ------- Student Agent Specific -------- #

USER_SYSTEM_PROMPT = """You are a participant in a gameplay study. You are playing your first {game} endgame puzzle, and you have an AI assistant you can ask questions to at any point while playing. You will NOT have access to this AI assistant for any future endgame puzzles.

# Who you are
{persona}
You are {age} years old. Your highest level of education completed is: {education}. Your occupation: {occupation}. How often you use AI tools (like ChatGPT, Claude, or Gemini): {ai_frequency}.
Stay in character as this specific person — your curiosity, confidence, patience, comfort with AI assistants, and the way you phrase things should reflect this background.

# Your voice (VERY IMPORTANT)
The words you use must give away this person's education — a high-school student and a data scientist should NOT sound alike. Match your register to WHO YOU ARE above.

# How you behave while playing
- React to the game state the way THIS person would.
- When something is unclear, confusing, or you want to go deeper, ask the AI assistant a question in your own natural voice (like a real person typing to a chatbot — short, informal, not a polished essay).
- You will not always have a question. Behave naturally; do not force a question every turn.
- Let your background and comfort with AI shape how you engage: someone who uses AI tools often asks fluently and chases follow-ups; someone who rarely uses them is more hesitant, less sure how to phrase questions, and may ask for reassurance or restatements.
- You have only about 8 minutes for the WHOLE game, so you cannot ask about everything.
- Keep each message short, like a quick chat message (a sentence or two). Do NOT restate the assistant's whole answer back before asking.
- Never re-ask something the assistant has already explained."""


# ------- Behaviour manipulations -------- #
# Appended to the prompts above to induce a specific dialogue style. Four rules
# these follow, learned from what the first drafts fought with:
#
# 1. NAME THE LINE BEING OVERRIDDEN. Every one of these contradicts something in
#    the base prompt ("keep each message short" for the student, "be AS SUCCINCT
#    AS POSSIBLE" and the Socratic framing for the assistant). An unacknowledged
#    contradiction is resolved differently by different models — strong ones
#    follow the later, more specific instruction, weaker ones revert to the base
#    — which turns the manipulation into a model x condition interaction. Saying
#    which rule yields removes the ambiguity instead of hoping it resolves well.
# 2. CHANGE FORM, NOT FREQUENCY. Each says "when you ask ...", never "always
#    ask". The base prompt's "do not force a question every turn" is preserved,
#    so turn COUNT stays comparable across conditions and only the composition of
#    acts moves. An "ALWAYS ASK" variant would confound style with volume.
# 3. HOLD THE ANSWER POLICY CONSTANT. The assistant modes change whether the
#    reasoning is stated, not whether the answer is given — otherwise "explains"
#    vs "doesn't explain" is confounded with "withholds the move" vs "hands it
#    over", and Provide Correct Answer moves for the wrong reason.
# 4. BUDGET INSTEAD OF PROHIBITION. "Two or three sentences" and "under about
#    four sentences" beat "DO NOT ELABORATE": an absolute produces either
#    mechanical minimalism or silent non-compliance, and it flattens the persona
#    register the study depends on.
#
# Deliberately NOT shouted in capitals. The base prompts reserve capitals for a
# few genuine emphases; a wall of capitals competes with those rather than
# standing out, and reads as boilerplate to skim past.
STUDENT_MODES = {
    "default": "",
    # Targets Think Aloud. Does not mention proposing a candidate move, so it
    # should not also inflate Common Ground Question.
    "think": """

# For this session: think out loud
When you ask the assistant something, first say what you are thinking in your own
words — what you notice on the board, what you are weighing, or what you expect to
happen — and then ask. For this session that replaces the "keep each message short
(a sentence or two)" rule above: two or three sentences is right. Everything else
still holds — your own voice and register, no forced question on a turn where you
have nothing to ask, and no restating the assistant's answer back to it.""",
    # Reworded after the v1 "think" arm zeroed out Llama-3.3-70B: 15 turns in the
    # default condition -> 1 under the manipulation, while Sonnet was unaffected
    # (15 -> 16). Llama runs at ~1.00 acts per turn and 0% Think Aloud natively, so
    # "preface each question with two or three sentences of reasoning" asks for an
    # utterance shape it does not produce — and given "ask this way or don't ask",
    # it stopped asking. Under "solution", which wants something SHORTER, it asked
    # MORE than default (30 turns), confirming the format requirement was the cause
    # rather than any general suppression.
    #
    # Two changes: frequency is protected explicitly and FIRST, before any style
    # instruction, and the bar drops from "two or three sentences" to a single
    # clause. No worked example, deliberately — an example would model one question
    # type and shift that act's share along with Think Aloud.
    "think_v2": """

# For this session: say what you are thinking
Keep asking the assistant exactly as often as you otherwise would. Never skip a
question, and never hold one back, because you are unsure how to word the reasoning —
asking matters more than the wording.
When you do ask, put a few words of what you are thinking in front of the question:
what you noticed, what you are weighing, or what you expect to happen. A single
clause is enough; it does not have to be a full sentence. For this session that
relaxes the "keep each message short (a sentence or two)" rule above to about two
sentences, and your own voice and register still apply.""",
    # Few-shot imitation of how real participants typed. Motivated by a validity
    # finding: simulated and human students carry the SAME act labels but ask very
    # different questions, and the assistant answers accordingly. Simulated students
    # write long formal requests ("which of my legal moves will result in the most
    # discs, assuming my opponent plays optimally?"), which draw a bare verdict --
    # Move Verdict lands on 90.8% of simulated assistant turns versus 41.0% of human
    # ones, while Hint (2.6% vs 26.3%) and Prompt (2.0% vs 29.3%) nearly vanish.
    #
    # The examples are REAL participant messages from the Connect Four study,
    # deliberately a DIFFERENT game from the Othello runs this is used on: style
    # transfers, board content cannot leak. Median human message is 8 words.
    #
    # It teaches STYLE only. It deliberately does not say "ask concept questions to
    # get principles" or name any act -- that would train the outcome being measured
    # and make the elicitation test circular. Whether human-like phrasing recovers
    # human-like assistant behaviour is the empirical question.
    "human_style": """

# For this session: type the way a real participant does
Below are real messages that people in an earlier study (a different game — Connect
Four) actually sent the assistant. Copy how they WRITE — the length, the register,
the way they think out loud — not what they ask about. Your game is Othello and the
board is your own.

  "what about 2?"
  "Where should I play next"
  "was my last move good?"
  "I have one move left. How should I use it?"
  "Ah oh! Are there any traps I'm missing?"
  "I'm not seeing any big yellow threats. Can you see any?"
  "Ok, I could block yellow with column 5, row 3"
  "I'm thinking I have to do 4-4 now otherwise yellow will win right?"
  "Would row 6 not be a threat in the future?"
  "I only have one move left and don't see anything obvious."
  "Im not a confident player, is it wise to add a counter to 4,6"
  "ok, what move you suppose will be better?"
  "what would you suggest"
  "Which move would be the best one and why"

Notice how short they are — most are under a dozen words, several are three or four.
People often name a square they are considering and ask whether it is right, or just
say what they notice and stop. They do not spell out the full list of legal moves,
they do not ask for the provably optimal line, and they do not write in complete
formal sentences. Typos and half-sentences are fine.""",
    # human_style plus a terse think-aloud nudge. Motivated by what human_style did
    # and did not fix: Common Ground Question landed on the human value (27.9 -> 37.3
    # vs 36.5), but Think Aloud moved the WRONG way (29.1 -> 23.2 vs 34.5).
    #
    # That matters because of which assistant acts each one elicits in the human data.
    # Common Ground Question pulls Negative Feedback (1.47) and Worked Line (1.24);
    # only Think Aloud pulls Prompt (1.35) and Hint (1.31) -- the two acts the
    # simulated assistant almost never produces (2.0% and 2.6%, against human 29.3%
    # and 26.3%). Recovering Think Aloud is therefore the remaining lever on the
    # scaffolding gap.
    #
    # The wording problem: the human examples ALREADY contain think-aloud, and Think
    # Aloud still fell. The brevity instruction appears to win -- given a dozen words
    # the model spends them on the question. So this does NOT add a length budget on
    # top (that would re-lengthen the messages and undo the human-likeness it is built
    # on). It points at the examples that already do both and asks for the same trade:
    # a few words of what you notice, THEN the question, inside the same short message.
    "human_style_think": """

# For this session: type the way a real participant does
Below are real messages that people in an earlier study (a different game — Connect
Four) actually sent the assistant. Copy how they WRITE — the length, the register,
the way they think out loud — not what they ask about. Your game is Othello and the
board is your own.

  "what about 2?"
  "Where should I play next"
  "was my last move good?"
  "I have one move left. How should I use it?"
  "Ah oh! Are there any traps I'm missing?"
  "I'm not seeing any big yellow threats. Can you see any?"
  "Ok, I could block yellow with column 5, row 3"
  "I'm thinking I have to do 4-4 now otherwise yellow will win right?"
  "Would row 6 not be a threat in the future?"
  "I only have one move left and don't see anything obvious."
  "Im not a confident player, is it wise to add a counter to 4,6"
  "ok, what move you suppose will be better?"
  "what would you suggest"
  "Which move would be the best one and why"

Notice how short they are — most are under a dozen words, several are three or four.
People often name a square they are considering and ask whether it is right, or just
say what they notice and stop. They do not spell out the full list of legal moves,
they do not ask for the provably optimal line, and they do not write in complete
formal sentences. Typos and half-sentences are fine.

Notice too that several of them say what they are THINKING before they ask — "I'm not
seeing any big yellow threats", "I'm thinking I have to do 4-4 now", "I only have one
move left and don't see anything obvious". Do that as often as they do. It does not
make the message longer: it is a handful of words in front of the question, inside the
same short message, not an extra sentence added on.""",
    # KNOCKOUT fragment. Suppresses Knowledge Deficit Question -- asking about a
    # concept, a rule, the board, or clarifying what the assistant said.
    #
    # Chosen because it is the elicitation link with the strongest cross-population
    # agreement: KDQ -> Board Report is lift 1.51 in humans and 3.86 in simulation,
    # and KDQ -> Move Verdict is 0.18 in BOTH. Board Report is also the act that
    # trends positive with human transfer (+0.231) while Move Verdict trends negative
    # (-0.198), so suppressing KDQ makes a directional prediction that can fail:
    # board description should fall and bare verdicts should rise.
    #
    # Worded behaviourally, never by act name -- naming the taxonomy would teach the
    # coder's categories rather than the behaviour, and the manipulation check has to
    # stay independent of the manipulation.
    "no_concept_q": """

# For this session: do not ask the assistant to explain things
Never ask how the game works, what a term means, why a move is good or bad, or what
is happening on the board, and never ask the assistant to clarify or expand on
something it already said. You can still say what you are thinking, name a move you
are considering and ask whether it is right, or ask what to play. Just never ask for
an explanation.""",
    # KNOCKOUT fragment, second in the series. Suppresses Think Aloud -- saying what
    # you notice or are weighing before/while asking.
    #
    # Chosen for the same reason as no_concept_q: strong coupling that agrees across
    # populations. Think Aloud -> Hint is lift 1.55 in humans and 1.37 in simulation,
    # and Think Aloud -> Prompt is 1.58 human (0.93 sim, weaker). Hint and Prompt are
    # the two acts the simulated assistant almost never produces, so this asks whether
    # suppressing the student behaviour that summons them pushes them lower still.
    #
    # Note the asymmetry with no_concept_q: Think Aloud is not a question type, so
    # suppressing it removes the REASONING while leaving the request intact. Asking is
    # explicitly protected below, because the v1 "think" arm showed the reverse
    # instruction (demanding reasoning) could stop a model asking at all.
    "no_think_aloud": """

# For this session: keep your reasoning to yourself
Ask the assistant exactly as often as you otherwise would — this changes only what
you put in the message, never whether you send one. When you do write, just ask the
thing. Do not say what you notice on the board, what you are weighing, what you
expect to happen, or how you got to the question. No preamble and no reasoning: the
request alone.""",
    # Targets Solution Request as the taxonomy defines it: an OPEN request for the
    # move with no candidate proposed. The earlier wording ("directly ask your
    # questions") was ambiguous — a terse "what's a double threat?" satisfies it
    # while coding as Knowledge Deficit Question, i.e. the wrong act entirely.
    "solution": """

# For this session: ask for the answer, not the reasoning
When you ask the assistant something, ask straight out for what to do — "what
should I play here?", "what's the best move?" — in one short sentence. Do not
explain what you are thinking, and do not name a move of your own for the assistant
to check. Everything else still holds: your own voice and register, and no forced
question on a turn where you have nothing to ask.""",
}

ASSISTANT_MODES = {
    "default": "",
    "explain": (
        "For this participant, always include the reasoning behind your answer: name "
        "the feature of the position that decides it and why, in one or two sentences. "
        "For this session that overrides 'be AS SUCCINCT AS POSSIBLE' only so far as a "
        "bare answer is never enough — stay under about four sentences. It does not "
        "change WHEN you give the answer: keep following the rule about not providing "
        "answers unless explicitly asked."),
    # Pushes Board Report, the act that trends POSITIVE with human transfer
    # (rho +0.231, p=.014, n=112) while Move Verdict trends negative (-0.198).
    # Describing the position and leaving the move to the learner is the
    # assistance-dilemma prediction, and this is the cheap way to test it: it holds
    # the coach model fixed, so it stays comparable to the human data, which was
    # collected with this same model.
    #
    # Deliberately does NOT forbid answering -- that would confound "describes more"
    # with "withholds the answer", which is the same confound that made the first
    # no_exp draft uninterpretable. The answer policy is unchanged; only the amount
    # of board description moves.
    "describe": (
        "For this participant, always begin by describing what is actually on the "
        "board that bears on their question — which squares are held by whom, what is "
        "playable, what is threatened, how the counts stand — before anything else. "
        "One or two sentences of concrete position description, naming squares. For "
        "this session that overrides 'be AS SUCCINCT AS POSSIBLE' to the extent that a "
        "reply with no board description is never enough. It does not change WHEN you "
        "give the answer: keep following the rule about not providing answers unless "
        "explicitly asked."),
    "no_exp": (
        "For this participant, give only the conclusion — the move, or a one-line "
        "factual answer to what was asked — with no justification, board analysis, "
        "alternatives, or follow-up question. One sentence. If asked 'why', still "
        "answer in one sentence without elaborating. For this session that overrides "
        "the Socratic framing above. It does not change WHEN you give the answer: keep "
        "following the rule about not providing answers unless explicitly asked."),
}



# ------- Act gates: ENFORCE a knockout instead of asking for one -------- #
# Prompt-based knockouts only reduce a dose: "no_concept_q" cut Knowledge Deficit
# Question from 10.4% to 3.8%, not to zero, so a null downstream is ambiguous between
# "the act does not matter" and "not enough of it was removed".
#
# A gate closes that. The message is checked BEFORE it reaches the assistant and a
# violation is sent back as a ModelRetry, so the act cannot occur at all and the
# manipulation is exact rather than partial.
#
# Two deliberate limits:
#  * BLOCK is deliberately narrow and ALLOW protects the acts we do NOT want removed.
#    "what should I play?" opens with "what" but is a Solution Request; blocking it
#    would confound the knockout with a second suppression.
#  * The retry text never names the taxonomy. Telling the student "that was a
#    Knowledge Deficit Question" would teach it the coding scheme, and the
#    manipulation check has to stay independent of the manipulation.
#
# A gate is ENFORCEMENT, not instruction: it buys a clean causal contrast at the cost
# of ecological validity, since real participants are not gated. That is the right
# trade for a mechanism experiment and the wrong one for a fidelity claim.
# Model used by a classifier gate. gpt-5.4-mini is deliberately the SAME model that
# holds the OpenAI seat in the annotation panel, so the gate and the manipulation
# check share one definition of the act. A regex gate does not: measured against
# 3,119 panel-labelled simulated turns it caught only 33% of real concept questions
# (and 43% of human ones), while wrongly rejecting ~5% of Solution Requests -- which
# would have confounded the knockout with a second, unintended suppression.
GATE_MODEL = "openai/gpt-5.4-mini"

ACT_GATES = {
    # Knowledge Deficit Question: asking about a concept, a rule, the board, or
    # asking the assistant to clarify what it said.
    "concept_q": {
        # The act the classifier must not let through. Everything below is the
        # regex fallback, kept only for --gate_mode regex.
        "act": "Knowledge Deficit Question",
        "block": r"\b(why|explain|what do(es)? .*mean|what is|what are|how do(es)?|"
                 r"clarify|elaborate|don'?t understand|confused about|what happened)\b",
        "allow": r"\b(what should i|what would you|which move|what move|best move|"
                 r"what next|what now|where should i|should i play)\b",
        "retry": ("Rephrase that without asking for an explanation — say what you are "
                  "thinking, name a move you are weighing and ask about it, or ask what "
                  "to play."),
    },
    # Solution Request: asking the assistant to pick the move ("which one?",
    # "what should I play?", "is g2 the way back?").
    #
    # AMPLIFICATION, not suppression. concept_q failed because the act it removed
    # was only 7% of baseline messages -- there was nothing there to take away. This
    # gate targets the act the baseline is SATURATED in (~68% of messages), so it
    # fires often enough to actually reshape the consultation, and the displaced
    # messages have to land somewhere other than where they already are.
    #
    # The retry names the replacement rather than just the prohibition (rule 1): a
    # bare "don't ask that" gets answered with a hedged solution request, which is
    # the same act wearing a question mark.
    "solution_req": {
        "act": "Solution Request",
        "block": r"\b(what should i|which (one|move)|what move|best move|what next|"
                 r"what now|where should i|should i play|what do you recommend)\b",
        "allow": r"\b(why|what makes|how do you tell|in general|rule of thumb)\b",
        "retry": ("The assistant will not pick your move for you. Ask about the idea "
                  "behind the position instead — what makes a square risky, how to "
                  "compare two moves in general, what rule applies here — and then "
                  "choose the move yourself."),
    },

    # ---- REQUIREMENT gates: block anything that LACKS the target act ------------
    # The block gates above can only take an act away, and the displaced messages
    # land wherever the model finds easiest -- concept_q's went straight into
    # Solution Request. These name what the message must CONTAIN, so the target act
    # is the thing being manipulated rather than a side effect of removing another.
    #
    # Headroom against the human_style_q5 Kimi baseline (panel labels, n=50):
    #   Knowledge Deficit Question  24%  -> 76 points of room
    #   Think Aloud                 60%  -> 40 points
    #   Common Ground Question      70%  -- too near ceiling to manipulate
    "require_kdq": {
        "require": ["Knowledge Deficit Question"],
        "retry": ("Before you ask what to do, ask what you don't understand. Put a "
                  "real question about the idea in this message — what a term means, "
                  "why something works the way it does, what rule is operating here."),
    },
    "require_think": {
        "require": ["Think Aloud"],
        "retry": ("Say your reasoning out loud first. Walk through what you are seeing "
                  "and how you are weighing it, in this message, before you ask "
                  "anything."),
    },
    "require_both": {
        "require": ["Knowledge Deficit Question", "Think Aloud"],
        "all": True,       # both must be present, not either
        "retry": ("Two things have to be in this message: your reasoning worked "
                  "through out loud, AND a question about something you genuinely do "
                  "not understand — not which move to play, but what idea you are "
                  "missing."),
    },
}

# A gate that fires on most messages can exhaust the student agent's retry budget
# and kill the run outright. After this many consecutive rejections on one
# consultation the message is let through and counted as a passthrough, so the
# manipulation leaks in a way the summary records rather than losing the cell.
GATE_MAX_STREAK = 2


# ------- Persona vocabulary -------- #
# The preset personas use their own shorthand ("master", "tech",
# "multiple_daily"); the study's demographics form uses different values
# ("postgrad", "it", "several_times_day"). Both are accepted so a persona can be
# written either way and still render as prose in the system prompt.
EDU_LABELS = {
    "less_than_hs": "less than high school", "hs_incomplete": "some high school",
    "hs_graduate": "high school graduate", "some_college": "some college",
    "associate": "a two-year associate degree", "bachelors": "a bachelor's degree",
    "bachelor": "a bachelor's degree", "some_postgrad": "some postgraduate study",
    "postgrad": "a postgraduate degree", "master": "a master's degree",
    "phd": "a doctorate",
}
OCC_LABELS = {
    "hospitality": "hospitality or service", "healthcare": "health care",
    "manufacturing": "manufacturing, mining or construction", "retail": "retail and trade",
    "education": "education", "finance": "banking, finance or insurance",
    "transportation": "transportation", "government": "government or public administration",
    "it": "information technology", "tech": "information technology",
    "agriculture": "agriculture or forestry",
    "professional": "professional, scientific and technical services",
    "arts": "arts, entertainment and recreation", "other": "another field",
}
AIF_LABELS = {
    "never": "never", "less_than_monthly": "less than monthly",
    "about_monthly": "about monthly", "few_times_month": "a few times a month",
    "weekly": "weekly", "few_times_week": "a few times a week",
    "daily": "daily", "several_times_day": "several times a day",
    "multiple_daily": "several times a day",
}

# Questions the simulated participant may ask. The real batches ran 1-15 turns
# with a median around 5, so this is the observed centre rather than a guess.
DEFAULT_QUESTION_BUDGET = 10


# ------- What each model will accept -------- #
# Together rejects an output budget the model cannot honour, and it does so as a
# 4xx on the FIRST request of a run — so one wrong number loses a whole sweep
# entry before a single move is played:
#   422 "`inputs` tokens + `max_new_tokens` must be <= 32769"  (Qwen2.5-7B)
#   400 "Input validation error"                              (gpt-oss-20b, no detail)
# Note the constraint is on input PLUS output, so a budget that works on turn 1
# can fail on turn 20 as the transcript grows.
#
# Mostly the budgets check_models.py settled on (the max_tokens column of
# model_performance.csv) — but treat those as OPTIMISTIC here: check_models asks
# one question at a time, while this sim carries a whole multi-round transcript,
# so the input side is far larger and a value that never had to shrink there
# still can here. Qwen3.7-Plus is the measured example: 50_000 passes on turn 1
# and is rejected mid-game. Anything unlisted starts at DEFAULT_MAX_TOKENS;
# _adapt_to_provider() shrinks whatever is still too big, at the cost of
# restarting that puzzle.
DEFAULT_MAX_TOKENS = 100_000

MODEL_MAX_TOKENS = {
    "MiniMaxAI/MiniMax-M3": 200_000,
    "moonshotai/Kimi-K2.6": 200_000,
    "zai-org/GLM-5.2": 200_000,
    "openai/gpt-oss-120b": 200_000,
    "openai/gpt-oss-20b": 50_000,
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": 50_000,
    "Qwen/Qwen2.5-7B-Instruct-Turbo": 12_500,
    "google/gemma-4-31B-it": 200_000,
    "Qwen/Qwen3.7-Plus": 12_500,     # 50_000 was rejected mid-game, see above
}

# Models the provider will not serve without stream=True (400 streaming_required).
# Runtime detection covers the rest; listing them here only saves one wasted call.
STREAM_ONLY_MODELS = {
    "Qwen/Qwen3.7-Plus",
}


def model_max_tokens(model: str) -> int:
    """Seed output budget for a model, named with or without a provider prefix."""
    return MODEL_MAX_TOKENS.get(bare_model_name(model), DEFAULT_MAX_TOKENS)


def bare_model_name(model: str) -> str:
    """"together:org/model" -> "org/model", the name the provider itself knows."""
    return str(model).split(":", 1)[-1] if ":" in str(model) else str(model)
