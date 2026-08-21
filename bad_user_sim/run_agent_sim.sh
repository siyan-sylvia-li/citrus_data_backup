#!/usr/bin/env bash
# Run agentic_sim.py for every preset persona x every model in model_list.txt.
#
#   ./run_agent_sim.sh              # run the sweep (Ctrl-C stops it cleanly)
#   ./run_agent_sim.sh stop         # kill a sweep, including orphaned runs
#   ./run_agent_sim.sh status       # what is running right now
#   JOBS=4 STAGGER=10 ./run_agent_sim.sh
#   nohup ./run_agent_sim.sh > sweep.log 2>&1 &     # detached, survives logout
#
# Runs JOBS sims concurrently. The inner loop is over models, so the jobs in
# flight at any moment are mostly DIFFERENT models — spreading load instead of
# hammering one endpoint's rate limit. One log per run under logs/; a failing run
# does not stop the sweep.
set -uo pipefail

cd "$(dirname "$0")"          # agentic_sim.py reads preset_personas/ relatively
mkdir -p logs

# Which game this sweep is for; agentic_sim.py reads the same variable. Log and
# results names carry the game so a Connect Four sweep does not overwrite the
# Othello logs for the same preset x model. Othello keeps the original names,
# so nothing already collected moves.
#   SIM_GAME="Connect Four" ./run_agent_sim.sh
GAME_SLUG="$(tr 'A-Z ' 'a-z_' <<<"${SIM_GAME:-Othello}")"
[[ "$GAME_SLUG" == "othello" ]] && GAME_TAG="" || GAME_TAG="${GAME_SLUG}_"

# Behaviour condition (constants.STUDENT_MODES / ASSISTANT_MODES). agentic_sim.py
# writes each non-default condition to its own run root, so these only need to reach
# it on the command line and into the log/results names:
#   STUDENT_MODE=think ./run_agent_sim.sh
#   ASST_MODE=no_exp SIM_GAME="Connect Four" ./run_agent_sim.sh
STUDENT_MODE="${STUDENT_MODE:-default}"
ASST_MODE="${ASST_MODE:-default}"
# QUESTIONS_REQUIRED fixes the consultation count so arms differ only in phrasing.
QUESTIONS_REQUIRED="${QUESTIONS_REQUIRED:-0}"
# ASSISTANT_OFF=1 runs the no-assistance control (assisted round without a coach).
ASSISTANT_OFF="${ASSISTANT_OFF:-}"
# ASSISTANT_MODEL swaps the coach itself (dspy naming, e.g. anthropic/claude-sonnet-5).
ASSISTANT_MODEL="${ASSISTANT_MODEL:-openai/gpt-5.5}"
# SIM_TOOL_FRAMING=shared_context tells the student the assistant sees the board and
# remembers the thread; agentic_sim.py reads it directly from the environment.
export SIM_TOOL_FRAMING="${SIM_TOOL_FRAMING:-default}"
# GATE=<name> ENFORCES an act knockout: violating messages are rejected before they
# reach the assistant, so the act is absent rather than merely reduced.
GATE="${GATE:-}"
COND_TAG=""
for t in "$STUDENT_MODE" "$ASST_MODE"; do
  [[ "$t" != "default" ]] && COND_TAG="${COND_TAG}${t}_"
done
if [[ -n "$ASSISTANT_OFF" ]]; then COND_TAG="${COND_TAG}noai_"
elif (( QUESTIONS_REQUIRED > 0 )); then COND_TAG="${COND_TAG}q${QUESTIONS_REQUIRED}_"; fi
[[ -n "$GATE" ]] && COND_TAG="${COND_TAG}gate-${GATE}_"
[[ "$SIM_TOOL_FRAMING" != "default" ]] && COND_TAG="${COND_TAG}${SIM_TOOL_FRAMING}_"
[[ "$ASSISTANT_MODEL" != "openai/gpt-5.5" ]] && \
  COND_TAG="${COND_TAG}asst-${ASSISTANT_MODEL//[\/:]/_}_"
RESULTS="logs/${GAME_TAG}${COND_TAG}results.tsv"

# ---- stopping ----------------------------------------------------------------
# Killing this script is NOT enough on its own: each run is a separate python
# process, and it keeps going after its parent dies — that is how a "stopped"
# sweep ends up as a dozen orphans still spending quota. Everything below kills
# the runs themselves.
#
# The [a] in the pattern is deliberate: it is a character class that matches the
# literal string, so the pgrep/pkill process does not match its OWN command line
# and kill itself (or this script).
RUN_PATTERN="[a]gentic_sim\.py --preset"
SWEEP_PIDFILE="logs/.sweep.pid"

running_runs() { pgrep -f "$RUN_PATTERN" 2>/dev/null; }

# The sweep loop is found by PID FILE, not by pattern. A pattern like
# "run_agent_sim.sh" also matches any shell that merely MENTIONS the script —
# the terminal you typed the command in, an editor, a tail -f — and killing
# those is far worse than leaving a sweep running. (Learned the hard way: an
# earlier version of this killed the shell that invoked it.)
sweep_loop_pid() {
  local pid
  [[ -f "$SWEEP_PIDFILE" ]] || return 0
  pid=$(<"$SWEEP_PIDFILE")
  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  [[ "$pid" == "$$" ]] && return 0                     # us; do not self-kill
  # Confirm the PID is still THIS script and not something that reused the
  # number since the file was written.
  grep -qa "run_agent_sim" "/proc/$pid/cmdline" 2>/dev/null && echo "$pid"
}

stop_everything() {
  local pids sweep
  # The loop first, so it cannot launch new runs while we kill the current ones.
  sweep=$(sweep_loop_pid)
  [[ -n "$sweep" ]] && { echo "stopping sweep loop (pid $sweep)"; kill -TERM "$sweep" 2>/dev/null; }
  pids=$(running_runs || true)
  if [[ -z "$pids" ]]; then
    echo "no agentic_sim runs to stop"
    return 0
  fi
  echo "stopping $(wc -w <<<"$pids") run(s)..."
  kill -TERM $pids 2>/dev/null
  for _ in $(seq 10); do                 # SIGTERM first: a run mid-puzzle exits
    sleep 1                              # on its own and keeps finished puzzles
    [[ -z "$(running_runs || true)" ]] && break
  done
  pids=$(running_runs || true)
  [[ -n "$pids" ]] && { echo "force-killing: $pids"; kill -KILL $pids 2>/dev/null; }
  sleep 1
  echo "remaining: $(running_runs | wc -l)"
  rm -f "$SWEEP_PIDFILE"
}

case "${1:-}" in
  stop)
    stop_everything
    exit 0
    ;;
  status)
    if [[ -n "$(running_runs || true)" ]]; then
      pgrep -af "$RUN_PATTERN" | sed 's/--student_model //'
      echo "-- $(running_runs | wc -l) run(s) live"
    else
      echo "nothing running"
    fi
    [[ -f "$RESULTS" ]] && echo "-- $(grep -c '^ok' "$RESULTS") ok, "\
"$(grep -c '^FAIL' "$RESULTS") failed so far  ($RESULTS)"
    exit 0
    ;;
esac

JOBS="${JOBS:-8}"
# Seconds between launches, to keep the opening burst from tripping a 429 when
# every student sends its long system prompt at once. Deliberately small: this
# delay is paid once per run (60 runs x 20s was 20 minutes of pure waiting),
# while agentic_sim.py backs off inside the run for the collisions that matter.
# If the logs fill with "(HTTP 429: waiting Ns)" the account is saturated —
# lower JOBS, since more concurrency then buys only more waiting.
STAGGER="${STAGGER:-5}"
# Which personas to sweep. Discovered from preset_personas/ so a preset added by
# build_preset_personas.py is picked up without editing this script, and overridable
# for a targeted run:
#   PRESETS="1 2 3" ./run_agent_sim.sh          # just these three
#   PRESETS="$(seq 6 20)" ./run_agent_sim.sh    # a slice of the new ones
if [[ -n "${PRESETS:-}" ]]; then
  # tr first: `read -ra` consumes only ONE line, so a newline-separated value --
  # which is exactly what the documented PRESETS="$(seq 6 19)" produces -- would
  # silently collapse to just "6" and run a fraction of the intended sweep.
  read -ra PRESETS <<<"$(tr '\n' ' ' <<<"$PRESETS")"
else
  mapfile -t PRESETS < <(ls preset_personas/preset_*.json 2>/dev/null \
    | sed 's#.*/preset_##; s#\.json##' | sort -n)
fi
(( ${#PRESETS[@]} )) || { echo "no presets found in preset_personas/"; exit 1; }

# One sweep at a time: every run draws on the same Together quota, so a second
# sweep does not run twice as fast, it makes both wait. Stale runs from a killed
# sweep keep going (the parent dies, the children do not), which is easy to miss.
if [[ -n "$(running_runs || true)" ]]; then
  echo "agentic_sim.py is already running:"
  pgrep -af "$RUN_PATTERN" | sed 's/^/  /'
  echo "Stop them with './run_agent_sim.sh stop', or set ALLOW_OVERLAP=1 to add to them."
  [[ -n "${ALLOW_OVERLAP:-}" ]] || exit 1
fi

# Claim the pidfile only now that we know we are the sweep that runs. Doing it
# before the guard meant a second, REFUSED invocation overwrote the live sweep's
# pid and then deleted it on the way out — leaving `stop` unable to find the loop,
# which would go on launching runs after they were killed.
echo $$ > "$SWEEP_PIDFILE"
# Ctrl-C (or a plain `kill` of this script) takes the runs down with it, instead
# of leaving them orphaned.
trap 'echo; echo "interrupted — cleaning up"; trap - INT TERM; stop_everything; exit 130' INT TERM
trap 'rm -f "$SWEEP_PIDFILE"' EXIT

# Pinned, not inherited: bare `python` on PATH is base miniconda, which has no
# pydantic_ai — so the sweep would only work from an already-activated shell.
PYTHON="${PYTHON:-/data/home/siyanli/miniconda3/envs/citrus/bin/python}"

# model_list.txt lines look like "org/model||price" — keep the model, drop the price.
# '#' comments are stripped FIRST: without that, a comment line survives sed (it has
# no "||") and is launched as a model named "# ...", which fails per-run and pollutes
# the results file. Dropped models are kept in the file as comments, so filtering them
# has to work.
#
# MODELS overrides the file for a one-off sweep shape, so narrowing a run does not
# mean editing (and then having to restore) the curated list:
#   MODELS="anthropic:claude-sonnet-5 openai:gpt-5.6-luna" ./run_agent_sim.sh
if [[ -n "${MODELS:-}" ]]; then
  read -ra MODELS <<<"$(tr '\n' ' ' <<<"$MODELS")"   # newline-safe, as with PRESETS
else
  mapfile -t MODELS < <(grep -v '^[[:space:]]*#' model_list.txt \
    | sed 's/||.*//' | grep -v '^[[:space:]]*$')
fi
(( ${#MODELS[@]} )) || { echo "no models to run"; exit 1; }

: >"$RESULTS"

run_one() {
  local preset=$1 model=$2 tag
  tag="${GAME_TAG}${COND_TAG}preset${preset}_${model//\//_}"
  if "$PYTHON" agentic_sim.py --preset "$preset" --student_model "$model" \
       --student_mode "$STUDENT_MODE" --asst_mode "$ASST_MODE" \
       --questions_required "$QUESTIONS_REQUIRED" ${ASSISTANT_OFF:+--assistant_off} \
       --assistant_model "$ASSISTANT_MODEL" ${GATE:+--gate "$GATE"} \
       >"logs/$tag.log" 2>&1; then
    printf 'ok\t%s\n' "$tag" | tee -a "$RESULTS"
  else
    printf 'FAIL\t%s\tlogs/%s.log\n' "$tag" "$tag" | tee -a "$RESULTS"
  fi
}

total=$(( ${#PRESETS[@]} * ${#MODELS[@]} ))

# Guard against a habitual bare `./run_agent_sim.sh` now costing 10x what it used
# to. The preset pool went from 5 to 57, so the default sweep went from 60 runs to
# ~680 -- each one a full multi-round game against a paid endpoint. Anything past
# MAX_RUNS has to be asked for explicitly.
MAX_RUNS="${MAX_RUNS:-120}"
if (( total > MAX_RUNS )) && [[ -z "${CONFIRM_LARGE:-}" ]]; then
  echo "refusing to launch $total runs (${#PRESETS[@]} presets x ${#MODELS[@]} models) > MAX_RUNS=$MAX_RUNS."
  echo "This spends real quota on every run. To proceed, either narrow the sweep:"
  echo "    PRESETS=\"\$(seq 6 15)\" $0"
  echo "or opt in to the full size:"
  echo "    CONFIRM_LARGE=1 $0        # or raise MAX_RUNS"
  exit 1
fi

echo "launching $total runs, $JOBS at a time, ${STAGGER}s apart"

running=0
first=1
for preset in "${PRESETS[@]}"; do
  for model in "${MODELS[@]}"; do
    (( first )) || sleep "$STAGGER"
    first=0
    run_one "$preset" "$model" &
    if (( ++running >= JOBS )); then
      wait -n
      (( running-- ))
    fi
  done
done
wait

echo
echo "done: $(grep -c '^ok' "$RESULTS") ok, $(grep -c '^FAIL' "$RESULTS") failed  ($RESULTS)"
grep '^FAIL' "$RESULTS" || true
