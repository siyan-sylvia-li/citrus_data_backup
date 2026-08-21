#!/usr/bin/env bash
# Wait for the in-flight sweep to drain, then run the Connect Four sweep.
#
# Sweeps must not overlap: every run draws on the same Together quota, so two at
# once does not go twice as fast, it makes both wait -- and run_agent_sim.sh
# refuses to start on top of live runs anyway. This just queues the second one.
#
# Launch DETACHED so it outlives the shell that started it:
#   nohup setsid ./queue_cf_sweep.sh > logs/queue_cf.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"

RUN_PATTERN="[a]gentic_sim\.py --preset"

# Three consecutive quiet minutes, not one: the sweep loop staggers launches 5s
# apart and a run can exit just before the next starts, so a single quiet check
# would fire while the Othello sweep is merely between runs.
quiet=0
echo "$(date -Is) waiting for the current sweep to finish..."
while (( quiet < 3 )); do
  if pgrep -f "$RUN_PATTERN" >/dev/null 2>&1; then quiet=0; else quiet=$(( quiet + 1 )); fi
  sleep 60
done
echo "$(date -Is) sweep drained; starting Connect Four"

# Same five models as the Othello sweep, on purpose: holding the model set fixed
# across games is what lets a game effect be read within model instead of being
# confounded by which models were run where.
export MODELS="together:meta-llama/Llama-3.3-70B-Instruct-Turbo openai:gpt-5.6-luna anthropic:claude-sonnet-5 together:google/gemma-4-31B-it together:moonshotai/Kimi-K2.6"
export PRESETS="$(seq 6 23)"
export JOBS=10
export SIM_GAME="Connect Four"

./run_agent_sim.sh
echo "$(date -Is) Connect Four sweep finished"
