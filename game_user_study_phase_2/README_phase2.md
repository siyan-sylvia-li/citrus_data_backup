# game_user_study_phase_2

Minimal runnable copy of `game_user_study_phase_othello`, as the starting point for the
**vanilla vs. scaffolded** arms. Same Othello task, same coach, same rounds.

## What was copied

`app.py`, `prompt_filter.py`, `templates/`, `static/`, `requirements.txt`, deploy files,
and only the `games/othello/` files the app needs at runtime:

    engine.py  solver.py  llm_eval.py  precompute.py
    system_prompt.txt
    solution_*.json  puzzle_config_*.txt

## What was NOT copied, and why

- `recordings*/` — the phase-1 participant data. A new study writes its own.
- analysis scripts and notebooks (`*_analysis*.py`, `dialogue_act_*.py`, `*.ipynb`, `*.csv`)
  — those operate on collected data and belong with the analysis, not the app.
- puzzle authoring (`generate_puzzles.py`, `import_puzzles.py`, `othelloclub_*.txt`,
  `validate_engine.py`, `verify_against_source.py`, `test_othello.py`) — the puzzles are
  already solved; `solution_*.json` is what runtime reads.
- `.env` — **symlinked**, not duplicated, so the API keys live in one place.

## Verified

`import app` succeeds and registers 24 routes. `OTH_ROUNDS` is unchanged: round 0 is
`oc20260727` with the assistant and a 5-turn mandate; rounds 1-2 are `b220260706` and
`bg20260726` solo.

## Where the arm logic goes

The intervention is a pre-task step and a session flag. Points of contact:

1. **Assign the arm** at eligibility/consent, store it in `session`, and persist it into
   the participant directory so it survives into the recordings.
2. **Add the scaffolded pre-task page** — the Group A / Group B contrasting-cases
   exercise, its free-text response, the forced choice, and the correction. Vanilla gets
   a time-matched control page instead, or the study is confounded with time-on-task.
3. **Round-1 pop-up** — the existing round intro in `OTH_ROUNDS[0]["intro"]` is the
   natural hook for the "how will you use what you noticed" reminder.

The assistant itself does **not** change between arms: both get the same coach and the
same `system_prompt.txt`. The manipulation is entirely participant-side.
