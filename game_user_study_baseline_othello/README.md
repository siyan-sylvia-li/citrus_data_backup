# Othello study — no-AI baseline arm

Baseline (control) arm of `game_user_study_phase_othello`. Same puzzles, same
order, same engine, same scoring, same screeners. **One manipulation: there is no
in-game AI assistant at any point.**

Deploys under its own Cloud Run service (`game-study-baseline-othello`) and
therefore its own GCS bucket, so its data never mixes with the AI arm's.

## What differs from `game_user_study_phase_othello`

| | AI arm | this arm |
|---|---|---|
| Round 1 assistant | available after the gate | never |
| Round 1 timer | 480s (8 min) | **300s (5 min)** |
| Transfer puzzles (2, 3) | 90s each, no assistant | unchanged |
| Min. assistant exchanges | 5 before the finish button unlocks | not applicable (inert) |
| Post-TLX AI-assessment form | shown after round 1 | never shown |
| `conversation_p*.jsonl` | written | never written |

Round 1's 8 minutes budgeted both the ~4 min orientation cost the first batch
showed and the time spent chatting. Removing the assistant removes the chat
share; 5 minutes still clears the observed orientation cost plus the 30–65s per
subsequent decision. This is the only parameter besides AI presence that differs
between arms — factor it into any timing comparison.

## What is deliberately held constant

- **Prompt-writing pre-filter** (`prompt_filter.py`, median judge score ≥ 3.0).
  Both arms must recruit the same population, so the screener is untouched — the
  participant still writes a prompt for an AI agent, they just never use one.
- **Confidence gate** on round 1: unaided first move + confidence rating;
  optimal *and* confident (4–5) screens out. Identical criterion, so both arms
  admit the same people. Surviving it here simply resumes play — a wrong first
  move still resets the board, a correct one still keeps its progress, which
  keeps the scored attempt comparable across arms.
- **NASA-TLX** after round 1 and after the transfer block.
- All three Prolific screen-out codes.

## Dead-but-kept code

`/api/chat`, `_solution_hint()`, `load_system_prompt()`, `parse_conversation_history()`,
`OTH_MIN_AI_TURNS`, the `{% if render_chat %}` blocks in `templates/othello.html`,
and `templates/post_survey.html` are all unreachable — every round sets
`"ai": False`, so `ai_enabled()` is always False. They are left in place so
`app.py` stays diffable against the AI arm; flipping a round's `ai` flag is all
it takes to re-enable them.

## Running

```sh
./run_local.sh                 # http://localhost:5001, data -> ./recordings-local
PROJECT_ID=citrus-user-study ./deploy.sh
PROJECT_ID=citrus-user-study ./download_data.sh
```

Needs `TOGETHER_API_KEY` + `OPENAI_API_KEY` in `.env` for the judge panel only.
Without them the prompt filter fails open and admits everyone.

## Copy to review

The participant-facing wording was rewritten and is **draft** — `consent.html`
(study description, ~12 min breakdown, data-collection statement),
`othello_primer.html`, and the two round-intro popups in `OTH_ROUNDS`
(`app.py`). The primer previously said "two puzzles" when there are three; that
is corrected here.
