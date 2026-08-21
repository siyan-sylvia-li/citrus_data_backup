import dspy
import dotenv

dotenv.load_dotenv()

import sys; sys.path.insert(0, "../game_user_study_phase_othello/games/othello")
import llm_eval

import threading
import litellm

MODEL_LIST = [l.strip() for l in open("model_list.txt") if l.strip()]
PRICING = {m: float(p) for m, p in (l.split("||") for l in MODEL_LIST)}

# Start generous and never truncate by choice: 200k is above every model here,
# so nothing is cut short by OUR number. A model whose window cannot take that
# says so, and _shrink drops it a step for that model only — the limit then
# comes from the model, not from us. (Qwen2.5-7B, 32k total, is the one that
# needs it; Kimi-K2.6 needed >32k of reasoning and broke when it was capped.)
DEFAULT_MAX_TOKENS = 200_000

# Timeout has to scale WITH that budget: a model allowed 200k tokens may spend
# many minutes producing them. 300s was too tight and timed out MiniMax-M3,
# which had completed fine before. num_retries stays at 1 because retries
# multiply the wait — at 2, a stuck call burns 3 x timeout before it surfaces.
LM_KWARGS = dict(temperature=0.0, num_retries=1, timeout=900)

_local = threading.local()
_stream_only: set = set()          # learned at runtime
_budget: dict = {}                 # model -> max_tokens, shrunk on demand
_budget_lock = threading.Lock()

def _kwargs(model_name):
    return dict(LM_KWARGS, max_tokens=_budget.get(model_name, DEFAULT_MAX_TOKENS))

def _lm(model_name):
    """One dspy.LM per (model, budget) per thread."""
    cache = getattr(_local, "lms", None)
    if cache is None:
        cache = _local.lms = {}
    key = (model_name, _budget.get(model_name, DEFAULT_MAX_TOKENS))
    if key not in cache:
        cache[key] = dspy.LM("together_ai/" + model_name, **_kwargs(model_name))
    return cache[key]

def _stream_call(model_name, payload):
    """Streaming-only models: talk to litellm directly and join the chunks.

    stream=True through dspy.LM does not work — litellm returns a
    CustomStreamWrapper and dspy tries to subscript it.
    """
    parts = []
    for chunk in litellm.completion(model="together_ai/" + model_name,
                                    messages=payload, stream=True,
                                    **_kwargs(model_name)):
        try:
            piece = chunk.choices[0].delta.content
        except (AttributeError, IndexError, KeyError):
            piece = None
        if piece:
            parts.append(piece)
    return "".join(parts)

def _shrink(model_name):
    """Quarter this model's output budget; False once there's no room left."""
    with _budget_lock:
        current = _budget.get(model_name, DEFAULT_MAX_TOKENS)
        if current <= 4096:
            return False
        _budget[model_name] = max(4096, current // 4)
        print(f"    (shrinking {model_name} max_tokens -> {_budget[model_name]})",
              flush=True)
        return True

def ask(model_name, payload):
    for _ in range(5):                       # room for several downshifts
        try:
            if model_name in _stream_only:
                return _stream_call(model_name, payload)
            return _lm(model_name)(messages=payload)
        except Exception as exc:
            msg = str(exc).lower()
            # Provider refuses non-streaming: remember and switch for good.
            if "streaming" in msg and model_name not in _stream_only:
                _stream_only.add(model_name)
                continue
            # Budget larger than the model's window: shrink and retry.
            if ("context" in msg or "max_tokens" in msg) and _shrink(model_name):
                continue
            raise
    raise RuntimeError(f"{model_name}: exhausted retries")

# Parallel across models; moves within the puzzle stay sequential. Failures are
# recorded per run (with the exception text) and never abort the batch.
rows = llm_eval.evaluate_models(
    list(PRICING), ["oc20260727"], ask=ask,
    max_workers=3,          # raise carefully: Together rate-limits per account
    messages=True,
    # repeats=3,            # temperature must be > 0 for repeats to differ
)

# .get() throughout: a failed run only carries {model, puzzle, repeat, error},
# so indexing the summary keys directly raises KeyError and loses the whole batch.
performance = {
    r["model"]: {
        "optimal_moves": r.get("optimal_moves"),
        "final_margin": r.get("final_margin"),
        "discs_lost": r.get("discs_lost"),
        "pricing": PRICING[r["model"]],
        "error": r.get("error") or r.get("aborted"),
        "max_tokens": _budget.get(r["model"], DEFAULT_MAX_TOKENS),
    }
    for r in rows
}

import pandas
performance_df = pandas.DataFrame(performance.values(), index=performance.keys())
performance_df.to_csv("model_performance.csv")


ok = {m: p for m, p in performance.items() if p["error"] is None}

def pareto_front(points, senses):
    """Indices of non-dominated rows. senses: "min" or "max" per column.

    Two objectives and a handful of models — not worth a dependency, and the
    paretoset package isn't installed in this env.
    """
    def better_or_equal(a, b):
        return all(a[i] <= b[i] if s == "min" else a[i] >= b[i]
                   for i, s in enumerate(senses))
    keep = []
    for i, p in enumerate(points):
        dominated = any(j != i and better_or_equal(q, p) and q != p
                        for j, q in enumerate(points))
        keep.append(not dominated)
    return keep

# Wanted: capable-looking models that still FAIL the puzzle.
# min final_margin (plays badly) x max pricing (expensive => presumed capable).
model_performance = pandas.DataFrame(
    {"final_margin": [p["final_margin"] for p in ok.values()],
     "pricing": [p["pricing"] for p in ok.values()]},
    index=list(ok),
)
mask = pareto_front(list(model_performance.itertuples(index=False, name=None)),
                    ["min", "max"])
paretoset_models = model_performance[mask]
paretoset_models.sort_values("pricing", ascending=False)


# "Best model that still fails" stated directly: of the models that did NOT win,
# the most expensive one. Self-contained so it works even if the Pareto cell
# above didn't run.
ok = {m: p for m, p in performance.items() if p["error"] is None}
failures = {m: p for m, p in ok.items() if p["final_margin"] <= 0}
errored = {m: p for m, p in performance.items() if p["error"] is not None}

print(f"{len(ok)}/{len(performance)} models ran; {len(failures)} of those failed oc20260727")
print("chance ≈ 44 discs lost, worst possible = 58, perfect = 0\n")
print(f"{'model':<45}{'margin':>8}{'discs lost':>12}{'$/Mtok':>9}")
for m, p in sorted(failures.items(), key=lambda kv: -kv[1]["pricing"]):
    flag = "  <- at/below chance" if p["discs_lost"] >= 44 else ""
    print(f"{m:<45}{p['final_margin']:>+8}{p['discs_lost']:>12}{p['pricing']:>9}{flag}")

if errored:
    print("\ndid not run (not evidence of failing the task):")
    for m, p in errored.items():
        print(f"  {m}: {p['error'][:90]}")

if failures:
    pick = max(failures.items(), key=lambda kv: kv[1]["pricing"])
    near = min(failures.items(), key=lambda kv: kv[1]["discs_lost"])
    print(f"\nmost expensive failing model: {pick[0]}  "
          f"(${pick[1]['pricing']}/Mtok, {pick[1]['discs_lost']} discs lost)")
    print(f"closest to solving it:        {near[0]}  "
          f"({near[1]['discs_lost']} discs lost, margin {near[1]['final_margin']:+d})")

