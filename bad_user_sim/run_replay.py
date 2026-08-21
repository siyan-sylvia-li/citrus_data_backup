"""Run the counterfactual replay experiment. Append-only; never touches study data.

For each stimulus pair, ask the study's coach BOTH the participant's original utterance
and its rewrite, at the identical reconstructed board with the identical conversation
history, then label both replies with the same fine panel used everywhere else.

Design notes that matter for reading the output:

* The original is REPLAYED rather than taken from the recording. The recorded reply came
  from a live session in August and cannot be reproduced -- see below -- so replaying
  both sides keeps the comparison inside one set of draws from the same coach.

* The coach is NOT deterministic. An early check suggested it was, but that was
  dspy.LM's default cache returning stored completions; with cache=False, two calls on
  an identical board and message agree at ~0.03 text similarity, and Board Report
  appears on roughly 0-3 of 8 replications. Caching is therefore disabled in ask().

* Despite that, there is ONE call per condition per pair. Replications and pairs are
  interchangeable for estimating a mean difference, and the pairs already exist: at
  N=150 the SE of the paired difference in Board Report rate is about .054, ample for
  the effects at issue. The cost is that no INDIVIDUAL pair is interpretable -- only
  the average over pairs is. Do not read single rows as evidence.

* `human` and `sim` stimuli are separate strata, never pooled. Human is the primary
  estimate; sim is a large-n replication over a different utterance distribution.

Results are appended, and every row records which pair it came from, so an interrupted
run resumes without redoing work and without overwriting anything.

    python run_replay.py --dry-run          # what would run, no API calls
    python run_replay.py                    # run everything outstanding
    python run_replay.py --contrast request_type --source human
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import threading
from collections import Counter
from pathlib import Path

import dotenv

BASE = Path(__file__).resolve().parent
dotenv.load_dotenv(BASE / ".env")

OUT_DIR = BASE / "replay_experiment"
STIMULI = OUT_DIR / "stimuli.jsonl"
RESULTS = OUT_DIR / "results.jsonl"


DEFAULT_COACH = "openai/gpt-5.5"     # app.py CHAT_MODEL -- what the human study used


def key(row, coach=DEFAULT_COACH):
    """Identifies a (pair, coach), so a resumed run skips what is already done.

    The coach is part of the key because the same stimuli are replayed across several
    assistant models: every act-level result is otherwise conditional on gpt-5.5, and
    holding the stimuli fixed while varying the model is what bounds that. Rows written
    before the model sweep have no `coach` field and are treated as gpt-5.5.
    """
    return (row["pid"], row["ply"], row["contrast"], row["source_pole"],
            row["original"][:80], coach)


def done_keys():
    if not RESULTS.exists():
        return set()
    out = set()
    for line in open(RESULTS):
        if not line.strip():
            continue
        r = json.loads(line)
        k = tuple(r["key"])
        if len(k) == 5:                       # pre-sweep row: implicitly gpt-5.5
            k = k + (DEFAULT_COACH,)
        out.add(k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--contrast", default=None)
    ap.add_argument("--source", default=None, choices=("human", "sim"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--per-cell", type=int, default=None,
                    help="cap pairs per (contrast, source, direction) so every coach "
                         "in the sweep sees the same stimulus mix")
    ap.add_argument("--workers", type=int, default=8,
                    help="pairs in flight; each pair is 2 coach + 6 panel calls")
    ap.add_argument("--coach", default=DEFAULT_COACH,
                    help="assistant model; default is the one the human study used")
    a = ap.parse_args()

    if not STIMULI.exists():
        print(f"no {STIMULI}; run build_stimuli.py first", file=sys.stderr)
        return 1
    stim = [json.loads(l) for l in open(STIMULI) if l.strip()]
    if a.contrast:
        stim = [s for s in stim if s["contrast"] == a.contrast]
    if a.source:
        stim = [s for s in stim if s.get("source") == a.source]

    already = done_keys()
    todo = [s for s in stim if tuple(key(s, a.coach)) not in already]
    if a.per_cell:
        # Deterministic subsample: same stimuli for every coach, so the model
        # comparison is within-stimulus rather than across different item sets.
        seen, keep = Counter(), []
        for st in sorted(stim, key=lambda x: (x["contrast"], x.get("source"),
                                              x["source_pole"], x["original"])):
            cell = (st["contrast"], st.get("source"), st["source_pole"])
            if seen[cell] >= a.per_cell:
                continue
            seen[cell] += 1
            if tuple(key(st, a.coach)) not in already:
                keep.append(st)
        todo = keep
    if a.limit:
        todo = todo[:a.limit]

    cells = Counter((s["contrast"], s.get("source"), s["source_pole"]) for s in todo)
    print(f"coach: {a.coach}")
    print(f"{len(stim)} stimulus pair(s); {len(already)} already run "
          f"(all coaches); {len(todo)} to do for this coach")
    for k, v in sorted(cells.items()):
        print(f"   {k[0]:<14}{k[1]:<7}{k[2]:<16}{v:>5}")
    print(f"\n~{2 * len(todo)} coach calls + ~{6 * len(todo)} panel calls")
    if a.dry_run or not todo:
        return 0

    import dspy
    import replay_probe as rp
    from assistant_acts_othello import OthelloFineAssistantSuite
    from assistant_agent import AssistantAgent
    suite = OthelloFineAssistantSuite()

    def ask(t, message):
        s = rp.board_at(t["moves"], t["ply"])
        coach = AssistantAgent(a.coach, s, game="othello")
        # dspy.LM caches by default. The coach is NOT deterministic -- at a fixed board
        # and message, Board Report appears on roughly 0-3 of 8 uncached replications --
        # so a cache hit would silently return a stored draw instead of a fresh one and
        # manufacture agreement between conditions that happen to share an input.
        coach.lm = dspy.LM(a.coach, cache=False,
                           **{k: v for k, v in coach.lm.kwargs.items() if k != "cache"})
        for pu, pa in t["history"]:
            coach.conversation_history.append({"role": "user", "content": pu})
            coach.conversation_history.append({"role": "assistant", "content": pa})
        return coach.answer(message)

    # Warm the game modules on the main thread. assistant_agent.load_game_modules
    # imports the study's engine under a mangled name (othello__engine); eight threads
    # importing it at once can each get a half-initialised module and fail with
    # "cannot import name 'BLACK'". One synchronous construction populates the cache.
    _warm = rp.board_at(todo[0]["moves"], todo[0]["ply"])
    AssistantAgent(a.coach, _warm, game="othello")

    OUT_DIR.mkdir(exist_ok=True)
    lock = threading.Lock()
    counter = {"n": 0}

    def one(t):
        """A pair is independent of every other, so pairs parallelise cleanly."""
        try:
            ra, rb = ask(t, t["original"]), ask(t, t["rewrite"])
            la = suite(utterance=ra.strip())
            lb = suite(utterance=rb.strip())
        except Exception as e:
            with lock:
                print(f"  {type(e).__name__}: {e}", file=sys.stderr)
            return None
        return dict(key=list(key(t, a.coach)), coach=a.coach,
                    contrast=t["contrast"], source=t.get("source"),
                    source_pole=t["source_pole"], target_pole=t["target_pole"],
                    pid=t["pid"], ply=t["ply"],
                    original=t["original"], rewrite=t["rewrite"],
                    reply_original=ra, reply_rewrite=rb,
                    acts_original=sorted(la.get("final") or []),
                    acts_rewrite=sorted(lb.get("final") or []),
                    seats_original=la.get("per_model"), seats_rewrite=lb.get("per_model"),
                    recorded_acts=t.get("recorded_acts"))

    with open(RESULTS, "a") as w:               # append: never clobber a prior run
        with concurrent.futures.ThreadPoolExecutor(a.workers) as ex:
            for row in ex.map(one, todo):
                if row is None:
                    continue
                with lock:
                    w.write(json.dumps(row) + "\n")
                    w.flush()
                    counter["n"] += 1
                    # Deliberately NOT printing per-pair acts. A single pair is one
                    # Bernoulli draw per condition and reads like a result when it is
                    # not; only the average over pairs is interpretable.
                    if counter["n"] % 10 == 0:
                        print(f"  {counter['n']}/{len(todo)} pairs", flush=True)
    print(f"\nwrote {counter['n']} row(s) to {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
