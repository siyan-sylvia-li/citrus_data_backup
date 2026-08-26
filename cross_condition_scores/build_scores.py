"""One scores.csv per condition, plus a combined file, for cross-condition comparison.

Reuses game_user_study_phase_2/score_from_logs.py for the scoring itself (moves_*.jsonl is
the source of truth, not the app's oth_score_*.json summaries), and adds what that script
cannot know on its own:

  * CONDITION membership. Three conditions live inside one directory -- phase 2's vanilla,
    v2 and v9 arms are separated by intervention.json, not by folder -- so filtering has to
    happen per participant, which `score_from_logs.py --dir` cannot do.

  * THE `ai` FLAG. Every copy of score_from_logs.py hardcodes R1 as has_AI=True, but the
    baseline studies played R1 with no assistant at all (0 conversation files against 152
    R1 move logs). Left alone, the CSV would claim the no-AI baselines were assisted. The
    flag is therefore taken from the condition definition and the script's value ignored.

    python build_scores.py            # writes per-condition CSVs + scores_all.csv
    python build_scores.py --list     # just show what each condition matches
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

CITRUS = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent

# Import the scorer from phase 2. All four copies in the repo have identical ROUNDS and
# scoring, so one is enough -- and using one means a scoring change cannot silently apply
# to some conditions and not others.
_spec = importlib.util.spec_from_file_location(
    "sfl", CITRUS / "game_user_study_phase_2" / "score_from_logs.py")
sfl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sfl)


def _variant(d: Path):
    f = d / "intervention.json"
    return json.load(open(f)).get("variant") if f.is_file() else None


def _form(d: Path):
    f = d / "intervention.json"
    return json.load(open(f)).get("form") if f.is_file() else None


def _has(d: Path, name: str) -> bool:
    return (d / name).is_file()


# (condition, population, root, R1 had an assistant, per-participant filter)
CONDITIONS = [
    ("baseline_adopters", "adopters", "game_user_study_baseline_othello/recordings-download-adopters",
     False, lambda d: True),
    ("vanilla_adopters", "adopters", "game_user_study_phase_othello/recordings-download",
     True, lambda d: True),
    ("baseline_non_adopters", "non-adopters", "game_user_study_baseline_othello/recordings-download",
     False, lambda d: True),
    # Phase 2's three arms share one directory. Vanilla = reached demographics but has no
    # intervention.json; the two scaffolded arms are told apart by the variant field.
    ("vanilla_non_adopters", "non-adopters", "game_user_study_phase_2/recordings-download",
     True, lambda d: _has(d, "demographics.json") and not _has(d, "intervention.json")),
    ("intervention_v2", "non-adopters", "game_user_study_phase_2/recordings-download",
     True, lambda d: _has(d, "demographics.json") and _variant(d) == "v2_think_pair"),
    ("intervention_v9", "non-adopters", "game_user_study_phase_2/recordings-download",
     True, lambda d: _has(d, "demographics.json") and _variant(d) == "v9_two_rows"),
    ("intervention_johnny", "non-adopters", "game_user_study_intervention_iter/recordings-download",
     True, lambda d: _has(d, "demographics.json")),
    ("intervention_modelled_ta", "non-adopters", "game_user_study_intervention_iter/recordings-modelled-ta",
     True, lambda d: _has(d, "demographics.json")),
    ("intervention_esr", "non-adopters", "game_user_study_intervention_iter/recordings-esr",
     True, lambda d: _has(d, "demographics.json")),
    # The 3-arm run shares one directory, so the arms are separated per participant the
    # same way phase 2's are. Vanilla is identified by the ABSENCE of intervention.json.
    ("3arm_vanilla", "non-adopters", "game_user_study_intervention_iter/recordings-3arm",
     True, lambda d: _has(d, "demographics.json") and not _has(d, "intervention.json")),
    ("3arm_johnny", "non-adopters", "game_user_study_intervention_iter/recordings-3arm",
     True, lambda d: _form(d) == "johnny_connect_four_boost"),
    ("3arm_v9_reasons", "non-adopters", "game_user_study_intervention_iter/recordings-3arm",
     True, lambda d: _variant(d) == "v9_reasons"),
    # The Aug-25/26 run (folder recordings-v9-vanilla) is another two-arm batch sharing one
    # directory, split the same way: vanilla by the ABSENCE of intervention.json, scaffolded
    # by the variant field. Its assisted-only participants were moved out to
    # recordings-v9-assisted-only-legacy, so everything left here ran the full protocol
    # (plus dropouts, which `reached_demographics` filters).
    ("v9run_vanilla", "non-adopters", "game_user_study_intervention_iter/recordings-v9-vanilla",
     True, lambda d: _has(d, "demographics.json") and not _has(d, "intervention.json")),
    ("v9run_v9_reasons", "non-adopters", "game_user_study_intervention_iter/recordings-v9-vanilla",
     True, lambda d: _variant(d) == "v9_reasons"),
]

# Batches that were COLLECTED separately but are ANALYSED as one condition. The 3-arm
# run's vanilla and johnny arms are the same protocol as the earlier runs of each, so they
# pool into those conditions rather than standing as conditions of their own -- which is
# what the notebook's ORDER list expects.
#
# Pooling is not free: the batches sit in different recruitment windows (phase-2 vanilla
# 08-18/19, 3-arm 08-21, the v9 run 08-25/26) and the earlier johnny batch ran
# ASSISTED-ONLY, so pooled johnny has 35 R1 rounds but only the 13 newest have a transfer
# block. `batch` keeps the original key on every row, so any of this can be split back
# apart.
POOL_INTO = {
    "3arm_vanilla": "vanilla_non_adopters",
    "3arm_johnny": "intervention_johnny",
    "v9run_vanilla": "vanilla_non_adopters",
    # NOTE the presentation caveat: 3arm's v9_reasons showed the contrast as TEXT TABLES,
    # while the Aug-26 run showed chat-style screenshots with truncated assistant replies
    # (INTERVENTION_REPLY_WORDS=5). The variant key was deliberately not bumped -- same
    # table, same messages, different rendering -- so these pool as one condition and the
    # `batch` column is the only thing that separates them. Split on it before claiming a
    # presentation effect either way.
    "v9run_v9_reasons": "3arm_v9_reasons",
}

EXCLUDE = {"test2", "Fede", "p1", "p3", "t1"}          # pilots and colleagues


def rows_for(root: Path, keep, r1_has_ai: bool, condition: str, population: str):
    """One row per participant-round, mirroring score_from_logs.main()'s schema."""
    out = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name in EXCLUDE or not keep(d):
            continue
        demo = sfl.read_json(d / "demographics.json") or {}
        gate = sfl.read_json(d / "gate_poc20260727.json") or {}
        conv = sfl.read_jsonl(d / "conversation_poc20260727.jsonl")
        iv = sfl.read_json(d / "intervention.json") or {}
        pt = sfl.read_json(d / "prompt_task.json") or {}
        mine, transfer = [], []
        for puzzle, label, _script_has_ai, is_transfer in sfl.ROUNDS:
            moves = sfl.read_jsonl(d / f"moves_p{puzzle}.jsonl")
            if not moves:
                continue
            sub = sfl.read_json(d / f"game_submission_p{puzzle}.json") or {}
            has_ai = r1_has_ai and not is_transfer
            row = {
                "condition": POOL_INTO.get(condition, condition),
                "batch": condition, "population": population,
                "pid": d.name, "round": label, "puzzle": puzzle, "ai": has_ai,
                "transfer": is_transfer,
                "form": iv.get("form"), "variant": iv.get("variant"),
                # The screens that DEFINE the population, carried so the analysis can
                # apply its own exclusions instead of inheriting mine. Carry-over
                # participants skipped the prompt task (a different protocol) and are
                # excluded from phase-2 results on procedural grounds.
                # Recruitment timestamps, so a batch can be identified in the analysis
                # instead of being hard-coded into the data layer.
                "consented_at": demo.get("consented_at"),
                "pretask_at": iv.get("submitted_at"),
                "prompt_rule": pt.get("rule"),
                "prompt_mean_score": pt.get("mean_score"),
                "prompt_passed": pt.get("passed"),
                "carryover": pt.get("rule") == "carried over from phase 1",
                "reached_demographics": bool(demo.get("genai_usage")),
                "effort_median": (iv.get("effort") or {}).get("median_score"),
                "skill_rating": demo.get("skill_rating"), "age": demo.get("age"),
                "education": demo.get("education"), "genai_usage": demo.get("genai_usage"),
                "gate_correct": gate.get("first_move_optimal"),
                "gate_confidence": gate.get("confidence"),
                "ai_turns": len(conv) if has_ai else 0,
                "end_reason": sub.get("end_reason"),
                **sfl.score_round(moves),
            }
            mine.append(row)
            if is_transfer:
                transfer.append(row)
        summary = sfl.transfer_summary(transfer)
        for row in mine:
            row.update(summary)
        out += mine
    return out


def build(write: bool = True, verbose: bool = True):
    """Score every condition and return one long DataFrame.

    Importable from a notebook in any working directory -- all paths are resolved from
    this file's location, not from cwd:

        from build_scores import build
        df = build()                      # writes the CSVs too
        df = build(write=False)           # just the DataFrame

    One row per participant-round. Filter with `~df.carryover & df.reached_demographics`
    to reproduce the exclusions used in the phase-2 analysis.
    """
    import pandas as pd

    everything, report = [], []
    for cond, pop, rel, r1_ai, keep in CONDITIONS:
        root = CITRUS / rel
        if not root.is_dir():
            report.append((cond, pop, "MISSING DIRECTORY", 0, 0, 0))
            continue
        rows = rows_for(root, keep, r1_ai, cond, pop)
        pids = {r["pid"] for r in rows}
        report.append((cond, pop, rel, len(pids),
                       sum(r["round"] == "R1" for r in rows),
                       len({r["pid"] for r in rows if r["transfer_label"]})))
        everything += rows

    if verbose:
        print(f"{'condition':<26}{'population':<14}{'participants':>13}{'R1 rounds':>11}"
              f"{'both solo rounds':>18}")
        for cond, pop, rel, n, r1, full in report:
            note = ("   <- nothing matched" if not n
                    else f"   -> pooled into {POOL_INTO[cond]}" if cond in POOL_INTO else "")
            print(f"{cond:<26}{pop:<14}{n:>13}{r1:>11}{full:>18}{note}")

    df = pd.DataFrame(everything)
    lead = ["condition", "batch", "population", "pid", "round", "puzzle", "ai", "transfer"]
    df = df[lead + [c for c in df.columns if c not in lead]]
    if write:
        # Keyed on the pooled `condition`, not on the CONDITIONS entry, so a pooled batch
        # lands in the file its analysis key names instead of one of its own.
        for cond, part in df.groupby("condition"):
            part.to_csv(OUT / f"scores_{cond}.csv", index=False)
        df.to_csv(OUT / "scores_all.csv", index=False)
        if verbose:
            print(f"\nwrote {df.condition.nunique()} per-condition CSVs and scores_all.csv "
                  f"({len(df)} participant-rounds) to {OUT}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="show matches, write nothing")
    a = ap.parse_args()
    build(write=not a.list)


if __name__ == "__main__":
    main()
