"""Manipulation check: did the behaviour prompts actually move the act they target?

Annotates each arm's conversations, then compares act shares against the DEFAULT
condition restricted to the same (preset, model) cells. Matched and paired, because
act shares differ far more by model than by anything these prompts do — an unmatched
baseline would mostly measure which models happen to be in each arm.

  student arms  (Othello)       think    -> Think Aloud
                                solution -> Solution Request
  assistant arms (Connect Four) explain  -> Direct Instruction / Hint
                                no_exp   -> Provide Correct Answer only

Each side uses the annotator whose prompt matches its game: the student scheme is
Othello-worded, the assistant scheme Connect Four-worded, so they are not
interchangeable.

    python manip_check.py                  # annotate what is missing, then report
    python manip_check.py --report-only    # no API calls; report on what exists
    python manip_check.py --side student    # or: assistant, both (default)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

BASE = Path(__file__).resolve().parent

# load_dotenv() with no argument searches upward from the CALLING FILE's directory.
# Pointing it at this package's .env explicitly is what makes the script work when it
# is invoked from somewhere else — the first version of this lived in a scratch dir,
# found no .env, and every annotator failed with "missing API key" while still writing
# annotation files full of empty labels.
import dotenv  # noqa: E402
dotenv.load_dotenv(BASE / ".env")

# Same relabels the notebook applies, so shares are comparable with it.
USER_ACT_REMAP = {"Correct Answer": "Think Aloud",
                  "Social Coordination Action": "Metacomment",
                  "Forced Choice": "Metacomment"}

# How many decisions of the solo round count. The human round is a fixed three
# scored moves; the simulator plays the puzzle out (2-10 on Connect Four), so an
# untruncated count is scored out of a different denominator per run.
TRUNC_DECISIONS = 3

SIDES = {
    "student": {
        "solo": ["b220260706", "bg20260726"],
        # Othello has a signed disc margin, which keeps the full ordering instead of
        # collapsing to a pass/fail — a run that lost by 2 is not a run that lost by 22.
        "outcome": "margin",
        "outcome_label": "solo margin sum (ceiling +16)",
        "puzzle": "oc20260727",
        "field": "annotation_user",
        "speaker": "user",
        "annotator": ("dialogue_act_annotation.py", "DialogueActSuite"),
        "remap": USER_ACT_REMAP,
        # Two families. The free-count arms let the student ask as often as it liked,
        # so they confound phrasing with volume. The _q5 arms fix every run at five
        # consultations, so only phrasing varies -- those are the interpretable ones,
        # and their baseline is `default_q5`, NOT the free-count default.
        "arms": {"default": "test_agentic_run",
                 "think": "test_agentic_run__think",
                 "think_v2": "test_agentic_run__think_v2",
                 "solution": "test_agentic_run__solution",
                 "default_q5": "test_agentic_run__q5",
                 "think_v2_q5": "test_agentic_run__think_v2_q5",
                 "solution_q5": "test_agentic_run__solution_q5",
                 "noai": "test_agentic_run__noai",
                 "human_style_q5": "test_agentic_run__human_style_q5",
                 "human_style_think_q5": "test_agentic_run__human_style_think_q5",
                 # Knockout series: one shared baseline, two treatments. All three use
                 # the most human-like configuration available (shared_context framing
                 # + real-participant examples) so the suppression is tested against a
                 # realistic student, not the artificial default.
                 "hs_shared": "test_agentic_run__human_style_q5_shared_context",
                 "no_concept_q": "test_agentic_run__human_style+no_concept_q_q5_shared_context",
                 "no_think_aloud": "test_agentic_run__human_style+no_think_aloud_q5_shared_context",
            # Amplification arms (require gates). Unlike the no_* suppression arms
            # these target an act the baseline is thin in, so the manipulation has
            # somewhere to move.
            "require_kdq": "test_agentic_run__human_style_q5_gate-require_kdq_shared_context",
            "require_think": "test_agentic_run__human_style_q5_gate-require_think_shared_context"},
        "targets": {"think": "Think Aloud", "think_v2": "Think Aloud",
                    "solution": "Solution Request",
                    "think_v2_q5": "Think Aloud", "solution_q5": "Solution Request",
                    # Few-shot human phrasing has no single target act: it is meant to
                    # shift the whole elicitation profile, so the act table is read as
                    # a distribution against the human one, not as one hypothesis.
                    "human_style_q5": "Common Ground Question",
                    # human_style raised Common Ground Question to the human value but
                    # LOST Think Aloud (29.1 -> 23.2, human 34.5). This arm adds a terse
                    # think-aloud nudge on top; Think Aloud is the target it must recover.
                    "human_style_think_q5": "Think Aloud",
                    # For a knockout the target is the act that must DISAPPEAR.
                    "no_concept_q": "Knowledge Deficit Question",
                    "no_think_aloud": "Think Aloud",
                    # Amplification: the target is the act that must RISE.
                    "require_kdq": "Knowledge Deficit Question",
                    "require_think": "Think Aloud"},
        # arm -> which arm it should be compared against. Anything unlisted uses
        # "default". A fixed-count arm compared against the free-count default would
        # reintroduce the volume confound at the baseline instead of the treatment.
        "baseline": {"think_v2_q5": "default_q5", "solution_q5": "default_q5",
                     "noai": "default_q5", "human_style_q5": "default_q5",
                     "human_style_think_q5": "default_q5",
                     "no_concept_q": "hs_shared", "no_think_aloud": "hs_shared",
                     "require_kdq": "hs_shared", "require_think": "hs_shared"},
    },
    "assistant": {
        "solo": ["w4p6"],
        # Connect Four is win/lose with no margin, so the graded measure is the count
        # of optimal decisions -- truncated, matching the human 3-move round.
        "outcome": "optimal",
        "outcome_label": f"solo optimal decisions, first {TRUNC_DECISIONS} (0-3)",
        "puzzle": "15",
        "field": "annotation_assistant",
        "speaker": "assistant",
        "annotator": ("dialogue_act_annotation_assistant.py", "AssistantDialogueActSuite"),
        "remap": {},
        "arms": {"default": "test_agentic_run_connect_four",
                 "explain": "test_agentic_run_connect_four__explain",
                 "no_exp": "test_agentic_run_connect_four__no_exp"},
        "targets": {"explain": "Direct Instruction", "no_exp": "Provide Correct Answer"},
    },
}


def load_suite(filename, cls):
    spec = importlib.util.spec_from_file_location("da_mod", BASE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, cls)()


def usable(ann):
    """True if this annotation came from a panel that actually answered.

    A run whose annotators all failed (no API key, provider down) still produces a
    complete file of records with n_valid == 0. Those must be re-annotated, not
    skipped as done — otherwise one broken run poisons the arm permanently and the
    report quietly compares empty label sets.
    """
    if ann is None:
        return True                     # genuinely empty turn; nothing to label
    if isinstance(ann, dict):
        if ann.get("n_valid") is not None:
            return ann["n_valid"] >= 1
        return bool(ann.get("per_model")) and any(
            v is not None for v in ann["per_model"].values())
    return True                          # legacy list form predates n_valid


def needs_work(path, field):
    if not path.exists() or not path.stat().st_size:
        return True
    for line in open(path):
        if not line.strip():
            continue
        t = json.loads(line)
        if field not in t or not usable(t.get(field)):
            return True
    return False


def annotate_arm(root, cfg, suite, only=None):
    """`only` restricts to a set of run-directory names.

    Used for the DEFAULT arm, which holds hundreds of runs while a manipulation arm
    holds ten. Only the cells the arms share are ever compared, so annotating the
    rest is pure API spend — on the assistant side that is ~200 runs of Connect Four
    to answer a question about 10.
    """
    convo = f"conversation_p{cfg['puzzle']}.jsonl"
    ann_f = f"annotated_conversation_p{cfg['puzzle']}.jsonl"
    if not root.exists():
        print(f"  {root.name}: missing — skipped")
        return 0
    todo = [d for d in sorted(root.iterdir())
            if d.is_dir() and (only is None or d.name in only)
            and (d / convo).exists() and (d / convo).stat().st_size
            and needs_work(d / ann_f, cfg["field"])]
    for n, d in enumerate(todo, 1):
        src = d / ann_f if (d / ann_f).exists() and (d / ann_f).stat().st_size else d / convo
        turns = [json.loads(l) for l in open(src) if l.strip()]
        print(f"  [{n}/{len(todo)}] {d.name} ({len(turns)} turns)", flush=True)
        out = []
        for t in turns:
            utt = (t.get(cfg["speaker"]) or "").strip()
            t[cfg["field"]] = suite(utterance=utt) if utt else None
            out.append(t)
        tmp = (d / ann_f).with_suffix(".jsonl.tmp")
        with open(tmp, "w") as w:
            for t in out:
                w.write(json.dumps(t) + "\n")
        tmp.replace(d / ann_f)          # atomic: a crash cannot leave a half file
    return len(todo)


def arm_shares(root, cfg):
    """{run_dir_name: ({act: share%}, n_turns)} from annotated files."""
    ann_f = f"annotated_conversation_p{cfg['puzzle']}.jsonl"
    out = {}
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not (d / ann_f).exists():
            continue
        acts, n_turns, any_usable = Counter(), 0, False
        for line in open(d / ann_f):
            if not line.strip():
                continue
            t = json.loads(line)
            n_turns += 1
            ann = t.get(cfg["field"])
            if not ann:
                continue
            if not usable(ann):
                continue
            any_usable = True
            final = ann["final"] if isinstance(ann, dict) else (ann[0] or [])
            for a in dict.fromkeys(cfg["remap"].get(x, x) for x in final):
                acts[a] += 1
        if any_usable and sum(acts.values()):
            out[d.name] = ({a: 100 * c / sum(acts.values()) for a, c in acts.items()},
                           n_turns)
    return out


def outcome_value(run_dir, cfg):
    """The solo-round outcome for one run, or None if it did not finish the round.

    margin  : signed disc difference summed over the solo puzzles (Othello).
    optimal : optimal decisions among each solo puzzle's FIRST TRUNC_DECISIONS,
              recomputed from the move log rather than read from summary
              optimal_moves, because the simulator plays the puzzle out while the
              human round is a fixed three moves.
    """
    total = 0
    for pz in cfg["solo"]:
        summ = run_dir / f"summary_p{pz}.json"
        if not summ.exists():
            return None
        if cfg["outcome"] == "margin":
            v = json.load(open(summ)).get("final_margin")
            if v is None:
                return None
            total += v
        else:
            mv = run_dir / f"moves_p{pz}.jsonl"
            if not mv.exists():
                return None
            rows = sorted((json.loads(l) for l in open(mv) if l.strip()),
                          key=lambda r: r.get("decision_number") or 0)[:TRUNC_DECISIONS]
            total += sum(bool(r.get("optimal")) for r in rows)
    return float(total)


def report_outcomes(side, cfg):
    """Does the manipulation move the OUTCOME, not just the dialogue?

    Paired on (preset, model) over every cell both arms completed — including runs
    that never consulted, which the act tables necessarily drop. A silent run still
    has an outcome, and excluding it here would bias exactly the comparison the
    manipulation is supposed to inform.
    """
    roots = {a: BASE / r for a, r in cfg["arms"].items()}
    print("\n" + "=" * 90)
    print(f"{side.upper()} — OUTCOME: {cfg['outcome_label']}, paired on (preset, model)")
    print("=" * 90)
    base_of = cfg.get("baseline", {})
    base = {}
    if roots["default"].exists():
        for d in roots["default"].iterdir():
            if d.is_dir():
                v = outcome_value(d, cfg)
                if v is not None:
                    base[d.name] = v
    print(f"{'arm':>12}  {'baseline':<12} {'n':>3} {'base':>8} {'arm':>8} {'diff':>8} "
          f"{'p':>8} {'d':>7}  95% CI")
    def outcomes_of(arm):
        got = {}
        if roots[arm].exists():
            for d in roots[arm].iterdir():
                if d.is_dir():
                    v = outcome_value(d, cfg)
                    if v is not None:
                        got[d.name] = v
        return got

    all_outcomes = {a: outcomes_of(a) for a in cfg["arms"]}
    for arm in cfg["arms"]:
        if arm == "default" or not roots[arm].exists():
            continue
        bname = base_of.get(arm, "default")
        base = all_outcomes[bname]
        pairs = []
        for d in sorted(roots[arm].iterdir()):
            if not d.is_dir() or d.name not in base:
                continue
            v = outcome_value(d, cfg)
            if v is not None:
                pairs.append((base[d.name], v))
        if len(pairs) < 3:
            print(f"{arm:>12}<-{bname:<12} {len(pairs):>3}   (too few paired cells)")
            continue
        b = np.array([p[0] for p in pairs]); x = np.array([p[1] for p in pairs])
        diff = x - b
        try:
            p = stats.wilcoxon(x, b).pvalue
        except ValueError:
            p = np.nan
        sd = diff.std(ddof=1)
        dz = diff.mean() / sd if sd > 0 else np.nan          # paired Cohen's d_z
        half = (stats.t.ppf(0.975, len(diff) - 1) * sd / np.sqrt(len(diff))
                if sd > 0 else 0.0)
        print(f"{arm:>12}<-{bname:<12} {len(pairs):>3} {b.mean():8.2f} {x.mean():8.2f} "
              f"{diff.mean():+8.2f} {p:8.3f} {dz:+7.2f}  "
              f"[{diff.mean() - half:+.2f}, {diff.mean() + half:+.2f}]")
    # The smallest effect this n could detect, so a null is read as "underpowered"
    # or "small effect" on evidence rather than by assumption.
    if base:
        sd_b = np.std(list(base.values()), ddof=1)
        for n in (10, 20, 40):
            mde = 2.8 * sd_b / np.sqrt(n)       # ~80% power, two-sided, paired-ish
            print(f"  reference: at n={n} paired cells, ~80% power needs a shift of "
                  f"{mde:.1f} ({cfg['outcome']} units; default SD={sd_b:.1f})")
    print("  Read alongside the act tables: an arm that changed how often the student "
          "consulted\n  has moved two things at once, and this column cannot separate them.")


def model_of(run_dir_name):
    """'preset_7anthropic_claude-sonnet-5' -> 'anthropic_claude-sonnet-5'."""
    import re
    m = re.match(r"^preset_(\d+)(.+)$", run_dir_name)
    return m.group(2) if m else run_dir_name


def ineligible_models(data, target, base="default"):
    """Models that never produce `target` in the DEFAULT condition.

    A manipulation aimed at an act a model does not emit at all is undefined for
    that model, not merely weak: it cannot comply, so it either ignores the prompt
    or — as Llama-3.3-70B did on `think` — stops talking rather than produce a shape
    it does not have. Pooling those runs measures the change in model mix. They are
    excluded and named, never dropped silently.
    """
    by_model = {}
    for cell, (shares, _) in data[base].items():
        by_model.setdefault(model_of(cell), []).append(shares.get(target, 0.0))
    return {m for m, vals in by_model.items() if max(vals) == 0.0}


def report(side, cfg):
    roots = {a: BASE / r for a, r in cfg["arms"].items()}
    data = {a: arm_shares(r, cfg) for a, r in roots.items()}
    print("\n" + "=" * 90)
    print(f"{side.upper()} ARMS — act share (% of a run's acts), paired on (preset, model)")
    print("=" * 90)
    base_of = cfg.get("baseline", {})
    for arm, target in cfg["targets"].items():
        base = base_of.get(arm, "default")
        cells = sorted(set(data[arm]) & set(data[base]))
        if not cells:
            print(f"\n--- {arm}: no matched {base} cells (arm has {len(data[arm])}) ---")
            continue
        skip = ineligible_models(data, target, base)
        dropped = sorted({model_of(c) for c in cells} & skip)
        cells = [c for c in cells if model_of(c) not in skip]
        if not cells:
            print(f"\n--- {arm}: every matched model is ineligible for {target!r} ---")
            continue
        acts = sorted({a for c in cells
                       for a in list(data[arm][c][0]) + list(data[base][c][0])})
        print(f"\n--- {arm} vs {base}   target: {target}   n={len(cells)} matched cells ---")
        if dropped:
            print(f"    excluded (0% {target} in the default condition, so the "
                  f"manipulation is undefined for them): {', '.join(dropped)}")
        print(f"{'act':>30} {base + '%':>10} {arm + '%':>11} {'diff':>8} {'p':>8}")
        for a in acts:
            b = np.array([data[base][c][0].get(a, 0.0) for c in cells])
            x = np.array([data[arm][c][0].get(a, 0.0) for c in cells])
            if b.std() == 0 and x.std() == 0 and np.allclose(b, x):
                continue
            try:
                p = stats.wilcoxon(x, b).pvalue
            except ValueError:                    # all differences zero
                p = np.nan
            mark = "  <== TARGET" if a == target else ""
            print(f"{a:>30} {b.mean():9.1f} {x.mean():10.1f} {x.mean() - b.mean():+8.1f} "
                  f"{p:8.3f}{mark}")
        tb = np.mean([data[base][c][1] for c in cells])
        tx = np.mean([data[arm][c][1] for c in cells])
        print(f"{'turns per run':>30} {tb:9.1f} {tx:10.1f} {tx - tb:+8.1f}"
              f"   <- should NOT move; the prompts change form, not frequency")
        # Whether the student consulted AT ALL is a frequency effect, and it is
        # invisible in shares because a silent run contributes no acts.
        #
        # Counted over every run directory the two arms have in common, NOT over
        # `cells`. `cells` is the set with usable annotations, which by construction
        # excludes silent runs — so scoping this to it reported "think 6/6" for an
        # arm that was actually 6/10, hiding the very effect the line exists to
        # surface.
        conv = f"conversation_p{cfg['puzzle']}.jsonl"
        both = sorted({d.name for d in roots[arm].iterdir() if d.is_dir()} &
                      {d.name for d in roots[base].iterdir() if d.is_dir()}) \
            if roots[arm].exists() and roots[base].exists() else []
        for label, root in ((base, roots[base]), (arm, roots[arm])):
            live = sum(1 for n in both
                       if (root / n / conv).exists() and (root / n / conv).stat().st_size)
            print(f"{'runs with any conversation':>30} {label:>10}: {live}/{len(both)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--side", choices=["student", "assistant", "both"], default="both")
    args = ap.parse_args()

    sides = ["student", "assistant"] if args.side == "both" else [args.side]
    for side in sides:
        cfg = SIDES[side]
        if not args.report_only:
            suite = load_suite(*cfg["annotator"])
            # The manipulation arms define which cells matter; the default arm is
            # then annotated only where it overlaps them.
            wanted = set()
            for arm, rel in cfg["arms"].items():
                if arm == "default":
                    continue
                root = BASE / rel
                if root.exists():
                    wanted |= {d.name for d in root.iterdir() if d.is_dir()}
            for arm, rel in cfg["arms"].items():
                print(f"annotating {side}/{arm} ...", flush=True)
                n = annotate_arm(BASE / rel, cfg, suite,
                                 only=wanted if arm == "default" else None)
                print(f"  {n} run(s) annotated", flush=True)
        report(side, cfg)
        report_outcomes(side, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
