"""Assistant-side fine coding for the require_kdq amplification arm.

The prediction is fixed in advance BY THE HUMAN DATA, not by the module's own
pre-registration (which was falsified -- see human_othello_assistant.py):

    Board Report  UP    (human rho +0.231 with transfer; KDQ elicits it at 1.51)
    Move Verdict  DOWN  (human rho -0.198; Solution Request elicits it at 2.06,
                         and require_kdq cut Solution Request 11.0% -> 1.4%)

CAVEAT TO CARRY INTO THE RESULT: require_kdq moved two acts, not one. Think Aloud
also fell (31.4% -> 24.2% act-share, p .047; 59% -> 38% turn-share). Think Aloud
suppresses Move Verdict in humans (0.61), so a Move Verdict DROP here has two
possible sources and this design cannot separate them. A Board Report RISE is the
cleaner test, since Think Aloud barely moves Board Report (1.13, ns).

require_think is excluded: its manipulation check failed (+2.7 act-share points,
p .875, 28% gate leak), so its assistant side is uninterpretable.

Othello-worded fine panel, same 12 codes and same three seats as the human pass, so
prevalences sit on one scale.

    python asst_panel_kdq.py                # annotate, then report
    python asst_panel_kdq.py --report-only
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

BASE = Path(__file__).resolve().parent
import dotenv  # noqa: E402
dotenv.load_dotenv(BASE / ".env")

ARMS = {
    "baseline": "test_agentic_run__human_style_q5_shared_context",
    "require_kdq": "test_agentic_run__human_style_q5_gate-require_kdq_shared_context",
}
PUZZLE = "poc20260727"
CONVO = f"conversation_{PUZZLE}.jsonl"
ANN = f"annotated_conversation_{PUZZLE}.jsonl"
FIELD = "annotation_assistant_fine"
FOCUS = ("Board Report", "Move Verdict")


def usable(rec):
    """All seats failed -> a file of empty labels, not data. Caught on turn 1."""
    per = (rec or {}).get("per_model") or {}
    return any(v is not None for v in per.values())


def cells(root):
    return sorted(d for d in (BASE / root).iterdir()
                  if d.is_dir() and "Kimi" in d.name and (d / CONVO).exists())


def annotate():
    from assistant_acts_othello import OthelloFineAssistantSuite
    suite = OthelloFineAssistantSuite()
    done = 0
    for arm, root in ARMS.items():
        for d in cells(root):
            src = d / ANN if (d / ANN).exists() and (d / ANN).stat().st_size else d / CONVO
            rows = [json.loads(l) for l in open(src) if l.strip()]
            todo = [t for t in rows
                    if FIELD not in t and (t.get("assistant") or "").strip()]
            if not todo:
                continue
            print(f"  {arm}/{d.name}: {len(todo)} turns", flush=True)
            for t in rows:
                if FIELD in t or not (t.get("assistant") or "").strip():
                    continue
                rec = suite(utterance=t["assistant"].strip())
                if not usable(rec):
                    print("every panel seat failed -- check .env; refusing to write "
                          "empty labels", file=sys.stderr)
                    return None
                t[FIELD] = rec
                done += 1
            tmp = (d / ANN).with_suffix(".jsonl.tmp")
            with open(tmp, "w") as w:
                for t in rows:
                    w.write(json.dumps(t) + "\n")
            tmp.replace(d / ANN)      # atomic; preserves annotation_user
    print(f"annotated {done} assistant turn(s)")
    return done


def shares(root):
    """Per cell: share of that cell's assistant turns bearing each act."""
    out = {}
    for d in cells(root):
        f = d / ANN
        if not f.exists():
            continue
        acts, n = Counter(), 0
        for line in open(f):
            if not line.strip():
                continue
            t = json.loads(line)
            a = t.get(FIELD)
            if not a or not (t.get("assistant") or "").strip():
                continue
            n += 1
            acts.update(a.get("final") or [])
        if n:
            preset = d.name.split("together")[0].replace("preset_", "")
            out[preset] = ({k: v / n for k, v in acts.items()}, n)
    return out


def report():
    data = {a: shares(r) for a, r in ARMS.items()}
    paired = sorted(set(data["baseline"]) & set(data["require_kdq"]))
    print(f"\n{'=' * 86}")
    print("ASSISTANT ACTS — turn-share per cell, paired on preset")
    print(f"{'=' * 86}")
    print(f"{len(paired)} paired cells "
          f"(baseline {len(data['baseline'])}, require_kdq {len(data['require_kdq'])})")
    nb = sum(data["baseline"][p][1] for p in paired)
    nk = sum(data["require_kdq"][p][1] for p in paired)
    print(f"{nb} baseline / {nk} require_kdq assistant turns\n")

    acts = sorted({k for a in ARMS for c in data[a].values() for k in c[0]})
    print(f"{'act':<30}{'baseline%':>10}{'kdq%':>8}{'diff':>8}{'p':>8}  ")
    rows = []
    for act in acts:
        b = np.array([data["baseline"][p][0].get(act, 0.0) for p in paired])
        k = np.array([data["require_kdq"][p][0].get(act, 0.0) for p in paired])
        if not np.any(b) and not np.any(k):
            continue
        # Paired on preset: same persona in both arms, so the pairing removes
        # persona variance rather than treating the two arms as independent samples.
        try:
            p = stats.wilcoxon(b, k).pvalue if np.any(b - k) else 1.0
        except ValueError:
            p = 1.0
        rows.append((act, 100 * b.mean(), 100 * k.mean(), 100 * (k.mean() - b.mean()), p))
    for act, bm, km, d, p in sorted(rows, key=lambda r: -abs(r[3])):
        tag = "  <== PREDICTED " + ("UP" if act == "Board Report" else "DOWN") \
              if act in FOCUS else ""
        print(f"{act:<30}{bm:>10.1f}{km:>8.1f}{d:>+8.1f}{p:>8.3f}{tag}")
    print("\n  Wilcoxon signed-rank on paired cells. Board Report is the clean test;")
    print("  a Move Verdict drop is confounded with the Think Aloud fall this arm")
    print("  also produced (31.4 -> 24.2 act-share, p .047).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if not args.report_only and annotate() is None:
        return 1
    report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
