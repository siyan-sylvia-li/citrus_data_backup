"""Subtype pass (scheme v3): one subtype per (utterance, frozen v1 parent act).

Reads the v1 labels from participant_utterances.csv and NEVER modifies them --
each row's `utt_type` is passed to the annotator as fixed context. Output is
participant_utterances_v3.csv: the v1 file with a `subtype` column added, so
every v1-level aggregate (including the Think Aloud share) is unchanged by
construction and the subtypes are a strict refinement.

Resumable; skips (pid, utt_ind, parent) triples already annotated.
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import dotenv
import dspy
import pandas as pd

import dialogue_act_subtypes as V3

BASE = Path("recordings-download")
OUT_JSONL = Path("annotations_v3_subtypes.jsonl")
OUT_CSV = Path("participant_utterances_v3.csv")


def items():
    """One record per (turn, v1 act), with the assistant's preceding reply as context."""
    df = pd.read_csv("participant_utterances.csv")
    convo = {}
    out = []
    for _, r in df.dropna(subset=["utt_type"]).iterrows():
        if not V3.SUBTYPES.get(r.utt_type):
            continue                      # parent carries no subtypes (Metacomment)
        if r.pid not in convo:
            f = BASE / r.pid / "conversation_p15.jsonl"
            convo[r.pid] = [json.loads(l) for l in open(f) if l.strip()] if f.exists() else []
        ex = convo[r.pid]
        ctx = (ex[r.utt_ind - 1].get("assistant") or "") if r.utt_ind > 0 and r.utt_ind - 1 < len(ex) else ""
        out.append(dict(pid=r.pid, utt_ind=int(r.utt_ind), utt=str(r.utt),
                        parent=r.utt_type, ctx=ctx))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dotenv.load_dotenv()
    # Fresh completions: the disk cache would otherwise make re-runs non-independent
    # (see the note in replicate_v1_labels.py -- it silently faked a replication).
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

    out_jsonl = Path(args.out) if args.out else OUT_JSONL
    todo = items()
    if args.limit:
        todo = todo[:args.limit]
    done = set()
    if out_jsonl.exists() and not args.overwrite:
        for l in open(out_jsonl):
            r = json.loads(l)
            done.add((r["pid"], r["utt_ind"], r["parent"]))
        todo = [t for t in todo if (t["pid"], t["utt_ind"], t["parent"]) not in done]
    elif args.overwrite and out_jsonl.exists():
        out_jsonl.unlink()
    print(f"{len(done)} done | {len(todo)} to go", flush=True)

    suite = V3.SubtypeSuite()
    fh = open(out_jsonl, "a")
    n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for it, r in zip(todo, ex.map(lambda i: suite(utterance=i["utt"], parent_act=i["parent"],
                                                     assistant_context=i["ctx"]), todo)):
            fh.write(json.dumps(dict(pid=it["pid"], utt_ind=it["utt_ind"], utt=it["utt"],
                                     parent=it["parent"], subtype=r["final"], votes=r["votes"],
                                     per_model=r["per_model"], n_valid=r["n_valid"],
                                     confidence=r["confidence"], unanimous=r["unanimous"],
                                     needs_review=r["needs_review"],
                                     responsive=V3.responsive(it["ctx"]),
                                     grounded=V3.grounded(it["utt"]))) + "\n")
            fh.flush()
            n += 1
            if n % 50 == 0:
                print(f"  {n}/{len(todo)}", flush=True)
    fh.close()

    recs = [json.loads(l) for l in open(out_jsonl)]
    sub = {(r["pid"], r["utt_ind"], r["parent"]): r for r in recs}

    # v1 file + subtype column. The v1 columns are copied through untouched.
    df = pd.read_csv("participant_utterances.csv")
    key = list(zip(df["pid"], df["utt_ind"], df["utt_type"]))
    df["subtype"] = [sub[k]["subtype"] if k in sub else None for k in key]
    df["subtype_conf"] = [sub[k]["confidence"] if k in sub else None for k in key]
    df["subtype_review"] = [sub[k]["needs_review"] if k in sub else None for k in key]
    df["responsive"] = [sub[k]["responsive"] if k in sub else None for k in key]
    df["grounded"] = [sub[k]["grounded"] if k in sub else None for k in key]
    out_csv = Path(args.out).with_suffix(".csv") if args.out else OUT_CSV
    df.to_csv(out_csv, index=False)

    # --- integrity check: the parent level MUST be untouched ---
    v1 = pd.read_csv("participant_utterances.csv")
    same = (v1["utt_type"].fillna("~").tolist() == df["utt_type"].fillna("~").tolist()
            and len(v1) == len(df))
    print(f"\nwrote {out_csv} ({len(df)} rows)")
    print(f"parent labels identical to v1: {same}   <- the Think Aloud share cannot move")
    print(f"subtype coverage: {df['subtype'].notna().sum()}/{v1['utt_type'].notna().sum()} act instances")

    macro, by_parent, exact = V3.subtype_kappa(recs)
    print(f"\nsubtype agreement (within parent): macro Fleiss kappa = {macro:.3f}")
    for p in by_parent:
        print(f"   {p:<32} kappa {by_parent[p]:+.3f}   all-3-agree {exact[p]:.0%}")
    print(f"needs_review (no majority): {sum(r['needs_review'] for r in recs)}/{len(recs)}")

    print("\nsubtype distribution:")
    for parent in V3.SUBTYPES:
        d = df[df["utt_type"] == parent]["subtype"].value_counts()
        tot = int(d.sum())
        if not tot:
            continue
        print(f"  {parent} (n={tot}):")
        for k, v in d.items():
            print(f"      {k:<18}{v:>5}  {v/tot:>5.0%}")


if __name__ == "__main__":
    sys.exit(main())
