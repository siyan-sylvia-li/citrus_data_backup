#!/usr/bin/env python
"""Generate preset_personas/preset_*.json from the clustered persona pool.

Source: citrus_all/citrus_simulator/selected_personas.json, `kmeans` list — one
persona per k-means cluster, with `weight` = cluster size. That list is already
where presets 1-4 came from (preset 5, the k-pop high-schooler, was hand-written
and is not in the pool), so extending from it keeps the sample construction
consistent instead of mixing two selection methods.

The sibling `kcenter` list is NOT used: it is the outlier-seeking selection, so it
contains entries that cannot play a study participant — organisations ("A web
development company", "An AI-powered language learning app"), non-English personas,
bare names with no description ("Heather A. Cameron"), and a "group of young kids".
Curating it is a separate judgement call; see CANDIDATES_REJECTED below for the
same problem inside kmeans.

WHY THE DEMOGRAPHICS ARE A HAND TABLE, NOT SAMPLED
--------------------------------------------------
The student prompt (constants.USER_SYSTEM_PROMPT) leans hard on the profile:
"The words you use must give away this person's education". So the demographics
have to be COHERENT with the persona text — sampling education independently would
hand "An attorney who specializes in litigation" a less-than-high-school education
and make the register instruction self-contradictory. Each row below is therefore
assigned to fit its persona, and the script prints the resulting marginals next to
the human study's so the drift is visible rather than assumed away.

VOCABULARY
----------
Values use the STUDY's demographics-form vocabulary (`bachelors`, `postgrad`, `it`,
`several_times_day`) rather than the older preset shorthand (`bachelor`, `master`,
`tech`, `multiple_daily`). constants.py accepts both, but the study values make the
simulated profiles directly comparable to the humans' demographics.json. Note that
presets 1-5 use the shorthand, and presets 4-5 use `retired` / `student` for
occupation, which is not an option on the study form at all (it renders raw via
OCC_LABELS.get(x, x)). Those five are left untouched so already-collected runs stay
valid.

`known_concepts` is deliberately omitted: nothing in this simulator reads it (grep
the package — it is a leftover from the Bitcoin-concepts simulator), and carrying it
forward would imply a lever that does not exist.

Usage:
    python build_preset_personas.py            # write new presets, skip existing
    python build_preset_personas.py --dry-run  # print what would be written
    python build_preset_personas.py --force    # overwrite (will NOT touch 1-5)
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

POOL = Path("/data/home/siyanli/citrus_all/citrus_simulator/selected_personas.json")
OUT_DIR = Path(__file__).parent / "preset_personas"

# Presets that already exist and must never be rewritten: runs have been collected
# against them, and changing a profile would silently redefine those runs.
FROZEN = {1, 2, 3, 4, 5}

# Personas from the kmeans pool that are already in use as presets 1-4.
ALREADY_PRESETS = {
    "A mental health professional who offers guidance and tools for healing and growth",
    "A data scientist who brings a quantitative and analytical approach to problem-solving",
    "A talented musician who contributes his skills and knowledge during the listening parties",
    "A fellow retired athlete who has explored the world together and experienced different sports cultures",
}

# Dropped from the pool, with the reason. Kept explicit so the exclusion is a
# recorded decision rather than a silent filter.
CANDIDATES_REJECTED = {
    "我是一个对体育并不怎么感兴趣的音乐家，专注于古典音乐创作和钢琴演奏。对于比赛和体育新闻我常常保持一种非常中立的态度，"
    "但我对于人们在自己的专业领域取得成就感到敬佩。":
        "non-English persona — would produce a non-English conversation, which the "
        "dialogue-act annotator's prompt and examples are not written for",
}

# (persona substring -> profile). The key is a substring match against the pool
# entry so the long descriptions do not have to be repeated verbatim.
# age / education / occupation / ai_frequency, chosen to fit the persona.
PROFILES = [
    ("skeptical about new knowledge",            52, "hs_graduate",   "other",          "never"),
    ("devoted Christian",                        45, "bachelors",     "education",      "less_than_monthly"),
    ("technology entrepreneur",                  38, "postgrad",      "professional",   "several_times_day"),
    ("fellow collector who focuses",             61, "bachelors",     "other",          "few_times_month"),
    ("graduate student from a different",        27, "some_postgrad", "education",      "daily"),
    ("concerned resident with insider",          49, "associate",     "government",     "few_times_month"),
    ("university student with big future",       20, "some_college",  "other",          "daily"),
    ("A parent who has experienced",             43, "some_college",  "other",          "weekly"),
    ("aspiring actor",                           26, "bachelors",     "arts",           "few_times_week"),
    ("local retailer committed",                 47, "hs_graduate",   "retail",         "less_than_monthly"),
    ("software developer specializing",          34, "bachelors",     "it",             "several_times_day"),
    ("finance professional skilled",             39, "postgrad",      "finance",        "few_times_week"),
    ("journalist from the country",              36, "bachelors",     "other",          "daily"),
    ("increasing school graduation rates",       51, "postgrad",      "education",      "weekly"),
    ("aging person who is considering",          68, "hs_graduate",   "other",          "never"),
    ("background in software development",       41, "bachelors",     "it",             "few_times_week"),
    ("politician who advocates",                 55, "postgrad",      "government",     "few_times_month"),
    ("Critical Political Scientist",             44, "postgrad",      "education",      "few_times_week"),
    ("research scientist diligently",            37, "postgrad",      "professional",   "few_times_week"),
    ("busy professional looking to sell",        42, "bachelors",     "professional",   "few_times_week"),
    ("preservation of the community's",          58, "some_college",  "other",          "less_than_monthly"),
    ("film enthusiast",                          31, "bachelors",     "arts",           "daily"),
    ("healthcare worker who goes above",         35, "associate",     "healthcare",     "few_times_month"),
    ("food and beverage industry",               46, "hs_graduate",   "hospitality",    "less_than_monthly"),
    ("favorite travel destinations",             33, "bachelors",     "other",          "few_times_week"),
    ("mobilizer who works closely",              40, "bachelors",     "other",          "weekly"),
    ("scholarly student deep into historical",   23, "some_college",  "education",      "daily"),
    ("dietician",                                34, "bachelors",     "healthcare",     "few_times_week"),
    ("young professional working in a different", 29, "bachelors",    "professional",   "several_times_day"),
    ("ecologist enthusiast",                     30, "bachelors",     "professional",   "few_times_week"),
    ("runs a blog dedicated to reviewing",       32, "some_college",  "other",          "daily"),
    ("linguistic academic",                      48, "postgrad",      "education",      "few_times_week"),
    ("follows Rip religiously",                  24, "some_college",  "retail",         "several_times_day"),
    ("balance stakeholder demands",              38, "bachelors",     "it",             "several_times_day"),
    ("talented and young athlete",               21, "hs_graduate",   "other",          "daily"),
    ("firsthand experience in international",    35, "postgrad",      "government",     "few_times_week"),
    ("literary fiction author",                  44, "bachelors",     "arts",           "few_times_month"),
    ("devoted football fan",                     28, "hs_graduate",   "manufacturing",  "weekly"),
    ("love for the sport since their early",     50, "associate",     "transportation", "less_than_monthly"),
    ("representative from a government agency",  47, "bachelors",     "government",     "few_times_month"),
    ("passionate about gaming",                  19, "some_college",  "other",          "several_times_day"),
    ("researcher with background in Computer",   33, "postgrad",      "it",             "several_times_day"),
    ("theory-oriented science professor",        56, "postgrad",      "education",      "weekly"),
    ("target audience and brand messaging",      45, "bachelors",     "professional",   "few_times_week"),
    ("attentive history teacher",                42, "postgrad",      "education",      "few_times_month"),
    ("digital artist",                           29, "bachelors",     "arts",           "daily"),
    ("A farmer who is exploring",                53, "hs_graduate",   "agriculture",    "less_than_monthly"),
    ("beginning to learn about software",        26, "some_college",  "it",             "daily"),
    ("blending different cultural influences",   52, "bachelors",     "arts",           "few_times_month"),
    ("directly affected by infrastructure",      60, "hs_graduate",   "manufacturing",  "never"),
    ("instrumental in the company's success",    40, "bachelors",     "professional",   "few_times_week"),
    ("attorney who specializes in litigation",   49, "postgrad",      "professional",   "few_times_week"),
]

# Human study marginals (n=130 with demographics), for the drift report.
# Source: game_study_data/recordings-download/*/demographics.json.
HUMAN = {
    "education": {"bachelors": 56, "postgrad": 42, "some_college": 12, "hs_graduate": 8,
                  "associate": 8, "some_postgrad": 3, "hs_incomplete": 1},
    "occupation": {"it": 40, "education": 18, "other": 16, "professional": 16, "retail": 12,
                   "arts": 11, "healthcare": 6, "government": 4, "finance": 3,
                   "hospitality": 2, "manufacturing": 2},
    "ai_frequency": {"few_times_week": 34, "several_times_day": 27, "daily": 20,
                     "few_times_month": 19, "less_than_monthly": 10, "weekly": 9,
                     "never": 6, "about_monthly": 5},
}


def build():
    pool = json.load(open(POOL))["kmeans"]
    by_key = {}
    for entry in pool:
        p = entry["persona"]
        if p in ALREADY_PRESETS or p in CANDIDATES_REJECTED:
            continue
        by_key[p] = entry

    profiles, unmatched = [], []
    for key, age, edu, occ, aif in PROFILES:
        hits = [p for p in by_key if key in p]
        if len(hits) != 1:
            unmatched.append((key, len(hits)))
            continue
        profiles.append({"persona": hits[0], "age": age, "education": edu,
                         "occupation": occ, "ai_frequency": aif,
                         "cluster": by_key[hits[0]]["cluster"],
                         "cluster_weight": by_key[hits[0]]["weight"]})
    leftover = sorted(set(by_key) - {p["persona"] for p in profiles})
    return profiles, unmatched, leftover


def report(profiles):
    n = len(profiles)
    print(f"\n=== marginals: {n} new presets vs human study (n=130) ===")
    for field, human in HUMAN.items():
        h_tot = sum(human.values())
        sim = Counter(p[field] for p in profiles)
        keys = sorted(set(sim) | set(human), key=lambda k: -human.get(k, 0))
        print(f"\n{field:>14}   {'sim%':>6} {'human%':>7}   diff")
        for k in keys:
            s = 100 * sim.get(k, 0) / n
            h = 100 * human.get(k, 0) / h_tot
            flag = "  <-- sim only" if k not in human else ""
            print(f"{k:>14}   {s:5.1f}% {h:6.1f}%   {s - h:+5.1f}{flag}")
    ages = [p["age"] for p in profiles]
    print(f"\n           age   sim mean {sum(ages)/n:.1f} (range {min(ages)}-{max(ages)})"
          f"   human mean 40.2 (range 20-74)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite existing (never presets 1-5)")
    args = ap.parse_args()

    profiles, unmatched, leftover = build()
    if unmatched:
        print("!! PROFILES keys that did not match exactly one pool entry:")
        for k, n in unmatched:
            print(f"   {n} hits: {k!r}")
    if leftover:
        print(f"!! {len(leftover)} pool personas have no profile row:")
        for p in leftover:
            print(f"   {p[:100]}")
    if unmatched or leftover:
        print("\nrefusing to write — every pool persona needs exactly one profile row")
        return 1

    start = max(FROZEN) + 1
    written = skipped = 0
    for i, prof in enumerate(profiles, start):
        out = OUT_DIR / f"preset_{i}.json"
        if i in FROZEN:
            skipped += 1
            continue
        if out.exists() and not args.force:
            skipped += 1
            continue
        body = {k: prof[k] for k in ("persona", "age", "education", "occupation", "ai_frequency")}
        if args.dry_run:
            print(f"would write {out.name}: {body['persona'][:70]}  "
                  f"({body['age']}, {body['education']}, {body['occupation']}, {body['ai_frequency']})")
        else:
            out.write_text(json.dumps(body, indent=4) + "\n")
        written += 1

    report(profiles)
    verb = "would write" if args.dry_run else "wrote"
    print(f"\n{verb} {written} preset(s) as preset_{start}..preset_{start + len(profiles) - 1}"
          f"; skipped {skipped}")
    print(f"total presets available after this: {len(list(OUT_DIR.glob('preset_*.json')))}")
    print(f"excluded from the pool by hand: {len(CANDIDATES_REJECTED)} "
          f"({'; '.join(CANDIDATES_REJECTED.values())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
