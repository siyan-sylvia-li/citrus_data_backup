"""Claude's independent labels (4th rater; coded blind to kappa_panel_key.json) for the 80
kappa_sample turns, now with ProvideInfo (user supplies info/context/source the assistant lacks).
Info-provision turns were moved OUT of Think Aloud. Writes kappa_sample_claude.csv.
NOTE: Claude is an LLM -> a model rater, supplements but does not replace the human coding.
Panel has no ProvideInfo (it used the 6-act scheme), so ProvideInfo is scored vs human only.
"""
import csv

ACTS = ["SolReq", "CommonGrd", "KnowDef", "ThinkAloud", "Ack", "Meta", "ProvideInfo"]

L = {
    0: ["KnowDef"], 1: [], 2: ["CommonGrd", "ThinkAloud"], 3: ["Ack", "ThinkAloud", "CommonGrd"],
    4: ["SolReq"], 5: ["CommonGrd"], 6: ["SolReq"], 7: ["CommonGrd"], 8: ["SolReq"],
    9: ["Ack", "CommonGrd", "ProvideInfo"], 10: ["ProvideInfo"], 11: ["CommonGrd"],
    12: ["Ack", "Meta", "ThinkAloud"], 13: ["SolReq"], 14: ["CommonGrd"],
    15: ["CommonGrd", "ThinkAloud"], 16: ["Ack", "ThinkAloud", "CommonGrd"],
    17: ["KnowDef", "ThinkAloud"], 18: ["CommonGrd", "ThinkAloud", "ProvideInfo"], 19: ["KnowDef"],
    20: ["KnowDef", "Meta", "ProvideInfo"], 21: ["Meta"], 22: ["Ack", "SolReq"],
    23: ["CommonGrd", "ThinkAloud"], 24: ["Ack", "KnowDef"], 25: ["CommonGrd", "Meta"],
    26: ["KnowDef", "CommonGrd"], 27: ["KnowDef"], 28: ["SolReq"], 29: ["CommonGrd"],
    30: ["ThinkAloud", "CommonGrd"], 31: ["Ack", "ThinkAloud"], 32: ["SolReq"],
    33: ["Ack", "ProvideInfo"], 34: ["SolReq"], 35: ["ThinkAloud"], 36: ["ThinkAloud", "CommonGrd"],
    37: ["Ack", "ThinkAloud"], 38: ["SolReq"], 39: [], 40: ["CommonGrd", "ThinkAloud", "SolReq"],
    41: ["CommonGrd"], 42: ["Meta", "KnowDef"], 43: ["Ack", "SolReq"], 44: ["KnowDef"],
    45: ["CommonGrd", "Ack"], 46: ["CommonGrd", "ThinkAloud"], 47: ["CommonGrd"],
    48: ["Ack", "CommonGrd", "ThinkAloud"], 49: ["ThinkAloud"], 50: ["KnowDef"],
    51: ["SolReq", "ProvideInfo"], 52: ["CommonGrd"], 53: ["ThinkAloud", "CommonGrd"],
    54: ["CommonGrd", "ThinkAloud"], 55: ["KnowDef"], 56: ["SolReq", "CommonGrd"], 57: ["KnowDef"],
    58: ["Ack", "CommonGrd", "ThinkAloud"], 59: ["KnowDef", "ThinkAloud"], 60: ["ProvideInfo"],
    61: ["ThinkAloud"], 62: ["KnowDef", "CommonGrd"], 63: ["ThinkAloud", "CommonGrd"],
    64: ["Meta", "CommonGrd"], 65: ["SolReq"], 66: ["Ack", "Meta", "CommonGrd"], 67: ["KnowDef"],
    68: [], 69: ["Ack", "SolReq"], 70: ["SolReq", "ProvideInfo"], 71: ["SolReq"], 72: ["Ack"],
    73: ["CommonGrd"], 74: ["KnowDef", "ThinkAloud"], 75: ["SolReq"], 76: ["Ack", "CommonGrd"],
    77: ["Meta"], 78: ["CommonGrd"], 79: ["ThinkAloud", "ProvideInfo"],
}

with open("kappa_sample_claude.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["turn_id"] + ACTS + ["NoAct"])
    for tid in sorted(L):
        acts = L[tid]
        w.writerow([tid] + [1 if a in acts else "" for a in ACTS] + [1 if not acts else ""])
print(f"wrote kappa_sample_claude.csv ({len(L)} turns; "
      f"{sum('ProvideInfo' in v for v in L.values())} ProvideInfo, {sum(1 for v in L.values() if not v)} NoAct)")
