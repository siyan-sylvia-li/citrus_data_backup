"""Small-scale StudyChat run: annotate consented students' assignment user-turns with OUR 5-act
taxonomy, then test whether R1 reasoning predicts the exam (no_llm) transfer, controlling for
assignment (llm) performance -- the phase-1 model, on real students with objective grades.

Run:  /Users/siyanli/Documents/CITRUS/document_study_og/.venv/bin/python run_studychat.py
Scale up by raising MAX_SCAN / N_STUDENTS / MAX_TURNS_PER_STUDENT below.
"""
import pathlib, dotenv, random
dotenv.load_dotenv(pathlib.Path(__file__).resolve().parent.parent / "wildchat_analysis" / ".env")

from typing import Literal
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, numpy as np, dspy
import statsmodels.api as sm
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from tqdm import tqdm

# ---------------- knobs ----------------
MAX_SCAN              = 300000  # safety cap on interactions scanned; loop stops early once N_STUDENTS eligible are found
MIN_TURNS            = 5        # min user turns for a student to be eligible
N_STUDENTS           = 100      # students to sample
MAX_TURNS_PER_STUDENT = 30      # turns sampled per student (annotation cost)
SEED = 0
REASON = {"Think Aloud", "Common Ground Question"}
REPO = "wmcnicho/StudyChat"

# ---------------- our taxonomy + classifier signature (copied from check_sharechat.ipynb) ----------------
SCHEME = {
    "Think Aloud": "S", "Conversational Acknowledgment": "S", "Knowledge Deficit Question": "S",
    "Misconception": "S", "Common Ground Question": "S", "Vague Answer": "S", "Partial Answer": "S",
    "Social Coordination Action": "S", "Metacomment": "S", "Read Aloud": "S", "Solution Request": "S",
    "Provide Information": "S",
    "Forced Choice": "T", "Repetition": "T", "Prompt": "T",
}
EXAMPLES = {
    "Forced Choice": '"Would that be random, uniformed, or clumped?"', "Prompt": '"So 200 times one is what?"',
    "Repetition": 'S: "Is it commensalism?" T: "Commensalism."',
    "Common Ground Question": '"Aren\'t they more lined up, like more in order?"',
    "Conversational Acknowledgment": '"Ok." "No sir." "Yes ma\'am."',
    "Knowledge Deficit Question": '"What do you mean by it doesn\'t have a skeleton?"',
    "Metacomment": '"I don\'t know." "Yes, I understand."',
    "Misconception": '"I always used to get diploid and haploid mixed up."',
    "Partial Answer": '"It has to do with the cells."', "Read Aloud": '"Question 7: Plot growth pattern."',
    "Social Coordination Action": '"No, I didn\'t hear about that."',
    "Think Aloud": '"500 equals 50 and 50 divided by 500 gives 10."',
    "Vague Answer": '"Because it helps to, umm, you know."',
    "Solution Request": '"What\'s the best move?" "What now?" "Just tell me what to play." "Any advice for the next move?"',
    "Provide Information": '"Here is my code: def f(): ..." "The assignment says to load the dataset remotely." "I got MSE 0.00, R2 1.0."',
}
TAXONOMY = "".join(f"DIALOGUE ACT: {k}, EXAMPLES: {EXAMPLES[k]}\n" for k in SCHEME)
DialogueAct = Literal['Think Aloud', 'Conversational Acknowledgment', 'Knowledge Deficit Question',
                      'Common Ground Question', 'Metacomment', 'Solution Request', 'Provide Information']

class DialogueActClassifierSignature(dspy.Signature):
    """Given an utterance from a user in a tutoring or teaching conversation with an AI assistant (the user is learning a topic, or working through it with the assistant's help), first split it into clauses, then classify each clause into the dialogue acts that apply per the taxonomy provided. Output all applicable acts.

Disambiguating the question-type acts (these are easily confused — read carefully):
- "Solution Request": an OPEN request for the assistant to explain, teach, or give the answer, with NO specific candidate proposed. E.g. "explain how photosynthesis works", "what's the answer?", "walk me through it", "just tell me the steps".
- "Common Ground Question": the user proposes their OWN answer, understanding, or approach and asks the assistant to confirm/evaluate it. E.g. "so mitosis makes two identical cells, right?", "is my understanding correct?".
- "Knowledge Deficit Question": asks about a concept, term, definition, or a specific step — or clarifies something the assistant just said — NOT a request for the whole answer. E.g. "what does 'derivative' mean?", "why does that step work?".
- "Think Aloud": the user narrates their OWN reasoning or works through the problem out loud. E.g. "so if the cell loses water it should shrink...", "okay, 12 times 8 is 96, then I add 4".
- "Metacomment": the user comments on their own state of knowledge/understanding, not the content. E.g. "I don't get it", "oh, I see now", "I'm confused".
- "Provide Information": the user supplies information/context the assistant lacks — pasting their code, the assignment text, an error, an output, or data — rather than asking or reasoning. Often co-occurs with a Solution Request ("here's my code — fix it"). E.g. a pasted code block, "the assignment says to...", "I got MSE 0.00, R2 1.0"."""
    utterance = dspy.InputField(desc="The utterance to be classified.")
    taxonomy = dspy.InputField(desc="Taxonomy of dialogue acts, with example utterances.")
    dialogue_acts: list[DialogueAct] = dspy.OutputField(desc="The list of dialogue acts applicable to this utterance.")

# ---------------- load grades + stream conversations ----------------
_g = lambda f: hf_hub_download(REPO, f, repo_type="dataset")
grades = pd.concat([pd.read_csv(_g("scores/f24_grades_released_normalized.csv")),
                    pd.read_csv(_g("scores/s25_grades_released_normalized.csv"))], ignore_index=True)
consented = set(grades["userId"])
print(f"consented students with grades: {len(grades)}")

random.seed(SEED)
by_user = {}
n_scanned = 0
for i, ex in enumerate(load_dataset(REPO, split="train", streaming=True)):
    n_scanned = i + 1
    if i >= MAX_SCAN:
        break
    u, p = ex.get("userId"), ex.get("prompt")
    if u in consented and isinstance(p, str) and p.strip():
        by_user.setdefault(u, []).append(p)
    if i % 1000 == 0 and sum(len(v) >= MIN_TURNS for v in by_user.values()) >= N_STUDENTS:
        break                                    # enough eligible students collected -> stop early
elig = [u for u, v in by_user.items() if len(v) >= MIN_TURNS]
print(f"consented students seen: {len(by_user)} | eligible (>= {MIN_TURNS} turns): {len(elig)}")
students = random.sample(elig, min(N_STUDENTS, len(elig)))
def _pick(v):   # MAX_TURNS_PER_STUDENT=None -> use all of a student's turns
    return v if (MAX_TURNS_PER_STUDENT is None or len(v) <= MAX_TURNS_PER_STUDENT) else random.sample(v, MAX_TURNS_PER_STUDENT)
tasks = [(u, p) for u in students for p in _pick(by_user[u])]
print(f"scanned {n_scanned} interactions | eligible consented students (>= {MIN_TURNS} turns): {len(elig)}")
print(f"annotating {len(tasks)} turns across {len(students)} students")

# ---------------- annotate (our taxonomy, single model) ----------------
_single = dspy.ChainOfThought(DialogueActClassifierSignature)
_single.set_lm(dspy.LM("openai/gpt-5.4-mini", temperature=0.0, max_tokens=2048, timeout=120))
def _classify(t):
    try:
        acts = _single(utterance=t, taxonomy=TAXONOMY).dialogue_acts
        return [k for k in SCHEME if k in set(acts)]
    except Exception:
        return None
with ThreadPoolExecutor(max_workers=8) as ex:
    labels = list(tqdm(ex.map(lambda up: _classify(up[1]), tasks), total=len(tasks), desc="acts"))

# ---------------- per-student profile + outcome OLS ----------------
rows = [{"userId": u, "reasoning": (bool(set(a) & REASON) if a else None),
         "solreq": ("Solution Request" in a if a else None),
         "provideinfo": ("Provide Information" in a if a else None)} for (u, _), a in zip(tasks, labels)]
d = pd.DataFrame(rows).dropna()
nt = sum(a is not None for a in labels)
ac = Counter(x for a in labels if a for x in a)
print("\nact coverage:", {k: f"{ac.get(k,0)/nt:.0%}" for k in SCHEME if ac.get(k, 0)})

per = d.groupby("userId").agg(reasoning=("reasoning", "mean"), solreq=("solreq", "mean"),
                              provideinfo=("provideinfo", "mean"), n=("reasoning", "size")).reset_index()
m = per.merge(grades[["userId", "llm", "no_llm"]], on="userId").dropna(subset=["llm", "no_llm"])
m.to_csv("studychat_small_perstudent.csv", index=False)
print(f"outcome sample: {len(m)} students  ->  studychat_small_perstudent.csv")
if len(m) < 8:
    import sys
    print(f"** Only {len(m)} students in the outcome sample -- too few for correlations/OLS.")
    print("   Raise MAX_SCAN / N_STUDENTS or lower MIN_TURNS, then rerun.")
    sys.exit(0)
print(f"corr(reasoning,llm)={np.corrcoef(m.reasoning,m.llm)[0,1]:+.2f}  "
      f"corr(llm,no_llm)={np.corrcoef(m.llm,m.no_llm)[0,1]:+.2f}  "
      f"corr(reasoning,no_llm)={np.corrcoef(m.reasoning,m.no_llm)[0,1]:+.2f}")

z = lambda a: (a - np.mean(a)) / np.std(a, ddof=1)
def ols(cols, names, y, lbl):
    X = sm.add_constant(np.column_stack([z(np.asarray(c, float)) for c in cols]))
    r = sm.OLS(z(np.asarray(y, float)), X).fit(cov_type="HC3"); ci = r.conf_int()
    print(f"{lbl} R2={r.rsquared:.3f}: " +
          "  ".join(f"{n} beta={r.params[i+1]:+.3f} 95%CI[{ci[i+1,0]:+.2f},{ci[i+1,1]:+.2f}] p={r.pvalues[i+1]:.3f}"
                    for i, n in enumerate(names)))
print("\nDV = exam (no_llm), standardized, HC3 robust SE:")
ols([m.reasoning], ["reasoning"], m.no_llm, " reasoning only        ")
ols([m.reasoning, m.llm], ["reasoning", "assign(llm)"], m.no_llm, " + assignment(llm) ctrl")
