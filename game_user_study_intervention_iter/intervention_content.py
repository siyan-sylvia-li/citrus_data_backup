"""Contrasting-case MATERIAL for the contrasting_cases form.

Separated from app.py so a form module can import the tables without importing the Flask
app, and so editing the material never touches routing code.

Every message is VERBATIM from a phase-1 Connect Four participant; variants differ only in
which real conversations appear and in what order, never in invented copy. The wording is
the measurement instrument -- it is the study owner's to edit, not implementation detail.
"""
from __future__ import annotations

# ---- Scaffolded arm: contrasting-cases intervention --------------------------
# Real Connect Four messages from phase-1 participants, shown verbatim. Group A asks
# the assistant for the answer; Group B narrates its own reasoning, names specific
# candidate moves, and asks how a move already played turned out.
#
# The two columns are LENGTH-MATCHED on purpose (A: 100 words / 10 messages, B: 96 / 9)
# and A's third participant is the single longest message in the table. Without that,
# the obvious reading of the contrast is "Group B writes more" -- which the phase-1 data
# says is not the mechanism: message length predicts transfer only through thinking
# aloud, and drops to nothing once thinking aloud is in the model (b=+0.044, p=.85).
# Variants are named and recorded per participant, because the material will be
# iterated during collection and "which table did this person see" must be recoverable
# from the data rather than from memory. Every message is VERBATIM from a phase-1
# Connect Four participant; variants differ only in which real conversations appear and
# in what order, never in invented copy.
#
# v0_live  -- the table the first participants saw.
# v2_think_pair -- rows 1 and 2 lead with a participant narrating their own reasoning
#   ("I was thinking of putting one in 3-3 / it would prevent yellow..."). Chosen after
#   a simulated-reader pretest (intervention_pretest.py, 18 readers/variant): naming of
#   the think-aloud dimension rose 78% -> 100% and of request type 83% -> 94%, with the
#   correct forced choice at 94%. Those rates are a CEILING, not a prediction -- the two
#   real participants on v0 named request type 2/2 and think-aloud 0/2 -- so the pretest
#   ranks variants, it does not estimate what humans will notice.
_A_TERSE = ["What do you think would be the best move right now?", "And now?", "Now?"]
_A_REPEAT = ["Which move would be the best one and why", "What about next move?",
             "what about 6", "so which one would be the best move?", "What about now?",
             "Give me the best move"]
_A_LONG = ["Analyze the current Connect Four board carefully. I am Red and need to "
           "choose the best first three columns to keep Red on track to force a win. "
           "Please identify the strongest move now, explain the key threat it creates, "
           "and give the best follow-up columns after the opponent responds. Use "
           "1-indexed column numbers."]
_B_CANDIDATE = ["I am about to move into column 3. will this work",
                "what would you suggest",
                "so just block up column 4 here. what's best to win with 2 moves left",
                "ok what next", "any risks to column 6"]
_B_PLAN = ["Where is the biggest threat from Red?",
           "But if I let Red place in that location, I can put a disc on top of it in 5, "
           "then 2 and 3 gets me a win."]
_B_RETRO = ["Was my previous move effective?",
            "That worked well to block the immediate threat - it felt like the only "
            "option, am I right?"]
_B_THINK1 = ["I was thinking of putting one in 3-3",
             "it would prevent yellow and open up one opportuhity for me"]
_B_THINK2 = ["I was considering column 3, is that a good option?", "so column 4 first?",
             "is it better to focus on blocking or building my own win?"]

# --- v9 blocks: Group B attaches a REASON to each move ------------------------
# _A_OUTSOURCE replaces the long elaborate prompt used in v0/v2. Participants' own
# observations showed that one backfiring -- three of four WRONG forced choices cited
# Group A's detail as a virtue ("quite in-depth instructions and are very detailed",
# "a full single prompt to get the AI to lay out a plan"). This one is equally long but
# ends "Reply with only the column number", so the outsourcing is unmistakable.
_A_OUTSOURCE = [
    "Analyze the current board and tell me the best sequence of my next three moves "
    "for Red. Please list only the column numbers in order and briefly explain why "
    "they are optimal.",
    "Given Yellow's last move, what is now the best second move for Red? Reply with "
    "only the column number.",
]
# Both are PREFIXES of real conversations, not curated selections.
_B_REASON_MID = [
    "I\u2019m considering row 3 because I think that gives me a chance to win "
    "horizontally or diagonally. Is that the best move?",
    "Thanks! Should I do column 3 now? It gives me a chance to win diagonally going "
    "from the bottom right up too",
]
_B_REASON_SHORT = [
    "If I add to column 2, will that reduce my possibilities rather than improve my "
    "chances?",
    "how about 4, blocking yellow's progress and adding to the diagonal?",
]

# Both variants keep Group A LONGER than Group B (v0: 100w/10 msgs vs 96w/9;
# v2: 100w/10 vs 86w/8). That direction matters: if B looked longer, the obvious
# reading of the contrast would be "write more", and phase-1 says length predicts
# transfer only through thinking aloud, dropping to nothing once it is controlled
# (b=+0.044, p=.85).

# --- v9_reasons: the same four blocks, each message paired with the assistant reply it
# actually received in the phase-1 conversation, stored in FULL and verbatim (markdown
# and all). Shortening for display is the form's job, not the material's, so the length
# shown can be changed without touching what was really said -- see REPLY_WORDS in
# interventions/forms/contrasting_cases.py. Only
# v9_reasons uses these; every other variant keeps the message-only blocks above and
# renders the reply as an ellipsis, so this change cannot move an earlier variant.
# Source conversations: _A_REPEAT p400, _A_OUTSOURCE p633, _B_REASON_MID p265,
# _B_REASON_SHORT p828.
_A_OUTSOURCE_R = [
    ('Analyze the current board and tell me the best sequence of my next three moves for Red. Please list only the column numbers in order and briefly explain why they are optimal.',
     '4, then depends on Yellow’s replies.\n\nColumn 4 is optimal because it’s the only move that preserves Red’s forced win: it builds a central threat while preventing Yellow from taking control of that column. Your next two Red moves should be chosen based on which immediate threat Yellow blocks or creates.'),
    ("Given Yellow's last move, what is now the best second move for Red? Reply with only the column number.",
     '4'),
]
_B_REASON_MID_R = [
    ('I’m considering row 3 because I think that gives me a chance to win horizontally or diagonally. Is that the best move?',
     'If by “row 3” you mean **drop in column 3** so it lands on row 3: it looks tempting, but **it’s not the best**—it doesn’t create a strong enough forced threat.\n\nThe strongest move here is **column 4**. Dropping there lands at **c4–r3** and sets up Red’s threats better while avoiding giving Yellow the tempo to defend/counter.\n\nSo: **play column 4.**'),
    ('Thanks! Should I do column 3 now? It gives me a chance to win diagonally going from the bottom right up too',
     'Column 3 is tempting, but check the diagonal you mentioned: after **c3**, the diagonal up-right runs through **c4–r4 / c5–r5**, while the other diagonal hits **c4–r2**, which is already Yellow—so it’s not a clean winning threat.\n\nThe key move is **column 4**: it stacks on your Red at **c4–r3**, creating a much stronger threat structure. So I’d play **column 4**, not column 3.'),
]
_A_REPEAT_R = [
    ('Which move would be the best one and why',
     'Best move: **Column 4**.\n\nWhy: Red’s disc lands at **row 3**, blocking Yellow’s dangerous **two-in-a-column** in column 4. If Red ignores it, Yellow can play there and start creating a strong vertical threat. Column 4 also keeps Red in the center, which gives the best chance to build threats in the remaining moves.'),
    ('What about next move?',
     'Think about **Yellow’s immediate threat** first: Yellow already has discs in **column 4 rows 1–2**, and Red is at **row 3**, so that’s contained for now.\n\nFor Red’s next move, ask: **Can I create a threat Yellow must answer, rather than just block?**  \nLook at the center columns—especially **column 4**—because playing there can stack on your existing Red at **4,3** and keep control of the strongest area.'),
    ('what about 6',
     'Column **2** is tempting: it would give Red **three vertical discs** in column 2, threatening to win at **col 2, row 4** next.\n\nBut it’s only a **single threat**—Yellow can simply play **column 2** to block it. So it doesn’t keep as much pressure as the key center move.'),
    ('so which one would be the best move?',
     'The best move is **column 4**.\n\nWhy: it puts Red at **c4 row 4**, building a strong central vertical stack above your Red at **c4 row 3** while also avoiding giving Yellow an easy defensive reset. It’s the move that keeps your winning chances/forced win alive with only **2 moves left**.'),
    ('What about now?',
     'Look for a move that creates an **immediate 4-in-a-row threat**, not just a block.\n\nKey area: your Reds at **c3 r1, c4 r2, c5 r3?** — check the diagonal possibilities and which column would complete/support one. Also make sure Yellow doesn’t already have a direct win you must block.'),
    ('Give me the best move',
     'Best move: **column 5**.\n\nIt lands at **c5 row 4**, creating a strong **diagonal threat** with your Reds around the center and avoiding giving Yellow the key square. With only 1 move left, this is the move that keeps Red’s forced win alive.'),
]
_B_REASON_SHORT_R = [
    ('If I add to column 2, will that reduce my possibilities rather than improve my chances?',
     'Column 2 would **create a vertical threat**: Red would have discs at **c2 rows 1–3**, needing **c2 row 4** next.\n\nBut ask: **can Yellow immediately block that by playing column 2?** Yes — Yellow could drop into **c2 row 4**, stopping it.\n\nSo column 2 improves one obvious threat, but it may be a bit “single-purpose”: it gives Yellow a clear block and may not build a harder-to-stop double threat. Before choosing it, compare: *does another move create two threats at once or force Yellow’s reply?*'),
    ("how about 4, blocking yellow's progress and adding to the diagonal?",
     'Yes—column **4** is doing two useful things:\n\n- It lands at **c4 row 3**, directly sitting on top of Yellow’s **c4 row 1–2**, so it stops Yellow from building that central vertical.\n- It also makes a diagonal line with your **c5 row 2** and **c6 row 1**: **c4-r3, c5-r2, c6-r1**.\n\nSo compared with columns 2 or 5, column 4 looks more forcing: it **blocks Yellow while improving Red’s shape**.'),
]

INTERVENTION_VARIANTS = {
    "v0_live": [(_A_REPEAT, _B_CANDIDATE), (_A_TERSE, _B_PLAN), (_A_LONG, _B_RETRO)],
    "v2_think_pair": [(_A_TERSE, _B_THINK1), (_A_REPEAT, _B_THINK2), (_A_LONG, _B_PLAN)],
    # v9: two rows, and Group B gives a REASON rather than only naming a candidate.
    # v0/v2 made candidate-naming the repeated feature and showed a justification once;
    # that is what participants copied -- scaffolded think-aloud turns carried a stated
    # reason in ~14% of cases against vanilla's ~25%, and the coached act did not
    # predict transfer while the spontaneous one did. Here 4 of 5 Group B messages
    # attach a consequence to the move. Group A stays LONGER (83 words vs 76) so
    # verbosity is not available as the perceived difference.
    "v9_two_rows": [(_A_OUTSOURCE, _B_REASON_MID), (_A_REPEAT, _B_REASON_SHORT)],
    # v9_reasons: the SAME two tables as v9_two_rows. The key exists because the
    # correction sentence changed -- it now requires the participant to have stated what
    # the move would do, where the earlier wording ("said what they were thinking",
    # "asked and reflected about specific moves") was satisfiable by a bare coordinate.
    # The correction text is not itself versioned, so bumping the variant is what keeps
    # `which material did this person see` recoverable from the data: the 15 participants
    # recorded as v9_two_rows saw the old sentence, v9_reasons participants see the new one.
    "v9_reasons":  [(_A_OUTSOURCE_R, _B_REASON_MID_R),
                    (_A_REPEAT_R, _B_REASON_SHORT_R)],
}
