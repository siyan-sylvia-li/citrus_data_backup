"""Forest plots of the elicitation matrix, ported from bad_user_sim/elicitation_analysis.ipynb.

The drawing code is that notebook's `plot_elicitation` VERBATIM -- same panels, axis limits,
colours, separation handling and caret logic -- so a plot produced here is directly
comparable with the ones already in that notebook rather than a lookalike. Only the data
source differs: `elicitation.load_turns()` covers all five Othello datasets and carries a
`population` column, so the same figure can be drawn for adopters and non-adopters.

    from elicitation import load_turns
    from elicitation_plot import plot_elicitation
    T = load_turns()
    plot_elicitation(T[T.population == "non-adopters"], "non-adopters")
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from elicitation import PACTS, SHORT, RELIABLE

SEPARATION = (1e-3, 1e3)   # ORs outside this are separation artifacts, not estimates

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PANEL_ACTS = ["Board Report", "Move Verdict", "General Principle", "Prompt"]
XLIM = (0.06, 16.0)          # log axis; wider CIs are drawn clipped with a caret
C_UP, C_DOWN, C_NS = "#d1622a", "#2f7f6d", "#9aa0a6"


def elicit_matrix(d):
    '''One row per (assistant act, participant act): OR, CI, p, separation flag.'''
    rows = []
    for aa in PANEL_ACTS:
        y = d.asst.apply(lambda s, t=aa: int(t in s))
        if y.nunique() < 2:
            continue
        X = sm.add_constant(d[[SHORT[p] for p in PACTS]])
        m = sm.Logit(y, X).fit(disp=0, cov_type="cluster",
                               cov_kwds={"groups": d.pid})
        ci = np.exp(m.conf_int())
        for pa in PACTS:
            k = SHORT[pa]
            orr = float(np.exp(m.params[k]))
            rows.append(dict(
                asst=aa, pact=k, OR=orr, lo=float(ci.loc[k, 0]),
                hi=float(ci.loc[k, 1]), p=float(m.pvalues[k]),
                # Quasi-complete separation: the MLE ran to +/-inf, so the OR and its
                # CI are artifacts. Flagged and drawn hollow rather than dropped.
                sep=not (SEPARATION[0] < orr < SEPARATION[1]),
                co=int(((X[k] == 1) & (y == 1)).sum()), n1=int((X[k] == 1).sum()),
                n_asst=int(y.sum())))
    return pd.DataFrame(rows)


def plot_elicitation(d, game):
    mat = elicit_matrix(d)
    acts = [a for a in PANEL_ACTS if a in set(mat.asst)]
    ypos = {SHORT[p]: len(PACTS) - 1 - i for i, p in enumerate(PACTS)}   # top-down

    fig, axes = plt.subplots(1, len(acts), figsize=(2.6 * len(acts) + 1.6, 3.4),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, aa in zip(axes, acts):
        sub = mat[mat.asst == aa]
        ax.axvline(1, color="#4a4a4a", lw=1, zorder=1)
        for _, r in sub.iterrows():
            yy = ypos[r.pact]
            col = C_NS if (r.p >= .05 or r.sep) else (C_UP if r.OR > 1 else C_DOWN)
            if r.sep:
                # No estimate to draw: park it at the edge it ran off to.
                x = XLIM[1] * 0.92 if r.OR > 1 else XLIM[0] * 1.08
                ax.plot([x], [yy], marker="o", ms=6, mfc="white", mec=col, mew=1.4,
                        zorder=3)
                ax.text(x, yy + 0.3, f"sep {r.co}/{r.n1}", fontsize=6, color=col,
                        ha="center")
                continue
            # Clamp to the axis, then say so. A caret is drawn only for the bound
            # that actually leaves the axis, and an estimate whose point OR is off
            # scale is labelled with its value instead of a dot mashed on the edge.
            lo, hi = float(np.clip(r.lo, *XLIM)), float(np.clip(r.hi, *XLIM))
            if hi > lo:
                ax.plot([lo, hi], [yy, yy], color=col, lw=2,
                        solid_capstyle="round", alpha=.85, zorder=2)
            if r.lo < XLIM[0]:
                ax.plot([XLIM[0]], [yy], marker="<", ms=5, color=col, zorder=3)
            if r.hi > XLIM[1]:
                ax.plot([XLIM[1]], [yy], marker=">", ms=5, color=col, zorder=3)
            if XLIM[0] < r.OR < XLIM[1]:
                ax.plot([r.OR], [yy], marker="o", ms=7, color=col, mec="white",
                        mew=1.2, zorder=4)
            else:
                off_hi = r.OR >= XLIM[1]
                x = XLIM[1] * 0.97 if off_hi else XLIM[0] * 1.03
                ax.plot([x], [yy], marker=">" if off_hi else "<", ms=8, color=col,
                        zorder=4)
                ax.text(x, yy + 0.28, f"{r.OR:.3g}", fontsize=6.5, color=col,
                        ha="right" if off_hi else "left")
        n_asst = int(sub.n_asst.iloc[0])
        dagger = "" if aa in RELIABLE else " †"
        ax.set_title(f"{aa}{dagger}\n{100 * n_asst / len(d):.0f}% of turns",
                     fontsize=8.5, color="#222" if aa in RELIABLE else "#777")
        ax.set_xscale("log")
        ax.set_xlim(*XLIM)
        ax.set_xticks([0.1, 1, 10])
        ax.set_xticklabels(["0.1", "1", "10"], fontsize=8)
        ax.set_xticks([0.25, 0.5, 2, 4], minor=True)
        ax.set_xticklabels([], minor=True)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.7, len(PACTS) - 0.3)
        ax.grid(axis="x", color="#e8e8e8", lw=.6, which="both", zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#bbb")

    axes[0].set_yticks([ypos[SHORT[p]] for p in PACTS])
    axes[0].set_yticklabels([SHORT[p] for p in PACTS], fontsize=8.5)
    fig.suptitle(f"{game.upper()} — what each participant act elicits  "
                 f"(n={len(d)} turns, {d.pid.nunique()} participants)",
                 fontsize=10.5, y=1.04)
    dag_note = " † = kappa<.6." if any(a not in RELIABLE for a in acts) else ""
    fig.supxlabel("odds ratio, adjusted for all six participant acts (log scale; "
                  f"caret = off scale, labelled where the point OR is).{dag_note}",
                  fontsize=8, color="#555", y=-0.06)
    fig.legend(handles=[
        Line2D([], [], color=C_UP, marker="o", lw=2, label="elicits (p<.05)"),
        Line2D([], [], color=C_DOWN, marker="o", lw=2, label="suppresses (p<.05)"),
        Line2D([], [], color=C_NS, marker="o", lw=2, label="not significant"),
        Line2D([], [], color=C_NS, marker="o", lw=0, mfc="white", mec=C_NS,
               label="separation (no estimate)")],
        loc="lower center", ncol=4, frameon=False, fontsize=8,
        bbox_to_anchor=(0.5, -0.19))
    plt.tight_layout()
    plt.show()
    return mat
