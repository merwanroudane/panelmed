"""Journal-style tables: one DataFrame, rendered as text, LaTeX, Markdown, HTML.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

WIDTH = 86


def stars(p: float) -> str:
    """Significance stars at the 1, 5 and 10 percent levels."""
    if p is None or not np.isfinite(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def _tex(s: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"),
                 ("#", r"\#"), ("^", r"\^{}"), ("*", r"$\ast$")):
        s = s.replace(a, b)
    return s


class Table:
    """A results table with a title, a model block, grouped rows and notes.

    ``frame`` must have the columns ``estimate, se, z, p, ci_low, ci_high``;
    its index holds the row labels. ``sections`` groups rows under headings.
    Printing the object gives the text table; ``as_latex()``, ``as_markdown()``
    and ``as_html()`` render the same numbers.
    """

    def __init__(self, title: str, info: Sequence[Tuple[str, str]], frame: pd.DataFrame,
                 notes: Sequence[str], sections: Optional[List[Tuple[str, List[str]]]] = None,
                 level: float = 0.95, row_header: str = ""):
        self.title, self.info, self.frame = title, list(info), frame
        self.notes, self.level, self.row_header = list(notes), level, row_header
        self.sections = sections or [("", list(frame.index))]

    # ------------------------------------------------------------------ text
    def as_text(self) -> str:
        lab = max([len(str(i)) for i in self.frame.index] + [len(self.row_header), 10]) + 1
        lab = min(lab, 40)
        w = max(WIDTH, lab + 71)
        out = ["=" * w, self.title.center(w), "=" * w]
        half = (len(self.info) + 1) // 2
        left, right = self.info[:half], self.info[half:]
        for k in range(half):
            a = f"{left[k][0]:<19}: {left[k][1]}"
            b = f"{right[k][0]:<17}: {right[k][1]:>12}" if k < len(right) else ""
            out.append(a.ljust(w - 32) + " " + b)
        ci = f"[{100 * self.level:.0f}% Conf. Int.]"
        out.append("-" * w)
        out.append(f"{self.row_header:<{lab}}{'Coef.':>10}{'Std.Err.':>10}{'z':>9}{'P>|z|':>8}"
                   f"{ci:>22}")
        out.append("-" * w)
        for k, (head, rows) in enumerate(self.sections):
            if head:
                if k:
                    out.append("-" * w)
                out.append(head)
            for r in rows:
                v = self.frame.loc[r]
                z = f"{v['z']:>9.3f}" if np.isfinite(v["z"]) else f"{'':>9}"
                p = f"{v['p']:>8.3f}" if np.isfinite(v["p"]) else f"{'':>8}"
                out.append(f"{str(r)[:lab - 1]:<{lab}}{v['estimate']:>10.4f}{v['se']:>10.4f}{z}{p}"
                           f"{v['ci_low']:>11.4f}{v['ci_high']:>11.4f} {stars(v['p'])}")
        out.append("=" * w)
        out.append("Significance: *** p<0.01, ** p<0.05, * p<0.10")
        out += self.notes
        return "\n".join(out)

    __str__ = as_text
    __repr__ = as_text

    def _repr_html_(self) -> str:
        return self.as_html()

    # ----------------------------------------------------------------- latex
    def as_latex(self, float_format: str = "%.4f", stars_: bool = True, notes: bool = True,
                 caption: Optional[str] = None, label: str = "tab:panelmed") -> str:
        lines = [r"% \newcommand{\sym}[1]{\ensuremath{^{#1}}}   (add to the preamble)",
                 r"\begin{table}[htbp]\centering",
                 r"\caption{%s}\label{%s}" % (_tex(caption or self.title), label),
                 r"\begin{tabular}{lcccc}", r"\toprule",
                 r" & Estimate & Std. Err. & $z$ & %d\%% CI \\" % round(100 * self.level),
                 r"\midrule"]
        for k, (head, rows) in enumerate(self.sections):
            if head:
                if k:
                    lines.append(r"\midrule")
                lines.append(r"\multicolumn{5}{l}{\textit{%s}} \\" % _tex(head))
            for r in rows:
                v = self.frame.loc[r]
                st = r"\sym{%s}" % stars(v["p"]) if stars_ and stars(v["p"]) else ""
                z = (float_format % v["z"]) if np.isfinite(v["z"]) else ""
                lines.append("%s & %s%s & (%s) & %s & [%s, %s] \\\\" % (
                    _tex(str(r)), float_format % v["estimate"], st, float_format % v["se"], z,
                    float_format % v["ci_low"], float_format % v["ci_high"]))
        lines += [r"\bottomrule", r"\end{tabular}"]
        if notes:
            txt = " ".join(_tex(n) for n in self.notes)
            lines += [r"\begin{minipage}{0.9\linewidth}\footnotesize",
                      r"\textit{Notes:} Standard errors in parentheses. \sym{***} $p<0.01$, "
                      r"\sym{**} $p<0.05$, \sym{*} $p<0.10$. " + txt, r"\end{minipage}"]
        lines.append(r"\end{table}")
        return "\n".join(lines)

    # -------------------------------------------------------------- markdown
    def as_markdown(self) -> str:
        ci = f"{100 * self.level:.0f}% CI"
        out = [f"**{self.title}**", "",
               f"| {self.row_header or ' '} | Coef. | Std.Err. | z | P>\\|z\\| | {ci} |",
               "|---|---:|---:|---:|---:|:---:|"]
        for head, rows in self.sections:
            if head:
                out.append(f"| *{head}* | | | | | |")
            for r in rows:
                v = self.frame.loc[r]
                z = f"{v['z']:.3f}" if np.isfinite(v["z"]) else ""
                p = f"{v['p']:.3f}" if np.isfinite(v["p"]) else ""
                out.append(f"| {r} | {v['estimate']:.4f}{stars(v['p'])} | {v['se']:.4f} | {z} | "
                           f"{p} | [{v['ci_low']:.4f}, {v['ci_high']:.4f}] |")
        out += ["", "Significance: \\*\\*\\* p<0.01, \\*\\* p<0.05, \\* p<0.10. "
                + " ".join(self.notes)]
        return "\n".join(out)

    # ------------------------------------------------------------------ html
    def as_html(self) -> str:
        ci = f"{100 * self.level:.0f}% CI"
        out = ['<table class="panelmed-table">', f"<caption>{self.title}</caption>",
               f"<thead><tr><th>{self.row_header}</th><th>Coef.</th><th>Std.Err.</th>"
               f"<th>z</th><th>P&gt;|z|</th><th>{ci}</th></tr></thead>", "<tbody>"]
        for head, rows in self.sections:
            if head:
                out.append(f'<tr class="section"><td colspan="6"><em>{head}</em></td></tr>')
            for r in rows:
                v = self.frame.loc[r]
                z = f"{v['z']:.3f}" if np.isfinite(v["z"]) else ""
                p = f"{v['p']:.3f}" if np.isfinite(v["p"]) else ""
                out.append(f"<tr><td>{r}</td><td>{v['estimate']:.4f}{stars(v['p'])}</td>"
                           f"<td>{v['se']:.4f}</td><td>{z}</td><td>{p}</td>"
                           f"<td>[{v['ci_low']:.4f}, {v['ci_high']:.4f}]</td></tr>")
        out += ["</tbody>", "</table>"]
        return "\n".join(out)


def inference_frame(est, se, level: float, index, null: Optional[Sequence[bool]] = None):
    """Tidy frame ``estimate, se, z, p, ci_low, ci_high`` from estimates and SEs.

    Rows flagged ``False`` in ``null`` get no z statistic or p-value (a test
    against zero is not of interest for them, e.g. the mediated share).
    """
    from scipy.stats import norm
    est, se = np.asarray(est, float), np.asarray(se, float)
    crit = norm.ppf(0.5 + level / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, est / se, np.nan)
    p = 2.0 * norm.sf(np.abs(z))
    if null is not None:
        keep = np.asarray(null, bool)
        z, p = np.where(keep, z, np.nan), np.where(keep, p, np.nan)
    return pd.DataFrame({"estimate": est, "se": se, "z": z, "p": p,
                         "ci_low": est - crit * se, "ci_high": est + crit * se},
                        index=pd.Index(list(index)))
