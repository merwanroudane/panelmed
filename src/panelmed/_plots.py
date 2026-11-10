"""Figures. Every function returns ``(fig, ax)`` and never calls ``plt.show()``.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ._colors import get_cmap, resolve_colorscale

JOURNAL_RC = {
    "figure.figsize": (6.5, 4.0), "figure.dpi": 110, "savefig.dpi": 300,
    "savefig.bbox": "tight", "font.family": "serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 1.6,
}
# colour paired with a marker and a line style, so figures survive greyscale
STYLE = {"direct": ("#1f4e9c", "o", "-"), "indirect": ("#c0392b", "s", "--"),
         "total": ("#1e8449", "D", "-."), "controlled": ("#7d3c98", "^", ":")}
ROLE = {"CDE": "controlled", "CDD": "controlled", "NDE": "direct", "NDD": "direct",
        "NIE": "indirect", "NID": "indirect", "TE": "total", "TD": "total"}
LONG = {"CDE": "Controlled direct", "NDE": "Natural direct", "NIE": "Natural indirect",
        "TE": "Total", "CDD": "Controlled direct", "NDD": "Natural direct",
        "NID": "Natural indirect", "TD": "Total"}


def _plt():
    import matplotlib.pyplot as plt
    return plt


def plot_coefficients(res, level: float = 0.95, ax=None):
    """Dot-and-whisker plot of the structural coefficients with Wald intervals."""
    plt = _plt()
    fr = res.to_frame(level)
    with plt.rc_context(JOURNAL_RC):
        fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(
            figsize=(6.5, 0.38 * len(fr) + 1.4))
        pos = np.arange(len(fr))[::-1]
        for k, (lab, row) in enumerate(fr.iterrows()):
            col = "#1f4e9c" if k < res.sY else "#c0392b"
            mk = "o" if k < res.sY else "s"
            ax.errorbar(row["estimate"], pos[k], xerr=[[row["estimate"] - row["ci_low"]],
                        [row["ci_high"] - row["estimate"]]], fmt=mk, color=col, capsize=3,
                        markersize=5, linewidth=1.4)
        ax.axvline(0, lw=0.8, color="0.3")
        ax.set_yticks(pos)
        ax.set_yticklabels(fr.index)
        ax.set_xlabel(f"Coefficient ({100 * level:.0f}% confidence interval)")
        ax.set_title("Structural coefficients: outcome (circles) and mediator (squares)")
        ax.grid(axis="y", visible=False)
    return fig, ax


def plot_moments(res, ax=None):
    """Diagnostic: standardised sample moments at the estimate.

    Each bar is sqrt(N) * mean(psi_j) / sd(psi_j). Under valid moment
    conditions the bars behave like standard normal draws; the dashed lines
    mark +/- 1.96. With identity weighting the moments are not set to zero
    one by one, so this is the picture of how well the over-identified
    system fits.
    """
    plt = _plt()
    psi = res.psi
    N = psi.shape[0]
    sd = psi.std(axis=0)
    z = np.sqrt(N) * psi.mean(axis=0) / np.where(sd > 0, sd, 1.0)
    ny = res.n_moments_y
    with plt.rc_context(JOURNAL_RC):
        fig, ax = (ax.figure, ax) if ax is not None else plt.subplots()
        idx = np.arange(z.size)
        ax.bar(idx[:ny], z[:ny], color="#1f4e9c", width=0.9, label="Outcome moments")
        ax.bar(idx[ny:], z[ny:], color="#c0392b", width=0.9, hatch="//", edgecolor="white",
               label="Mediator moments")
        for v in (-1.96, 1.96):
            ax.axhline(v, ls="--", lw=0.9, color="0.3")
        ax.axhline(0, lw=0.8, color="0.3")
        share = float((np.abs(z) > 1.96).mean())
        ax.set_xlabel("Moment condition")
        ax.set_ylabel("Standardised sample moment")
        ax.set_title(f"Moment diagnostic: {100 * share:.0f}% of {z.size} moments beyond "
                     r"$\pm$1.96")
        ax.legend(loc="upper right")
    return fig, ax


def plot_effects(eff, ax=None):
    """Dot-and-whisker plot of the decomposition, one row per effect."""
    plt = _plt()
    fr = eff.frame[~eff.frame["kind"].str.contains("/")]
    with plt.rc_context(JOURNAL_RC):
        fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(
            figsize=(6.5, 0.42 * len(fr) + 1.4))
        pos = np.arange(len(fr))[::-1]
        for k, (lab, row) in enumerate(fr.iterrows()):
            col, mk, _ = STYLE[ROLE[row["kind"]]]
            ax.errorbar(row["estimate"], pos[k], xerr=[[row["estimate"] - row["ci_low"]],
                        [row["ci_high"] - row["estimate"]]], fmt=mk, color=col, capsize=3,
                        markersize=6, linewidth=1.5)
        ax.axvline(0, lw=0.8, color="0.3")
        ax.set_yticks(pos)
        ax.set_yticklabels([f"{LONG[k]} ({lab})" for lab, k in zip(fr.index, fr["kind"])])
        ax.set_xlabel(f"Effect on the outcome ({100 * eff.level:.0f}% confidence interval)")
        ax.set_title("Decomposition of the total effect")
        ax.grid(axis="y", visible=False)
    return fig, ax


def plot_profile(eff, kinds: Optional[Sequence[str]] = None, ax=None):
    """Period-specific effects with confidence bands (as Figure A.1 of the paper)."""
    plt = _plt()
    fr = eff.frame[~eff.frame["kind"].str.contains("/")]
    if fr["period"].nunique() < 2:
        raise ValueError("Build the effects with period='all' (or a list) to plot a profile.")
    kinds = list(kinds) if kinds is not None else [k for k in eff.kinds if ROLE[k] != "controlled"]
    with plt.rc_context(JOURNAL_RC):
        fig, ax = (ax.figure, ax) if ax is not None else plt.subplots()
        for k in kinds:
            sub = fr[fr["kind"] == k]
            col, mk, ls = STYLE[ROLE[k]]
            x = np.arange(len(sub))
            ax.fill_between(x, sub["ci_low"], sub["ci_high"], color=col, alpha=0.20, lw=0)
            ax.plot(x, sub["estimate"], color=col, marker=mk, ls=ls, markersize=4,
                    label=f"{LONG[k]} ({k})")
        ax.axhline(0, lw=0.8, color="0.3")
        step = max(1, int(np.ceil(len(x) / 8)))
        ax.set_xticks(x[::step])
        ax.set_xticklabels(list(sub["period"])[::step])
        ax.set_xlabel("Period")
        ax.set_ylabel(f"Effect ({100 * eff.level:.0f}% band)")
        ax.set_title("Period-specific effects")
        ax.legend()
    return fig, ax


def plot_dose_response(res, grid=None, d_star: float = 0.0, m: float = 0.0,
                       period="last", level: float = 0.95, ax=None):
    """Natural direct, natural indirect and total effect of ``d`` versus ``d_star``.

    ``grid`` holds the treatment values ``d`` (constant histories); default:
    25 points between the 5th and 95th percentiles of the observed treatment.
    """
    plt = _plt()
    D = res.model.design.D
    grid = np.linspace(*np.percentile(D, [5, 95]), 25) if grid is None else np.asarray(grid, float)
    rows = {k: [] for k in ("NDE", "NIE", "TE")}
    for g in grid:
        fr = res.effects(d=g, d_star=d_star, m=m, period=period, level=level).frame
        for k in rows:
            rows[k].append(fr.loc[k, ["estimate", "ci_low", "ci_high"]].to_numpy(float))
    with plt.rc_context(JOURNAL_RC):
        fig, ax = (ax.figure, ax) if ax is not None else plt.subplots()
        for k, vals in rows.items():
            v = np.array(vals)
            col, mk, ls = STYLE[ROLE[k]]
            ax.fill_between(grid, v[:, 1], v[:, 2], color=col, alpha=0.20, lw=0)
            ax.plot(grid, v[:, 0], color=col, ls=ls, label=f"{LONG[k]} ({k})")
        ax.axhline(0, lw=0.8, color="0.3")
        ax.set_xlabel(f"Treatment history d (compared with d* = {d_star:g})")
        ax.set_ylabel(f"Effect on the outcome ({100 * level:.0f}% band)")
        ax.set_title("Dose-response of the direct, indirect and total effects")
        ax.legend()
    return fig, ax


def _surface_values(res, grid, kind, m, period):
    D = res.model.design.D
    grid = np.linspace(*np.percentile(D, [5, 95]), 21) if grid is None else np.asarray(grid, float)
    Z = np.empty((grid.size, grid.size))
    N, T = res.model.nobs
    periods, avg = res._periods(period)
    ki = ("CDE", "NDE", "NIE", "TE").index(kind)
    from ._utils import as_history
    mh = as_history(m, N, T, "m")
    for i, ds in enumerate(grid):
        for j, d in enumerate(grid):
            a = res._cf.level(res._beta, as_history(d, N, T, "d"), as_history(ds, N, T, "d"),
                              mh, res.sY)[:, periods, ki]
            Z[i, j] = a.mean()
    return grid, Z


def plot_surface(res, grid=None, kind: str = "TE", m: float = 0.0, period="last",
                 colorscale: str = "Parula", interactive: bool = False, ax=None):
    """Effect of history ``d`` versus ``d*`` over a grid of both, as a heatmap.

    Parameters
    ----------
    grid : 1-D array, optional
        Values used for both ``d`` and ``d*``.
    kind : {"TE", "NDE", "NIE", "CDE"}
    colorscale : str, default "Parula"
        One of Parula, Jet, Turbo, BlueRed, Sinha.
    interactive : bool
        Return a plotly 3-D surface instead of a matplotlib heatmap.
    """
    grid, Z = _surface_values(res, grid, kind, m, period)
    if interactive:
        try:
            import plotly.graph_objects as go
        except ImportError as e:  # pragma: no cover
            raise ImportError("interactive=True requires plotly: pip install "
                              "\"panelmed[plotly]\"") from e
        fig = go.Figure(go.Surface(x=grid, y=grid, z=Z, colorscale=resolve_colorscale(colorscale),
                                   colorbar=dict(title=kind)))
        fig.update_layout(template="plotly_white", title=f"{LONG[kind]} effect ({kind})",
                          scene=dict(xaxis_title="d", yaxis_title="d*", zaxis_title=kind,
                                     camera=dict(eye=dict(x=1.5, y=-1.6, z=0.9))))
        return fig
    plt = _plt()
    with plt.rc_context(JOURNAL_RC):
        fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(5.6, 4.4))
        im = ax.pcolormesh(grid, grid, Z, cmap=get_cmap(colorscale), shading="auto")
        cs = ax.contour(grid, grid, Z, levels=[0.0], colors="white", linewidths=1.2)
        ax.clabel(cs, fmt="0", fontsize=8)
        fig.colorbar(im, ax=ax, label=f"{LONG[kind]} effect ({kind})")
        ax.set_xlabel("Treatment history d")
        ax.set_ylabel("Reference history d*")
        ax.set_title(f"{LONG[kind]} effect of d versus d*")
        ax.grid(False)
    return fig, ax
