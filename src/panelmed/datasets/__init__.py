"""Data generators shipped with the package.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

__all__ = ["simulate_panel", "load_us_counties", "US_COUNTIES_COVARIATES", "SIM_OUTCOME",
           "SIM_MEDIATOR", "SIM_THETA", "SIM_GAMMA", "SIM_TRUE_EFFECTS"]

#: Covariates of the county application, in the order used by the paper (Section 6).
US_COUNTIES_COVARIATES = ["log_pop", "ui_share", "share_under18", "share_over65", "births",
                          "deaths", "mig_international", "mig_domestic"]


def load_us_counties(prepared: bool = True) -> pd.DataFrame:
    """U.S. county panel, 3,058 counties, 2000-2019 (annual).

    The data of the empirical application in Wu and Xiao (2026, Section 6),
    assembled by Cerqua, Letta and Pinto (2026) from public U.S. statistics.

    Parameters
    ----------
    prepared : bool, default True
        ``True`` returns the analysis panel built as in Section 6 of the
        paper: years 2001-2019 (T = 19), outcome ``y`` = log personal income
        per capita, mediator ``m`` = log average wage per job, treatment
        ``d`` = workplace employment rate, and eight covariates measured in
        the previous calendar year; every variable is winsorised at its
        pooled 1% and 99% quantiles and then standardised, so effects are in
        outcome standard deviations. ``False`` returns the raw 2000-2019 file.

    Returns
    -------
    DataFrame
        Long format, one row per county and year. Prepared columns:
        ``county_id, county_name, year, y, m, d`` and the covariates listed
        in ``US_COUNTIES_COVARIATES``. Raw columns: ``county_id, year,
        state_fips, county_name, income_pc`` (USD), ``avg_wage`` (USD per
        job), ``employment_rate`` (jobs per resident aged 18-65),
        ``population``, ``ui_share`` (% of personal income from unemployment
        insurance), ``share_under18``, ``share_over65`` (% of population),
        ``births``, ``deaths``, ``mig_international``, ``mig_domestic`` (per
        1,000 residents).

    References
    ----------
    Cerqua, A., Letta, M. and Pinto, G. (2026). On the (mis)use of machine
    learning with panel data. Oxford Bulletin of Economics and Statistics,
    88(3), 506-518. <https://doi.org/10.1111/obes.70019>

    Examples
    --------
    >>> df = load_us_counties()
    >>> df["county_id"].nunique(), df["year"].nunique()
    (3058, 19)
    """
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "us_counties.csv.gz")
    raw = pd.read_csv(path)
    if not prepared:
        return raw
    d = raw.sort_values(["county_id", "year"]).reset_index(drop=True)
    d["y"] = np.log(d["income_pc"])
    d["m"] = np.log(d["avg_wage"])
    d["d"] = d["employment_rate"]
    d["log_pop"] = np.log(d["population"])
    for c in US_COUNTIES_COVARIATES:                     # covariates dated t - 1
        d[c] = d.groupby("county_id")[c].shift(1)
    d = d[d["year"] >= 2001].copy()
    for c in ["y", "m", "d"] + US_COUNTIES_COVARIATES:
        lo, hi = d[c].quantile([0.01, 0.99])
        s = d[c].clip(lo, hi)
        d[c] = (s - s.mean()) / s.std()
    cols = ["county_id", "county_name", "year", "y", "m", "d"] + US_COUNTIES_COVARIATES
    return d[cols].reset_index(drop=True)

#: Outcome and mediator dictionaries of the Monte Carlo design (Section 5.1).
SIM_OUTCOME = ["d", "d**2", "L1_d", "d*x3", "m", "L1_m"]
SIM_MEDIATOR = ["d", "d**2", "L1_d"]
SIM_THETA = np.array([2.0, 0.5, 2.0, 2.0, 1.0, 0.5])
SIM_GAMMA = np.array([1.0, 0.5, 0.5])
#: True effects of Eq. (23): histories 1_T versus 0_T, evaluated at t = T.
SIM_TRUE_EFFECTS = {"CDE": 4.5, "NDE": 4.5, "NIE": 3.0, "TE": 7.5,
                    "NDD": 3.0, "NID": 2.0, "TD": 5.0}


def _nuisances(design: str):
    e = np.exp
    if design == "baseline":                                     # Eq. (A.16)
        def f0(x):
            return ((x[..., 0] ** 2 - 1) + (np.cosh(x[..., 0]) - e(0.5)) + np.sin(2 * x[..., 1])
                    + np.tanh(2 * x[..., 2]) + np.sin(x[..., 0] * x[..., 1]))

        def g0(x):
            return ((x[..., 0] ** 2 - 1) + 2 * (np.cos(2 * x[..., 1]) - e(-2.0))
                    + np.sin(2 * x[..., 3]) + 2 * (e(x[..., 4]) - e(0.5)))

        def h0(x):
            return (x[..., 0] ** 2 - 1) - x[..., 0] + (x[..., 1] ** 2 - 1) + x[..., 2]
    elif design == "alternative":                                # Eq. (A.17)
        def f0(x):
            return ((x[..., 0] ** 2 - 1) + (x[..., 1] ** 2 - 1) + 0.2 * x[..., 0] ** 3 * np.abs(x[..., 0])
                    + 2 * x[..., 0] / (1 + x[..., 0] ** 2) + np.arcsinh(x[..., 2])
                    + np.arctan(x[..., 0] * x[..., 1]))

        def g0(x):
            return ((x[..., 1] ** 2 - 1) + 10 * (e(-x[..., 0] ** 2) - 3 ** -0.5)
                    + 2 * x[..., 3] / (1 + x[..., 3] ** 2)
                    + 2 * (np.maximum(x[..., 4], 0) ** 2 - 0.5))

        def h0(x):
            return (1.5 * (x[..., 0] ** 2 - 1) + (x[..., 1] ** 2 - 1) + (x[..., 2] ** 2 - 1)
                    + x[..., 2])
    else:
        raise ValueError("`design` must be 'baseline' or 'alternative'.")
    return f0, g0, h0


def simulate_panel(n: int = 1600, t: int = 10, design: str = "baseline",
                   seed: Optional[int] = None, as_frame: bool = False):
    """Simulate the Monte Carlo panel of Wu and Xiao (2026), Appendix D.1.

    Nonlinear time-varying confounding, correlated unit effects sharing a
    common factor, equation-specific period effects, and a treatment that
    responds to the previous outcome and mediator shocks (sequential rather
    than strict exogeneity). Use ``SIM_OUTCOME`` and ``SIM_MEDIATOR`` as the
    dictionaries; the true coefficients are ``SIM_THETA`` and ``SIM_GAMMA``.

    Parameters
    ----------
    n, t : int
        Number of units and periods.
    design : {"baseline", "alternative"}
        Nuisance functions of Eq. (A.16) or Eq. (A.17).
    seed : int, optional
    as_frame : bool
        Return a long ``DataFrame`` instead of a dict of ``(N, T)`` arrays.

    Returns
    -------
    dict with keys ``y``, ``m``, ``d`` (``(N, T)``) and ``x`` (``(N, T, 5)``),
    or a long DataFrame with columns ``unit, time, y, m, d, x1..x5``.

    Examples
    --------
    >>> sim = simulate_panel(n=50, t=5, seed=0)
    >>> sim["y"].shape, sim["x"].shape
    ((50, 5), (50, 5, 5))
    """
    rng = np.random.default_rng(seed)
    f0, g0, h0 = _nuisances(design)
    A = rng.normal(size=n)                               # common factor, drawn once per unit
    aD, aM, aY = (A + rng.normal(size=n) for _ in range(3))
    xi = rng.normal(size=(n, t))
    eM = rng.normal(size=(n, t))
    eY = rng.normal(size=(n, t))
    nu = rng.normal(size=(n, t, 5))
    X = np.empty((n, t, 5))
    X[:, 0] = nu[:, 0]
    for s in range(1, t):
        X[:, s] = 0.5 * X[:, s - 1] + np.sqrt(1 - 0.25) * nu[:, s]
    per = np.arange(1, t + 1)
    lamY = np.sin(2 * np.pi * per / t)                   # period effects, common across units
    lamM = lamD = 0.5 * np.cos(2 * np.pi * per / t)
    eD = xi.copy()
    eD[:, 1:] += eY[:, :-1] + eM[:, :-1]                 # feedback from last period's shocks
    D = h0(X) + aD[:, None] + lamD + eD
    M = np.empty((n, t))
    Y = np.empty((n, t))
    gx, fx = g0(X), f0(X)
    for s in range(t):
        d_lag = D[:, s - 1] if s > 0 else 0.0            # values before period 1 are zero
        m_lag = M[:, s - 1] if s > 0 else 0.0
        M[:, s] = (D[:, s] + 0.5 * D[:, s] ** 2 + 0.5 * d_lag + gx[:, s] + aM + lamM[s]
                   + eM[:, s])
        Y[:, s] = (2 * D[:, s] + 0.5 * D[:, s] ** 2 + 2 * d_lag + 2 * D[:, s] * X[:, s, 2]
                   + M[:, s] + 0.5 * m_lag + fx[:, s] + aY + lamY[s] + eY[:, s])
    if not as_frame:
        return {"y": Y, "m": M, "d": D, "x": X}
    df = pd.DataFrame({
        "unit": np.repeat(np.arange(1, n + 1), t), "time": np.tile(per, n),
        "y": Y.ravel(), "m": M.ravel(), "d": D.ravel()})
    for j in range(5):
        df[f"x{j + 1}"] = X[:, :, j].ravel()
    return df
