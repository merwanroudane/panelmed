"""Public estimator, results objects and the one-call wrapper.

Implements Wu, P. and Xiao, Z. (2026), "Causal Mediation Analysis for Panel
Data with Double/Debiased Machine Learning", arXiv:2610.21339.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import linalg, stats

from ._core import (DERIV_KINDS, LEVEL_KINDS, Counterfactual, Design, IdentificationError,
                    effect_inference, fit_once)
from ._tables import Table, inference_frame
from ._utils import (Dictionary, as_history, make_env, polynomial_basis, term_depends_on,
                     term_nonlinear_in_d, to_panel)

PAPER = "Wu & Xiao (2026)"


def _boot_one(P: Design, ids: np.ndarray, cfg: dict, seed) -> Optional[np.ndarray]:
    """One cross-sectional bootstrap replication: the whole procedure is repeated."""
    try:
        return fit_once(P.take(ids), rng=np.random.default_rng(seed), **cfg)["beta"]
    except (IdentificationError, linalg.LinAlgError):
        return None


class PanelMediation:
    """DML-GMM causal mediation analysis for panels with two-way fixed effects.

    The outcome and the mediator follow the partially linear model of Eq. (4):

        Y_it = tau_Y(D, M, X)' theta + f0(X) + alpha_i + lambda_t + eps_it
        M_it = tau_M(D, X)'   gamma + g0(X) + alpha_i + lambda_t + eps_it

    Parameters
    ----------
    y, m, d : (N, T) arrays, or column names when ``data`` is given
        Outcome, continuous mediator and treatment.
    x : (N, T, K) array or list of column names, optional
        Time-varying covariates entering the nuisance functions.
    outcome : list of str or callable, or Dictionary
        Outcome dictionary tau_Y, e.g. ``["d", "L1_d", "d*x3", "m", "L1_m"]``.
    mediator : list of str or callable, or Dictionary
        Mediator dictionary tau_M, e.g. ``["d", "L1_d"]``. It cannot contain ``m``.
    data : DataFrame, optional
        Long-format balanced panel; then pass ``unit`` and ``time``.
    x_names : list of str, optional
        Covariate names for array input (default ``x1, x2, ...``).
    basis_degree : int, default 3
        Degree of the polynomial basis approximating f0, g0 and h0.
    basis_max_vars : int, optional
        Keep only monomials with at most this many distinct covariates.
    basis_lags : int, default 0
        Number of covariate lags entering the basis.
    basis : callable, optional
        Custom basis ``X (N, T, K) -> (N, T, p)``; overrides the polynomial.
    iv_interact : "auto" or list of covariate names
        Covariates interacted with the lagged treatment residual in the
        instruments. ``"auto"`` uses the covariates that interact with the
        treatment in each dictionary.
    iv_square : "auto" or bool
        Add the squared lagged treatment residual as an instrument. ``"auto"``
        adds it when the dictionary is nonlinear in the treatment.

    References
    ----------
    Wu, P. and Xiao, Z. (2026). Causal Mediation Analysis for Panel Data with
    Double/Debiased Machine Learning. arXiv:2610.21339.
    <https://doi.org/10.48550/arXiv.2610.21339>

    Examples
    --------
    >>> from panelmed import PanelMediation, simulate_panel
    >>> sim = simulate_panel(n=120, t=5, seed=1)
    >>> model = PanelMediation(sim["y"], sim["m"], sim["d"], sim["x"],
    ...                        outcome=["d", "m"], mediator=["d"], basis_degree=1)
    >>> res = model.fit(n_folds=3, seed=1)
    >>> res.params.shape
    (3,)
    """

    def __init__(self, y, m, d, x=None, *, outcome, mediator,
                 data: Optional[pd.DataFrame] = None, unit=None, time=None,
                 x_names: Optional[Sequence[str]] = None, basis_degree: int = 3,
                 basis_max_vars: Optional[int] = None, basis_lags: int = 0,
                 basis: Optional[Callable] = None, iv_interact="auto", iv_square="auto"):
        pan = to_panel(y, m, d, x, data=data, unit=unit, time=time, x_names=x_names)
        self.names, self.x_names = pan["names"], pan["x_names"]
        self.units, self.times = pan["units"], pan["times"]
        self.tau_y, self.tau_m = Dictionary(outcome), Dictionary(mediator)
        self.L = max(self.tau_y.max_lag, self.tau_m.max_lag)
        Y, M, D, X = pan["Y"], pan["M"], pan["D"], pan["X"]
        N, T = Y.shape
        env = make_env(D, M, X, self.x_names, self.L)
        env_m = make_env(D, None, X, self.x_names, self.L)
        TY, TM = self.tau_y.evaluate(env), self.tau_m.evaluate(env_m)
        rng = np.random.default_rng(12345)
        d_keys = ["d"] + [f"L{k}_d" for k in range(1, self.L + 1)]
        if not term_depends_on(self.tau_m, env_m, d_keys, rng).any():
            raise ValueError("The mediator dictionary does not depend on the treatment `d`.")
        if not term_depends_on(self.tau_y, env, ["m"] + [f"L{k}_m" for k in range(1, self.L + 1)],
                               rng).any():
            raise ValueError("The outcome dictionary does not contain the mediator `m`; "
                             "there is no indirect path to estimate.")
        if basis is not None:
            B = np.asarray(basis(X), dtype=float)
            if B.ndim != 3 or B.shape[:2] != (N, T):
                raise ValueError("`basis` must return an (N, T, p) array.")
            self.basis_label = f"custom ({B.shape[2]} terms)"
        elif X.shape[2] == 0:
            B = np.zeros((N, T, 0))
            self.basis_label = "none (no covariates)"
        else:
            B = polynomial_basis(X, basis_degree, basis_max_vars, basis_lags)
            self.basis_label = f"polynomial deg. {basis_degree} ({B.shape[2]} terms)"

        def interactions(dic, e):
            if iv_interact != "auto":
                unknown = [c for c in iv_interact if c not in self.x_names]
                if unknown:
                    raise ValueError(f"`iv_interact` names not among the covariates: {unknown}.")
                return [self.x_names.index(c) for c in iv_interact]
            dep_d = term_depends_on(dic, e, d_keys, rng)
            return [j for j, nm in enumerate(self.x_names)
                    if (term_depends_on(dic, e, [nm], rng) & dep_d).any()]

        def square(dic, e):
            return term_nonlinear_in_d(dic, e) if iv_square == "auto" else bool(iv_square)

        self.design = Design(
            Y, M, D, X, TY, TM, B,
            endo_y=term_depends_on(self.tau_y, env, ["d", "m"], rng),
            endo_m=term_depends_on(self.tau_m, env_m, ["d"], rng),
            int_y=interactions(self.tau_y, env), int_m=interactions(self.tau_m, env_m),
            sq_y=square(self.tau_y, env), sq_m=square(self.tau_m, env_m))
        self.param_names = ([f"theta[{n}]" for n in self.tau_y.names]
                            + [f"gamma[{n}]" for n in self.tau_m.names])

    @property
    def nobs(self):
        return self.design.Y.shape

    def fit(self, n_folds: int = 5, max_lags: Optional[int] = 2, penalty="plugin",
            method: str = "dml", weighting: str = "identity",
            seed: Optional[int] = None) -> "PanelMediationResults":
        """Estimate the structural parameters by Algorithm 1.

        Parameters
        ----------
        n_folds : int, default 5
            Number of folds K >= 3. One fold evaluates the score, the next one
            supplies the period means (auxiliary fold), the rest train the
            nuisance functions.
        max_lags : int or None, default 2
            Residual dates kept in each instrument vector; ``None`` uses the
            whole available history.
        penalty : "plugin", "cv" or float, default "plugin"
            l1 penalty of the first stage: ``sqrt(2 log(2p) / n)`` (the paper's
            rule), 5-fold cross-validation, or a fixed value.
        method : "dml" or "naive", default "dml"
            ``"naive"`` is the benchmark of Section 5: no cross-fitting,
            full-sample demeaning, lagged levels as instruments.
        weighting : "identity" or "optimal", default "identity"
            One-step GMM (the paper's choice) or two-step efficient GMM.
        seed : int, optional
            Seed of the random fold assignment.

        Returns
        -------
        PanelMediationResults
        """
        cfg = dict(n_folds=n_folds, max_lags=max_lags, penalty=penalty, method=method,
                   weighting=weighting)
        out = fit_once(self.design, rng=np.random.default_rng(seed), **cfg)
        return PanelMediationResults(self, out, cfg, seed)


class PanelMediationResults:
    """Structural estimates of the DML-GMM estimator and access to the effects.

    Attributes
    ----------
    params, bse, zvalues, pvalues : Series
        Coefficients of the outcome (``theta``) and mediator (``gamma``)
        dictionaries with sandwich standard errors, Eq. (A.15).
    cov : DataFrame
        Estimated covariance matrix of the coefficients.
    n_moments : int
        Number of stacked moment conditions.
    j_stat, j_df, j_pvalue : float
        Hansen J test of the over-identifying restrictions (two-step GMM only).
    """

    def __init__(self, model: PanelMediation, out: Dict, cfg: Dict, seed):
        self.model, self.cfg, self.seed = model, cfg, seed
        nm = model.param_names
        self._beta, self._phi = out["beta"], out["phi"]
        self.params = pd.Series(out["beta"], index=nm, name="coef")
        self.cov = pd.DataFrame(out["cov"], index=nm, columns=nm)
        self.bse = pd.Series(np.sqrt(np.diag(out["cov"])), index=nm, name="se")
        self.zvalues = (self.params / self.bse).rename("z")
        self.pvalues = pd.Series(2 * stats.norm.sf(np.abs(self.zvalues)), index=nm, name="p")
        self.n_moments, self.n_moments_y = out["n_moments"], out["n_moments_y"]
        self.jacobian_cond = out["cond"]
        self.j_stat, self.j_df = out["j_stat"], out["j_df"]
        self.j_pvalue = (float(stats.chi2.sf(self.j_stat, self.j_df))
                         if np.isfinite(self.j_stat) and self.j_df > 0 else np.nan)
        self.psi = out["psi"]
        self.sY = len(model.tau_y)
        self._cf = Counterfactual(model.design, model.tau_y, model.tau_m, model.x_names)
        self._boot = None

    # --------------------------------------------------------------- tables
    def conf_int(self, level: float = 0.95) -> pd.DataFrame:
        """Wald confidence intervals of the structural coefficients."""
        c = stats.norm.ppf(0.5 + level / 2)
        return pd.DataFrame({"lower": self.params - c * self.bse,
                             "upper": self.params + c * self.bse})

    def to_frame(self, level: float = 0.95) -> pd.DataFrame:
        """Tidy frame: estimate, se, z, p, ci_low, ci_high."""
        return inference_frame(self._beta, self.bse.to_numpy(), level, self.params.index)

    def _info(self):
        N, T = self.model.nobs
        c, m = self.cfg, self.model
        info = [("Outcome", m.names["y"]), ("Mediator", m.names["m"]),
                ("Treatment", m.names["d"]), ("Covariates", str(len(m.x_names))),
                ("Basis", m.basis_label),
                ("Estimator", "DML-GMM" if c["method"] == "dml" else "Naive-GMM"),
                ("Units (N)", f"{N}"), ("Periods (T)", f"{T}"),
                ("Folds (K)", f"{c['n_folds']}" if c["method"] == "dml" else "-"),
                ("Instrument lags", "all" if c["max_lags"] is None else f"{c['max_lags']}"),
                ("Moments", f"{self.n_moments}"),
                ("Weighting", "two-step" if c["weighting"] == "optimal" else "identity")]
        return info

    def summary(self, level: float = 0.95) -> Table:
        """Journal-style table of the structural coefficients."""
        fr = self.to_frame(level)
        sY = self.sY
        notes = ["Sandwich standard errors from the stacked individual scores (Eq. A.15); "
                 "N -> infinity, T fixed.",
                 f"Moments: {self.n_moments_y} outcome + {self.n_moments - self.n_moments_y} "
                 f"mediator; Jacobian condition number {self.jacobian_cond:.1f}."]
        if np.isfinite(self.j_stat):
            notes.append(f"Hansen J = {self.j_stat:.3f}, df = {self.j_df}, "
                         f"p = {self.j_pvalue:.3f} (H0: moment conditions valid).")
        return Table(f"Panel mediation, structural parameters  ({PAPER})", self._info(), fr,
                     notes, [("Outcome equation (theta)", list(fr.index[:sY])),
                             ("Mediator equation (gamma)", list(fr.index[sY:]))],
                     level, "Parameter")

    def to_latex(self, **kw) -> str:
        """LaTeX (booktabs) table of the structural coefficients."""
        return self.summary().as_latex(**kw)

    def to_markdown(self) -> str:
        """Markdown table of the structural coefficients."""
        return self.summary().as_markdown()

    # -------------------------------------------------------------- effects
    def _periods(self, period):
        T = self.model.nobs[1]
        if isinstance(period, str):
            if period == "last":
                return [T - 1], False
            if period == "all":
                return list(range(T)), False
            if period == "average":
                return list(range(T)), True
            raise ValueError("`period` must be 'last', 'all', 'average', an int or a list.")
        per = [period] if np.isscalar(period) else list(period)
        if any((not float(p).is_integer()) or p < 1 or p > T for p in per):
            raise ValueError(f"Periods are numbered 1..{T}.")
        return [int(p) - 1 for p in per], False

    def _build(self, kinds, raw_fn, periods, average, level, se, title, hist_note):
        """Shared machinery of ``effects`` and ``derivatives``."""
        nk = len(kinds)
        tlab = [str(self.model.times[t]) for t in periods]

        def ell_fn(beta):
            a = raw_fn(beta)                                  # (N, P, 4)
            return a.mean(axis=1) if average else a.reshape(a.shape[0], -1)

        est, infl = effect_inference(ell_fn, self._beta, self._phi)
        groups = [""] if average else tlab
        labels, kind_col, per_col = [], [], []
        for g in groups:
            for k in kinds:
                labels.append(k if (average or len(groups) == 1) else f"{k}[{g}]")
                kind_col.append(k)
                per_col.append("average" if average else g)
        # mediated share = indirect / total (ratio; delta method on the influence functions)
        ind, tot = kinds[2], kinds[3]
        shares, sh_infl, sh_lab = [], [], []
        for gi, g in enumerate(groups):
            i_ind, i_tot = gi * nk + 2, gi * nk + 3
            r = est[i_ind] / est[i_tot] if est[i_tot] != 0 else np.nan
            shares.append(r)
            sh_infl.append((infl[:, i_ind] - r * infl[:, i_tot]) / est[i_tot])
            nm = f"{ind}/{tot}"
            sh_lab.append(nm if (average or len(groups) == 1) else f"{nm}[{g}]")
            kind_col.append(nm)
            per_col.append("average" if average else g)
        est_all = np.concatenate([est, shares])
        infl_all = np.column_stack([infl] + sh_infl)
        labels_all = labels + sh_lab
        N = infl_all.shape[0]
        cov = infl_all.T @ infl_all / N / N                  # Eq. (21), divided by N
        draws = None
        if se == "bootstrap":
            if self._boot is None:
                raise ValueError("Run `res.bootstrap(...)` before asking for bootstrap SEs.")
            ids, betas = self._boot["ids"], self._boot["betas"]
            dr = np.empty((len(betas), len(est)))
            for b in range(len(betas)):
                dr[b] = ell_fn(betas[b])[ids[b]].mean(axis=0)
            sh = np.column_stack([dr[:, gi * nk + 2] / dr[:, gi * nk + 3]
                                  for gi in range(len(groups))])
            draws = np.column_stack([dr, sh])
            se_vec = draws.std(axis=0, ddof=1)
        elif se == "analytic":
            se_vec = np.sqrt(np.diag(cov))
        else:
            raise ValueError("`se` must be 'analytic' or 'bootstrap'.")
        is_share = np.array(["/" in k for k in kind_col])
        frame = inference_frame(est_all, se_vec, level, labels_all, null=~is_share)
        frame["kind"], frame["period"] = kind_col, per_col
        return EffectsResult(self, frame, cov, infl_all, draws, level, se, title, hist_note,
                             kinds, groups)

    def effects(self, d=1.0, d_star=0.0, m=0.0, period="last", level: float = 0.95,
                se: str = "analytic") -> "EffectsResult":
        """History-specific level effects: CDE, NDE, NIE, TE (Definition 1).

        Parameters
        ----------
        d, d_star : scalar, length-T vector or (N, T) array
            Treatment histories compared (``d`` versus ``d_star``). A scalar
            means that value in every period.
        m : scalar, length-T vector or (N, T) array
            Mediator history held fixed for the controlled direct effect.
        period : "last", "all", "average", int or list of int
            Period(s) t at which the effects are evaluated (1-based).
            ``"all"`` gives one row per period, ``"average"`` the mean over
            periods.
        level : float
            Confidence level.
        se : "analytic" or "bootstrap"
            Influence-function standard errors, Eq. (21), or the cross-sectional
            bootstrap (run ``bootstrap`` first).

        Returns
        -------
        EffectsResult
        """
        N, T = self.model.nobs
        dh, ds, mh = (as_history(v, N, T, nm) for v, nm in
                      ((d, "d"), (d_star, "d_star"), (m, "m")))
        periods, avg = self._periods(period)
        sY = self.sY

        def raw(beta):
            return self._cf.level(beta, dh, ds, mh, sY)[:, periods, :]

        note = f"Treatment history d versus d*; mediator fixed at m for the CDE."
        return self._build(LEVEL_KINDS, raw, periods, avg, level, se,
                           "Panel mediation, level effects", note)

    def derivatives(self, d=1.0, m=0.0, period="last", wrt: Optional[int] = None,
                    level: float = 0.95, se: str = "analytic") -> "EffectsResult":
        """History-specific derivative effects: CDD, NDD, NID, TD (Definition 2).

        The derivative is taken with respect to the treatment of period
        ``wrt`` (1-based); by default the effect's own period, i.e. the
        contemporaneous derivative of Eq. (24).

        Parameters are as in :meth:`effects`; ``d`` is the history at which
        the derivative is evaluated.
        """
        N, T = self.model.nobs
        dh, mh = as_history(d, N, T, "d"), as_history(m, N, T, "m")
        periods, avg = self._periods(period)
        j = None if wrt is None else int(wrt) - 1
        if j is not None and (j < 0 or j > min(periods)):
            raise ValueError("`wrt` must be a period no later than the effect period.")
        sY = self.sY

        def raw(beta):
            return self._cf.derivative(beta, dh, mh, sY, periods, j)

        note = "Derivative with respect to the treatment of the effect's own period." \
            if wrt is None else f"Derivative with respect to the treatment of period {wrt}."
        return self._build(DERIV_KINDS, raw, periods, avg, level, se,
                           "Panel mediation, derivative effects", note)

    # ------------------------------------------------------------ bootstrap
    def bootstrap(self, n_boot: int = 199, seed: Optional[int] = None,
                  n_jobs: int = 1, verbose: bool = False) -> "PanelMediationResults":
        """Cross-sectional bootstrap (Kapetanios, 2008) of the whole procedure.

        Units are drawn with replacement with their complete trajectories, and
        every step (first stages, fold-specific transformations, GMM) is
        repeated. The draws are stored; ``effects(..., se="bootstrap")`` then
        uses them. The same ``seed`` gives the same resamples for two models
        on the same data, so their estimates can be compared replication by
        replication with :func:`compare_effects`.

        References
        ----------
        Kapetanios, G. (2008). A bootstrap procedure for panel data sets with
        many cross-sectional units. Econometrics Journal, 11(2), 377-395.
        <https://doi.org/10.1111/j.1368-423X.2008.00243.x>
        """
        from joblib import Parallel, delayed
        N = self.model.nobs[0]
        ss = np.random.SeedSequence(seed)
        rng = np.random.default_rng(ss.spawn(1)[0])
        ids = rng.integers(0, N, size=(n_boot, N))
        child = ss.spawn(n_boot + 1)[1:]                      # independent stream per draw
        P = self.model.design
        res = Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0)(
            delayed(_boot_one)(P, ids[b], self.cfg, child[b]) for b in range(n_boot))
        ok = [b for b, r in enumerate(res) if r is not None]
        if len(ok) < 2:
            raise RuntimeError("The bootstrap failed in (almost) every replication.")
        betas = np.array([res[b] for b in ok])
        self._boot = dict(ids=ids[ok], betas=betas, n_requested=n_boot, seed=seed)
        self.bse_boot = pd.Series(betas.std(axis=0, ddof=1), index=self.params.index,
                                  name="se_boot")
        return self

    # ---------------------------------------------------------------- plots
    def plot_coefficients(self, **kw):
        """Dot-and-whisker plot of the structural coefficients."""
        from ._plots import plot_coefficients
        return plot_coefficients(self, **kw)

    def plot_moments(self, **kw):
        """Diagnostic plot: standardised sample moments at the estimate."""
        from ._plots import plot_moments
        return plot_moments(self, **kw)

    def plot_dose_response(self, grid=None, **kw):
        """Level effects of treatment history ``d`` versus 0 across a grid of ``d``."""
        from ._plots import plot_dose_response
        return plot_dose_response(self, grid, **kw)

    def plot_surface(self, grid=None, **kw):
        """Heatmap of an effect over the (d, d*) grid, Parula colours."""
        from ._plots import plot_surface
        return plot_surface(self, grid, **kw)


class EffectsResult:
    """Causal effects with their joint covariance matrix.

    Attributes
    ----------
    frame : DataFrame
        One row per effect: estimate, se, z, p, ci_low, ci_high, kind, period.
    cov : DataFrame
        Joint covariance from the influence functions (Theorem 2). It may be
        singular because the total effect is the sum of two other rows.
    influence : ndarray (N, n_effects)
        Estimated influence function of each effect.
    draws : ndarray or None
        Bootstrap draws, when ``se="bootstrap"``.
    """

    def __init__(self, parent, frame, cov, influence, draws, level, se, title, hist_note,
                 kinds, groups):
        self.parent, self.frame, self.influence, self.draws = parent, frame, influence, draws
        self.cov = pd.DataFrame(cov, index=frame.index, columns=frame.index)
        self.level, self.se_type, self.title, self.note = level, se, title, hist_note
        self.kinds, self.groups = kinds, groups

    @property
    def estimates(self) -> pd.Series:
        return self.frame["estimate"]

    @property
    def se(self) -> pd.Series:
        return self.frame["se"]

    def to_frame(self) -> pd.DataFrame:
        """Tidy frame of the effects."""
        return self.frame.copy()

    def conf_int(self) -> pd.DataFrame:
        """Wald confidence intervals at the level chosen when the effects were built."""
        return self.frame[["ci_low", "ci_high"]].rename(
            columns={"ci_low": "lower", "ci_high": "upper"})

    def summary(self) -> Table:
        """Journal-style table of the effects and the mediated share."""
        p = self.parent
        fr = self.frame
        share = fr.index[fr["kind"].str.contains("/")]
        main = fr.index[~fr["kind"].str.contains("/")]
        if self.se_type == "bootstrap":
            b = p._boot
            src = (f"Standard errors: cross-sectional bootstrap, {len(b['betas'])} of "
                   f"{b['n_requested']} replications; Wald intervals.")
        else:
            src = "Standard errors: estimated influence functions (Eq. 21, Theorem 3)."
        notes = [src, self.note,
                 "Total = natural direct + natural indirect. No test against zero is "
                 "reported for the mediated share."]
        return Table(f"{self.title}  ({PAPER})", p._info(), fr, notes,
                     [("Effects", list(main)), ("Mediated share", list(share))],
                     self.level, "Estimand")

    def to_latex(self, **kw) -> str:
        """LaTeX (booktabs) table of the effects."""
        return self.summary().as_latex(**kw)

    def to_markdown(self) -> str:
        """Markdown table of the effects."""
        return self.summary().as_markdown()

    def wald(self, rows: Sequence[str], value: Union[float, Sequence[float]] = 0.0) -> Dict:
        """Joint Wald test that the selected effects equal ``value``.

        Uses a generalised inverse, so effects linked by the exact
        decomposition (a singular covariance) are handled; the degrees of
        freedom are the rank of the selected covariance block (Section 4).

        Returns
        -------
        dict with ``stat``, ``df`` and ``pvalue``.
        """
        rows = [rows] if isinstance(rows, str) else list(rows)
        diff = self.frame.loc[rows, "estimate"].to_numpy() - np.asarray(value, dtype=float)
        if self.draws is not None:
            idx = [self.frame.index.get_loc(r) for r in rows]
            V = np.atleast_2d(np.cov(self.draws[:, idx], rowvar=False))
        else:
            V = self.cov.loc[rows, rows].to_numpy()
        sv = linalg.svdvals(V)
        df = int((sv > sv[0] * 1e-9).sum())
        stat = float(diff @ linalg.pinvh(V, rtol=1e-9) @ diff)
        return {"stat": stat, "df": df, "pvalue": float(stats.chi2.sf(stat, df))}

    def plot(self, **kw):
        """Dot-and-whisker plot of the decomposition (direct, indirect, total)."""
        from ._plots import plot_effects
        return plot_effects(self, **kw)

    def plot_profile(self, **kw):
        """Period-by-period profile with confidence bands (needs ``period='all'``)."""
        from ._plots import plot_profile
        return plot_profile(self, **kw)

    def __repr__(self) -> str:
        return self.summary().as_text()


def compare_effects(a: EffectsResult, b: EffectsResult, level: float = 0.95) -> Table:
    """Paired difference ``b - a`` between two specifications on the same panel.

    The two sets of effects are estimated on the same units, so the standard
    error of the difference uses the difference of the unit-level influence
    functions. When both were bootstrapped with the same ``seed`` and
    ``n_boot``, the within-replication differences are used instead, as in
    Panel C of Table 2 of the paper.
    """
    if list(a.frame.index) != list(b.frame.index):
        raise ValueError("The two EffectsResult objects must contain the same effects.")
    if a.influence.shape[0] != b.influence.shape[0]:
        raise ValueError("The two specifications must be estimated on the same units.")
    diff = (b.frame["estimate"] - a.frame["estimate"]).to_numpy()
    N = a.influence.shape[0]
    same_boot = (a.draws is not None and b.draws is not None
                 and a.draws.shape == b.draws.shape
                 and np.array_equal(a.parent._boot["ids"], b.parent._boot["ids"]))
    if same_boot:
        se = (b.draws - a.draws).std(axis=0, ddof=1)
        src = f"Standard errors: within-replication bootstrap differences ({len(a.draws)} draws)."
    else:
        se = np.sqrt(((b.influence - a.influence) ** 2).mean(axis=0) / N)
        src = "Standard errors: difference of the unit-level influence functions."
    fr = inference_frame(diff, se, level, a.frame.index)
    return Table(f"Paired difference between specifications (B - A)  ({PAPER})",
                 b.parent._info(), fr, [src], level=level, row_header="Estimand")


def panelmed(y, m, d, x=None, *, outcome, mediator, data=None, unit=None, time=None,
             n_folds: int = 5, max_lags: Optional[int] = 2, penalty="plugin",
             method: str = "dml", weighting: str = "identity", seed: Optional[int] = None,
             **model_kw) -> PanelMediationResults:
    """Fit the panel mediation model in one call.

    A thin wrapper around ``PanelMediation(...).fit(...)``; see those two for
    the meaning of every argument.

    Examples
    --------
    >>> from panelmed import panelmed, simulate_panel
    >>> df = simulate_panel(n=120, t=5, seed=1, as_frame=True)
    >>> res = panelmed("y", "m", "d", ["x1", "x2"], data=df, unit="unit", time="time",
    ...                outcome=["d", "m"], mediator=["d"], basis_degree=1,
    ...                n_folds=3, seed=1)
    >>> sorted(res.effects().frame["kind"])[:2]
    ['CDE', 'NDE']
    """
    model = PanelMediation(y, m, d, x, outcome=outcome, mediator=mediator, data=data,
                           unit=unit, time=time, **model_kw)
    return model.fit(n_folds=n_folds, max_lags=max_lags, penalty=penalty, method=method,
                     weighting=weighting, seed=seed)
