"""Numerical core: DML-GMM for the panel mediation model.

Implements Algorithm 1 and Sections 3-4 of

    Wu, P. and Xiao, Z. (2026). "Causal Mediation Analysis for Panel Data with
    Double/Debiased Machine Learning". arXiv:2610.21339.

Equation numbers in the comments refer to that paper. Column convention for
every first-differenced array ``dQ = Q[:, 1:] - Q[:, :-1]``: column ``c``
holds the difference dated ``t = c + 2`` (paper periods are 1-based).

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import linalg

from ._utils import Dictionary, make_env

LEVEL_KINDS = ("CDE", "NDE", "NIE", "TE")
DERIV_KINDS = ("CDD", "NDD", "NID", "TD")


class IdentificationError(ValueError):
    """Raised when the stacked moment Jacobian does not have full column rank."""


# --------------------------------------------------------------------------- #
# Prepared data
# --------------------------------------------------------------------------- #
@dataclass
class Design:
    """Level arrays shared by estimation and by the counterfactual maps."""
    Y: np.ndarray            # (N, T) outcome
    M: np.ndarray            # (N, T) mediator
    D: np.ndarray            # (N, T) treatment
    X: np.ndarray            # (N, T, LX) covariates
    TY: np.ndarray           # (N, T, sY) outcome dictionary at observed histories
    TM: np.ndarray           # (N, T, sM) mediator dictionary at observed histories
    B: np.ndarray            # (N, T, p) covariate basis for f0, g0, h0
    endo_y: np.ndarray       # terms of tau_Y depending on current d or m
    endo_m: np.ndarray       # terms of tau_M depending on current d
    int_y: List[int] = field(default_factory=list)   # covariates interacted, outcome IV
    int_m: List[int] = field(default_factory=list)   # covariates interacted, mediator IV
    sq_y: bool = False       # squared treatment residual in the outcome IV
    sq_m: bool = False       # squared treatment residual in the mediator IV

    def take(self, idx: np.ndarray) -> "Design":
        """Design for a resample of units (complete trajectories)."""
        return Design(self.Y[idx], self.M[idx], self.D[idx], self.X[idx], self.TY[idx],
                      self.TM[idx], self.B[idx], self.endo_y, self.endo_m, self.int_y,
                      self.int_m, self.sq_y, self.sq_m)


def _diff(q: np.ndarray) -> np.ndarray:
    return q[:, 1:] - q[:, :-1]


def _project(z: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Least-squares projection of the columns of ``v`` on the span of ``z``.

    Uses the Gram matrix of the standardised instruments with a pseudo-inverse,
    so duplicated or collinear instrument columns are harmless.
    """
    if z.shape[1] == 0:
        return np.zeros_like(v)
    scale = np.sqrt((z ** 2).mean(axis=0))
    scale[scale < 1e-12] = 1.0
    zs = z / scale
    gram_inv = linalg.pinvh(zs.T @ zs, rtol=1e-10)
    return zs @ (gram_inv @ (zs.T @ v))


def _lasso(x: np.ndarray, y: np.ndarray, penalty) -> np.ndarray:
    """l1-penalised least squares without intercept on standardised columns.

    ``penalty="plugin"`` uses lambda = sqrt(2 log(2p) / n_obs) (Appendix D.1);
    a float fixes lambda; ``"cv"`` selects it by 5-fold cross-validation.
    The objective is (1/(2n))||y - xb||^2 + lambda * ||b||_1, as in 'glmnet'.
    """
    from sklearn.linear_model import Lasso, LassoCV
    n, p = x.shape
    sd = x.std(axis=0)
    sd[sd < 1e-12] = 1.0
    xs = x / sd
    if isinstance(penalty, str) and penalty == "cv":
        fit = LassoCV(cv=5, fit_intercept=False, max_iter=100000, tol=1e-6).fit(xs, y)
    else:
        lam = np.sqrt(2.0 * np.log(2.0 * p) / n) if penalty == "plugin" else float(penalty)
        if lam <= 0:
            coef, *_ = linalg.lstsq(xs, y)
            return coef / sd
        fit = Lasso(alpha=lam, fit_intercept=False, max_iter=100000, tol=1e-7,
                    precompute=True).fit(xs, y)
    return fit.coef_ / sd


def _first_stage(dy, dT, dB, t_lev, endo, penalty):
    """Penalised IV fit of one differenced structural equation on a training sample.

    Regresses the period-demeaned ``dy`` on the differenced dictionary and the
    differenced basis for t = 3..T. Terms flagged in ``endo`` are projected on
    the instrument span (long differences and lagged first differences of the
    same terms, plus the exogenous regressors) before the l1 fit.

    Returns the dictionary coefficients (used only as the preliminary copy of
    gamma) and the basis coefficients that define the nuisance fit.
    """
    n = dy.shape[0]
    s = dT.shape[2]

    def dm(a):  # training-sample cross-sectional mean, period by period
        return a - a.mean(axis=0, keepdims=True)

    y = dm(dy)[:, 1:].reshape(-1)                      # equations t = 3..T
    reg = dm(dT)[:, 1:, :].reshape(y.size, s)
    bas = dm(dB)[:, 1:, :].reshape(y.size, -1)
    x = np.concatenate([reg, bas], axis=1)
    if endo.any():
        te = t_lev[:, :, endo]
        long_d = te[:, 1:-1, :] - te[:, [0], :]        # tau_{t-1} - tau_1, t = 3..T
        lag_d = dT[:, :-1, :][:, :, endo]              # Delta tau_{t-1},   t = 3..T
        zex = dm(np.concatenate([long_d, lag_d], axis=2)).reshape(y.size, -1)
        z = np.concatenate([zex, reg[:, ~endo], bas], axis=1)
        x = x.copy()
        cols = np.flatnonzero(endo)
        x[:, cols] = _project(z, x[:, cols])
        y = _project(z, y)
    coef = _lasso(x, y, penalty)
    return coef[:s], coef[s:]


# --------------------------------------------------------------------------- #
# Algorithm 1
# --------------------------------------------------------------------------- #
def fit_once(P: Design, n_folds: int = 5, max_lags: Optional[int] = 2,
             penalty="plugin", method: str = "dml", weighting: str = "identity",
             rng: Optional[np.random.Generator] = None) -> Dict:
    """One pass of the DML-GMM estimator (or of the naive benchmark).

    Parameters
    ----------
    P : Design
    n_folds : int
        Number of folds K >= 3 (score-evaluation, auxiliary and training roles).
    max_lags : int or None
        Number of most recent residual dates kept in each instrument vector;
        ``None`` keeps the whole history.
    penalty : "plugin", "cv" or float
    method : "dml" or "naive"
    weighting : "identity" (one-step) or "optimal" (two-step)
    rng : numpy Generator used for the random fold assignment.

    Returns
    -------
    dict with ``beta``, ``cov``, the individual scores ``psi`` and the
    influence function ``phi`` of the structural estimator.
    """
    rng = np.random.default_rng() if rng is None else rng
    N, T = P.Y.shape
    sY, sM = P.TY.shape[2], P.TM.shape[2]
    dY, dM, dD = _diff(P.Y), _diff(P.M), _diff(P.D)
    dTY, dTM, dB = _diff(P.TY), _diff(P.TM), _diff(P.B)

    rY = np.empty((N, T - 1)); rM = np.empty((N, T - 1))
    gY = np.empty((N, T - 1, sY)); gM = np.empty((N, T - 1, sM))
    UD = np.empty((N, T - 1)); UM = np.empty((N, T - 1))
    gamma_prelim = []

    if method == "dml":
        if n_folds < 3:
            raise ValueError("`n_folds` must be at least 3 (Algorithm 1).")
        if N < 2 * n_folds:
            raise ValueError(f"Too few units (N = {N}) for {n_folds} folds.")
        folds = np.array_split(rng.permutation(N), n_folds)          # step 1
        for k in range(n_folds):
            ev = folds[k]
            aux = folds[(k + 1) % n_folds]                           # step 2(a): k'
            tr = np.concatenate([folds[j] for j in range(n_folds)
                                 if j not in (k, (k + 1) % n_folds)])
            # step 2(b): nuisances from the training sample only
            g_tilde, b_g = _first_stage(dM[tr], dTM[tr], dB[tr], P.TM[tr], P.endo_m, penalty)
            _, b_f = _first_stage(dY[tr], dTY[tr], dB[tr], P.TY[tr], P.endo_y, penalty)
            dBt = dB[tr] - dB[tr].mean(axis=0, keepdims=True)
            dDt = dD[tr] - dD[tr].mean(axis=0, keepdims=True)
            b_h, *_ = linalg.lstsq(dBt.reshape(-1, dB.shape[2]), dDt.reshape(-1))
            gamma_prelim.append(g_tilde)

            # step 2(c): operator D_k of Eq. (8), auxiliary-fold demeaning
            def star(a, ev=ev, aux=aux):
                return a[ev] - a[aux].mean(axis=0, keepdims=True)

            sB = star(dB)
            gY[ev] = star(dTY); gM[ev] = star(dTM)
            rY[ev] = star(dY) - sB @ b_f                 # Delta Y* - Delta f*
            rM[ev] = star(dM) - sB @ b_g                 # Delta M* - Delta g*
            UD[ev] = star(dD) - sB @ b_h                 # U^D,  step 3
            UM[ev] = rM[ev] - gM[ev] @ g_tilde           # U^M(gamma tilde), step 3
        inst_d, inst_m = UD, UM
    elif method == "naive":
        g_tilde, b_g = _first_stage(dM, dTM, dB, P.TM, P.endo_m, penalty)
        _, b_f = _first_stage(dY, dTY, dB, P.TY, P.endo_y, penalty)
        gamma_prelim.append(g_tilde)

        def dm(a):
            return a - a.mean(axis=0, keepdims=True)

        sB = dm(dB)
        gY[:] = dm(dTY); gM[:] = dm(dTM)
        rY[:] = dm(dY) - sB @ b_f
        rM[:] = dm(dM) - sB @ b_g
        inst_d = dm(P.D)[:, 1:]                          # demeaned levels dated 2..T
        inst_m = dm(P.M)[:, 1:]
    else:
        raise ValueError("`method` must be 'dml' or 'naive'.")

    sq = inst_d ** 2
    sq = (sq - sq.mean(axis=0)) / np.where(sq.std(axis=0) > 0, sq.std(axis=0), 1.0)

    # Instruments A^Y_it, A^M_it: lagged residuals dated p < t (step 3)
    blocks = []          # (equation, Z (N, kz), residual (N,), regressors (N, s))
    for c in range(1, T - 1):                            # equation period t = c + 2
        lags = list(range(c - 1, -1, -1))                # residual columns, newest first
        if max_lags is not None:
            lags = lags[:max_lags]
        zy, zm = [], []
        for l in lags:                                   # residual dated p = l + 2
            zy += [inst_d[:, l], inst_m[:, l]]
            zy += [inst_d[:, l] * P.X[:, l + 1, j] for j in P.int_y]
            if P.sq_y:
                zy.append(sq[:, l])
            zm.append(inst_d[:, l])
            zm += [inst_d[:, l] * P.X[:, l + 1, j] for j in P.int_m]
        if P.sq_m:
            zm.append(sq[:, c - 1])                      # most recent date only
        blocks.append(("y", np.column_stack(zy), rY[:, c], gY[:, c, :]))
        blocks.append(("m", np.column_stack(zm), rM[:, c], gM[:, c, :]))
    blocks.sort(key=lambda b: b[0] != "y")               # outcome moments first

    # Stacked affine score psi_i(beta) = a_i - G_i beta  (step 4)
    s = sY + sM
    a = np.concatenate([z * r[:, None] for _, z, r, _ in blocks], axis=1)
    q = a.shape[1]
    Gbar = np.zeros((q, s))
    pos = 0
    for eq, z, _, g in blocks:
        cols = slice(0, sY) if eq == "y" else slice(sY, s)
        Gbar[pos:pos + z.shape[1], cols] = z.T @ g / N
        pos += z.shape[1]
    abar = a.mean(axis=0)

    def psi_of(beta):
        parts = []
        for eq, z, r, g in blocks:
            b = beta[:sY] if eq == "y" else beta[sY:]
            parts.append(z * (r - g @ b)[:, None])
        return np.concatenate(parts, axis=1)

    sv = linalg.svdvals(Gbar)
    if q < s or sv[-1] < 1e-10 * sv[0]:
        raise IdentificationError(
            f"The moment Jacobian has rank below {s} ({q} moments). Add instrument "
            "lags (`max_lags`), interactions or use a smaller dictionary.")

    def solve(omega):
        gw = Gbar.T @ omega
        return linalg.solve(gw @ Gbar, gw @ abar, assume_a="pos")

    omega = np.eye(q)
    beta = solve(omega)                                  # Eq. (12), one-step
    if weighting == "optimal":
        p1 = psi_of(beta)
        omega = linalg.pinvh(p1.T @ p1 / N)
        beta = solve(omega)
    elif weighting != "identity":
        raise ValueError("`weighting` must be 'identity' or 'optimal'.")

    psi = psi_of(beta)
    Psi = psi.T @ psi / N
    G = -Gbar                                            # d psi / d beta'
    A = linalg.solve(G.T @ omega @ G, G.T @ omega, assume_a="pos")
    V = A @ Psi @ A.T                                    # Eq. (A.15)
    phi = -(psi - psi.mean(axis=0)) @ A.T                # influence function, Eq. (16)
    pbar = psi.mean(axis=0)
    j_stat = float(N * pbar @ linalg.pinvh(Psi) @ pbar) if weighting == "optimal" else np.nan
    return dict(beta=beta, cov=V / N, psi=psi, phi=phi, A=A, G=G, omega=omega,
                n_moments=q, n_moments_y=sum(z.shape[1] for e, z, _, _ in blocks if e == "y"),
                j_stat=j_stat, j_df=q - s, cond=float(sv[0] / sv[-1]),
                gamma_prelim=np.mean(gamma_prelim, axis=0))


# --------------------------------------------------------------------------- #
# Counterfactual maps and unit-level effect contributions (Sections 3.2 and 4)
# --------------------------------------------------------------------------- #
class Counterfactual:
    """Counterfactual mediator and outcome maps of Eqs. (13), (14) and (17)."""

    def __init__(self, P: Design, tau_y: Dictionary, tau_m: Dictionary,
                 x_names: Sequence[str]):
        self.P, self.tau_y, self.tau_m, self.x_names = P, tau_y, tau_m, list(x_names)
        self.L = max(tau_y.max_lag, tau_m.max_lag)

    def _tm(self, dh):
        return self.tau_m.evaluate(make_env(dh, None, self.P.X, self.x_names, self.L))

    def _ty(self, dh, mh):
        return self.tau_y.evaluate(make_env(dh, mh, self.P.X, self.x_names, self.L))

    def nested(self, a, b, theta, gamma):
        """Y^beta_it{a, M^beta(b)} - Y_it for every unit and period."""
        m_cf = self.P.M + (self._tm(b) - self.P.TM) @ gamma          # Eq. (17), mediator
        return (self._ty(a, m_cf) - self.P.TY) @ theta               # Eq. (17), outcome

    def fixed(self, a, mh, theta):
        """Y^beta_it(a, m) - Y_it for a prespecified mediator history."""
        return (self._ty(a, mh) - self.P.TY) @ theta

    def level(self, beta, d, d_star, m, sY):
        """Unit contributions l^CDE, l^NDE, l^NIE, l^TE; shape (N, T, 4)."""
        th, ga = beta[:sY], beta[sY:]
        y_dd = self.nested(d, d, th, ga)
        y_ds = self.nested(d, d_star, th, ga)
        y_ss = self.nested(d_star, d_star, th, ga)
        cde = self.fixed(d, m, th) - self.fixed(d_star, m, th)
        nde, nie = y_ds - y_ss, y_dd - y_ds
        return np.stack([cde, nde, nie, nde + nie], axis=2)

    def derivative(self, beta, d, m, sY, periods, wrt=None, h=1e-4):
        """Unit contributions l^CDD, l^NDD, l^NID, l^TD; shape (N, len(periods), 4).

        The derivative is taken with respect to the treatment of period
        ``wrt`` (default: the effect's own period) by central differences,
        which are exact for dictionaries that are quadratic in the treatment.
        """
        th, ga = beta[:sY], beta[sY:]
        N = d.shape[0]
        out = np.empty((N, len(periods), 4))
        for k, t in enumerate(periods):
            j = t if wrt is None else wrt
            up, dn = d.copy(), d.copy()
            up[:, j] += h
            dn[:, j] -= h
            cdd = (self.fixed(up, m, th) - self.fixed(dn, m, th))[:, t] / (2 * h)
            ndd = (self.nested(up, d, th, ga) - self.nested(dn, d, th, ga))[:, t] / (2 * h)
            nid = (self.nested(d, up, th, ga) - self.nested(d, dn, th, ga))[:, t] / (2 * h)
            out[:, k, :] = np.column_stack([cdd, ndd, nid, ndd + nid])
        return out


def effect_inference(ell_fn, beta: np.ndarray, phi_beta: np.ndarray):
    """Plug-in estimate and influence function of a vector of causal effects.

    ``ell_fn(beta)`` returns the (N, p) unit contributions. Implements
    Eqs. (18)-(21): phi_i = l_i - theta_hat + J * phi_beta_i, with J the
    Jacobian of the mean contribution with respect to beta.
    """
    ell = ell_fn(beta)
    est = ell.mean(axis=0)
    J = np.empty((est.size, beta.size))
    for k in range(beta.size):
        step = 1e-5 * max(1.0, abs(beta[k]))
        up, dn = beta.copy(), beta.copy()
        up[k] += step
        dn[k] -= step
        J[:, k] = (ell_fn(up).mean(axis=0) - ell_fn(dn).mean(axis=0)) / (2 * step)
    infl = ell - est + phi_beta @ J.T
    return est, infl
