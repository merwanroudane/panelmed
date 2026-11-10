"""Input handling: panel reshaping, structural dictionaries, covariate basis.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

import itertools
import re
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

Term = Union[str, Callable[[Dict[str, np.ndarray]], np.ndarray]]

_SAFE = {k: getattr(np, k) for k in (
    "exp", "log", "sin", "cos", "tan", "tanh", "sinh", "cosh", "sqrt", "abs",
    "arctan", "arcsinh", "maximum", "minimum", "where", "sign", "power", "pi")}
_LAG = re.compile(r"\bL(\d+)_")
_RESERVED = {"d", "m"} | set(_SAFE)


# --------------------------------------------------------------------------- #
# Structural dictionaries tau_Y and tau_M (Section 2.3 of the paper)
# --------------------------------------------------------------------------- #
class Dictionary:
    """A researcher-specified dictionary of structural terms.

    Each term is either a string expression or a callable. Expressions are
    written with the names ``d`` (treatment), ``m`` (mediator), the covariate
    names, and the lag prefix ``L1_``, ``L2_``, ... (``L1_d`` is the treatment
    one period earlier). Values dated before period 1 are zero, as in the paper.

    Parameters
    ----------
    terms : sequence of str or callable
        For example ``["d", "d**2", "L1_d", "d*x3", "m", "L1_m"]``. A callable
        receives a dict mapping names to ``(N, T)`` arrays and returns an
        ``(N, T)`` array.
    names : sequence of str, optional
        Labels used in tables. Defaults to the expressions themselves.
    lags : int, optional
        Largest lag used. Inferred from string terms; pass it explicitly when
        a callable uses lags.

    Examples
    --------
    >>> tau = Dictionary(["d", "d**2", "L1_d"])
    >>> len(tau), tau.max_lag
    (3, 1)
    """

    def __init__(self, terms: Sequence[Term], names: Optional[Sequence[str]] = None,
                 lags: Optional[int] = None):
        if isinstance(terms, Dictionary):
            names = names or terms.names
            lags = lags if lags is not None else terms.max_lag
            terms = terms.terms
        if isinstance(terms, str) or callable(terms):
            terms = [terms]
        self.terms: List[Term] = list(terms)
        if len(self.terms) == 0:
            raise ValueError("A dictionary needs at least one term.")
        self._code = []
        auto, found = [], 0
        for j, t in enumerate(self.terms):
            if isinstance(t, str):
                expr = t.replace("^", "**").strip()
                found = max([found] + [int(v) for v in _LAG.findall(expr)])
                try:
                    self._code.append(compile(expr, f"<term {t!r}>", "eval"))
                except SyntaxError as e:
                    raise ValueError(f"Dictionary term {t!r} is not a valid expression.") from e
                auto.append(t.replace(" ", ""))
            elif callable(t):
                self._code.append(None)
                auto.append(getattr(t, "__name__", f"term{j + 1}"))
            else:
                raise TypeError("Dictionary terms must be strings or callables.")
        self.names = list(names) if names is not None else auto
        if len(self.names) != len(self.terms):
            raise ValueError("`names` must have one label per dictionary term.")
        self.max_lag = int(found if lags is None else max(found, lags))

    def __len__(self) -> int:
        return len(self.terms)

    def __repr__(self) -> str:
        return f"Dictionary({self.names})"

    def evaluate(self, env: Dict[str, np.ndarray]) -> np.ndarray:
        """Evaluate every term; returns an array with a trailing term axis."""
        shape = np.shape(env["d"])
        cols = []
        for t, code in zip(self.terms, self._code):
            if code is None:
                v = t(env)
            else:
                try:
                    v = eval(code, {"__builtins__": {}}, {**_SAFE, **env})
                except NameError as e:
                    raise ValueError(
                        f"Dictionary term {t!r} uses an unknown name ({e}). Available "
                        f"names: {sorted(k for k in env)}. Note that the mediator "
                        "dictionary cannot contain the mediator `m`.") from e
            cols.append(np.broadcast_to(np.asarray(v, dtype=float), shape))
        return np.stack(cols, axis=-1)


def lag_array(q: np.ndarray, k: int) -> np.ndarray:
    """Shift an ``(N, T)`` array ``k`` periods back; pre-sample values are 0."""
    if k == 0:
        return q
    out = np.zeros_like(q)
    if k < q.shape[1]:
        out[:, k:] = q[:, :-k]   # column t holds the value dated t-k
    return out


def make_env(d: np.ndarray, m: Optional[np.ndarray], x: np.ndarray,
             x_names: Sequence[str], max_lag: int) -> Dict[str, np.ndarray]:
    """Name -> ``(N, T)`` array map used to evaluate dictionaries."""
    env: Dict[str, np.ndarray] = {"d": d}
    if m is not None:
        env["m"] = m
    for j, nm in enumerate(x_names):
        env[nm] = x[:, :, j]
    for k in range(1, max_lag + 1):
        env[f"L{k}_d"] = lag_array(d, k)
        if m is not None:
            env[f"L{k}_m"] = lag_array(m, k)
        for j, nm in enumerate(x_names):
            env[f"L{k}_{nm}"] = lag_array(x[:, :, j], k)
    return env


def term_depends_on(dic: Dictionary, env: Dict[str, np.ndarray], keys: Sequence[str],
                    rng: np.random.Generator) -> np.ndarray:
    """Boolean vector: does each term change when ``env[keys]`` are perturbed?"""
    base = dic.evaluate(env)
    env2 = dict(env)
    hit = False
    for k in keys:
        if k in env2:
            env2[k] = env2[k] + rng.normal(size=np.shape(env2[k]))
            hit = True
    if not hit:
        return np.zeros(len(dic), dtype=bool)
    alt = dic.evaluate(env2)
    scale = 1.0 + np.abs(base).max(axis=(0, 1))
    return np.abs(alt - base).max(axis=(0, 1)) > 1e-10 * scale


def term_nonlinear_in_d(dic: Dictionary, env: Dict[str, np.ndarray]) -> bool:
    """True when some term is nonlinear in the contemporaneous treatment."""
    e_up, e_dn = dict(env), dict(env)
    e_up["d"] = env["d"] + 1.0
    e_dn["d"] = env["d"] - 1.0
    base = dic.evaluate(env)
    second = dic.evaluate(e_up) - 2.0 * base + dic.evaluate(e_dn)
    return bool(np.abs(second).max() > 1e-8 * (1.0 + np.abs(base).max()))


# --------------------------------------------------------------------------- #
# Covariate basis for the nuisance functions f0, g0, h0
# --------------------------------------------------------------------------- #
def polynomial_exponents(n_vars: int, degree: int,
                         max_vars: Optional[int] = None) -> np.ndarray:
    """Exponent matrix of all monomials of total degree 1..``degree``.

    ``max_vars`` keeps only monomials involving at most that many distinct
    variables (the application in the paper uses ``max_vars=2``).

    Examples
    --------
    >>> polynomial_exponents(5, 3).shape[0]
    55
    >>> polynomial_exponents(8, 3, max_vars=2).shape[0]
    108
    """
    rows = []
    for deg in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(n_vars), deg):
            e = np.bincount(combo, minlength=n_vars)
            if max_vars is None or int((e > 0).sum()) <= max_vars:
                rows.append(e)
    return np.array(rows, dtype=int).reshape(-1, n_vars)


def polynomial_basis(x: np.ndarray, degree: int = 3, max_vars: Optional[int] = None,
                     lags: int = 0) -> np.ndarray:
    """Monomial basis of the covariates, shape ``(N, T, p)``, no intercept."""
    if lags > 0:
        x = np.concatenate(
            [x] + [np.stack([lag_array(x[:, :, j], k) for j in range(x.shape[2])], axis=2)
                   for k in range(1, lags + 1)], axis=2)
    expo = polynomial_exponents(x.shape[2], degree, max_vars)
    out = np.ones(x.shape[:2] + (expo.shape[0],))
    for j in range(x.shape[2]):
        col = x[:, :, j][:, :, None]
        for power in range(1, degree + 1):
            mask = expo[:, j] == power
            if mask.any():
                out[:, :, mask] *= col ** power
    return out


# --------------------------------------------------------------------------- #
# Panel input
# --------------------------------------------------------------------------- #
def _check_name(nm: str) -> str:
    if not str(nm).isidentifier() or nm in _RESERVED or _LAG.match(nm + "_"):
        raise ValueError(
            f"Covariate name {nm!r} cannot be used in dictionary expressions. Rename "
            "the column to a plain identifier (letters, digits, underscore) other "
            "than 'd' and 'm'.")
    return str(nm)


def to_panel(y, m, d, x=None, data: Optional[pd.DataFrame] = None, unit=None, time=None,
             x_names: Optional[Sequence[str]] = None) -> dict:
    """Return balanced ``(N, T)`` arrays from wide arrays or a long DataFrame."""
    if data is not None:
        if unit is None or time is None:
            raise ValueError("With `data`, pass the `unit` and `time` column names.")
        xcols = [] if x is None else ([x] if isinstance(x, str) else list(x))
        cols = [y, m, d] + xcols
        missing = [c for c in [unit, time] + cols if c not in data.columns]
        if missing:
            raise ValueError(f"Columns not found in `data`: {missing}.")
        df = data[[unit, time] + cols].sort_values([unit, time])
        if df[cols].isna().any().any():
            bad = [c for c in cols if df[c].isna().any()]
            raise ValueError(f"Missing values in {bad}; drop or impute them first.")
        if df.duplicated([unit, time]).any():
            raise ValueError("`data` has duplicated (unit, time) rows.")
        counts = df.groupby(unit, sort=False)[time].size()
        n, t = counts.shape[0], df[time].nunique()
        if not (counts == t).all():
            raise ValueError(
                "The panel is unbalanced: the method requires every unit to be observed "
                f"in every period ({int((counts != t).sum())} of {n} units are incomplete).")
        arr = df[cols].to_numpy(dtype=float).reshape(n, t, len(cols))
        out = dict(Y=arr[:, :, 0], M=arr[:, :, 1], D=arr[:, :, 2], X=arr[:, :, 3:],
                   units=np.asarray(counts.index), times=np.sort(df[time].unique()),
                   names=dict(y=str(y), m=str(m), d=str(d)))
        x_names = [str(c) for c in xcols]
    else:
        Y, M, D = (np.asarray(a, dtype=float) for a in (y, m, d))
        if Y.ndim != 2 or M.shape != Y.shape or D.shape != Y.shape:
            raise ValueError("`y`, `m` and `d` must be (N, T) arrays of the same shape.")
        n, t = Y.shape
        if x is None:
            X = np.zeros((n, t, 0))
        else:
            X = np.asarray(x, dtype=float)
            if X.ndim == 2:
                X = X[:, :, None]
            if X.shape[:2] != (n, t):
                raise ValueError("`x` must have shape (N, T) or (N, T, K).")
        for nm, a in (("y", Y), ("m", M), ("d", D), ("x", X)):
            if not np.isfinite(a).all():
                raise ValueError(f"`{nm}` contains NaN or infinite values.")
        if x_names is None:
            x_names = [f"x{j + 1}" for j in range(X.shape[2])]
        out = dict(Y=Y, M=M, D=D, X=X, units=np.arange(n), times=np.arange(1, t + 1),
                   names=dict(y="y", m="m", d="d"))
    if len(x_names) != out["X"].shape[2]:
        raise ValueError("`x_names` must have one name per covariate.")
    out["x_names"] = [_check_name(nm) for nm in x_names]
    n, t = out["Y"].shape
    if t < 3:
        raise ValueError(f"The method needs at least T = 3 periods; got T = {t}.")
    return out


def as_history(v, n: int, t: int, name: str) -> np.ndarray:
    """Broadcast a scalar, a length-T vector or an (N, T) array to (N, T)."""
    a = np.asarray(v, dtype=float)
    if a.ndim == 0:
        return np.full((n, t), float(a))
    if a.shape == (t,):
        return np.broadcast_to(a, (n, t)).copy()
    if a.shape == (n, t):
        return a.copy()
    raise ValueError(f"`{name}` must be a scalar, a length-{t} vector or an ({n}, {t}) array.")
