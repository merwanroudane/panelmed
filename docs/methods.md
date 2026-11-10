# Methods: how the code maps to the paper

**panelmed** implements Wu, P. and Xiao, Z. (2026), "Causal Mediation Analysis for
Panel Data with Double/Debiased Machine Learning", arXiv:2610.21339
(<https://doi.org/10.48550/arXiv.2610.21339>). Equation, algorithm and appendix
numbers below refer to that paper.

Author of the package: Dr Merwan Roudane (merwanroudane920@gmail.com).

## 1. Model

For unit `i = 1..N` and period `t = 1..T` (balanced panel, `N` large, `T` fixed):

```
Y_it = tau_Y(D_i,t-L:t, M_i,t-L:t, X_i,t-L:t)' theta0 + f0(X) + alpha^Y_i + lambda^Y_t + eps^Y_it
M_it = tau_M(D_i,t-L:t, X_i,t-L:t)'           gamma0 + g0(X) + alpha^M_i + lambda^M_t + eps^M_it
D_it = h0(X) + alpha^D_i + lambda^D_t + eps^D_it
```

`tau_Y` and `tau_M` are dictionaries chosen by the researcher; `f0`, `g0`, `h0` are
unknown functions; values dated before period 1 are zero.

## 2. Step -> equation map

| Code (`src/panelmed/`) | Paper | What it computes |
|---|---|---|
| `_utils.Dictionary`, `make_env` | Section 2.3 | `tau_Y`, `tau_M` at observed or counterfactual histories; pre-sample values set to 0 |
| `_utils.polynomial_basis` | App. D.1, E.1 | monomial library for `f0`, `g0`, `h0` (55 terms for 5 covariates at degree 3; 108 for 8 covariates with at most 2 distinct variables) |
| `_core._diff` | Eq. (7) | first differences, removing the unit effects |
| `_core.fit_once`, fold loop | Algorithm 1, steps 1-2(a) | random split into `K` folds; evaluation fold `k`, auxiliary fold `k' = k+1` (cyclic), training sample = the rest |
| `_core._first_stage` | Algorithm 1 step 2(b); App. D.1 | l1-penalised IV fit of the differenced, period-demeaned equation for `t = 3..T`; gives `f0_hat`, `g0_hat` and the preliminary `gamma_tilde` |
| `_core._project` | App. D.1 | projection of the response and of the endogenous regressors on the instrument span |
| `_core._lasso` | App. D.1 | `lambda = sqrt(2 log(2p) / n_obs)`, standardised regressors, no intercept, every coefficient penalised |
| least squares for `h0` in `fit_once` | App. D.1 | `Delta D` on the differenced basis, `t = 2..T`, training-sample demeaning |
| `star(...)` in `fit_once` | Eq. (8) | operator `D_k`: first difference minus the auxiliary-fold period mean |
| `rY`, `rM`, `UD`, `UM` | Algorithm 1 step 3 | `U^Y`, `U^M`, `U^D` and `U^M(gamma_tilde)` |
| instrument loop in `fit_once` | Algorithm 1 step 3; App. D.1 | `A^Y_it`, `A^M_it` from lagged residuals dated `p < t`, truncated at `max_lags`, with the interaction and squared-residual augmentations |
| `a`, `Gbar`, `solve` | Eq. (12) | stacked affine score and the closed-form GMM estimator |
| `V = A Psi A'` | Eq. (A.15) | sandwich covariance of the coefficients |
| `phi` | Eq. (16) | influence function `-A (psi_i - psi_bar)` |
| `Counterfactual.nested`, `.fixed` | Eqs. (13), (14), (17) | `M^beta_it(d)` and `Y^beta_it(d, m)` |
| `Counterfactual.level` | Definition 1, Section 4 | unit contributions to CDE, NDE, NIE, TE |
| `Counterfactual.derivative` | Definition 2, Section 4 | unit contributions to CDD, NDD, NID, TD |
| `_core.effect_inference` | Eqs. (18)-(21) | plug-in effects and their influence functions |
| `EffectsResult.wald` | Section 4 (after Theorem 3) | Wald test with a generalised inverse for singular covariance |
| `PanelMediationResults.bootstrap` | Section 4 (end); App. E.1 | cross-sectional bootstrap repeating the whole procedure |
| `fit(method="naive")` | Section 5.1; App. D.1 | Naive-GMM benchmark |
| `datasets.simulate_panel` | App. D.1, D.3 | Monte Carlo data-generating processes (A.16), (A.17) |

## 3. Implementation choices and where they come from

**First-stage instruments.** Appendix D.1 lists, for the simulation, the long
differences `tau_j,t-1 - tau_j,1` of the contemporaneous dictionary terms and the
lagged first differences of two of them. The package applies one rule to any
dictionary: for every term that depends on the current treatment or mediator, the
long difference and the lagged first difference enter the instrument set, together
with the remaining (predetermined) dictionary terms and the basis. For the paper's
dictionaries this spans exactly the instrument space described in Appendix D.1,
because the lagged first differences of `d` and `m` coincide with the lagged terms
already included.

**Final-stage instruments.** Per retained residual date `p`: the outcome equation
uses `U^D_p`, `U^M_p`, `U^D_p * X_p,j` for each covariate `j` that interacts with the
treatment in `tau_Y`, and the squared treatment residual when `tau_Y` is nonlinear
in the treatment. The mediator equation uses `U^D_p`, `U^D_p * X_p,j`, and one
squared residual (most recent date) when `tau_M` is nonlinear. The squared residual
is centred and standardised over the whole sample. With these rules the moment
counts of the paper are reproduced: 60 + 23 = 83 in the simulation, 99 and 627 in
the two empirical specifications. `iv_interact` and `iv_square` override the
automatic choice.

**Projection.** The paper's code uses a QR basis of the instrument matrix; the
package uses the pseudo-inverse of the Gram matrix of the standardised instruments.
Both give the least-squares projection on the same span.

**Lasso.** 'glmnet' with `alpha = 1` is replaced by the coordinate-descent lasso of
'scikit-learn', which minimises the same objective
`(1/(2n)) ||y - Xb||^2 + lambda ||b||_1` on regressors scaled to unit standard
deviation.

**Derivatives.** Derivative effects are computed by central differences in the
treatment history (step `1e-4`), exact for dictionaries that are quadratic in the
treatment. The Jacobian of the mean contribution with respect to the coefficients
is also computed by central differences; the contributions are polynomial in the
coefficients, so the result is exact up to rounding.

**Variance.** Coefficient standard errors use the uncentred score covariance of
Eq. (A.15). Effect standard errors use the centred influence function of Eq. (21).

## 4. Additions that are not in the paper

| Addition | Reference |
|---|---|
| Two-step efficient GMM and the J test of over-identifying restrictions (`weighting="optimal"`) | Hansen (1982), <https://doi.org/10.2307/1912775> |
| Analytical standard error of the mediated share (indirect / total) by the delta method on the influence functions | follows from Theorem 2 of the paper |
| Paired comparison of two specifications from unit-level influence functions (`compare_effects`) | follows from Theorem 2; the bootstrap version is Panel C of Table 2 |
| Condition number of the moment Jacobian, and an explicit error when the rank condition fails | rank condition of Section 3.1 |
| Standardised-moment diagnostic figure | - |

## 5. What is not implemented

- Vector-valued mediators (Appendix A.1) and growing mediator dimension (Appendix A.2).
- Unbalanced panels; the paper treats balanced panels only.
- First-stage learners other than the penalised linear-in-basis fit. The paper's
  theory covers general learners under rate conditions, but under sequential
  exogeneity it only specifies penalised IV/GMM first stages.
- The empirical application of Section 6: the county panel is not distributed with
  the paper.

## 6. Validation

`examples/monte_carlo_table1.py` reruns the design of Table 1. No reference
implementation of the mediation estimator is public, so agreement is checked
against the paper's reported moments counts, true effects (Eq. 23) and Monte Carlo
table rather than against other code.
