# Changelog

## [Unreleased]

### Fixed
- Figures use a constrained layout, so long axis labels are no longer clipped
  when a figure is saved without `bbox_inches="tight"`.
- Year labels on the period-profile figure no longer overlap.
- Long parameter names are printed in full in the text tables.

### Added
- Documentation site under `docs/`.

## [0.1.0] - 2026-11-09

### Added
- DML-GMM estimator of the panel mediation model with two-way fixed effects
  (Wu and Xiao, 2026, Algorithm 1): cross-fitting with an auxiliary fold,
  penalised IV first stages, lagged residual instruments, stacked GMM.
- Level effects (CDE, NDE, NIE, TE) and derivative effects (CDD, NDD, NID, TD)
  with influence-function standard errors and joint Wald tests.
- Cross-sectional bootstrap of the whole procedure.
- Naive-GMM benchmark, two-step GMM with the Hansen J test, mediated share with
  delta-method standard errors, paired comparison of specifications.
- Text, LaTeX, Markdown and HTML tables; coefficient, decomposition, profile,
  dose-response, surface and moment-diagnostic figures.
- Monte Carlo data generator of the paper.
