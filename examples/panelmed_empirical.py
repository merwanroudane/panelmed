"""Real-data example: employment, wages and income in U.S. counties, 2001-2019.

Follows the empirical application of Wu and Xiao (2026), Section 6: how much
of the effect of the workplace employment rate (treatment) on personal income
per capita (outcome) is transmitted by the average wage per job (mediator)?

Run it with:  python panelmed_empirical.py [bootstrap replications]

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

from panelmed import PanelMediation, compare_effects, load_us_counties  # noqa: E402
from panelmed.datasets import US_COUNTIES_COVARIATES as X  # noqa: E402

N_BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "assets")
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
    fig.savefig(os.path.join(OUT, name + ".pdf"))


# Step 1 - load the shipped county panel (already transformed as in the paper)
df = load_us_counties()
print(df.head(3).round(3).to_string(index=False))
print(f"\nCounties: {df['county_id'].nunique()}, years: {df['year'].min()}-{df['year'].max()}\n")

common = dict(data=df, unit="county_id", time="year", basis_degree=3, basis_max_vars=2)

# Step 2 - Specification I: homogeneous effects
spec1 = PanelMediation("y", "m", "d", X, outcome=["d", "m"], mediator=["d"], **common)
res1 = spec1.fit(n_folds=5, max_lags=2, seed=2026)
print(res1.summary())

# Step 3 - Specification II: effects vary with county characteristics
inter = [f"d*{c}" for c in X]
spec2 = PanelMediation("y", "m", "d", X, outcome=["d", "m"] + inter,
                       mediator=["d"] + inter, **common)
res2 = spec2.fit(n_folds=5, max_lags=2, seed=2026)
print()
print(res2.summary())

# Step 4 - average derivative effects over 2001-2019, Eq. (24)
se = "analytic"
if N_BOOT > 0:
    res1.bootstrap(n_boot=N_BOOT, seed=7, n_jobs=-2)
    res2.bootstrap(n_boot=N_BOOT, seed=7, n_jobs=-2)       # same seed: same county resamples
    se = "bootstrap"
eff1 = res1.derivatives(d=0, period="average", se=se)
eff2 = res2.derivatives(d=0, period="average", se=se)
print()
print(eff1.summary())
print()
print(eff2.summary())

# Step 5 - does the decomposition change between the two specifications?
print()
print(compare_effects(eff1, eff2))

# Step 6 - figures
save(res1.plot_coefficients()[0], "counties_coefficients")
save(eff2.plot()[0], "counties_decomposition")
save(res2.derivatives(d=0, period="all").plot_profile()[0], "counties_profile")
save(res2.plot_dose_response()[0], "counties_dose_response")
save(res2.plot_surface()[0], "counties_surface")
save(res1.plot_moments()[0], "counties_moments")
print("\nFigures written to docs/assets/")
