"""Complete walkthrough on the Monte Carlo panel of Wu and Xiao (2026).

Every table and figure shown in the README is produced by this script.
Run it with:  python panelmed_walkthrough.py

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
import os
import warnings

import matplotlib

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

from panelmed import PanelMediation, compare_effects, simulate_panel  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "assets")
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
    fig.savefig(os.path.join(OUT, name + ".pdf"))


# Step 1 - the data: a long panel, one row per unit and period
df = simulate_panel(n=1600, t=10, seed=2026, as_frame=True)
print(df.head().round(3).to_string(index=False))
print(f"\nUnits: {df['unit'].nunique()}, periods: {df['time'].nunique()}\n")

# Step 2 - the model: outcome and mediator dictionaries
model = PanelMediation(
    "y", "m", "d", ["x1", "x2", "x3", "x4", "x5"],
    data=df, unit="unit", time="time",
    outcome=["d", "d**2", "L1_d", "d*x3", "m", "L1_m"],
    mediator=["d", "d**2", "L1_d"],
)

# Step 3 - estimation (Algorithm 1)
res = model.fit(n_folds=10, max_lags=2, seed=2026)
print(res.summary())

# Step 4 - level effects of treatment history 1 versus 0, last period
eff = res.effects(d=1, d_star=0, m=0, period="last")
print()
print(eff.summary())
print("\nJoint Wald test, H0: NDE = NIE = 0 :", eff.wald(["NDE", "NIE"]))

# Step 5 - derivative effects, averaged over periods as in Eq. (24)
der = res.derivatives(d=1, m=0, period="average")
print()
print(der.summary())

# Step 6 - the naive benchmark, for comparison
naive = model.fit(method="naive")
print()
print(naive.effects(d=1, d_star=0, m=0).summary())

# Step 7 - bootstrap standard errors (the whole procedure is repeated)
res.bootstrap(n_boot=99, seed=1, n_jobs=-2)
print()
print(res.effects(d=1, d_star=0, m=0, se="bootstrap").summary())

# Step 8 - a second specification and the paired difference
model_b = PanelMediation(
    "y", "m", "d", ["x1", "x2", "x3", "x4", "x5"],
    data=df, unit="unit", time="time",
    outcome=["d", "L1_d", "m", "L1_m"], mediator=["d", "L1_d"],
)
res_b = model_b.fit(n_folds=10, seed=2026)
print()
print(compare_effects(eff, res_b.effects(d=1, d_star=0, m=0)))

# Step 9 - the figures
save(res.plot_coefficients()[0], "coefficients")
save(eff.plot()[0], "decomposition")
save(res.derivatives(d=1, period="all").plot_profile()[0], "profile")
save(res.plot_dose_response()[0], "dose_response")
save(res.plot_surface()[0], "surface")
save(res.plot_moments()[0], "moments")
print("\nFigures written to docs/assets/")
