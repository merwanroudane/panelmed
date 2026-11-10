"""Monte Carlo check against Table 1 of Wu and Xiao (2026).

Reproduces the design of Section 5.1 / Appendix D.1 (N = 1600, T = 10, ten
folds, two instrument lags, identity weighting) for the DML-GMM estimator and
the Naive-GMM benchmark, and prints mean, bias, SD, RMSE, average estimated
standard error (ASE) and coverage of the nominal 95% Wald interval (CP).

Usage:
    python monte_carlo_table1.py [replications] [n_jobs] [design] [N]

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from panelmed import PanelMediation, simulate_panel
from panelmed.datasets import (SIM_GAMMA, SIM_MEDIATOR, SIM_OUTCOME, SIM_THETA,
                               SIM_TRUE_EFFECTS)

EFFECTS = ["CDE", "NDE", "NIE", "TE", "NDD", "NID", "TD"]
TRUTH = np.concatenate([SIM_THETA, SIM_GAMMA, [SIM_TRUE_EFFECTS[k] for k in EFFECTS]])


def one_replication(seed, n, design):
    warnings.filterwarnings("ignore")
    sim = simulate_panel(n=n, t=10, design=design, seed=seed)
    model = PanelMediation(sim["y"], sim["m"], sim["d"], sim["x"],
                           outcome=SIM_OUTCOME, mediator=SIM_MEDIATOR)
    out = {}
    for method in ("dml", "naive"):
        res = model.fit(n_folds=10, max_lags=2, method=method, seed=seed)
        lev = res.effects(d=1, d_star=0, m=0, period="last").frame
        der = res.derivatives(d=1, m=0, period="last").frame
        est = np.concatenate([res.params.to_numpy(), lev.loc[EFFECTS[:4], "estimate"],
                              der.loc[EFFECTS[4:], "estimate"]])
        se = np.concatenate([res.bse.to_numpy(), lev.loc[EFFECTS[:4], "se"],
                             der.loc[EFFECTS[4:], "se"]])
        out[method] = (est, se)
    return out


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else -2
    design = sys.argv[3] if len(sys.argv) > 3 else "baseline"
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 1600
    t0 = time.time()
    runs = Parallel(n_jobs=n_jobs)(delayed(one_replication)(1000 + r, n, design)
                                   for r in range(reps))
    names = ([f"theta[{k}]" for k in SIM_OUTCOME] + [f"gamma[{k}]" for k in SIM_MEDIATOR]
             + EFFECTS)
    rows = []
    for j, nm in enumerate(names):
        for method, label in (("dml", "DML-GMM"), ("naive", "Naive-GMM")):
            est = np.array([r[method][0][j] for r in runs])
            se = np.array([r[method][1][j] for r in runs])
            cover = np.abs(est - TRUTH[j]) <= 1.959964 * se
            rows.append(dict(Estimand=nm, Method=label, Mean=est.mean(),
                             Bias=est.mean() - TRUTH[j], SD=est.std(ddof=1),
                             RMSE=np.sqrt(((est - TRUTH[j]) ** 2).mean()), ASE=se.mean(),
                             CP=cover.mean()))
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(f"Design: {design}, N = {n}, T = 10, {reps} replications, "
          f"{time.time() - t0:.0f} s")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return table


if __name__ == "__main__":
    main()
