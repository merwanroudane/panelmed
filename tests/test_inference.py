"""Inference: reproducibility, bootstrap, Wald tests, two-step GMM."""
import numpy as np
import pytest

from panelmed import PanelMediation, compare_effects, simulate_panel


def test_same_seed_is_bit_identical(model):
    a, b = model.fit(n_folds=5, seed=11), model.fit(n_folds=5, seed=11)
    assert np.array_equal(a.params.to_numpy(), b.params.to_numpy())
    c = model.fit(n_folds=5, seed=12)
    assert not np.array_equal(a.params.to_numpy(), c.params.to_numpy())


def test_total_is_sum_and_wald_handles_singular_covariance(res):
    e = res.effects(d=1, d_star=0)
    f = e.frame["estimate"]
    np.testing.assert_allclose(f["TE"], f["NDE"] + f["NIE"], atol=1e-12)
    w3 = e.wald(["NDE", "NIE", "TE"])
    w2 = e.wald(["NDE", "NIE"])
    assert w3["df"] == 2                      # TE = NDE + NIE: rank 2, not 3
    np.testing.assert_allclose(w3["stat"], w2["stat"], rtol=1e-5)
    assert 0.0 <= w2["pvalue"] <= 1.0


def test_two_step_gmm_reports_hansen_j(model):
    r = model.fit(n_folds=5, weighting="optimal", seed=3)
    assert np.isfinite(r.j_stat) and r.j_df == r.n_moments - 9
    assert 0.0 <= r.j_pvalue <= 1.0


def test_naive_benchmark_runs_and_differs(model, res):
    r = model.fit(method="naive")
    assert r.n_moments == res.n_moments
    assert not np.allclose(r.params.to_numpy(), res.params.to_numpy())


def _small():
    s = simulate_panel(n=150, t=5, seed=5)
    return PanelMediation(s["y"], s["m"], s["d"], s["x"][:, :, :2], outcome=["d", "m"],
                          mediator=["d"], basis_degree=2)


def test_bootstrap_is_reproducible_and_independent_of_n_jobs():
    mod = _small()
    a = mod.fit(n_folds=3, seed=1).bootstrap(n_boot=6, seed=9, n_jobs=1)
    b = mod.fit(n_folds=3, seed=1).bootstrap(n_boot=6, seed=9, n_jobs=2)
    assert np.array_equal(a._boot["betas"], b._boot["betas"])
    e = a.effects(se="bootstrap")
    assert e.draws.shape == (6, 5) and (e.frame["se"] > 0).all()
    with pytest.raises(ValueError, match="bootstrap"):
        mod.fit(n_folds=3, seed=1).effects(se="bootstrap")


def test_compare_effects_paired_difference():
    s = simulate_panel(n=150, t=5, seed=5)
    x = s["x"][:, :, :2]
    a = PanelMediation(s["y"], s["m"], s["d"], x, outcome=["d", "m"], mediator=["d"],
                       basis_degree=2).fit(n_folds=3, seed=1)
    b = PanelMediation(s["y"], s["m"], s["d"], x, outcome=["d", "m", "d*x1"],
                       mediator=["d", "d*x1"], basis_degree=2).fit(n_folds=3, seed=1)
    tab = compare_effects(a.derivatives(period="average"), b.derivatives(period="average"))
    assert list(tab.frame.index) == ["CDD", "NDD", "NID", "TD", "NID/TD"]
    assert (tab.frame["se"] > 0).all()
