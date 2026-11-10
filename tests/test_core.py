"""Numerical core: paper targets, closed forms and recovery of the known truth."""
import numpy as np
import pytest

from panelmed import PanelMediation, polynomial_basis, simulate_panel
from panelmed._utils import polynomial_exponents
from panelmed.datasets import SIM_GAMMA, SIM_MEDIATOR, SIM_OUTCOME, SIM_THETA


def test_basis_dimensions_match_the_paper():
    # 55 monomials (simulation, App. D.1) and 108 (application, App. E.1)
    assert polynomial_exponents(5, 3).shape[0] == 55
    assert polynomial_exponents(8, 3, max_vars=2).shape[0] == 108
    x = np.random.default_rng(0).normal(size=(4, 3, 2))
    b = polynomial_basis(x, degree=2)
    np.testing.assert_allclose(b[:, :, 2], x[:, :, 0] ** 2, rtol=1e-12)


def test_moment_counts_match_appendix_d1():
    # T = 10, two lags: 60 outcome + 23 mediator moments for 9 parameters
    s = simulate_panel(n=300, t=10, seed=0)
    r = PanelMediation(s["y"], s["m"], s["d"], s["x"], outcome=SIM_OUTCOME,
                       mediator=SIM_MEDIATOR).fit(n_folds=5, seed=0)
    assert (r.n_moments_y, r.n_moments - r.n_moments_y) == (60, 23)


def test_moment_counts_match_appendix_e1():
    # L = 0, T = 19, 8 covariates: 99 moments (Spec. I) and 627 (Spec. II)
    rng = np.random.default_rng(1)
    n, t, k = 240, 19, 8
    x = rng.normal(size=(n, t, k))
    a = rng.normal(size=(n, 1))
    d = x[:, :, 0] + a + rng.normal(size=(n, t))
    m = 0.5 * d + a + rng.normal(size=(n, t))
    y = d + m + a + rng.normal(size=(n, t))
    names = [f"x{j + 1}" for j in range(k)]
    kw = dict(basis_degree=1)
    r1 = PanelMediation(y, m, d, x, outcome=["d", "m"], mediator=["d"], **kw).fit(seed=0)
    assert r1.n_moments == 99
    r2 = PanelMediation(y, m, d, x, outcome=["d", "m"] + [f"d*{c}" for c in names],
                        mediator=["d"] + [f"d*{c}" for c in names], **kw).fit(seed=0)
    assert r2.n_moments == 627 and len(r2.params) == 19


def test_effects_equal_closed_forms(res, sim):
    """App. D.1 closed forms hold to 1e-8 at the estimated coefficients."""
    th, ga = res.params.to_numpy()[:6], res.params.to_numpy()[6:]
    xbar = sim["x"][:, -1, 2].mean()
    lev = res.effects(d=1, d_star=0, m=0).frame["estimate"]
    der = res.derivatives(d=1, m=0).frame["estimate"]
    nde = th[0] + th[1] + th[2] + th[3] * xbar
    nie = ga.sum() * (th[4] + th[5])
    np.testing.assert_allclose(lev["CDE"], nde, atol=1e-8)
    np.testing.assert_allclose(lev["NDE"], nde, atol=1e-8)
    np.testing.assert_allclose(lev["NIE"], nie, atol=1e-8)
    np.testing.assert_allclose(lev["TE"], nde + nie, atol=1e-8)
    np.testing.assert_allclose(der["NDD"], th[0] + 2 * th[1] + th[3] * xbar, atol=1e-6)
    np.testing.assert_allclose(der["NID"], (ga[0] + 2 * ga[1]) * th[4], atol=1e-6)
    np.testing.assert_allclose(der["TD"], der["NDD"] + der["NID"], atol=1e-10)


def test_effect_se_equals_delta_method_when_no_unit_variation(res):
    """NID = (g1 + 2 g2) * th5 has no unit-level variation: its SE is the delta method."""
    b = res.params.to_numpy()
    grad = np.zeros(9)
    grad[4] = b[6] + 2 * b[7]
    grad[6], grad[7] = b[4], 2 * b[4]
    n = res.psi.shape[0]
    centred = res._phi.T @ res._phi / n / n     # Eq. (21) centres the score; (A.15) does not
    se = res.derivatives(d=1).frame.loc["NID", "se"]
    np.testing.assert_allclose(se, np.sqrt(grad @ centred @ grad), rtol=1e-5)
    assert abs(se / np.sqrt(grad @ res.cov.to_numpy() @ grad) - 1) < 0.05


def test_identical_histories_give_zero_effects(res):
    e = res.effects(d=0.7, d_star=0.7).frame["estimate"]
    np.testing.assert_allclose(e[["CDE", "NDE", "NIE", "TE"]], 0.0, atol=1e-10)


@pytest.mark.slow
def test_recovers_truth_and_error_shrinks_with_n():
    """Known-truth DGP: max abs error below 0.35 at N=500, smaller on average at N=2000."""
    truth = np.concatenate([SIM_THETA, SIM_GAMMA])
    err = []
    for n in (500, 2000):
        e = []
        for seed in range(4):
            s = simulate_panel(n=n, t=8, seed=100 + seed)
            r = PanelMediation(s["y"], s["m"], s["d"], s["x"], outcome=SIM_OUTCOME,
                               mediator=SIM_MEDIATOR).fit(n_folds=5, seed=seed)
            e.append(np.abs(r.params.to_numpy() - truth))
        err.append(np.mean(e, axis=0))
    assert err[0].max() < 0.35
    assert err[1].mean() < err[0].mean()
