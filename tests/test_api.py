"""Input contracts, tables, figures, colours."""
import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import panelmed as pm  # noqa: E402
from panelmed import PanelMediation, panelmed, simulate_panel  # noqa: E402


def test_dataframe_and_arrays_give_identical_numbers():
    df = simulate_panel(n=150, t=5, seed=2, as_frame=True)
    s = simulate_panel(n=150, t=5, seed=2)
    kw = dict(outcome=["d", "L1_d", "m"], mediator=["d"], basis_degree=2)
    a = panelmed("y", "m", "d", ["x1", "x2"], data=df.sample(frac=1, random_state=0),
                 unit="unit", time="time", n_folds=3, seed=4, **kw)
    b = panelmed(s["y"], s["m"], s["d"], s["x"][:, :, :2], n_folds=3, seed=4, **kw)
    np.testing.assert_allclose(a.params.to_numpy(), b.params.to_numpy(), rtol=1e-12)
    assert list(a.params.index) == ["theta[d]", "theta[L1_d]", "theta[m]", "gamma[d]"]


def test_input_errors_name_the_problem():
    s = simulate_panel(n=60, t=5, seed=2)
    y = s["y"].copy()
    y[0, 0] = np.nan
    with pytest.raises(ValueError, match="`y` contains NaN"):
        PanelMediation(y, s["m"], s["d"], outcome=["d", "m"], mediator=["d"])
    with pytest.raises(ValueError, match="at least T = 3"):
        PanelMediation(s["y"][:, :2], s["m"][:, :2], s["d"][:, :2], outcome=["d", "m"],
                       mediator=["d"])
    with pytest.raises(ValueError, match="cannot contain the mediator"):
        PanelMediation(s["y"], s["m"], s["d"], outcome=["d", "m"], mediator=["d", "m"])
    with pytest.raises(ValueError, match="no indirect path"):
        PanelMediation(s["y"], s["m"], s["d"], outcome=["d"], mediator=["d"])
    df = simulate_panel(n=60, t=5, seed=2, as_frame=True).iloc[:-1]
    with pytest.raises(ValueError, match="unbalanced"):
        PanelMediation("y", "m", "d", data=df, unit="unit", time="time",
                       outcome=["d", "m"], mediator=["d"])
    with pytest.raises(ValueError, match="at least 3"):
        PanelMediation(s["y"], s["m"], s["d"], outcome=["d", "m"],
                       mediator=["d"]).fit(n_folds=2)


def test_tables_render_in_every_format(res):
    for obj in (res, res.effects(period="average")):
        txt = str(obj.summary())
        assert "Std.Err." in txt and "Significance" in txt
        assert "\\toprule" in obj.to_latex() and "\\hline" not in obj.to_latex()
        assert obj.to_markdown().count("|") > 20
        assert "<table" in obj.summary().as_html()
    fr = res.to_frame()
    assert list(fr.columns) == ["estimate", "se", "z", "p", "ci_low", "ci_high"]
    assert res.conf_int().shape == (9, 2)


def test_period_options(res):
    assert len(res.effects(period="all").frame) == 6 * 5       # 4 effects + share, per period
    assert len(res.effects(period=[2, 3]).frame) == 2 * 5
    avg = res.derivatives(period="average").frame
    assert list(avg.index) == ["CDD", "NDD", "NID", "TD", "NID/TD"]
    with pytest.raises(ValueError, match="numbered"):
        res.effects(period=99)


def test_figures_return_fig_and_ax(res):
    import matplotlib.pyplot as plt
    calls = (res.plot_coefficients, res.plot_moments, res.effects().plot,
             res.effects(period="all").plot_profile,
             lambda: res.plot_dose_response(grid=np.linspace(-1, 1, 4)),
             lambda: res.plot_surface(grid=np.linspace(-1, 1, 4)))
    for call in calls:
        fig, ax = call()
        assert fig is ax.figure
        plt.close(fig)


def test_county_panel_loads_and_is_balanced():
    raw = pm.load_us_counties(prepared=False)
    assert raw.shape == (61160, 15) and raw["year"].min() == 2000
    df = pm.load_us_counties()
    assert df["county_id"].nunique() == 3058 and df["year"].nunique() == 19
    assert not df.isna().any().any()
    np.testing.assert_allclose(df[["y", "m", "d"]].std().to_numpy(), 1.0, rtol=1e-10)


def test_colour_maps():
    assert pm.parula_colors(64)[0] == "#3E26A8" and len(pm.parula_colors(10)) == 10
    scale = pm.resolve_colorscale("Parula", 8)
    assert scale[0][0] == 0.0 and scale[-1][0] == 1.0 and len(scale) == 8
    for f in (pm.matlab_jet_colors, pm.turbo_colors, pm.bluered_colors, pm.sinha_colors):
        assert len(f(5)) == 5
    with pytest.raises(ValueError):
        pm.resolve_colorscale("nope")
