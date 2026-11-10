import warnings

import pytest

from panelmed import PanelMediation, simulate_panel
from panelmed.datasets import SIM_MEDIATOR, SIM_OUTCOME

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def sim():
    return simulate_panel(n=400, t=6, seed=7)


@pytest.fixture(scope="session")
def model(sim):
    return PanelMediation(sim["y"], sim["m"], sim["d"], sim["x"],
                          outcome=SIM_OUTCOME, mediator=SIM_MEDIATOR)


@pytest.fixture(scope="session")
def res(model):
    return model.fit(n_folds=5, seed=3)
