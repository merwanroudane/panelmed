"""panelmed: causal mediation analysis for panel data with double/debiased machine learning.

Implements Wu, P. and Xiao, Z. (2026), "Causal Mediation Analysis for Panel
Data with Double/Debiased Machine Learning", arXiv:2610.21339
<https://doi.org/10.48550/arXiv.2610.21339>.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
GitHub: https://github.com/merwanroudane
"""
from importlib.metadata import PackageNotFoundError, version as _v

from ._colors import (bluered_colors, get_cmap, matlab_jet_colors, parula_cmap,
                      parula_colors, resolve_colorscale, sinha_colors, turbo_colors)
from ._core import IdentificationError
from ._results import (EffectsResult, PanelMediation, PanelMediationResults,
                       compare_effects, panelmed)
from ._utils import Dictionary, polynomial_basis
from .datasets import load_us_counties, simulate_panel

__all__ = [
    "PanelMediation", "PanelMediationResults", "EffectsResult", "panelmed",
    "compare_effects", "Dictionary", "polynomial_basis", "simulate_panel",
    "load_us_counties",
    "IdentificationError", "parula_colors", "parula_cmap", "matlab_jet_colors",
    "turbo_colors", "bluered_colors", "sinha_colors", "get_cmap", "resolve_colorscale",
]

try:
    __version__ = _v("panelmed")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.1.0"
__author__ = "Dr Merwan Roudane"
__email__ = "merwanroudane920@gmail.com"
