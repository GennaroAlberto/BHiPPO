"""BayesHiPPO: Bayesian time-relevance measures for polynomial memory."""

from .hippo import HiPPOMeasure, HiPPOMatrices, HiPPOLayer, hippo_scan, make_hippo_matrices
from .bayesian import BayesianHiPPOFilter, BayesianProjectionPosterior, MeasureMixture
from .basis import legendre_basis, reconstruct_legs, reconstruct_legt
from .window_priors import WindowPriorGrid, make_window_prior_grid, posterior_window_weights

__all__ = [
    "HiPPOMeasure",
    "HiPPOMatrices",
    "HiPPOLayer",
    "hippo_scan",
    "make_hippo_matrices",
    "BayesianHiPPOFilter",
    "BayesianProjectionPosterior",
    "MeasureMixture",
    "legendre_basis",
    "reconstruct_legs",
    "reconstruct_legt",
    "WindowPriorGrid",
    "make_window_prior_grid",
    "posterior_window_weights",
]
