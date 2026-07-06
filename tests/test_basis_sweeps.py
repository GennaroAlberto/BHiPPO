import torch

from bayeshippo.basis import legendre_basis, reconstruct_legs
from bayeshippo.bayesian import BayesianProjectionPosterior
from bayeshippo.data import make_synthetic_series
from bayeshippo.hippo import hippo_scan
from bayeshippo.metrics import mse


def test_legendre_basis_shape_and_orthogonality_monte_carlo():
    u = torch.linspace(0, 1, 2001)
    phi = legendre_basis(u, 5)
    assert phi.shape == (2001, 5)
    gram = phi.T @ phi / phi.shape[0]
    assert torch.allclose(torch.diag(gram), torch.ones(5), atol=5e-3)
    off = gram - torch.diag(torch.diag(gram))
    assert off.abs().max() < 5e-3


def test_posterior_predictive_moments_diag():
    u = torch.linspace(0, 1, 25)
    phi = legendre_basis(u, 4)
    post = BayesianProjectionPosterior(mean=torch.ones(4), cov=0.1 * torch.ones(4))
    mean, var = post.predictive_moments(phi)
    assert mean.shape == (25,)
    assert var.shape == (25,)
    assert torch.all(var >= 0)


def test_synthetic_reconstruction_smoke():
    series = make_synthetic_series(128, kind="smooth", noise_std=0.1, seed=0)
    coeff = hippo_scan(series.observed, 8, "legs", return_sequence=False)
    rec = reconstruct_legs(coeff, series.times, final_time=1.0)
    val = mse(rec, series.clean)
    assert torch.isfinite(val)
