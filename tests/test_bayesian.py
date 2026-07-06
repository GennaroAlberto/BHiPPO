import torch

from bayeshippo.bayesian import BayesianProjectionPosterior


def test_projection_evidence_mass_controls_variance():
    y = torch.tensor([1.0, -2.0])
    p1 = BayesianProjectionPosterior.from_projection(y, obs_var=2.0, prior_cov=1e6, evidence_mass=1.0)
    p10 = BayesianProjectionPosterior.from_projection(y, obs_var=2.0, prior_cov=1e6, evidence_mass=10.0)
    assert torch.allclose(p1.mean, y, atol=1e-5)
    assert torch.allclose(p10.mean, y, atol=1e-5)
    assert torch.all(p10.cov < p1.cov)


def test_weighted_sample_posterior_matches_linear_regression_limit():
    # y = 1 + 2 x in basis [1, x]
    x = torch.tensor([-1.0, 0.0, 1.0])
    phi = torch.stack([torch.ones_like(x), x], dim=-1)
    y = 1.0 + 2.0 * x
    post = BayesianProjectionPosterior.from_weighted_samples(
        phi, y, obs_var=1e-4, prior_cov=1e6, jitter=1e-8
    )
    assert torch.allclose(post.mean, torch.tensor([1.0, 2.0]), atol=1e-3)
    assert post.cov.shape == (2, 2)
