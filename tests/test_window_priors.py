import torch

from bayeshippo.window_priors import (
    continuous_log_uniform_complexity,
    make_window_prior_grid,
    mixture_age_density,
    posterior_window_weights,
)


def test_log_uniform_grid_normalizes():
    prior = make_window_prior_grid(1.0, 256.0, 16, kind="log_uniform")
    assert prior.theta.shape == (16,)
    assert torch.all(prior.theta > 0)
    assert torch.allclose(prior.weights.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(prior.weights, torch.ones(16) / 16, atol=1e-6)


def test_posterior_window_weights_favor_low_loss():
    prior = make_window_prior_grid(1.0, 8.0, 4, kind="log_uniform")
    losses = torch.tensor([2.0, 1.0, 0.0, 3.0])
    w = posterior_window_weights(losses, prior.weights, temperature=0.1)
    assert int(torch.argmax(w)) == 2
    assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-6)


def test_mixture_age_density_nonnegative_and_recent_biased():
    prior = make_window_prior_grid(1.0, 100.0, 64, kind="log_uniform")
    ages = torch.tensor([0.5, 5.0, 50.0])
    q = mixture_age_density(ages, prior.theta, prior.weights)
    assert torch.all(q >= 0)
    assert q[0] > q[1] > q[2]


def test_log_uniform_complexity_scale_free():
    c1 = continuous_log_uniform_complexity(1.0, 1024.0, 8.0, 0.25)
    c2 = continuous_log_uniform_complexity(1.0, 1024.0, 128.0, 0.25)
    assert abs(c1 - c2) < 1e-5
