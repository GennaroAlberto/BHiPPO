import torch

from bayeshippo.bounds import (
    gaussian_kl_diag,
    gaussian_kl_full,
    linear_readout_variance,
    lipschitz_readout_loss_gap_bound,
    pac_bayes_pinsker_bound,
)
from bayeshippo.kalman import KalmanState, LinearGaussianFilter


def test_gaussian_kl_diag_zero_for_matching_gaussians():
    m = torch.zeros(3)
    v = torch.ones(3)
    kl = gaussian_kl_diag(m, v)
    assert torch.allclose(kl, torch.tensor(0.0))


def test_gaussian_kl_full_zero_for_matching_gaussians():
    m = torch.zeros(2)
    cov = torch.eye(2)
    kl = gaussian_kl_full(m, cov)
    assert torch.allclose(kl, torch.tensor(0.0), atol=1e-5)


def test_pac_bayes_bound_is_valid_range():
    bound = pac_bayes_pinsker_bound(empirical_loss=0.1, kl=2.0, n=100, delta=0.05)
    assert 0.1 <= float(bound) <= 1.0


def test_linear_readout_variance_diag_matches_manual():
    W = torch.tensor([[1.0, 2.0], [0.5, -1.0]])
    p = torch.tensor([0.25, 0.5])
    exact = linear_readout_variance(W, p)
    manual = (W.square() * p.unsqueeze(0)).sum()
    assert torch.allclose(exact, manual)


def test_lipschitz_gap_uses_trace_covariance():
    cov = torch.tensor([0.25, 0.75])
    gap = lipschitz_readout_loss_gap_bound(cov, readout_lipschitz=2.0, loss_lipschitz=3.0)
    assert torch.allclose(gap, torch.tensor(6.0))


def test_kalman_scalar_update_matches_closed_form_and_reduces_covariance():
    kf = LinearGaussianFilter(jitter=0.0)
    state = KalmanState(mean=torch.tensor([0.0]), cov=torch.tensor([[1.0]]))
    H = torch.tensor([[1.0]])
    R = torch.tensor([[1.0]])
    y = torch.tensor([2.0])
    out = kf.update(state, y, H, R)
    # K = 1/(1+1) = 1/2, mean = 1, cov = 1/2.
    assert torch.allclose(out.mean, torch.tensor([1.0]))
    assert torch.allclose(out.cov, torch.tensor([[0.5]]))
    assert out.cov.item() < state.cov.item()
