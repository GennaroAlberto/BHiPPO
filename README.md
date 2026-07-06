# BayesHiPPO

**BayesHiPPO** is a research package and LaTeX manuscript for the project:

> **Learning What to Remember: Bayesian Time-Relevance Measures for HiPPO Memory**

The key reframing is that HiPPO does not merely compute polynomial memory. It
**fixes a prior over which past times are relevant**. Deterministic HiPPO then
returns the flat-prior posterior mean of projection coefficients under that
fixed relevance law. Bayesian HiPPO makes the missing probabilistic structure
explicit:

- `mu_t`: a normalized **relevance distribution** over past times;
- `nu_t = alpha_t mu_t`: an unnormalized **information measure**;
- `c_t | data`: a posterior over polynomial memory coefficients;
- `mu_t | data`: optionally, a posterior over memory measures/timescales.

This matters because deterministic HiPPO is invariant to rescaling the measure:
it cannot distinguish one noisy observation from many consistent observations
with the same normalized relevance profile. Bayesian HiPPO can, via posterior
covariance and evidence mass.

## What is in this repository

```text
bayeshippo/
  hippo.py          # LegS/LegT/LagT matrices and recurrences
  basis.py          # shifted Legendre basis evaluation and reconstruction
  bayesian.py       # coefficient posterior, empirical Gram posterior, mixtures
  bounds.py         # PAC-Bayes/readout-stability utilities
  kalman.py         # linear-Gaussian Kalman updates for coefficient dynamics
  data.py           # synthetic signals for sweeps
  metrics.py        # MSE, NLL, coverage, ESS
  sweep.py          # grid/JSONL reproducibility helpers
experiments/
  sweep_noisy_reconstruction.py  # deterministic vs Bayesian shrinkage/calibration
  sweep_measure_mixture.py       # infer/average over relevance measures
  sweep_window_prior_copying.py  # log-uniform/window-length prior on copying-style memory
  plot_sweep.py                  # quick plotting from CSV outputs
paper/
  main.tex         # reframed proof-heavy manuscript
  refs.bib
tests/
```

## Install

```bash
cd bayeshippo_project
python -m pip install -e .[dev,experiments]
pytest -q
```

The current test suite passes locally:

```bash
PYTHONPATH=. pytest -q
# 11 passed
```

## First experiments

### 1. Noisy reconstruction sweep

This experiment tests the paper's first concrete imperfection claim:

deterministic HiPPO returns a point estimate and ignores evidence mass, while
Bayesian HiPPO can shrink noisy high-order coefficients and report uncertainty.

```bash
PYTHONPATH=. python experiments/sweep_noisy_reconstruction.py --quick --out runs/noisy_quick
PYTHONPATH=. python experiments/plot_sweep.py runs/noisy_quick/summary.csv
```

Full grid:

```bash
PYTHONPATH=. python experiments/sweep_noisy_reconstruction.py --out runs/noisy_full
```

Key outputs:

- `mse_det_clean`: deterministic HiPPO reconstruction error against the clean signal;
- `mse_bayes_clean`: Bayesian posterior-mean reconstruction error;
- `nll_bayes_noisy`: predictive Gaussian negative log likelihood;
- `coverage95_bayes_noisy`: empirical 95% interval coverage;
- `posterior_trace`: memory uncertainty.

### 2. Measure-mixture sweep

This experiment tests the main reframing:

deterministic HiPPO chooses a fixed relevance prior; Bayesian HiPPO can infer a
posterior over relevance priors.

```bash
PYTHONPATH=. python experiments/sweep_measure_mixture.py --quick --out runs/mixture_quick
PYTHONPATH=. python experiments/plot_sweep.py runs/mixture_quick/summary.csv
```

The component bank currently contains:

```text
LegS, LegT(theta=0.05), LegT(theta=0.10), LegT(theta=0.25),
LegT(theta=0.50), LegT(theta=1.00)
```

Key outputs:

- `best_component`: best fixed relevance measure on that trial;
- `weights`: posterior weights over the measure bank;
- `mix_loss`: mixture reconstruction loss;
- `legs_loss`: fixed LegS baseline;
- `ess`: effective number of active measures.


### 3. Window-prior Copying-style sweep

This experiment tests the newest hypothesis: instead of choosing the LegT window
length `theta`, use a weakly-informative prior over windows. A log-uniform prior
on `[theta_min, M]` approximates the scale-invariant improper prior
`dtheta/theta` after finite truncation. This is the right object for Copying-like
tasks where the gap length is unknown.

```bash
PYTHONPATH=. python experiments/sweep_window_prior_copying.py --quick --out runs/window_copy_quick
```

Important outputs:

- `posterior_mass_covering_required`: posterior mass on windows long enough to include the prefix;
- `posterior_log_mean_theta`: geometric posterior mean window length;
- `mix_loss` / `mix_acc`: mixture reconstruction performance;
- `short_theta_loss` and `long_theta_loss`: misspecified fixed-window baselines.

The unbounded flat prior over `theta in (0, infinity)` is intentionally not used: for finite sequences the posterior can be improper if losses flatten as `theta -> infinity`. The package uses proper truncations such as `theta in [1, cT]`, which approximate noninformativeness over the observable range.

## Paper thesis

The publishable thesis is not simply "HiPPO is Bayesian." The stronger claim is:

> HiPPO fixes a time-relevance prior. Bayesian HiPPO turns that prior into an
> inferable object and separates relevance from evidence.

The manuscript develops four pieces of theory:

1. **Bayesian decision theory:** HiPPO's measure is a prior over query time.
2. **Conjugate inference:** deterministic HiPPO is the flat-prior posterior mean.
3. **Imperfection theorem:** normalized deterministic projections discard evidence mass.
4. **Adaptive measures:** finite mixtures of HiPPO measures satisfy an exponential-weights regret bound.

## Next engineering steps

The package is now ready for sweep-style experiments. The next additions should be:

1. a Character Trajectories / timescale-shift loader;
2. forecasting readouts with NLL/CRPS/coverage metrics;
3. true online posterior weight updates instead of offline scoring;
4. a full neural Copying task model that learns the readout while using the window-prior memory bank;
4. baselines against deterministic HiPPO, LMU, S4/Mamba-style state-space layers, and uncertainty-aware SSM variants;
5. Hydra or YAML-based config management once the first sweeps stabilize.
