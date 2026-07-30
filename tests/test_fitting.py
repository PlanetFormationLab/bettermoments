"""
Tests for the analytical fitting paths (curve_fit and MCMC starting points).
"""

import numpy as np
import pytest

from bettermoments import profiles as pr
from bettermoments import mcmc_sampling as ms


X = np.linspace(-4000.0, 4000.0, 120)
TRUTH = {'gaussian': [500.0, 400.0, 2.0],
         'gaussian_cont': [500.0, 400.0, 2.0, 0.2],
         'gaussthick': [500.0, 400.0, 2.0, 1.0],
         'gausshermite': [500.0, 400.0, 2.0, 0.05, 0.05],
         'doublegauss': [-500.0, 400.0, 2.0, 1500.0, 400.0, 1.0]}


def make_spectrum(model, seed=0):
    rng = np.random.default_rng(seed)
    return getattr(pr, model)(X, *TRUTH[model]) + rng.normal(0, 0.02, X.size)


@pytest.mark.parametrize('model', sorted(TRUTH))
def test_estimate_p0_lengths(model):
    y = make_spectrum(model)
    assert len(ms.estimate_p0(X, y, model)) == pr.free_params(model)


@pytest.mark.parametrize('model', ['gaussian_cont', 'gaussthick_cont',
                                   'doublegauss_cont', 'gausshermite_cont'])
def test_estimate_p0_and_priors_cont_variants(model):
    """The _cont variants must get full-length p0 and priors (C5, C6)."""
    y = make_spectrum(model[:-5])
    assert len(ms.estimate_p0(X, y, model)) == pr.free_params(model)
    assert len(ms.default_priors(X, y, model)) == pr.free_params(model)


def test_default_priors_doublegauss_full_length():
    y = make_spectrum('doublegauss')
    priors = ms.default_priors(X, y, 'doublegauss')
    assert len(priors) == 6


@pytest.mark.parametrize('model', ['gaussian', 'gaussian_cont', 'gaussthick',
                                   'gausshermite'])
def test_curve_fit_recovers_truth(model):
    y = make_spectrum(model)
    p0, dp0 = ms.fit_spectrum(X, y, np.full(X.size, 0.02), model, mcmc=None)
    truth = TRUTH[model]
    assert np.allclose(p0[:2], truth[:2], rtol=0.1)
    assert np.all(np.isfinite(dp0))


def test_fit_spectrum_niter_validation():
    y = make_spectrum('gaussian')
    with pytest.raises(ValueError):
        ms.fit_spectrum(X, y, np.full(X.size, 0.02), 'gaussian', niter=0)


def test_fit_cube_unknown_model_raises():
    data = np.zeros((X.size, 2, 2))
    with pytest.raises(ValueError, match='Unknown'):
        ms.fit_cube(X, data, 0.02, 'not_a_model')


def test_collapse_doublegauss_orders_by_peak():
    """The primary component must be the brighter one, not the wider (C7)."""
    from bettermoments.methods import collapse_doublegauss
    rng = np.random.default_rng(4)
    # Brighter narrow component + fainter *broader* component.
    y = pr.doublegauss(X, -500.0, 300.0, 2.0, 1500.0, 800.0, 1.0)
    data = np.tile((y + rng.normal(0, 0.02, X.size))[:, None, None], (1, 2, 2))
    out = collapse_doublegauss(X, data, 0.02, mcmc=None)
    ggv0, ggFnu = out[0], out[4]
    ggv0b, ggFnub = out[6], out[10]
    assert np.all(ggFnu >= ggFnub)
    assert np.allclose(ggv0, -500.0, atol=100.0)
    assert np.allclose(ggv0b, 1500.0, atol=200.0)


def test_emcee_sampler_runs():
    y = make_spectrum('gaussian')
    p0, dp0 = ms.fit_spectrum(X, y, np.full(X.size, 0.02), 'gaussian',
                              mcmc='emcee', nburnin=50, nsteps=50)
    assert np.allclose(p0[0], 500.0, atol=100.0)
    assert np.all(dp0 > 0)
