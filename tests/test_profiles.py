"""
Tests for bettermoments.profiles: every model is callable with its declared
number of parameters, the ``_cont`` variants are consistent with their base
models, and ``build_cube`` works for every supported method.
"""

import numpy as np
import pytest

from bettermoments import profiles as pr


X = np.linspace(-5.0, 5.0, 101)

MODELS = ['gaussian', 'gaussian_cont', 'gaussthick', 'gaussthick_cont',
          'doublegauss', 'doublegauss_cont', 'gausshermite',
          'gausshermite_cont']


def _example_params(model):
    base = {'gaussian': [0.0, 1.0, 2.0],
            'gaussthick': [0.0, 1.0, 2.0, 0.5],
            'doublegauss': [0.0, 1.0, 2.0, 1.0, 0.5, 1.0],
            'gausshermite': [0.0, 1.0, 2.0, 0.1, 0.1]}
    if model.endswith('_cont'):
        return base[model[:-5]] + [0.3]
    return base[model]


@pytest.mark.parametrize('model', MODELS)
def test_model_callable_with_free_params(model):
    params = _example_params(model)
    assert len(params) == pr.free_params(model)
    y = getattr(pr, model)(X, *params)
    assert y.shape == X.shape
    assert np.all(np.isfinite(y))


@pytest.mark.parametrize('model', ['gaussian', 'gaussthick', 'doublegauss',
                                   'gausshermite'])
def test_cont_variant_adds_offset(model):
    params = _example_params(model)
    base = getattr(pr, model)(X, *params)
    cont = getattr(pr, model + '_cont')(X, *(params + [0.3]))
    assert np.allclose(cont, base + 0.3)


def test_free_params_unknown_model_raises():
    with pytest.raises(ValueError, match='Unknown model'):
        pr.free_params('not_a_model')


def test_gaussthick_zero_tau_is_finite():
    y = pr.gaussthick(X, 0.0, 1.0, 2.0, 0.0)
    assert np.all(np.isfinite(y))
    # In the tau -> 0 limit, the profile tends to the plain Gaussian.
    assert np.allclose(y, pr.gaussian(X, 0.0, 1.0, 2.0), atol=1e-5)


def test_gaussthick_peak_is_fnu():
    """The normalisation makes ``Fnu`` the peak for any optical depth."""
    for tau in [0.1, 1.0, 10.0]:
        y = pr.gaussthick(X, 0.0, 1.0, 2.0, tau)
        assert np.isclose(y.max(), 2.0, rtol=1e-3)


def test_doublegauss_takes_maximum():
    y = pr.doublegauss(X, -1.0, 1.0, 2.0, 1.0, 1.0, 1.0)
    a = pr.gaussian(X, -1.0, 1.0, 2.0)
    b = pr.gaussian(X, 1.0, 1.0, 1.0)
    assert np.allclose(y, np.maximum(a, b))


@pytest.mark.parametrize('method', ['gaussian', 'gaussthick', 'gausshermite',
                                    'doublegauss'])
def test_build_cube_all_methods(method):
    nparams = pr.free_params(method)
    params = _example_params(method)
    moments = np.empty((2 * nparams, 4, 5))
    for i, p in enumerate(params):
        moments[2 * i] = p
        moments[2 * i + 1] = 0.1
    cube = pr.build_cube(X, moments, method)
    assert cube.shape == (X.size, 4, 5)
    assert np.all(np.isfinite(cube))
    # Each spaxel is the corresponding 1D model.
    expected = getattr(pr, method)(X, *params)
    assert np.allclose(cube[:, 2, 3], expected)


def test_build_cube_wrong_shape_raises():
    with pytest.raises(ValueError):
        pr.build_cube(X, np.zeros((5, 4, 5)), 'gaussian')
