"""
Tests for bettermoments.methods: moment values recover known inputs and the
uncertainties behave correctly (scaling, determinism, masking).
"""

import numpy as np
import pytest

from bettermoments.methods import (collapse_zeroth, collapse_first,
                                   collapse_second, collapse_eighth,
                                   collapse_ninth, collapse_maximum,
                                   collapse_quadratic, collapse_width,
                                   collapse_percentiles, check_finite_errors,
                                   collapse_method_products)
from .conftest import V0, DV, FNU, RMS


def test_zeroth_recovers_integral(velax, clean_data):
    M0, dM0 = collapse_zeroth(velax, clean_data, RMS)
    expected = np.sqrt(np.pi) * FNU * DV
    assert np.allclose(M0, expected, rtol=0.01)
    assert np.all(dM0 > 0)


def test_zeroth_error_scales_with_rms(velax, clean_data):
    dM0_a = collapse_zeroth(velax, clean_data, RMS)[1]
    dM0_b = collapse_zeroth(velax, clean_data, 2 * RMS)[1]
    assert np.allclose(dM0_b, 2 * dM0_a)


def test_first_recovers_center(velax, clean_data):
    M1, dM1 = collapse_first(velax, clean_data, RMS)
    assert np.allclose(M1, V0, atol=0.05 * DV)
    assert np.all(dM1 > 0)


def test_first_is_deterministic(velax, noisy_data):
    a = collapse_first(velax, noisy_data, RMS)
    b = collapse_first(velax, noisy_data, RMS)
    assert np.array_equal(a[0], b[0], equal_nan=True)
    assert np.array_equal(a[1], b[1], equal_nan=True)


def test_first_empty_pixels_are_nan(velax, clean_data):
    data = clean_data.copy()
    data[:, 0, 0] = 0.0
    M1, dM1 = collapse_first(velax, data, RMS)
    assert np.isnan(M1[0, 0]) and np.isnan(dM1[0, 0])


def test_second_recovers_dispersion(velax, clean_data):
    M2, dM2 = collapse_second(velax, clean_data, RMS)
    assert np.allclose(M2, DV / np.sqrt(2), rtol=0.05)
    assert np.all(np.isfinite(dM2))


def test_eighth_and_ninth(velax, clean_data):
    M8, dM8 = collapse_eighth(velax, clean_data, RMS)
    assert np.allclose(M8, FNU, rtol=0.02)  # peak not exactly on the grid
    assert np.allclose(dM8, RMS)
    M9, dM9 = collapse_ninth(velax, clean_data, RMS)
    chan = np.diff(velax).mean()
    assert np.allclose(M9, V0, atol=chan)
    assert np.allclose(dM9, 0.5 * chan)
    M = collapse_maximum(velax, clean_data, RMS)
    assert len(M) == 4


def test_quadratic_recovers_center_and_peak(velax, clean_data):
    v0, dv0, Fnu, dFnu = collapse_quadratic(velax, clean_data, RMS)
    chan = np.diff(velax).mean()
    assert np.allclose(v0, V0, atol=0.5 * chan)
    assert np.allclose(Fnu, FNU, rtol=0.01)
    assert np.all(dv0 > 0) and np.all(dFnu > 0)


def test_quadratic_flat_spectra_are_nan(velax):
    data = np.zeros((velax.size, 4, 4))
    v0, dv0, Fnu, dFnu = collapse_quadratic(velax, data, RMS)
    assert np.all(np.isnan(v0)) and np.all(np.isnan(dv0))


def test_width_recovers_doppler_width(velax, clean_data):
    dV, ddV = collapse_width(velax, clean_data, RMS)
    assert np.allclose(dV, DV, rtol=0.02)
    assert np.all(np.isfinite(ddV))


def test_width_zero_pixels_are_nan(velax, clean_data):
    data = clean_data.copy()
    data[:, 0, 0] = 0.0
    dV, _ = collapse_width(velax, data, RMS)
    assert np.isnan(dV[0, 0])


def test_percentiles_center_and_brightness_invariance(velax, clean_data):
    wp50, dwp50 = collapse_percentiles(velax, clean_data, RMS)[:2]
    assert np.allclose(wp50, V0, atol=0.1 * DV)
    # Doubling the brightness at fixed rms must halve the uncertainty.
    dwp50_bright = collapse_percentiles(velax, 2 * clean_data, RMS)[1]
    assert np.allclose(dwp50_bright, 0.5 * dwp50, rtol=1e-6)


def test_check_finite_errors_non_square(velax):
    data = np.zeros((velax.size, 6, 9))
    data[25:35, 2:4, 3:6] = 1.0
    M0, dM0 = collapse_zeroth(velax, data, RMS)
    dM0 = np.where(M0 > 0, dM0, np.nan)
    out = check_finite_errors((M0, dM0))
    assert out[0].shape == (6, 9)


def test_collapse_method_products_unknown_raises():
    with pytest.raises(ValueError, match='Unknown method'):
        collapse_method_products('not_a_method')


def test_acf_inflates_moment_uncertainties(velax, clean_data):
    """A positive ACF must inflate the M0 uncertainty over the diagonal."""
    acf = np.array([1.0, 0.5, 0.2])
    dM0_diag = collapse_zeroth(velax, clean_data, RMS)[1]
    dM0_acf = collapse_zeroth(velax, clean_data, RMS, acf=acf)[1]
    assert np.all(dM0_acf > dM0_diag)


def test_dM1_monte_carlo(velax):
    """dM1 must match the observed scatter of M1 over noise realisations."""
    line = FNU * np.exp(-((velax[:, None, None] - V0) / DV)**2)
    line = np.where(line > 5 * RMS, line, 0.0)
    rng = np.random.default_rng(2)
    samples = []
    for _ in range(200):
        noisy = np.where(line > 0.0, line + rng.normal(0, RMS, (velax.size,
                                                                1, 1)), 0.0)
        samples.append(collapse_first(velax, noisy, RMS)[0][0, 0])
    scatter = np.std(samples)
    predicted = collapse_first(velax, line, RMS)[1][0, 0]
    assert np.isclose(predicted, scatter, rtol=0.25)
