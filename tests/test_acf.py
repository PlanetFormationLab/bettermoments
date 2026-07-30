"""
Tests for the spectral ACF estimation and covariance construction.
"""

import numpy as np
import pytest
from scipy.ndimage import uniform_filter1d

from bettermoments import (estimate_spectral_acf, build_spectral_covariance,
                           estimate_RMS)


def white_noise(shape, seed=0):
    return np.random.default_rng(seed).normal(0.0, 1.0, shape)


def test_acf_white_noise_is_delta():
    acf = estimate_spectral_acf(white_noise((128, 16, 16)), rms=1.0,
                                threshold=np.inf)
    assert acf[0] == 1.0
    assert acf.size <= 2  # truncated almost immediately


def test_acf_detects_correlation():
    noise = uniform_filter1d(white_noise((128, 16, 16)), 3, axis=0)
    rms = estimate_RMS(noise)
    acf = estimate_spectral_acf(noise, rms=rms, threshold=np.inf)
    assert acf.size >= 2
    assert acf[1] > 0.3


def test_acf_ignores_blank_spectra():
    """NaN-padded pixels (zero-filled on load) must not poison the ACF (C2)."""
    noise = uniform_filter1d(white_noise((128, 16, 16)), 3, axis=0)
    rms = estimate_RMS(noise)
    reference = estimate_spectral_acf(noise, rms=rms, threshold=np.inf)
    padded = noise.copy()
    padded[:, :8, :] = 0.0  # what load_cube does to NaN pixels
    acf = estimate_spectral_acf(padded, rms=rms, threshold=np.inf)
    assert acf.size == reference.size
    assert np.allclose(acf, reference, atol=0.05)


def test_acf_all_blank_raises():
    with pytest.raises(ValueError):
        estimate_spectral_acf(np.zeros((64, 8, 8)), rms=1.0)


def test_build_covariance_positive_definite():
    acf = np.array([1.0, 0.6, 0.2])
    C = build_spectral_covariance(rms=0.1, acf=acf, nchan=32)
    assert C.shape == (32, 32)
    assert np.allclose(C, C.T)
    assert np.linalg.eigvalsh(C).min() > 0.0
    assert np.allclose(np.diag(C), 0.01, rtol=0.1)


def test_estimate_rms_overlap_guard():
    with pytest.raises(ValueError):
        estimate_RMS(np.zeros((8, 16, 16)), N=5)


def test_acf_correction_validates_monte_carlo():
    """dM0 with the ACF must match the observed scatter on correlated noise."""
    rng = np.random.default_rng(3)
    nv, n_mc = 64, 300
    chan = 100.0
    velax = np.arange(nv) * chan
    from bettermoments.methods import collapse_zeroth

    # Correlated noise via a running mean of 3 channels.
    noise = uniform_filter1d(rng.normal(0, 1, (nv, n_mc, 1)), 3, axis=0)
    rms = noise.std()
    acf = estimate_spectral_acf(noise, rms=rms, threshold=1e10)

    # Uniform unit weights: M0 = chan * sum(noise).
    data = np.where(np.abs(noise) > 0, noise + 10.0, 10.0)  # all "unmasked"
    M0, dM0 = collapse_zeroth(velax, data, rms, acf=acf)
    scatter = np.std(M0)
    assert np.isclose(dM0.mean(), scatter, rtol=0.2)

    # And the diagonal assumption must underestimate it.
    dM0_diag = collapse_zeroth(velax, data, rms)[1]
    assert dM0_diag.mean() < 0.8 * scatter
