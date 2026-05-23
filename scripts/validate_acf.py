"""
Monte Carlo validation of the spectral-correlation uncertainty propagation
added in response to issue #27.

For each tested collapse method, we:

  1. Build a cube with a known emission line and noise drawn from a known
     spectral covariance (sampled via Cholesky for an exact match).
  2. Run many noise realisations and measure the empirical scatter of the
     moment across realisations -- this is the "truth".
  3. Compare the empirical scatter to the analytic uncertainty returned by
     the method, both with ``acf=None`` (diagonal) and with the true ACF
     passed via ``acf=``.

The corrected (``acf=truth``) prediction should match the MC scatter to
within a few percent. The diagonal prediction will under-estimate, by
roughly the Priestley variance-inflation factor sqrt(S).

Sampling from the target covariance (rather than convolving white noise)
removes ACF-estimator bias and edge effects so the validation isolates the
propagation correctness.

Run as: ``python scripts/validate_acf.py``
"""

import os
import sys

import numpy as np
from scipy.linalg import cholesky

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bettermoments import build_spectral_covariance
from bettermoments.methods import collapse_zeroth, collapse_quadratic


def correlated_noise(shape, L, rng):
    """Sample noise with covariance ``L @ L.T`` from the lower-triangular L."""
    nv = L.shape[0]
    spatial = int(np.prod(shape[1:]))
    flat = rng.standard_normal((nv, spatial))
    return (L @ flat).reshape(shape)


def run_method(method, velax, profile, spatial_shape, sigma, acf_truth, L,
               nrlz, rng, extract_fn):
    """Return (MC_scatter, predicted_diag, predicted_acf) for one method."""
    cube_shape = (velax.size,) + spatial_shape
    values = []
    for _ in range(nrlz):
        cube = profile[:, None, None] + correlated_noise(cube_shape, L, rng)
        v, _ = extract_fn(method(velax, cube, rms=sigma))
        values.append(v)
    empirical = np.array(values).std(axis=0).mean()

    cube = profile[:, None, None] + correlated_noise(cube_shape, L, rng)
    _, dv_diag = extract_fn(method(velax, cube, rms=sigma))
    _, dv_acf = extract_fn(method(velax, cube, rms=sigma, acf=acf_truth))
    return empirical, float(dv_diag.mean()), float(dv_acf.mean())


def main():
    rng = np.random.default_rng(0)
    nv = 81
    spatial_shape = (16, 16)
    chan = 0.05
    velax = np.arange(nv) * chan - 2.0
    profile = 3.0 * np.exp(-0.5 * ((velax - 0.0) / 0.3) ** 2)
    sigma = 0.02
    nrlz = 400

    # Target ACF: realistic for Hanning-smoothed data.
    acf_truth = np.array([1.0, 0.5, 0.1])
    S = 1.0 + 2.0 * acf_truth[1:].sum()
    C = build_spectral_covariance(rms=sigma, acf=acf_truth, nchan=nv)
    L = cholesky(C, lower=True)

    cases = [
        ("collapse_zeroth   (M0 / dM0)",
         collapse_zeroth,
         lambda out: (out[0], out[1])),
        ("collapse_quadratic (v0 / dv0)",
         collapse_quadratic,
         lambda out: (out[0], out[1])),
        ("collapse_quadratic (Fnu / dFnu)",
         collapse_quadratic,
         lambda out: (out[2], out[3])),
    ]

    print("Cube: nv={}, spatial={}, sigma={:.3g}, n_realisations={}".format(
        nv, spatial_shape, sigma, nrlz))
    print("True ACF: {}   variance inflation S = {:.3f}, sqrt(S) = {:.3f}"
          .format(acf_truth, S, np.sqrt(S)))
    print("-" * 78)
    print("{:<33} {:>10} {:>10} {:>10} {:>8}".format(
        "method", "MC", "acf=None", "acf=truth", "verdict"))
    print("-" * 78)

    all_pass = True
    for name, method, extract in cases:
        emp, pred_diag, pred_acf = run_method(
            method, velax, profile, spatial_shape, sigma, acf_truth, L,
            nrlz, rng, extract)
        ratio = pred_acf / emp if emp > 0 else float("nan")
        ok = 0.9 <= ratio <= 1.1
        all_pass &= ok
        verdict = "PASS" if ok else "FAIL ({:.2f}x)".format(ratio)
        print("{:<33} {:>10.4g} {:>10.4g} {:>10.4g} {:>8}".format(
            name, emp, pred_diag, pred_acf, verdict))

    print("-" * 78)
    print("Overall:", "PASS" if all_pass else "FAIL")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
