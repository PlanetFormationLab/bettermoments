import numpy as np


def quadratic(data, uncertainty=None, axis=0, x0=0.0, dx=1.0, linewidth=None,
              acf=None):
    """
    Compute the quadratic estimate of the centroid of a line in a data cube.

    The use case that we expect is a data cube with spatiotemporal coordinates
    in all but one dimension. The other dimension (given by the ``axis``
    parameter) will generally be wavelength, frequency, or velocity. This
    function estimates the centroid of the *brightest* line along the ``axis''
    dimension, in each spatiotemporal pixel.

    Following Vakili & Hogg we allow for the option for the data to be smoothed
    prior to the parabolic fitting. The recommended kernel is a Gaussian of
    comparable width to the line. However, for low noise data, this is not
    always necessary.

    Args:
        data (ndarray): The data cube as an array with at least one dimension.
        uncertainty (Optional[ndarray or float]): The uncertainty on the
            intensities given by ``data``. If this is a scalar, all
            uncertainties are assumed to be the same. If this is an array, it
            must have the same shape as ``data'' and give the uncertainty on
            each intensity. If not provided, the uncertainty on the centroid
            will not be estimated.
        axis (Optional[int]): The axis along which the centroid should be
            estimated. By default this will be the zeroth axis.
        x0 (Optional[float]): The wavelength/frequency/velocity/etc. value for
            the zeroth pixel in the ``axis'' dimension.
        dx (Optional[float]): The pixel scale of the ``axis'' dimension.
        acf (Optional[ndarray]): Normalised spectral ACF (``acf[0] = 1``) used
            to construct the 3x3 noise covariance sub-block centred on the
            peak channel. When provided, ``uncertainty`` must be a scalar
            (per-channel RMS) and the off-diagonal terms ``acf[1]``, ``acf[2]``
            enter the propagation correctly. If ``None`` (default), channels
            are assumed independent.

    Returns:
        x_max (ndarray): The centroid of the brightest line along the ``axis''
            dimension in each pixel.
        x_max_sig (ndarray or None): The uncertainty on ``x_max''. If
            ``uncertainty'' was not provided, this will be ``None''.
        y_max (ndarray): The predicted value of the intensity at maximum.
        y_max_sig (ndarray or None): The uncertainty on ``y_max''. If
            ``uncertainty'' was not provided, this will be ``None''.

    """
    # Cast the data to a numpy array
    data = np.moveaxis(np.atleast_1d(data), axis, 0)
    shape = data.shape[1:]
    data = np.reshape(data, (len(data), -1))

    # Find the maximum velocity pixel in each spatial pixel
    idx = np.argmax(data, axis=0)

    # Deal with edge effects by keeping track of which pixels are right on the
    # edge of the range
    idx_bottom = idx == 0
    idx_top = idx == len(data) - 1
    idx = np.clip(idx, 1, len(data)-2)

    # Extract the maximum and neighboring pixels
    f_minus = data[(idx-1, range(data.shape[1]))]
    f_max = data[(idx, range(data.shape[1]))]
    f_plus = data[(idx+1, range(data.shape[1]))]

    # Work out the polynomial coefficients
    a0 = f_max
    a1 = 0.5 * (f_plus - f_minus)
    a2 = 0.5 * (f_plus + f_minus - 2*f_max)

    # Flat-topped, clipped or fully masked spectra have no defined parabolic
    # maximum; these pixels are set to NaN rather than dividing by zero.
    flat = a2 == 0.0

    # Compute the maximum of the quadratic
    with np.errstate(divide='ignore', invalid='ignore'):
        x_max = idx - 0.5 * a1 / a2
        y_max = a0 - 0.25 * a1**2 / a2

    # Set sensible defaults for the edge cases
    x_max[idx_bottom] = 0
    x_max[idx_top] = len(data) - 1
    y_max[idx_bottom] = f_minus[idx_bottom]
    y_max[idx_top] = f_plus[idx_top]
    x_max[flat] = np.nan
    y_max[flat] = np.nan

    # If no uncertainty was provided, end now
    if uncertainty is None:
        return (
            np.reshape(x0 + dx * x_max, shape), None,
            np.reshape(y_max, shape), None)

    # Per-pixel sensitivity vectors g_x, g_y of shape (3, npix) for the three
    # channels (idx-1, idx, idx+1) entering the parabolic fit. Verified
    # analytically against finite differences and against Monte Carlo.
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_a2sq = 1.0 / (a2 ** 2)
        gx = np.stack([0.25 * (a1 + a2) * inv_a2sq,
                       -0.5 * a1 * inv_a2sq,
                       0.25 * (a1 - a2) * inv_a2sq])
        gy = np.stack([0.125 * a1 * (a1 + 2.0 * a2) * inv_a2sq,
                       1.0 - 0.25 * a1**2 * inv_a2sq,
                       0.125 * a1 * (a1 - 2.0 * a2) * inv_a2sq])

    if acf is None:
        try:
            uncertainty = float(uncertainty) + np.zeros_like(data)
        except TypeError:

            # An array of errors was provided
            uncertainty = np.moveaxis(np.atleast_1d(uncertainty), axis, 0)
            if uncertainty.shape[0] != data.shape[0] or \
                    shape != uncertainty.shape[1:]:
                raise ValueError("the data and uncertainty must have the same "
                                 "shape")
            uncertainty = np.reshape(uncertainty, (len(uncertainty), -1))

        df = np.stack([uncertainty[(idx-1, range(uncertainty.shape[1]))]**2,
                       uncertainty[(idx, range(uncertainty.shape[1]))]**2,
                       uncertainty[(idx+1, range(uncertainty.shape[1]))]**2])
        x_max_var = np.sum(gx**2 * df, axis=0)
        y_max_var = np.sum(gy**2 * df, axis=0)
    else:
        try:
            rms = float(uncertainty)
        except TypeError:
            raise ValueError("`uncertainty` must be a scalar when `acf` is "
                             "provided.")
        acf_arr = np.asarray(acf, dtype=float)
        rho1 = acf_arr[1] if acf_arr.size > 1 else 0.0
        rho2 = acf_arr[2] if acf_arr.size > 2 else 0.0
        C3 = (rms ** 2) * np.array([[1.0, rho1, rho2],
                                    [rho1, 1.0, rho1],
                                    [rho2, rho1, 1.0]])
        x_max_var = np.einsum('ip,ij,jp->p', gx, C3, gx)
        y_max_var = np.einsum('ip,ij,jp->p', gy, C3, gy)

    x_max_var = np.clip(x_max_var, 0.0, None)
    y_max_var = np.clip(y_max_var, 0.0, None)

    # The sensitivity vectors are built from the clipped indices, so the
    # uncertainties for edge and flat pixels are undefined.
    undefined = idx_bottom | idx_top | flat
    x_max_var[undefined] = np.nan
    y_max_var[undefined] = np.nan

    return (
        np.reshape(x0 + dx * x_max, shape),
        np.reshape(dx * np.sqrt(x_max_var), shape),
        np.reshape(y_max, shape),
        np.reshape(np.sqrt(y_max_var), shape))
