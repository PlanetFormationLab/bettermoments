"""
Collapse a data cube down to a summary statistic using various methods. Now
returns statistical uncertainties for all statistics.

TODO:
    - Deal with the fact we're using three different convolution routines.
"""

import argparse
import numpy as np
import multiprocessing

# -- SUPPRESS WARNINGS -- #

import warnings
warnings.filterwarnings("ignore")

# -- DATA MANIPULATION -- #


def estimate_RMS(data, N=5):
    """
    Estimate the RMS noise from the first and last ``N`` channels of the
    data cube, using the central 50% of the spatial extent.

    Args:
        data (ndarray): 3D data cube with the spectral axis first.
        N (Optional[int]): Number of channels at each end to use for the
            noise estimate. Defaults to ``5``.

    Returns:
        float: Estimated RMS noise level.
    """
    x1, x2 = np.percentile(np.arange(data.shape[2]), [25, 75])
    y1, y2 = np.percentile(np.arange(data.shape[1]), [25, 75])
    x1, x2, y1, y2, N = int(x1), int(x2), int(y1), int(y2), int(N)
    rms = np.nanstd([data[:N, y1:y2, x1:x2], data[-N:, y1:y2, x1:x2]])
    return rms


def estimate_spectral_acf(data, N=5, max_lag=None, rms=None, threshold=2.0):
    r"""
    Estimate the normalised spectral autocorrelation function (ACF) of the
    noise from off-source spatial pixels, using the full spectral axis.

    Off-source pixels are identified as those whose peak absolute intensity
    across all channels is below ``threshold * rms``. Each such full-length
    spectrum is mean-subtracted and used to form the biased ACF estimator,

    .. math::
        \hat{\rho}(\tau) = \frac{\sum_{i} n_i n_{i+\tau}}{\sum_{i} n_i^2},

    averaged across all selected pixels. The returned ACF is truncated at the
    first lag at which it falls within the white-noise band
    :math:`\pm 2/\sqrt{N_{\rm chan}}`, where :math:`N_{\rm chan}` is the number
    of channels per spectrum. ``acf[0]`` is always 1.

    Note that with ``threshold = 2`` a fraction of true noise-only sightlines
    will be rejected (the maximum of ``N_chan`` Gaussian draws routinely
    exceeds 2 sigma), so the selection is conservative: it favours including
    only clearly off-source pixels at the cost of sample size, which is fine
    for a first-order ACF estimate.

    Args:
        data (ndarray): 3D data cube with the spectral axis first.
        N (Optional[int]): Number of channels at each end of the cube used to
            estimate ``rms`` when one is not supplied. Defaults to ``5``.
        max_lag (Optional[int]): Hard cap on the maximum lag returned. If
            ``None``, defaults to ``N_chan - 1``.
        rms (Optional[float]): Per-channel RMS used to define the off-source
            threshold. If ``None``, estimated via :func:`estimate_RMS`.
        threshold (Optional[float]): Pixels with peak ``|intensity| <
            threshold * rms`` across all channels are treated as off-source.
            Defaults to ``2.0``.

    Returns:
        ndarray: 1D array of normalised ACF values, ``acf[0] = 1``, length
            between ``1`` and ``max_lag + 1``.
    """
    N = int(N)
    if N < 2:
        raise ValueError("`N` must be at least 2 to estimate an ACF.")
    if rms is None:
        rms = estimate_RMS(data, N=N)
    if not np.isfinite(rms) or rms <= 0:
        raise ValueError("`rms` must be a positive, finite number.")

    peak = np.nanmax(np.abs(data), axis=0)
    offsource = np.isfinite(peak) & (peak < threshold * rms)
    if not offsource.any():
        raise ValueError(
            "No off-source pixels found at threshold {:.1f} * rms; "
            "try increasing `threshold`.".format(threshold))

    spectra = data[:, offsource]
    spectra = spectra[:, np.all(np.isfinite(spectra), axis=0)]
    if spectra.shape[1] == 0:
        raise ValueError("No finite off-source spectra found for ACF estimate.")
    nchan = spectra.shape[0]

    spectra = spectra - spectra.mean(axis=0, keepdims=True)
    var = np.sum(spectra * spectra, axis=0)

    if max_lag is None:
        max_lag = nchan - 1
    max_lag = int(min(max_lag, nchan - 1))
    noise_band = 2.0 / np.sqrt(nchan)

    acf = [1.0]
    for tau in range(1, max_lag + 1):
        cov = np.sum(spectra[:-tau] * spectra[tau:], axis=0)
        rho = np.mean(cov / np.where(var > 0, var, np.nan))
        if not np.isfinite(rho):
            break
        acf.append(float(rho))
        if abs(rho) < noise_band:
            break
    return np.array(acf)


def build_spectral_covariance(rms, acf, nchan, eps=1e-6):
    r"""
    Build the Toeplitz spectral noise covariance matrix,

    .. math::
        \mathbf{C}_{ij} = \sigma^2 \rho(|i - j|),

    where :math:`\rho` is the normalised autocorrelation function returned by
    :func:`estimate_spectral_acf` and :math:`\sigma` is the per-channel RMS.
    Lags beyond ``len(acf) - 1`` are taken to be zero.

    Truncated empirical ACFs need not yield a positive-definite Toeplitz
    matrix (the implied spectral density can go negative at high frequency).
    To guarantee a valid covariance suitable for Cholesky decomposition, the
    eigenvalues are clipped at a small fraction ``eps`` of the spectral
    radius before reconstruction.

    Args:
        rms (float): Per-channel RMS noise.
        acf (ndarray): 1D normalised ACF with ``acf[0] = 1``.
        nchan (int): Number of channels (size of the returned matrix).
        eps (Optional[float]): Eigenvalue floor as a fraction of the largest
            eigenvalue. Defaults to ``1e-6``.

    Returns:
        ndarray: ``(nchan, nchan)`` symmetric positive-definite covariance.
    """
    acf = np.asarray(acf, dtype=float)
    if acf.ndim != 1 or acf.size < 1 or acf[0] == 0:
        raise ValueError("`acf` must be a 1D array with non-zero acf[0].")
    row = np.zeros(nchan)
    n = min(acf.size, nchan)
    row[:n] = acf[:n] / acf[0]
    i = np.arange(nchan)
    lag = np.abs(i[:, None] - i[None, :])
    C = (rms ** 2) * np.where(lag < nchan, row[np.clip(lag, 0, nchan - 1)], 0.0)

    w, V = np.linalg.eigh(C)
    floor = eps * w.max()
    if w.min() < floor:
        w = np.maximum(w, floor)
        C = (V * w) @ V.T
        C = 0.5 * (C + C.T)
    return C


def smooth_data(data, smooth=0, polyorder=0):
    """
    Smooth the input data with a kernel of a width ``smooth``. If ``polyorder``
    is provided, will smooth with a Savitzky-Golay filter, while if
    ``polyorder=0``, the default, then only a top-hat kernel will be used. From
    experimentation, ``smooth=5`` with ``polyorder=3`` provides a good result
    for noisy, but spectrally resolved data.

    .. warning::
        When smoothing low resolution data, this can substantially alter the
        line profile, so measurements must be taken with caution.

    Args:
        data (array): Data to smooth.
        smooth (optional[int]): The width of the kernel for smooth in number of
            channels.
        polyorder (optional[int]): Polynomial order for the Savitzky-Golay
            filter. This must be smaller than ``smooth``. If not provided, the
            smoothing will only be a top-hat filter.

    Returns:
        smoothed_data (array): A smoothed copy of ``data``.
    """
    assert data.ndim == 3, "Data must have 3 dimensions to smooth."
    if smooth > 1:
        if polyorder > 0:
            from scipy.signal import savgol_filter
            smooth += 0 if smooth % 2 else 1
            smoothed_data = savgol_filter(data, smooth, polyorder=polyorder,
                                          mode='wrap', axis=0)
        else:
            from scipy.ndimage import uniform_filter1d
            a = uniform_filter1d(data, smooth, mode='wrap', axis=0)
            b = uniform_filter1d(data[::-1], smooth, mode='wrap', axis=0)[::-1]
            smoothed_data = np.mean([a, b], axis=0)
    else:
        smoothed_data = data.copy()
    return smoothed_data


def get_channel_mask(data, firstchannel=0, lastchannel=-1, user_mask=None):
    """
    Returns the channel mask (a mask for the zeroth axis) based on a first and
    last channel. A ``chan_mask`` can also be provided for more complex masks,
    however be warned that the ``firstchannel`` and ``lastchannel`` will always
    take precedence over ``chan_mask``.

    Args:
        data (array): The data array to use for masking.
        firstchannel (optional[int]): The first channel to include. Defaults to
            the first channel.
        lastchannel (optional[int]): The last channel to include. Defaults to
            the last channel. This can be both a positive value, or a negative
            value following the normal indexing conventions, i.e. ``-1``
            describes the last channel.
        user_mask (optional[array]): A 1D array with size ``data.shape[0]``
            detailing which channels to include in the moment map creation.

    Returns:
        channel_mask (array): A mask array the same shape as ``data``.
    """
    channels = np.arange(data.shape[0])
    channel_mask = np.ones(data.shape[0]) if user_mask is None else user_mask
    assert channel_mask.shape == channels.shape
    lastchannel = channels[lastchannel] if lastchannel < 0 else lastchannel
    assert 0 <= firstchannel < lastchannel <= data.shape[0]
    channel_mask = np.where(channels >= firstchannel, channel_mask, 0)
    channel_mask = np.where(channels <= lastchannel, channel_mask, 0)
    return np.where(channel_mask[:, None, None], np.ones(data.shape), 0.0)


def get_user_mask(data, user_mask_path=None):
    """
    Returns a mask based on a user-provided file. All positive values are
    included in the mask.

    Args:
        data (array): The data array to mask.
        user_mask_path (optional[str]): Path to the FITS cube containing the
            user-defined mask.

    Returns:
        user_mask (array): A mask array the same shape as ``data``.
    """
    if user_mask_path is None:
        user_mask = np.ones(data.shape)
    else:
        from .io import _get_data
        user_mask = np.where(_get_data(user_mask_path) > 0, 1.0, 0.0)
    assert user_mask.shape == data.shape
    return user_mask.astype('float')


def get_threshold_mask(data, clip=None, rms=None, smooth_threshold_mask=0,
                       noise_channels=5):
    """
    Returns a mask based on a sigma-clip to the input data. The most standard
    approach would be to use ``clip=3`` to mask out all pixels with intensities
    :math:`|I| \leq 3\sigma`. If you wanted to specify an asymmetric criteria
    then you can provide a tuple, ``clip=(-2, 3)`` which would mask out all
    pixels where :math:`-2\sigma \leq I \leq 3\sigma`.

    [Some discussion on the smooth_threshold_mask coming...]

    Args:
        data (array): The data array to mask.
        clip (optional[float/tuple]): The sigma clip to apply. If a single
            value is provided, this is taken to be a symmetric mask. If a tuple
            if provided, this is taking as a minimum and maximum clip value.
        rms (optional[float]): The RMS level to use for defining the noise. If
            not specified, will calculate it based on the standard deviation of
            the data.
        smooth_threshold_mask (optional[float]): Convolution kernel FWHM in
            pixels.
        noise_channels (optional[int]): Number of channels at the start and end
            of the velocity axis to use for estimating the noise.

    Returns:
        threshold_mask (array): A mask array the same shape as ``data``.
    """

    # No clipping required.

    if clip is None:
        return np.ones(data.shape)

    # Define the clippng range.

    clip = np.atleast_1d(clip)
    clip = np.array([-clip[0], clip[0]]) if clip.size == 1 else clip
    assert np.all(clip != 0.0), "Use `clip=None` to not use a threshold mask."

    # If we are making a Frankenmask, we must first smooth the cube to both
    # lower the background noise and extend the range of the real emission.
    # After the smoothing, we devide through by the RMS to generate a SNR mask.

    assert smooth_threshold_mask >= 0.0
    if smooth_threshold_mask > 0.0:
        from scipy.ndimage import gaussian_filter
        SNR = [gaussian_filter(c, sigma=smooth_threshold_mask) for c in data]
        SNR = np.array(SNR)
    else:
        SNR = data.copy()

    # Select the right RMS to use for the SNR calculate. If the RMS is provided,
    # which is the default, we use that, otherwise we calculate it based on the
    # standard deviation of (hopefully) line-free channels.

    if rms is None:
        SNR /= estimate_RMS(SNR, noise_channels)
    else:
        SNR /= rms

    # Return the mask.

    return np.logical_or(SNR < clip[0], SNR > clip[-1]).astype('float')


def get_combined_mask(user_mask, threshold_mask, channel_mask, combine='and'):
    """
    Return the combined user, threshold and channel masks, ``user_mask``,
    ``threshold_mask`` and ``velo_mask``, respectively. The user and threshold
    masks can be combined either through ``AND`` or ``OR``, which is controlled
    through the ``combine`` argument. This defaults to ``AND``, such that all
    mask requirements are met.

    Args:
        user_mask (array): User-defined mask from ``get_user_mask``.
        threshold_mask (array): Threshold mask from ``get_threshold_mask``.
        channel_mask (array): Channel mask from ``get_channel_mask``.

    Returns:
        combined_mask (array): A combined mask.

    """
    assert combine in ['and', 'or'], "Unknown `combine`: {}.".format(combine)
    combine = np.logical_and if combine == 'and' else np.logical_or
    combined_mask = combine(combine(user_mask, threshold_mask), channel_mask)
    return combined_mask.astype('float')


# -- COMAND LINE INTERFACE -- #


def main():
    """Command-line interface entry point for collapsing a FITS data cube."""

    # Parse all the command line arguments.

    parser = argparse.ArgumentParser()
    parser.add_argument('path',
                        help='Path to the FITS cube to collapse.')
    parser.add_argument('-clip', default=None, nargs='*', type=float,
                        help='Mask absolute values below this SNR.')
    parser.add_argument('-combine', default='and',
                        help='How to combine the masks if provided.')
    parser.add_argument('-firstchannel', default=0, type=int,
                        help='First channel to use when collapsing cube.')
    parser.add_argument('-lastchannel', default=-1, type=int,
                        help='Last channel to use when collapsing cube.')
    parser.add_argument('-mask', default=None,
                        help='Path to the mask FITS cube.')
    parser.add_argument('-method', default='quadratic',
                        help='Method used to collapse cube.')
    parser.add_argument('-noisechannels', default=5, type=int,
                        help='Number of end channels to use to estimate RMS.')
    parser.add_argument('-outname', default=None, type=str,
                        help='Filename prefix for the saved images.')
    parser.add_argument('-polyorder', default=0, type=int,
                        help='Polynomial order to use for SavGol filtering.')
    parser.add_argument('-processes', default=-1, type=int,
                        help='Number of process to use for analytical fits.')
    parser.add_argument('-rms', default=None, type=float,
                        help='Estimated uncertainty on each pixel.')
    parser.add_argument('-smooth', default=0, type=int,
                        help='Width of filter to smooth spectrally.')
    parser.add_argument('-smooththreshold', default=0.0, type=float,
                        help='Kernel in beam FWHM to smooth threshold map.')
    parser.add_argument('-stokes', default=0, type=int,
                        help='Stokes channel to use.')
    parser.add_argument('--acf', action='store_true',
                        help='Account for spectrally correlated noise: '
                             'auto-estimate the noise ACF from off-source '
                             'pixels (peak |intensity| < 2 * rms) and '
                             'propagate uncertainties through the full '
                             'covariance. Supported by zeroth, first, '
                             'second, quadratic, width, gaussian, '
                             'gaussthick, gausshermite, doublegauss.')
    parser.add_argument('--debug', action='store_true',
                        help='Return all intermediate products to help debug.')
    parser.add_argument('--nooverwrite', action='store_false',
                        help='Do not overwrite files.')
    parser.add_argument('--returnmask', action='store_true',
                        help='Return the masked used as a FITS file.')
    parser.add_argument('--returnmodel', action='store_true',
                        help='Return a model cube built from the moments.')
    parser.add_argument('--silent', action='store_true',
                        help='Do not see how the sausages are made.')

    args = parser.parse_args()

    # Check they all make sense.

    if args.noisechannels < 1:
        raise ValueError("`noisechannels` must an integer greater than 1.")

    args.combine = args.combine.lower()
    if args.combine not in ['and', 'or']:
        raise ValueError("`combine` must be `and` or `or`.")

    if not args.silent:
        import warnings
        warnings.filterwarnings("ignore")

    if args.processes == -1:
        args.processes = multiprocessing.cpu_count()

    # Read in the data and the user-defined mask.
    # If nothing is provided, include all pixels.

    if not args.silent:
        print("Loading up data...")
    from .io import load_cube
    data, velax = load_cube(args.path, args.stokes)

    # Load up the user-defined mask.

    if not args.silent and args.mask is not None:
        print("Loading up user-defined mask...")
    user_mask = get_user_mask(data=data, user_mask_path=args.mask)
    if args.debug:
        from .io import _save_user_mask
        _save_user_mask(user_mask, args)

    # Define the velocity mask based on first and last channels. If nothing is
    # provided, use all channels. A more extensive version is possible for the
    # non-command line version.

    if not args.silent and (args.firstchannel != 0 or args.lastchannel != -1):
        print("Defining channel-based mask...")
    channel_mask = get_channel_mask(data=data,
                                    firstchannel=args.firstchannel,
                                    lastchannel=args.lastchannel)
    if args.debug:
        from .io import _save_channel_mask
        _save_channel_mask(channel_mask, args)

    # Smooth the data in the spectral dimension. Uses by default a uniform
    # (boxcar) filter. If a `polyorder` is provided, assumes the user wants a
    # Savitzky-Golay filter. In this case, extend all even window sizes by one
    # to make sure it is an odd number.

    if not args.silent and args.smooth:
        print("Smoothing the data...")
    data = smooth_data(data=data,
                       smooth=args.smooth,
                       polyorder=args.polyorder)
    if args.debug:
        from .io import _save_smoothed_data
        _save_smoothed_data(data, args)

    # Calculate the RMS based on the first and last `noisechannels`, which is 5
    # by default. TODO: Test if there's a better way of doing this...

    if args.rms is None:
        if not args.silent:
            print("Estimating noise in the data...")
        args.rms = estimate_RMS(data, args.noisechannels)
        if not args.silent:
            print("Estimated RMS: {:.2e}.".format(args.rms))

    # Estimate the spectral noise ACF from line-free channels if requested,
    # so uncertainties can be propagated through the full covariance.

    acf = None
    if args.acf:
        if not args.silent:
            print("Estimating spectral noise ACF...")
        acf = estimate_spectral_acf(data, N=args.noisechannels, rms=args.rms)
        if not args.silent:
            S = 1.0 + 2.0 * acf[1:].sum()
            print("ACF: {}  (variance inflation S = {:.2f})"
                  .format(np.array2string(acf, precision=3), S))

    # Define the threshold mask. This includes the spatial smoothing of the
    # data for create Frankenmasks.

    if not args.silent and args.clip is not None:
        print("Calculating threshold-based mask...")
    threshold_mask = get_threshold_mask(data=data,
                                        clip=args.clip,
                                        rms=args.rms,
                                        smooth_threshold_mask=args.smooththreshold,
                                        noise_channels=args.noisechannels)
    if args.debug:
        from .io import _save_threshold_mask
        _save_threshold_mask(threshold_mask, args)

    # Combine the masks and apply to the data.

    if not args.silent:
        print("Masking the data...")
    combined_mask = get_combined_mask(user_mask=user_mask,
                                      threshold_mask=threshold_mask,
                                      channel_mask=channel_mask,
                                      combine=args.combine)
    if args.returnmask or args.debug:
        from .io import _save_mask
        _save_mask(combined_mask, args)
    if args.debug:
        from .io import _save_channel_count
        _save_channel_count(np.sum(combined_mask, axis=0), args)
    masked_data = data.copy() * combined_mask

    # Reverse the direction if the velocity axis is decreasing.

    if np.diff(velax).mean() < 0:
        masked_data = masked_data[::-1]
        velax = velax[::-1]

    # Calculate the moments.

    if not args.silent:
        print("Calculating maps...")

    if args.method == 'zeroth':
        from .methods import collapse_zeroth
        moments = collapse_zeroth(velax=velax,
                                  data=masked_data,
                                  rms=args.rms,
                                  acf=acf)

    elif args.method == 'first':
        from .methods import collapse_first
        moments = collapse_first(velax=velax,
                                 data=masked_data,
                                 rms=args.rms,
                                 acf=acf)

    elif args.method == 'second':
        from .methods import collapse_second
        moments = collapse_second(velax=velax,
                                  data=masked_data,
                                  rms=args.rms,
                                  acf=acf)

    elif args.method == 'eighth':
        from .methods import collapse_eighth
        moments = collapse_eighth(velax=velax,
                                  data=masked_data,
                                  rms=args.rms)

    elif args.method == 'ninth':
        from .methods import collapse_ninth
        moments = collapse_ninth(velax=velax,
                                 data=masked_data,
                                 rms=args.rms)

    elif args.method == 'maximum':
        from .methods import collapse_maximum
        moments = collapse_maximum(velax=velax,
                                   data=masked_data,
                                   rms=args.rms)

    elif args.method == 'quadratic':
        from .methods import collapse_quadratic
        moments = collapse_quadratic(velax=velax,
                                     data=masked_data,
                                     rms=args.rms,
                                     acf=acf)
        if args.clip is not None:
            temp = moments[2] / moments[3] >= max(args.clip)
            moments *= np.where(temp, 1.0, np.nan)[None, :, :]

    elif args.method == 'width':
        from .methods import collapse_width
        moments = collapse_width(velax=velax,
                                 data=masked_data,
                                 rms=args.rms,
                                 acf=acf)

    elif args.method == 'percentiles':
        from .methods import collapse_percentiles
        moments = collapse_percentiles(velax=velax,
                                       data=masked_data,
                                       rms=args.rms)

    elif args.method == 'gaussian':
        from .methods import collapse_gaussian
        print("Using {} CPUs.".format(args.processes))
        moments = collapse_gaussian(velax=velax,
                                    data=masked_data,
                                    rms=args.rms,
                                    ncpu=args.processes,
                                    acf=acf,
                                    mcmc=None)

    elif args.method == 'gaussthick':
        from .methods import collapse_gaussthick
        print("Using {} CPUs.".format(args.processes))
        moments = collapse_gaussthick(velax=velax,
                                      data=masked_data,
                                      rms=args.rms,
                                      ncpu=args.processes,
                                      acf=acf,
                                      mcmc=None)

    elif args.method == 'gausshermite':
        from .methods import collapse_gausshermite
        print("Using {} CPUs.".format(args.processes))
        moments = collapse_gausshermite(velax=velax,
                                        data=masked_data,
                                        rms=args.rms,
                                        ncpu=args.processes,
                                        acf=acf,
                                        mcmc=None)

    elif args.method == 'doublegauss':
        from .methods import collapse_doublegauss
        print("Using {} CPUs.".format(args.processes))
        moments = collapse_doublegauss(velax=velax,
                                       data=masked_data,
                                       rms=args.rms,
                                       ncpu=args.processes,
                                       acf=acf,
                                       mcmc=None)

    else:
        raise ValueError("Unknown method.")

    # Check for any NaN values in the uncertainty maps.

    if not args.silent:
        print("Checking for NaNs in error maps.")
    from .methods import check_finite_errors
    moments = check_finite_errors(moments)

    # Save as FITS files.
    
    if not args.silent:
        print("Saving moment maps...")
    from .io import save_to_FITS
    save_to_FITS(moments=moments,
                 method=args.method,
                 path=args.path,
                 outname=args.outname,
                 overwrite=args.nooverwrite)

    # If applicable, build a model cube from the decomposition.

    if args.returnmodel:
        if not args.silent:
            print("Building and saving model...")
        from .profiles import build_cube
        from .io import _save_model
        try:
            model = build_cube(x=velax, moments=moments, method=args.method)
        except ValueError:
            print("Model failed, returning empty data cube.")
            model = np.zeros(masked_data.shape)
        _save_model(model=model, args=args)


if __name__ == '__main__':
    main()
