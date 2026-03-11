"""
All the methods used for collapsing the cube.
"""

import numpy as np



# -- COLLAPSE METHODS -- *


def collapse_zeroth(velax, data, rms):
    r"""
    Collapses the cube by integrating along the spectral axis. It will return
    the integrated intensity along the spectral axis, ``M0``, and the
    associated uncertainty, ``dM0``. Following `Teague (2019)`_ these are
    calculated by,

    .. math::
        M_0 = \sum_{i}^N I_i \, \Delta v_{{\rm chan},\,i}

    and

    .. math::
        M_0 = \sqrt{\sum_{i\,(I_i > 0)}^N \sigma_i^2 \cdot \Delta v_{{\rm chan},\,i}^2}

    where :math:`\Delta v_i` and :math:`I_i` are the chanenl width and flux
    density at the :math:`i^{\rm th}` channel, respectively and the sum goes
    over the whole ``axis``.

    .. _Teague (2019): https://iopscience.iop.org/article/10.3847/2515-5172/ab2125

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Flux densities or brightness temperature array. Assumes
            that the first axis is the velocity axis.
        rms (float): Noise per pixel in same units as ``data``.

    Returns:
        ``M0`` (`ndarray`), ``dM0`` (`ndarray`):
            ``M0``, the integrated intensity along provided axis and ``dM0``,
            the uncertainty on ``M0`` in the same units as ``M0``.
    """
    chan = abs(np.diff(velax).mean())
    npix = np.sum(data != 0.0, axis=0)
    M0 = np.trapz(data, dx=chan, axis=0)
    dM0 = chan * rms * npix**0.5 * np.ones(M0.shape)
    if M0.shape != data[0].shape:
        raise ValueError("`data` not collapsed correctly." +
                         " Expected shape: {},".format(data[0].shape) +
                         " returned shape: {}.".format(M0.shape))
    if M0.shape != dM0.shape:
        raise ValueError("Mismatch in `M0` and `dM0` shapes: "
                         "{} and {}.".format(M0.shape, dM0.shape))
    return M0, dM0


def collapse_first(velax, data, rms):
    r"""
    Collapses the cube using the intensity weighted average velocity (or first
    moment map). For a symmetric line profile this will be the line center,
    however for highly non-symmetric line profiles, this will not give a
    meaningful result. Following `Teague (2019)`_, the line center is given by,

    .. math::
        M_1 = \frac{\sum_i^N I_i v_i}{\sum_i^N I_i}

    where :math:`v_i` and :math:`I_i` are the velocity and flux density at the
    :math:`i^{\rm th}` channel, respectively and the sum goes over the whole
    ``axis``. In addition, the uncertainty is given by,

    .. math::
        \delta M_1 = \sqrt{\sum_{i\,(I_i > 0)}^N \sigma_i^2 \cdot (v_i - M_1)^2}

    where :math:`\sigma_i` is the rms noise.

    .. _Teague (2019): https://iopscience.iop.org/article/10.3847/2515-5172/ab2125

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Flux densities or brightness temperature array. Assumes
            that the zeroth axis is the velocity axis.
        rms (float): Noise per pixel in same units as ``data``.

    Returns:
        ``M1`` (`ndarray`), ``dM1`` (`ndarray`):
            ``M1``, the intensity weighted average velocity in units of
            ``velax`` and ``dM1``, the uncertainty in the intensity weighted
            average velocity with same units as ``v0``.
    """
    chan = np.diff(velax).mean()
    vpix = chan * np.arange(data.shape[0]) + velax[0]
    vpix = vpix[:, None, None] * np.ones(data.shape)

    weights = 1e-10 * np.random.rand(data.size).reshape(data.shape)
    weights = np.where(data != 0.0, abs(data), weights)
    M1 = np.average(vpix, weights=weights, axis=0)
    dM1 = (vpix - M1[None, :, :]) * rms / np.sum(weights, axis=0)
    dM1 = np.sqrt(np.sum(dM1**2, axis=0))

    npix = np.sum(data != 0.0, axis=0)
    M1 = np.where(npix >= 1.0, M1, np.nan)
    dM1 = np.where(npix >= 1.0, dM1, np.nan)

    if M1.shape != data[0].shape:
        raise ValueError("`data` not collapsed correctly." +
                         " Expected shape: {},".format(data[0].shape) +
                         " returned shape: {}.".format(M1.shape))
    if M1.shape != dM1.shape:
        raise ValueError("Mismatch in `M1` and `dM1` shapes: "
                         "{} and {}.".format(M1.shape, dM1.shape))
    return M1, dM1


def collapse_second(velax, data, rms):
    r"""
    Collapses the cube using the intensity-weighted average velocity dispersion
    (or second moment). For a symmetric line profile this will be a measure of
    the line width. Following `Teague (2019)`_ this is calculated by,

    .. math::
        M_2 = \sqrt{\frac{\sum_i^N I_i (v_i - M_1)^2}{{\sum_i^N I_i}}}

    where :math:`M_1` is the first moment and :math:`v_i` and :math:`I_i` are
    the velocity and flux density at the :math:`i^{\rm th}` channel,
    respectively. The uncertainty is given by,

    .. math::
        \delta M_2 &= \frac{1}{2 M_2} \cdot \sqrt{\sum_{i\,(I_i > 0)}^N \sigma_i^2 \cdot \big[(v_i - M_1)^2 - M_2^2\big]^2}

    where :math:`\sigma_i` is the rms noise in the :math:`i^{\rm th}` channel.

    .. _Teague (2019): https://iopscience.iop.org/article/10.3847/2515-5172/ab2125

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Flux densities or brightness temperature array. Assumes
            that the first axis is the velocity axis.
        rms (float): Noise per pixel in same units as ``data``.

    Returns:
        ``M2`` (`ndarray`), ``dM2`` (`ndarray`):
            ``M2`` is the intensity weighted velocity dispersion with units of
            ``velax``.  ``dM2`` is the uncertainty of ``M2`` in the same units.
    """
    chan = np.diff(velax).mean()
    vpix = chan * np.arange(data.shape[0]) + velax[0]
    vpix = vpix[:, None, None] * np.ones(data.shape)

    weights = 1e-10 * np.random.rand(data.size).reshape(data.shape)
    weights = np.where(data != 0.0, abs(data), weights)

    M1 = collapse_first(velax=velax, data=data, rms=rms)[0]
    M1 = M1[None, :, :] * np.ones(data.shape)
    M2 = np.sum(weights * (vpix - M1)**2, axis=0) / np.sum(weights, axis=0)
    M2 = np.sqrt(M2)

    dM2 = ((vpix - M1)**2 - M2**2) * rms / np.sum(weights, axis=0)
    dM2 = np.sqrt(np.sum(dM2**2, axis=0)) / 2. / M2

    npix = np.sum(data != 0.0, axis=0)
    M2 = np.where(npix >= 1.0, M2, np.nan)
    dM2 = np.where(npix >= 1.0, dM2, np.nan)

    if M2.shape != data[0].shape:
        raise ValueError("`data` not collapsed correctly." +
                         " Expected shape: {},".format(data[0].shape) +
                         " returned shape: {}.".format(M2.shape))
    if M2.shape != dM2.shape:
        raise ValueError("Mismatch in `M2` and `dM2` shapes: "
                         "{} and {}.".format(M2.shape, dM2.shape))
    return M2, dM2


def collapse_eighth(velax, data, rms):
    """
    Take the peak value along the provided axis. The uncertainty is the RMS
    noise of the image.

    Args:
        velax (ndarray): Velocity axis of the cube. Not needed.
        data (ndarray): Flux densities or brightness temperature array. Assumes
            that the first axis is the velocity axis.
        rms (float): Noise per pixel in same units as ``data``.

    Returns:
        ``M8`` (`ndarray`), ``dM8`` (`ndarray`):
            The peak value, ``M8``, and the associated uncertainty, ``dM8``.
    """
    M8 = np.max(data, axis=0)
    loc = np.where(np.max(data, axis=0)==0)
    M8[loc] = np.nan
    dM8 = rms * np.ones(M8.shape)
    mask = M8 != 0.0
    return np.where(mask, M8, np.nan), np.where(mask, dM8, np.nan)


def collapse_ninth(velax, data, rms):
    """
    Take the velocity of the peak intensity along the provided axis. The
    uncertainty is half the channel width.

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Flux densities or brightness temperature array. Assumes
            that the first axis is the velocity axis.
        rms (float): Noise per pixel in same units as ``data``.

    Returns:
        ``M9`` (`ndarray`), ``dM9`` (`ndarray`):
            The velocity value of the peak value, ``M9``, and the associated
            uncertainty, ``dM9``.
    """
    M9 = velax[np.argmax(data, axis=0)]
    loc = np.where(np.max(data, axis=0)==0)
    M9[loc] = np.nan
    dM9 = 0.5 * abs(np.diff(velax).mean()) * np.ones(M9.shape)
    mask = np.max(data, axis=0) != 0.0
    return np.where(mask, M9, np.nan), np.where(mask, dM9, np.nan)


def collapse_percentiles(velax, data, rms):
    """
    Collapse the cube by taking intensity-weighted percentiles. This can be
    thought of as an extension to the first and second moments which are
    intensity weighted averages and standard deviations. The advantage here is
    that by looking at the 16th and 84th percentile individually, we learn
    something about the skewness of the line (i.e., red-shifted side compared
    to the blue-shifted side) without assuming a line profile.

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Masked intensity or brightness temperature array. The
            first axis must be the velocity axis.
        rms (float): Noise per pixel in same units as ``data``.

    Returns:
        tuple: Eight `ndarray` values: the intensity-weighted median
            (``wp50``, ``dwp50``), the blue- and red-shifted line widths
            (``wpdVb``, ``dwpdVb``, ``wpdVr``, ``dwpdVr``), and the line
            center based on the center of the 16th and 84th percentile
            (``wp1684``, ``dwp1684``).
    """
    from tqdm import tqdm

    # Dummy arrays.

    wp = np.ones((3, data.shape[1], data.shape[2]))
    dwp = np.ones(wp.shape)

    # Calculate the weighted percentiles.

    weights = np.cumsum(np.clip(data, a_min=0.0, a_max=None), axis=0)
    weights /= weights[-1]
    pcnts = np.array([0.16, 0.5, 0.84])

    # Calculate the offset in velocity axis for the cumuative sum.
    # This is because it calculates it for the value between velocity samples
    # so there's a half-channel offset in the returned percentiles.

    dv = 0.5 * np.diff(velax).mean()

    # Loop through the pixels fiding the right values.

    with tqdm(total=data.shape[1]*data.shape[2]) as pbar:
        for i in range(weights.shape[2]):
            for j in range(weights.shape[1]):
                if weights[-1, j, i] == 0.0:
                    wp[:, j, i] = np.nan
                    dwp[:, j, i] = np.nan
                else:
                    wgts = weights[:, j, i]
                    k = np.argmin(abs(wgts[:, None] - pcnts[None, :]), axis=0)
                    wp[:, j, i] = np.interp(pcnts, wgts, velax+dv)
                    dwp[:, j, i] = np.gradient(velax, wgts)[k] * rms
                pbar.update(1)

    # Calculate the useful quantities.

    wp50, dwp50 = wp[1], dwp[1]
    wpdVb, dwpdVb = np.sqrt(2) * (wp50 - wp[0]), np.hypot(dwp50, dwp[0])
    wpdVr, dwpdVr = np.sqrt(2) * (wp[2] - wp50), np.hypot(dwp[2], dwp50)
    wp1684 = 0.5 * (wp[0] + wp[2])
    dwp1684 = 0.5  * np.hypot(dwp[0], dwp[2])

    # Return the values.

    return wp50, dwp50, wpdVb, dwpdVb, wpdVr, dwpdVr, wp1684, dwp1684

def collapse_gaussian(velax, data, rms, indices=None, ncpu=1, **kwargs):
    r"""
    Collapse the cube by fitting a Gaussian line profile to each pixel. This
    function is a wrapper of `collapse_analytical` which provides more
    details about the arguments.

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Maksed intensity or brightness temperature array. The
            first axis must be the velocity axis.
        rms (float): Noise per pixel in same units as ``data``.
        indices (Optional[list]): A list of pixels described by
            ``(y_idx, x_idx)`` tuples to fit. If none are provided, will fit
            all pixels.
        ncpu (Optional[int]): Number of worker processes to use. Defaults to
            ``1`` (serial). Set higher to parallelise across CPU cores.

    Returns:
        tuple: Six `ndarray` values: the Gaussian center (``gv0``,
            ``dgv0``), the Doppler line width (``gdV``, ``dgdV``) and the
            line peak (``gFnu``, ``dgFnu``).
    """
    return collapse_analytical(velax=velax, data=data, rms=rms,
                               model_function='gaussian', indices=indices,
                               ncpu=ncpu, **kwargs)


def collapse_gaussthick(velax, data, rms, indices=None, ncpu=1, **kwargs):
    r"""
    Collapse the cube by fitting a Gaussian line profile with an optically
    thick core to each pixel. This function is a wrapper of
    `collapse_analytical` which provides more details about the arguments.

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Maksed intensity or brightness temperature array. The
            first axis must be the velocity axis.
        rms (float): Noise per pixel in same units as ``data``.
        indices (Optional[list]): A list of pixels described by
            ``(y_idx, x_idx)`` tuples to fit. If none are provided, will fit
            all pixels.
        ncpu (Optional[int]): Number of worker processes to use. Defaults to
            ``1`` (serial). Set higher to parallelise across CPU cores.

    Returns:
        tuple: Eight `ndarray` values: the Gaussian center (``gtv0``,
            ``dgtv0``), the Doppler width (``gtdV``, ``dgtdV``), the line
            peak (``gtFnu``, ``dgtFnu``), and the effective optical depth
            (``gttau``, ``dgttau``).
    """
    return collapse_analytical(velax=velax, data=data, rms=rms,
                               model_function='gaussthick', indices=indices,
                               ncpu=ncpu, **kwargs)


def collapse_gausshermite(velax, data, rms, indices=None, ncpu=1, **kwargs):
    r"""
    Collapse the cube by fitting a Gaussian line profile with an optically
    thick core to each pixel. This function is a wrapper of
    `collapse_analytical` which provides more details about the arguments.

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Maksed intensity or brightness temperature array. The
            first axis must be the velocity axis.
        rms (float): Noise per pixel in same units as ``data``.
        indices (Optional[list]): A list of pixels described by
            ``(y_idx, x_idx)`` tuples to fit. If none are provided, will fit
            all pixels.
        ncpu (Optional[int]): Number of worker processes to use. Defaults to
            ``1`` (serial). Set higher to parallelise across CPU cores.

    Returns:
        tuple: Ten `ndarray` values: the Gaussian center (``ghv0``,
            ``dghv0``), the line peak (``ghFnu``, ``dghFnu``), the Doppler
            width (``ghdV``, ``dghdV``), the ``h3`` asymmetry term
            (``ghh3``, ``dghh3``) and the ``h4`` kurtosis term (``ghh4``,
            ``dghh4``).
    """
    return collapse_analytical(velax=velax, data=data, rms=rms,
                               model_function='gausshermite', indices=indices,
                               ncpu=ncpu, **kwargs)


def collapse_doublegauss(velax, data, rms, indices=None, ncpu=1, **kwargs):
    r"""
    Collapse the cube by fitting two Gaussian line profiles to each pixel.
    The first Gaussian component will be the peak of the two components.
    This function is a wrapper of `collapse_analytical` which provides more
    details about the arguments.

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Maksed intensity or brightness temperature array. The
            first axis must be the velocity axis.
        rms (float): Noise per pixel in same units as ``data``.
        indices (Optional[list]): A list of pixels described by
            ``(y_idx, x_idx)`` tuples to fit. If none are provided, will fit
            all pixels.
        ncpu (Optional[int]): Number of worker processes to use. Defaults to
            ``1`` (serial). Set higher to parallelise across CPU cores.

    Returns:
        tuple: Twelve `ndarray` values: the primary Gaussian center
            (``ggv0``, ``dggv0``), line peak (``ggFnu``, ``dggFnu``) and
            Doppler width (``ggdV``, ``dggdV``), followed by the same for
            the secondary component (``ggv0b``, ``dggv0b``, ``ggFnub``,
            ``dggFnub``, ``ggdVb``, ``dggdVb``).
    """
    p = collapse_analytical(velax=velax, data=data, rms=rms,
                            model_function='doublegauss', indices=indices,
                            ncpu=ncpu, **kwargs)
    idx = np.argmax(p[2::6], axis=0)
    pf = [np.where(idx, p[i+6], p[i]) for i in range(6)]
    pb = [np.where(idx, p[i], p[i+6]) for i in range(6)]
    return np.concatenate([pf, pb])
    

def collapse_analytical(velax, data, rms, model_function, indices=None,
                        ncpu=1, **kwargs):
    r"""
    Collapse the cube by fitting an analytical form to each pixel, including
    the option to use an MCMC sampler which has been found to be more forgiving
    when it comes to noisy data. Parallelism is handled at the per-pixel level
    so work is distributed evenly across workers.

    For more information on ``kwargs``, see the ``fit_spectrum`` documentation
    in ``mcmc_sampling.py``.

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Masked intensity or brightness temperature array. The
            first axis must be the velocity axis.
        rms (float): Noise per pixel in same units as ``data``.
        model_function (str): Name of the model function to fit to the data.
            Must be a function within ``profiles.py``.
        indices (Optional[list]): A list of pixels described by
            ``(y_idx, x_idx)`` tuples to fit. If none are provided, will fit
            all pixels.
        ncpu (Optional[int]): Number of worker processes to use. Defaults to
            ``1`` (serial execution with no pool overhead). Set to a higher
            value to parallelise across CPU cores.

    Returns:
        results_array (ndarray): An array containing all the fits. The first
            axis contains the mean and standard deviation of each posterior
            distribution.
    """
    from .mcmc_sampling import fit_cube
    from .profiles import free_params

    # Unless provided, fit all spaxels where there are some finite values.
    # By default, we require each spectrum to have twice as many finite values
    # as there are free parameters in the model being fit.

    if indices is None:
        indices = _get_finite_pixels(data, 2.0 * free_params(model_function))

    # Fit each pixel, distributing work evenly across ncpu workers.

    results = fit_cube(velax, data, rms, model_function, indices,
                       ncpu=ncpu, **kwargs)
    results = results.reshape(results.shape[0], -1)

    # Populate arrays with results and return.

    results_arrays = np.ones((*data.shape[1:], results.shape[1])) * np.nan
    for idx, result in zip(indices, results):
        results_arrays[idx[0], idx[1]] = result
    results_arrays = np.rollaxis(results_arrays, -1, 0)
    return results_arrays


def collapse_quadratic(velax, data, rms):
    """
    Collapse the cube using the quadratic method presented in `Teague &
    Foreman-Mackey (2018)`_. Will return the line center, ``v0``, and the
    uncertainty on this, ``dv0``, as well as the line peak, ``Fnu``, and the
    uncertainty on that, ``dFnu``. This provides the sub-channel precision of
    :func:`bettermoments.collapse_cube.collapse_first` with the robustness to
    noise from :func:`bettermoments.collapse_cube.collapse_ninth`.

    .. _Teague & Foreman-Mackey (2018): https://iopscience.iop.org/article/10.3847/2515-5172/aae265

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Flux density or brightness temperature array. Assumes
            that the zeroth axis is the velocity axis.
        rms (float): Noise per pixel in same units as ``data``.

    Returns:
        ``v0`` (`ndarray`), ``dv0`` (`ndarray`), ``Fnu`` (`ndarray`), ``dFnu`` (`ndarray`):
            ``v0``, the line center in the same units as ``velax`` with ``dv0``
            as the uncertainty on ``v0`` in the same units as ``velax``.
            ``Fnu`` is the line peak in the same units as the
            ``data`` with associated uncertainties, ``dFnu``.
    """
    from bettermoments.quadratic import quadratic
    chan = np.diff(velax).mean()
    return np.squeeze(quadratic(data, x0=velax[0], dx=chan, uncertainty=rms))


def collapse_maximum(velax, data, rms):
    """
    A wrapper returning the result of both
    :func:`bettermoments.collapse_cube.collapse_eighth` and
    :func:`bettermoments.collapse_cube.collapse_ninth`.

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Flux densities or brightness temperature array. Assumes
            that the first axis is the velocity axis.
        rms (float): Noise per pixel in same units as ``data``.

    Returns:
        ``M8`` (`ndarray`), ``dM8`` (`ndarray`), ``M9`` (`ndarray`), ``dM9`` (`ndarray`):
            The peak value, ``M8``, and the associated uncertainty, ``dM8``.
            The velocity value of the peak value, ``M9``, and the associated
            uncertainty, ``dM9``.
    """
    M8, dM8 = collapse_eighth(velax=velax, data=data, rms=rms)
    M9, dM9 = collapse_ninth(velax=velax, data=data, rms=rms)
    return M8, dM8, M9, dM9


def collapse_width(velax, data, rms):
    r"""
    Returns an effective width, a rescaled ratio of the integrated intensity
    and the line peak. For a Gaussian line profile this would be the Doppler
    width as the total intensity is given by,

    .. math::
        M_0 = \sum_{i}^N I_i \, \Delta v_{{\rm chan},\,i}

    where :math:`\Delta v_i` and :math:`I_i` are the chanenl width and flux
    density at the :math:`i^{\rm th}` channel. If the line profile is Gaussian,
    then equally

    .. math::
        M_0 = \sqrt{\pi} \times F_{\nu} \times \Delta V

    where :math:`F_{\nu}` is the peak value of the line and :math:`\Delta V` is
    the Doppler width of the line. As :math:`M_0` and :math:`F_{\nu}` are
    readily calculated using
    :func:`bettermoments.collapse_cube.collapse_zeroth` and
    :func:`bettermoments.collapse_cube.collapse_quadratic`, respectively,
    :math:`\Delta V` can calculated through :math:`\Delta V = M_{0} \, / \,
    (\sqrt{\pi} \, F_{\nu})`. This should be more robust against noise than
    second moment maps.

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Flux densities or brightness temperature array. Assumes
            that the first axis is the velocity axis.
        rms (float): Noise per pixel in same units as ``data``.

    Returns:
        ``dV`` (`ndarray`), ``ddV`` (`ndarray`):
            The effective velocity dispersion, ``dV`` and ``ddV``, the
            associated uncertainty.
    """
    M0, dM0 = collapse_zeroth(velax=velax, data=data, rms=rms)
    _, _, Fnu, dFnu = collapse_quadratic(velax=velax, data=data, rms=rms)
    dV = M0 / Fnu / np.sqrt(np.pi)
    ddV = dV * np.hypot(dFnu / Fnu, dM0 / M0)
    return abs(dV), abs(ddV)


# -- HELPER FUNCTIONS -- #


def available_collapse_methods():
    """Prints the available methods for collapsing the datacube."""
    funcs = ['zeroth', 'first', 'second', 'eighth', 'ninth',
             'maximum', 'quadratic', 'width', 'percentiles', 'gaussian',
             'gaussthick', 'gausshermite', 'doublegauss']
    txt = 'Available methods are:\n'
    txt += '\n'
    txt += '\t {:12} (integrated intensity)\n'
    txt += '\t {:12} (intensity weighted average velocity)\n'
    txt += '\t {:12} (intensity weighted velocity dispersion)\n'
    txt += '\t {:12} (peak intensity)\n'
    txt += '\t {:12} (velocity channel of peak intensity)\n'
    txt += '\t {:12} (both collapse_eighth and collapse_ninth)\n'
    txt += '\t {:12} (quadratic fit to peak intensity)\n'
    txt += '\t {:12} (effective width for a Gaussian profile)\n'
    txt += '\t {:12} (intesity weighted percentiles)\n'
    txt += '\t {:12} (gaussian fit)\n'
    txt += '\t {:12} (gaussian with optically thick core fit)\n'
    txt += '\t {:12} (gaussian-hermite expansion fit)\n'
    txt += '\t {:12} (double gaussian fit)\n'
    txt += '\n'
    txt += 'Call the function with `collapse_{{method_name}}`.'
    print(txt.format(*funcs))


def collapse_method_products(method):
    """Prints the products from each ``collapse_method``."""
    returns = {}
    returns['zeroth'] = 'M0, dM0'
    returns['first'] = 'M1, dM1'
    returns['second'] = 'M2, dM2'
    returns['eighth'] = 'M8, dM8'
    returns['ninth'] = 'M9, dM9'
    returns['maximum'] = 'M8, dM8, M9, dM9'
    returns['quadratic'] = 'v0, dv0, Fnu, dFnu'
    returns['width'] = 'dV, ddV'
    returns['percentiles'] = 'wp50, dwp50, wpdVb, dwpdVb, wpdVr, dwpdVr, '
    returns['percentiles'] += 'wp1684, dwp1684'
    returns['gaussian'] = 'gv0, dgv0, gdV, dgdV, gFnu, dgFnu'
    returns['gaussthick'] = 'gtv0, dgtv0, gtdV, dgtdV, gtFnu, dgtFnu, '
    returns['gaussthick'] += 'gttau, dgttau'
    returns['gausshermite'] = 'ghv0, dghv0, ghdV, dghdV, ghFnu, dghFnu, '
    returns['gausshermite'] += 'ghh3, dghh3, ghh4, dghh4'
    returns['doublegauss'] = 'ggv0, dggv0, ggdV, dggdV, ggFnu, dggFnu,'
    returns['doublegauss'] += 'ggv0b, dggv0b, ggdVb, dggdVb, ggFnub, dggFnub'
    try:
        return returns[method]
    except KeyError:
        print('`{}` not found.'.format(method))
        available_collapse_methods()


def check_finite_errors(moments):
    """
    Check if the errors are finite, and if not, replace them with
    nearest-neighbor interpolated values.

    Args:
        moments (tuple): Tuple of arrays in ``(value, error, value, error, ...)``
            order, as returned by one of the ``collapse_*`` functions.

    Returns:
        verified_moments (tuple): Same structure as ``moments`` with any
            non-finite error values replaced by interpolated finite errors.
    """
    verified_moments = ()
    assert len(moments) % 2 == 0, "Odd number of moments."
    for value, error in zip(moments[::2], moments[1::2]):
        if not np.all(np.isfinite(value) == np.isfinite(error)):
            value, error = _interpolate_finite_errors(value, error)
        verified_moments += (value, error)
    return verified_moments


def _interpolate_finite_errors(value, error, fill_value=1.0):
    """Replace NaN errors with interpolated finite errors."""
    from scipy.interpolate import NearestNDInterpolator

    # Check the moments are square images.

    assert value.ndim == 2
    assert value.shape == error.shape
    assert value.shape[0] == value.shape[1]

    # If there are no finite values at all, replace with `fill_value`.

    if not np.any(np.isfinite(error)):
        return value, np.ones(value.shape) * fill_value
    
    # Find all finite errors to use for interpolation.

    axis = np.arange(value.shape[0])
    xx, yy = np.meshgrid(axis, axis)
    xx, yy, zz = xx.flatten(), yy.flatten(), error.flatten()
    mask = np.isfinite(zz)
    xx, yy, zz = xx[mask], yy[mask], zz[mask]

    # Interpolate errors onto regular grid and return where `value` is finite.

    xi, yi = np.meshgrid(axis, axis)
    zi = NearestNDInterpolator(np.array([yy, xx]).T, zz)
    zi = zi(np.array([xi, yi]).T)
    return value, np.where(np.isfinite(value), zi, np.nan)



def _get_finite_pixels(data, min_finite=3):
    """
    Returns the (yidx, xidx) tuple for each of the pixels which have at least
    ``min_finite`` finite samples along the zeroth axis. A good rule of thumb
    is twice the number of free parameters for the model.

    Args:
        data (array): The data that will be used for the fitting.
        min_finite (optional[int]): Minimum number of finite samples along the
            zeroth axis. Must be positive.

    Returns:
        indices: A list of (yidx, xidx) tuples of all finite pixels.
    """
    assert min_finite > 0, "Must have at least one finite sample to fit."""
    finite_spaxels = np.sum(data != 0.0, axis=0) > int(min_finite)
    indices = np.indices(data[0].shape).reshape(2, data[0].size).T
    return indices[finite_spaxels.flatten()]
