"""
Analytical forms to fit to the spectra. All functions take the velocity axis
as the first argument, then the free parameters. All functions have a ``_cont``
version which includes a baseline offset (the final free parameter).
"""

import numpy as np


def free_params(model_function):
    """
    Return the number of free parameters for the given model function.

    Args:
        model_function (str): Name of the model function, e.g.
            ``'gaussian'``, ``'gaussthick'``, ``'gausshermite'``,
            ``'doublegauss'``, or any of their ``_cont`` variants.

    Returns:
        int: Number of free parameters in the model.
    """
    nparams = {'gaussian': 3, 'gaussian_cont': 4, 'gaussthick': 4,
               'gaussthick_cont': 5, 'doublegauss': 6, 'doublegauss_cont': 7,
               'gausshermite': 5, 'gausshermite_cont': 6}
    try:
        return nparams[model_function]
    except KeyError:
        raise ValueError("Unknown model function '{}'; must be one of {}."
                         .format(model_function, sorted(nparams)))


def gaussian(x, *params):
    """
    Gaussian function with Doppler width.

    Args:
        x (arr): Velocity axis in [m/s].
        params (tuple): The line center in [m/s], the line Doppler width in
            [m/s] and the line peak in [Jy/beam].

    Returns:
        model (arr): Model spectrum in [Jy/beam].
    """
    assert len(params) == free_params('gaussian')
    model = params[2] * np.exp(-((x - params[0]) / params[1])**2)
    return model


def gaussian_cont(x, *params):
    """
    Gaussian function with Doppler width.

    Args:
        x (arr): Velocity axis in [m/s].
        params (tuple): The line center in [m/s], the line Doppler width in
            [m/s], the line peak in [Jy/beam] and the continuum offset in
            [Jy/beam].

    Returns:
        model (arr): Model spectrum in [Jy/beam].
    """
    assert len(params) == free_params('gaussian_cont')
    model = gaussian(x, *params[:-1]) + params[-1]
    return model


def doublegauss(x, *params):
    """
    Two Gaussian components with individual widths. Note that the model is
    the *maximum* of the two components at each velocity, not their sum: the
    two components are assumed to be optically thick layers along the line of
    sight (e.g. the front and back sides of a disk) rather than co-added,
    optically thin emission.

    Args:
        x (arr): Velocity axis in [m/s].
        params (tuple): The line center in [m/s], the line Doppler width in
            [m/s] and the line peak in [Jy/beam] for each of the two components.

    Returns:
        model (arr): Model spectrum in [Jy/beam].
    """
    assert len(params) == free_params('doublegauss')
    gaussian_a = gaussian(x, *params[:3])
    gaussian_b = gaussian(x, *params[3:])
    return np.max([gaussian_a, gaussian_b], axis=0)


def doublegauss_cont(x, *params):
    """
    The ``doublegauss`` function with continuum offset. See ``doublegauss``
    for more details.

    Args:
        x (arr): Velocity axis in [m/s].
        params (tuple): The line center in [m/s], the line Doppler width in
            [m/s] and the line peak in [Jy/beam] for each of the two
            components. The final parameter is the continuum offset in
            [Jy/beam].

    Returns:
        model (arr): Model spectrum in [Jy/beam].
    """
    assert len(params) == free_params('doublegauss_cont')
    return doublegauss(x, *params[:-1]) + params[-1]


def gaussthick(x, *params):
    r"""
    Gaussian profile with non-negligible optical depth,

    .. math::
        I(v) = I_{\nu} \big(1 - \exp(\mathcal{G}(v, v0, \Delta V, \tau))\big) \, / \, (1 - \exp(-\tau))

    where :math:`\mathcal{G}` is a Gaussian function. Note that :math:`\tau` is
    forced to be non-negative, but negative values will be clipped to 0.

    Args:
        x (arr): Velocity axis in [m/s].
        params (tuple): The line center in [m/s], the line Doppler width in
            [m/s], the line peak in [Jy/beam] and the optical depth.

    Returns:
        model (arr): Model spectrum in [Jy/beam].
    """
    assert len(params) == free_params('gaussthick')
    tau_peak = np.clip(params[3], a_min=1e-8, a_max=None)
    tau = gaussian(x, params[0], params[1], tau_peak)
    model = params[2] * (1.0 - np.exp(-np.clip(tau, a_min=0.0, a_max=1e10)))
    return model / (1.0 - np.exp(-tau_peak))


def gaussthick_cont(x, *params):
    """
    The ``gaussthick`` function with continuum offset. See ``gaussthick``
    for more details.

    Args:
        x (arr): Velocity axis in [m/s].
        params (tuple): The line center in [m/s], the line Doppler width in
            [m/s], the line peak in [Jy/beam], the optical depth and the
            continuum offset in [Jy/beam].

    Returns:
        model (arr): Model spectrum in [Jy/beam].
    """
    assert len(params) == free_params('gaussthick_cont')
    return gaussthick(x, *params[:-1]) + params[-1]


def _H3(x):
    """Third Hermite polynomial."""
    return (2 * x**3 - 3 * x) * 3**-0.5


def _H4(x):
    """Fourth Hermite polynomial."""
    return (4 * x**4 - 12 * x**2 + 3) * 24**-0.5


def gausshermite(x, *params):
    """
    Gauss-Hermite expansion with the Doppler width. This allows for a flexible
    line profile that purely a Gaussian, where the ``h3`` and ``h4`` terms
    quantify the skewness and kurtosis of the line as in
    `van der Marel & Franx (1993)`_.

    .. _van der Marel & Franx (1993): https://ui.adsabs.harvard.edu/abs/1993ApJ...407..525V/abstract

    Args:
        x (arr): Velocity axis in [m/s].
        params (tuple): The line center in [m/s], the Doppler width in
            [m/s], the line peak in [Jy/beam], the ``h3`` skewness
            coefficient and the ``h4`` kurtosis coefficient.

    Returns:
        model (arr): Model spectrum in [Jy/beam].
    """
    assert len(params) == free_params('gausshermite')
    xx = 2**0.5 * (x - params[0]) / params[1]
    model = params[2] * np.exp(-xx**2 / 2)
    model *= 1.0 + params[3] * _H3(xx) + params[4] * _H4(xx)
    return model


def gausshermite_cont(x, *params):
    """
    The ``gausshermite`` function with continuum offset. See ``gausshermite``
    for more details.

    Args:
        x (arr): Velocity axis in [m/s].
        params (tuple): The line center in [m/s], the Doppler width in
            [m/s], the line peak in [Jy/beam], the ``h3`` skewness
            coefficient, the ``h4`` kurtosis coefficient and the continuum
            offset in [Jy/beam].

    Returns:
        model (arr): Model spectrum in [Jy/beam].
    """
    assert len(params) == free_params('gausshermite_cont')
    xx = 2**0.5 * (x - params[0]) / params[1]
    model = params[2] * np.exp(-xx**2 / 2)
    model *= 1.0 + params[3] * _H3(xx) + params[4] * _H4(xx)
    model += params[5]
    return model


def build_cube(x, moments, method):
    """
    From a list of (N, M) moment maps, construct a model data cube on the
    coordinate systems of the provided cube.

    Args:
        x (arr): Velocity axis in [m/s].
        moments (array): Array of best-fit parameters.
        method (str): Method used to decompose the data.

    Returns:
        cube (array): 3D data cube.
    """

    # Test to see if the number of free parameters is correct.

    if moments.shape[0] != 2 * free_params(method):
        raise ValueError("Incorrect number of free parameters.")

    # Build the cube. We do this explicitly because I don't know how to
    # broadcast and unpack at once and perhaps leaves the option for more
    # complex emission profiles.

    if method == 'gaussian':
        v0, dV, Fnu = moments[::2]
        cube = gaussian(x[:, None, None],
                        v0[None, :, :],
                        dV[None, :, :],
                        Fnu[None, :, :])
    elif method == 'gaussthick':
        v0, dV, Fnu, tau = moments[::2]
        cube = gaussthick(x[:, None, None],
                          v0[None, :, :],
                          dV[None, :, :],
                          Fnu[None, :, :],
                          tau[None, :, :])
    elif method == 'gausshermite':
        v0, dV, Fnu, h3, h4 = moments[::2]
        cube = gausshermite(x[:, None, None],
                            v0[None, :, :],
                            dV[None, :, :],
                            Fnu[None, :, :],
                            h3[None, :, :],
                            h4[None, :, :])
    elif method == 'doublegauss':
        v0, dV, Fnu, v0b, dVb, Fnub = moments[::2]
        cube = doublegauss(x[:, None, None],
                           v0[None, :, :],
                           dV[None, :, :],
                           Fnu[None, :, :],
                           v0b[None, :, :],
                           dVb[None, :, :],
                           Fnub[None, :, :])
    else:
        raise ValueError(f"Cannot build cube for method {method}.")

    # Return the cube.

    return cube
