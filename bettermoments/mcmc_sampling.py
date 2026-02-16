"""
Functions to apply the fitting in an MCMC manner.
"""

import numpy as np
from tqdm import tqdm
from .profiles import free_params


# -- MCMC Functions -- #

def lnprior(params, priors):
    """
    Evaluate the log-prior probability for the given parameters.

    Args:
        params (array): Parameter values to evaluate.
        priors (list): List of prior specifications, one per parameter. Each
            entry is a list ``[arg1, arg2, type]`` where ``type`` is
            ``'flat'`` or ``'gaussian'``.

    Returns:
        float: Log-prior probability.
    """
    lnp = 0.0
    for param, prior in zip(params, priors):
        lnp += parse_prior(param, prior)
    return lnp


def parse_prior(p, prior):
    """
    Evaluate a single prior for a parameter value.

    Args:
        p (float): Parameter value.
        prior (list): Prior specification ``[arg1, arg2, type]``. For
            ``'flat'`` priors, ``arg1`` and ``arg2`` are the lower and upper
            bounds. For ``'gaussian'`` priors, ``arg1`` is the mean and
            ``arg2`` is the standard deviation.

    Returns:
        float: Log-prior probability for the parameter.
    """
    if prior[-1] == 'flat':
        valid = np.logical_and(p >= prior[0], p <= prior[1])
        return np.where(valid, -np.log(prior[1] - prior[0]), -np.inf)
    elif prior[-1] == 'gaussian':
        return -0.5 * ((p - prior[0]) / prior[1])**2
    else:
        raise ValueError("Unknown prior type '{}'.".format(prior[-1]))


def lnlike(params, x, y, dy, model_function):
    """
    Evaluate the log-likelihood for the given parameters.

    Args:
        params (array): Model parameter values.
        x (array): Velocity axis.
        y (array): Observed intensities.
        dy (array): Uncertainties on the observed intensities.
        model_function (callable): The model function to evaluate.

    Returns:
        float: Log-likelihood value.
    """
    y_mod = model_function(x, *params)
    return -0.5 * np.sum(((y - y_mod) / dy)**2)


def lnpost(params, x, y, dy, priors, model_function):
    """
    Evaluate the log-posterior probability for the given parameters.

    Args:
        params (array): Model parameter values.
        x (array): Velocity axis.
        y (array): Observed intensities.
        dy (array): Uncertainties on the observed intensities.
        priors (list): List of prior specifications (see ``lnprior``).
        model_function (callable): The model function to evaluate.

    Returns:
        float: Log-posterior probability.
    """
    lnp = lnprior(params, priors)
    if ~np.isfinite(lnp):
        return lnp
    return lnp + lnlike(params, x, y, dy, model_function)


# -- Sampling Functions -- #


def fit_cube(velax, data, rms, model_function, indices=None, **kwargs):
    """
    Cycle through the provided indices fitting each spectrum. Only spectra
    which have more more than twice the number of pixel compared to the number
    of free parameters in the model will be fit.

    For more information on ``kwargs``, see the ``fit_spectrum`` documentation.

    Args:
        velax (ndarray): Velocity axis of the cube.
        data (ndarray): Intensity or brightness temperature array. The first
            axis must be the velocity axis.
        rms (float): Noise per pixel in same units as ``data``.
        model_function (str): Name of the model function to fit to the data.
            Must be a function withing ``profiles.py``.
        indices (list): A list of pixels described by ``(y_idx, x_idx)`` tuples
            to fit. If none are provided, will fit all pixels.

    Returns:
        fits (ndarray): A ``(Npix, Ndim, 2)`` shaped array of the fits and
            associated uncertainties. The uncertainties will be interleaved
            with the best-fit values.
    """

    # Check the inputs.

    assert velax.size == data.shape[0], "Incorrect velax and data shape."
    try:
        _ = import_function(model_function)
        nparams = free_params(model_function)
    except ValueError as error_message:
        print(error_message)
    if indices is None:
        indices = np.indices(data[0].shape).reshape(2, data[0].size).T
    indices = np.atleast_2d(indices)
    indices = indices.T if indices.shape[1] != 2 else indices

    # Default axes.

    x = velax.copy()
    dy = np.ones(x.size) * rms

    # Cycle through the pixels and apply the fitting.

    fits = np.ones((indices.shape[0], 2, nparams)) * np.nan
    with tqdm(total=indices.shape[0]) as pbar:
        for i, idx in enumerate(indices):
            y = data[:, idx[0], idx[1]].copy()
            mask = np.logical_and(np.isfinite(y), y != 0.0)
            if len(y[mask]) > nparams * 2:
                fits[i] = fit_spectrum(x[mask], y[mask], dy[mask],
                                       model_function, **kwargs)
            pbar.update(1)
    return np.swapaxes(fits, 1, 2)


def fit_spectrum(x, y, dy, model_function, p0=None, priors=None, nwalkers=None,
                 nburnin=500, nsteps=500, mcmc='emcee', scatter=1e-3,
                 niter=1, returns='default', plots=False, **kwargs):
    """
    Fit the provided spectrum with ``model_function``. If ``mcmc`` is not
    specified, the results of the ``scipy.optimize.curve_fit`` optimization
    will be returned instead. Using ``plots=True`` is only recommended for
    debugging and when this function is not called as part of ``fit_cube``.

    Args:
        x (array): Velocity axis.
        y (array): Intensity axis.
        dy (array): Uncertainties on the intensity.
        model_function (str): Name of the model to fit to the spectrum. Must be
            a function defined in ``profiles.py``.
        p0 (Optional[array]): An array of starting positions.
        priors (Optioinal[list]): User-defined priors.
        nwalkers (Optional[int]): Number of walkers for the MCMC.
        nburnin (Optional[int]): Number of steps to discard as burnin.
        nsteps (Optional[int]): Number of steps to take beyond ``burnin`` to
            sample the posterior distribution.
        mcmc (Optional[str/None]): The MCMC package to import EnsembleSampler
            from: ``'emcee'`` or ``'zeus'``. If ``None``, will skip the MCMC
            sampling and return the ``scipy.optimize.curve_fit`` results.
        scatter (Optional[float]): Scatter to apply to ``p0`` values for the
            walkers.
        niter (Optional[int]): Number of MCMC iterations to run, each time
            using the median of the posterior samples as the new ``p0``.
            Between each iteration ``scipy.optimize.curve_fit`` is not called.
        returns (Optional[str]): What the function returns. ``'default'`` will
            return ``(mu, sig)`` for each parameter, ``'percentiles'`` will
            return the 16th, 50th and 84th percentiles for each marginalized
            posterior distribution, ``'samples'`` will return all posterior
            samples, while ``'sampler'`` will return the EnsembleSampler.
        plots (Optioanl[bool]): If ``True``, make diagnost plots.
        free_params (Optional[int]): The number of free parameters expected.

    Returns:
        Various depending on the value of ``returns``.
    """

    # Set the starting positions.

    if p0 is None:
        p0 = estimate_p0(x, y, model_function)

    # Try a parameter optimization.

    p0, cvar = optimize_p0(x, y, dy, model_function, p0)
    if mcmc is None:
        return p0, cvar

    # Run the sample niter times.

    priors = default_priors(x, y, model_function) if priors is None else priors
    for _ in range(niter):
        sampler = run_sampler(x, y, dy, p0, priors, model_function, nwalkers,
                              nburnin, nsteps, mcmc, scatter, **kwargs)
        samples = sampler.get_chain(discard=nburnin, flat=True)
        p0 = np.median(samples, axis=0)

    # Make dianostic plots.

    if plots:
        diagnostic_plots(sampler, nburnin)

    # Return the requested statisitics.

    percentiles = np.percentile(samples, [16, 50, 84], axis=0)
    if returns == 'default':
        return p0, 0.5 * (percentiles[2] - percentiles[0])
    elif returns == 'percentiles':
        return percentiles
    elif returns == 'samples':
        return samples
    elif returns == 'sampler':
        return sampler
    else:
        raise ValueError("Unknown returns value {}.".format(returns))


def run_sampler(x, y, dy, p0, priors, model_function, nwalkers=None,
                nburnin=500, nsteps=500, mcmc='emcee', scatter=1e-3,
                **kwargs):
    """
    Build and run the MCMC ensemble sampler.

    Args:
        x (array): Velocity axis.
        y (array): Observed intensities.
        dy (array): Uncertainties on the observed intensities.
        p0 (array): Starting parameter values.
        priors (list): List of prior specifications (see ``lnprior``).
        model_function (str): Name of the model function in ``profiles.py``.
        nwalkers (Optional[int]): Number of walkers. Defaults to
            ``2 * len(p0)``.
        nburnin (Optional[int]): Number of burn-in steps.
        nsteps (Optional[int]): Number of production steps.
        mcmc (Optional[str]): MCMC backend, ``'emcee'`` or ``'zeus'``.
        scatter (Optional[float]): Fractional scatter applied to ``p0`` to
            generate initial walker positions.

    Returns:
        sampler: The ``EnsembleSampler`` instance after running.
    """

    # Select the MCMC backend.

    if mcmc == 'emcee':
        import emcee
        EnsembleSampler = emcee.EnsembleSampler
    elif mcmc == 'zeus':
        import zeus
        EnsembleSampler = zeus.EnsembleSampler
    else:
        raise ValueError("Unknown MCMC package '{}'.".format(mcmc))

    # Default parameters for the EnsembleSampler.

    nwalkers = len(p0) * 2 if nwalkers is None else nwalkers
    p0 = random_p0(p0, scatter, nwalkers)
    progress = kwargs.pop('progress', False)
    moves = kwargs.pop('moves', None)
    pool = kwargs.pop('pool', None)
    args = [x, y, dy, priors, import_function(model_function)]

    # Build, run and return the EnsembleSampler.

    sampler = EnsembleSampler(nwalkers, p0.shape[1], lnpost,
                              args=args, moves=moves, pool=pool)
    sampler.run_mcmc(p0, nburnin+nsteps, progress=progress,
                     skip_initial_state_check=True, **kwargs)
    return sampler


# -- Starting Positions -- #

def _estimate_x0(x, y):
    """Estimate the line center."""
    return x[np.nanargmax(y)]


def _estimate_dx(x, y):
    """Estimate the Doppler width."""
    yy = np.where(np.isfinite(y), y, 0.0)
    return np.trapz(yy, x) / np.nanmax(y) / np.sqrt(np.pi)


def estimate_p0(x, y, model_function):
    """
    Estimate initial parameter values from the spectrum.

    Args:
        x (array): Velocity axis.
        y (array): Observed intensities.
        model_function (str): Name of the model function in ``profiles.py``.

    Returns:
        list: Estimated starting parameter values.
    """
    p0 = [_estimate_x0(x, y), _estimate_dx(x, y), np.max(y)]
    if 'doublegauss' == model_function:
        v0b = np.average(x, weights=y) 
        p0 += [v0b, p0[1], y[abs(x - v0b).argmin()]]
    elif 'gaussthick' == model_function:
        p0 += [0.5]
    elif 'gausshermite' == model_function:
        p0 += [0.0, 0.0]
    if '_cont' in model_function:
        p0 += [0.0]
    return p0


def optimize_p0(x, y, dy, model_function, p0, **kwargs):
    """
    Optimize starting parameters using ``scipy.optimize.curve_fit``.

    Args:
        x (array): Velocity axis.
        y (array): Observed intensities.
        dy (array): Uncertainties on the observed intensities.
        model_function (str): Name of the model function in ``profiles.py``.
        p0 (array): Initial parameter guesses.

    Returns:
        p0 (array), cvar (array): Optimized parameters and the square root
            of the diagonal of the covariance matrix.
    """
    from scipy.optimize import curve_fit
    model_function = import_function(model_function)
    try:
        kwargs['maxfev'] = kwargs.pop('maxfev', 10000)
        p0, cvar = curve_fit(model_function, x, y, sigma=dy, p0=p0, **kwargs)
        cvar = np.diag(cvar)**0.5
    except RuntimeError:
        cvar = np.ones(len(p0)) * np.nan
    return p0, cvar


def random_p0(p0, scatter, nwalkers):
    """
    Generate initial walker positions by adding scatter to ``p0``.

    Args:
        p0 (array): Best-fit parameter values.
        scatter (float): Fractional scatter to apply.
        nwalkers (int): Number of walkers.

    Returns:
        ndarray: Array of shape ``(nwalkers, ndim)`` with initial positions.
    """
    p0 = np.squeeze(p0)
    dp0 = np.random.randn(nwalkers * len(p0)).reshape(nwalkers, len(p0))
    dp0 = np.where(p0 == 0.0, 1.0, p0)[None, :] * (1.0 + scatter * dp0)
    return np.where(p0[None, :] == 0.0, dp0 - 1.0, dp0)


# -- Prior Functions -- #

def _x0_prior(x):
    """Default x0 prior."""
    return [x.min(), x.max(), 'flat']


def _dx_prior(x):
    """Default dx prior."""
    return [0.0, 0.25 * abs(x.max() - x.min()), 'flat']


def _A_prior(y):
    """Default A prior."""
    return [0.0, 2.0 * np.nanmax(y), 'flat']


def _tau_prior():
    """Default tau prior."""
    return [0.0, 1e3, 'flat']


def _h3_prior():
    """Default h3 prior."""
    return [-10, 10, 'flat']


def _h4_prior():
    """Default h4 prior."""
    return [-10, 10, 'flat']


def _cont_prior(y):
    """Default cont prior."""
    return [-2.0 * np.nanstd(y), 2.0 * np.nanstd(y), 'flat']


def default_priors(x, y, model_function):
    """
    Return default flat priors based on the data range and model type.

    Args:
        x (array): Velocity axis.
        y (array): Observed intensities.
        model_function (str): Name of the model function in ``profiles.py``.

    Returns:
        list: List of prior specifications, one per free parameter.
    """
    priors = [_x0_prior(x), _dx_prior(x), _A_prior(y)]
    if 'multi' in model_function:
        priors += [_x0_prior(x), _dx_prior(x), _A_prior(y)]
    elif 'thick' in model_function:
        priors += [_tau_prior()]
    elif 'hermite' in model_function:
        priors += [_h3_prior(), _h4_prior()]
    if '_cont' in model_function:
        priors += [_cont_prior(y)]
    return priors


# -- Helper Functions -- #

def import_function(function_name):
    """Checks to see if the function can be imported."""
    from bettermoments import profiles
    maybe_function = getattr(profiles, function_name, None)
    if maybe_function is None:
        raise ValueError("Unknown function {}.".format(function_name))
    return maybe_function


def verify_fits(fits, free_params=None):
    """
    Fill all failed fitting attempts with NaNs.

    Args:
        fits (list): List of fit results, where each element is an array of
            parameter values.
        free_params (Optional[int]): Number of free parameters. If not
            provided, inferred from the first successful fit.

    Returns:
        ndarray: Array of fit results with failed fits replaced by NaN.
    """
    if free_params is None:
        for p in fits:
            if np.all(np.isfinite(p)):
                empty = np.ones(np.array(p).shape) * np.nan
                break
    else:
        empty = np.ones(free_params) * np.nan
    fits = [p if np.all(np.isfinite(p)) else empty for p in fits]
    return np.squeeze(fits)


def diagnostic_plots(sampler, nburnin, mcmc='emcee'):
    """
    Make diagnostic plots from the MCMC sampler, including walker traces
    and a corner plot of the posterior samples.

    Args:
        sampler: The ``EnsembleSampler`` instance after running.
        nburnin (int): Number of burn-in steps to discard.
        mcmc (Optional[str]): MCMC backend used, ``'emcee'`` or ``'zeus'``.
    """
    import matplotlib.pyplot as plt
    for s, sample in enumerate(sampler.get_chain().T):
        fig, ax = plt.subplots()
        for walker in sample:
            ax.plot(walker, alpha=0.1, color='k')
        ax.axvline(nburnin, ls=':', color='r')
    import corner
    samples = sampler.get_chain(discard=nburnin, flat=True)
    corner.corner(samples, title_fmt='.4f', bins=30,
                  quantiles=[0.16, 0.5, 0.84], show_titles=True)
