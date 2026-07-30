"""
All the functions to deal with data I/O.
"""

import os

import scipy.constants as sc
from astropy.io import fits
import numpy as np

__all__ = ['load_cube', 'save_to_FITS']


def _output_path(path, suffix):
    """
    Return ``path`` with its extension replaced by ``suffix + '.fits'``. The
    extension is stripped case-insensitively for ``.fits`` and ``.fit`` files;
    other paths (e.g. a bare prefix from ``-outname``) have the suffix
    appended. Raises a ``ValueError`` rather than ever returning a path equal
    to the input, which would risk overwriting the original data.
    """
    base, ext = os.path.splitext(path)
    if ext.lower() not in ('.fits', '.fit'):
        base = path
    new_path = base + suffix + '.fits'
    if os.path.abspath(new_path) == os.path.abspath(path):
        raise ValueError("Output path matches input path: '{}'.".format(path))
    return new_path


# -- READ DATA -- #


def load_cube(path, stokes=0):
    """
    Load a FITS data cube and return the data and velocity axis.

    Args:
        path (str): Path to the FITS cube.
        stokes (Optional[int]): Stokes index to select if the cube has a
            Stokes axis. Defaults to ``0``.

    Returns:
        data (ndarray), velax (ndarray): The data cube with non-finite values
            replaced by zero, and the velocity axis in [m/s].
    """
    return _get_data(path, stokes=stokes), _get_velax(path)


def _get_data(path, fill_value=0.0, stokes=0):
    """Read the FITS cube."""
    data = np.squeeze(fits.getdata(path))
    if data.ndim == 4:
        stokes = int(stokes)
        if not 0 <= stokes < data.shape[0]:
            raise ValueError("stokes={} is out of range for a cube with {} "
                             "Stokes components.".format(stokes,
                                                         data.shape[0]))
        data = data[stokes]
    return np.where(np.isfinite(data), data, fill_value)


def _get_velax(path):
    """Read the velocity axis information."""
    return _read_velocity_axis(fits.getheader(path))


def _get_bunits(path):
    """Return the dictionary of units for each collapse_function result."""
    bunits = {}
    flux_unit = fits.getheader(path).get('bunit', '')
    if flux_unit == '':
        print("WARNING: No BUNIT found in '{}'; ".format(path)
              + "intensity units will be blank in the output headers.")

    # method='zeroth'

    bunits['M0'] = '{} m/s'.format(flux_unit)
    bunits['dM0'] = '{} m/s'.format(flux_unit)

    # method='first'

    bunits['M1'] = 'm/s'
    bunits['dM1'] = 'm/s'

    # method='second'

    bunits['M2'] = 'm/s'
    bunits['dM2'] = 'm/s'

    # method='eighth'

    bunits['M8'] = '{}'.format(flux_unit)
    bunits['dM8'] = '{}'.format(flux_unit)

    # method='ninth'

    bunits['M9'] = 'm/s'
    bunits['dM9'] = 'm/s'

    # method='quadratic'

    bunits['v0'] = 'm/s'
    bunits['Fnu'] = '{}'.format(flux_unit)
    bunits['dv0'] = 'm/s'
    bunits['dFnu'] = '{}'.format(flux_unit)

    # method='width'

    bunits['dV'] = 'm/s'
    bunits['ddV'] = 'm/s'

    # method='percentiles'

    bunits['wp50'] = 'm/s'
    bunits['dwp50'] = 'm/s'
    bunits['wpdVb'] = 'm/s'
    bunits['dwpdVb'] = 'm/s'
    bunits['wpdVr'] = 'm/s'
    bunits['dwpdVr'] = 'm/s'
    bunits['wp1684'] = 'm/s'
    bunits['dwp1684'] = 'm/s'

    # method='gaussian'

    bunits['gv0'] = 'm/s'
    bunits['gFnu'] = '{}'.format(flux_unit)
    bunits['gdV'] = 'm/s'
    bunits['dgv0'] = 'm/s'
    bunits['dgFnu'] = '{}'.format(flux_unit)
    bunits['dgdV'] = 'm/s'

    # method='gaussthick'

    bunits['gtv0'] = 'm/s'
    bunits['gtFnu'] = '{}'.format(flux_unit)
    bunits['gtdV'] = 'm/s'
    bunits['gttau'] = ''
    bunits['dgtv0'] = 'm/s'
    bunits['dgtFnu'] = '{}'.format(flux_unit)
    bunits['dgtdV'] = 'm/s'
    bunits['dgttau'] = ''

    # method='gausshermite'

    bunits['ghv0'] = 'm/s'
    bunits['ghFnu'] = '{}'.format(flux_unit)
    bunits['ghdV'] = 'm/s'
    bunits['ghh3'] = ''
    bunits['ghh4'] = ''
    bunits['dghv0'] = 'm/s'
    bunits['dghFnu'] = '{}'.format(flux_unit)
    bunits['dghdV'] = 'm/s'
    bunits['dghh3'] = ''
    bunits['dghh4'] = ''

    # method='doublegauss'

    bunits['ggv0'] = 'm/s'
    bunits['ggFnu'] = '{}'.format(flux_unit)
    bunits['ggdV'] = 'm/s'
    bunits['dggv0'] = 'm/s'
    bunits['dggFnu'] = '{}'.format(flux_unit)
    bunits['dggdV'] = 'm/s'
    bunits['ggv0b'] = 'm/s'
    bunits['ggFnub'] = '{}'.format(flux_unit)
    bunits['ggdVb'] = 'm/s'
    bunits['dggv0b'] = 'm/s'
    bunits['dggFnub'] = '{}'.format(flux_unit)
    bunits['dggdVb'] = 'm/s'

    # Mask

    bunits['mask'] = 'bool'

    # Models

    bunits['gaussian_model'] = '{}'.format(flux_unit)
    bunits['gaussthick_model'] = '{}'.format(flux_unit)
    bunits['gausshermite_model'] = '{}'.format(flux_unit)
    bunits['doublegauss_model'] = '{}'.format(flux_unit)

    return bunits


def _spectral_unit_scale(header):
    """Return the factor converting the spectral axis (CUNIT3) to SI."""
    unit = header.get('cunit3', '').strip().lower().replace(' s-1', '/s')
    scales = {'': 1.0, 'm/s': 1.0, 'km/s': 1e3,
              'hz': 1.0, 'khz': 1e3, 'mhz': 1e6, 'ghz': 1e9}
    try:
        return scales[unit]
    except KeyError:
        print("WARNING: Unknown CUNIT3 '{}'; assuming SI units.".format(unit))
        return 1.0


def _read_rest_frequency(header):
    """Read the rest frequency in [Hz]."""
    try:
        nu = header['restfreq']
    except KeyError:
        try:
            nu = header['restfrq']
        except KeyError:
            nu = header['crval3'] * _spectral_unit_scale(header)
    return nu


def _read_velocity_axis(header):
    """Return the velocity axis in [m/s] (converting from frequency and/or
    non-SI units where necessary)."""
    if 'freq' in header['ctype3'].lower():
        specax = _read_spectral_axis(header)
        nu = _read_rest_frequency(header)
        velax = (nu - specax) * sc.c / nu
    else:
        velax = _read_spectral_axis(header)
    return velax


def _read_spectral_axis(header):
    """Returns the spectral axis in [Hz] or [m/s]."""
    specax = (np.arange(header['naxis3']) - header['crpix3'] + 1.0)
    specax = header['crval3'] + specax * header['cdelt3']
    return specax * _spectral_unit_scale(header)


def _collapse_beamtable(path):
    """Returns the largest beam from the CASA beam table if present."""
    header = fits.getheader(path)
    if header.get('CASAMBM', False):
        try:
            with fits.open(path) as hdul:
                beam = np.max([b[:3] for b in hdul[1].data.view()], axis=0)
            return beam[0] / 3600., beam[1] / 3600., beam[2]
        except IndexError:
            print('WARNING: No beam table found despite CASAMBM flag.')
            return abs(header['cdelt1']), abs(header['cdelt2']), 0.0
    try:
        return header['bmaj'], header['bmin'], header['bpa']
    except KeyError:
        print("WARNING: No beam information found in '{}'; ".format(path)
              + "assuming a pixel-sized beam in the output headers.")
        return abs(header['cdelt1']), abs(header['cdelt2']), 0.0


# -- WRITE DATA -- #


def _write_header(path, bunit):
    """Write a new header for the saved file."""
    header = fits.getheader(path, copy=True)
    new_header = fits.PrimaryHDU().header
    new_header['SIMPLE'] = True
    new_header['BITPIX'] = -64
    new_header['NAXIS'] = 2
    beam = _collapse_beamtable(path)
    new_header['BMAJ'] = beam[0]
    new_header['BMIN'] = beam[1]
    new_header['BPA'] = beam[2]
    if bunit is not None:
        new_header['BUNIT'] = bunit
    else:
        new_header['BUNIT'] = header.get('BUNIT', '')
    for i in [1, 2]:
        for val in ['NAXIS', 'CTYPE', 'CRVAL', 'CDELT', 'CRPIX', 'CUNIT']:
            key = '%s%d' % (val, i)
            if key in header.keys():
                new_header[key] = header[key]

    # Copy the WCS rotation matrix (PC or CD form) and pole keywords so that
    # rotated or skewed frames keep the correct astrometric solution.

    for i in [1, 2]:
        for j in [1, 2]:
            for form in ['PC{}_{}', 'CD{}_{}']:
                key = form.format(i, j)
                if key in header.keys():
                    new_header[key] = header[key]
    for key in ['LONPOLE', 'LATPOLE']:
        if key in header.keys():
            new_header[key] = header[key]

    for key in ['RESTFRQ', 'RESTFREQ']:
        if key in header.keys():
            new_header['RESTFRQ'] = header[key]
            break
    try:
        new_header['SPECSYS'] = header['SPECSYS']
    except KeyError:
        pass

    # This tries to import the correct coordinate system (i.e., not getting
    # confused between J2000 and ICRS coordinates).

    try:
        new_header['EQUINOX'] = header['EQUINOX']
    except KeyError:
        pass
    for key in ['RADESYS', 'RADECSYS']:
        if key in header.keys():
            new_header['RADESYS'] = header[key]
            break

    new_header['COMMENT'] = 'made with bettermoments'
    return new_header


def _save_smoothed_data(data, args):
    """Save the smoothed data for inspection."""
    header = fits.getheader(args.path, copy=True)
    header['COMMENT'] = 'smoothed data used for moment map creation'
    header['COMMENT'] = 'made with bettermoments'
    header['COMMENT'] = '-smooth {}'.format(args.smooth)
    header['COMMENT'] = '-polyorder {}'.format(args.polyorder)
    new_path = _output_path(args.outname or args.path, '_smoothed_data')
    fits.writeto(new_path, data, header, overwrite=args.overwrite,
                 output_verify='silentfix')


def _save_mask(data, args):
    """Save the combined mask for inspection."""
    header = fits.getheader(args.path, copy=True)
    header['COMMENT'] = 'mask used for moment map creation'
    header['COMMENT'] = 'made with bettermoments'
    header['COMMENT'] = '-lastchannel {}'.format(args.lastchannel)
    header['COMMENT'] = '-firstchannel {}'.format(args.firstchannel)
    header['COMMENT'] = '-mask {}'.format(args.mask)
    header['COMMENT'] = '-clip {}'.format(args.clip)
    header['COMMENT'] = '-smooththreshold {}'.format(args.smooththreshold)
    header['COMMENT'] = '-combine {}'.format(args.combine)
    new_path = _output_path(args.outname or args.path, '_mask')
    fits.writeto(new_path, data, header, overwrite=args.overwrite,
                 output_verify='silentfix')


def _save_channel_count(data, args):
    """Save the number of channels used in each pixel."""
    header = fits.getheader(args.path, copy=True)
    header['BUNIT'] = 'channels'
    header['COMMENT'] = 'number of channels used in each pixel'
    header['COMMENT'] = 'made with bettermoments'
    header['COMMENT'] = '-lastchannel {}'.format(args.lastchannel)
    header['COMMENT'] = '-firstchannel {}'.format(args.firstchannel)
    header['COMMENT'] = '-mask {}'.format(args.mask)
    header['COMMENT'] = '-clip {}'.format(args.clip)
    header['COMMENT'] = '-smooththreshold {}'.format(args.smooththreshold)
    header['COMMENT'] = '-combine {}'.format(args.combine)
    new_path = _output_path(args.outname or args.path, '_channel_count')
    fits.writeto(new_path, data, header, overwrite=args.overwrite,
                 output_verify='silentfix')


def _save_threshold_mask(data, args):
    """Save the smoothed data for inspection."""
    header = fits.getheader(args.path, copy=True)
    header['COMMENT'] = 'user-defined threshold mask'
    header['COMMENT'] = 'made with bettermoments'
    header['COMMENT'] = '-clip {}'.format(args.clip)
    header['COMMENT'] = '-smooththreshold {}'.format(args.smooththreshold)
    header['COMMENT'] = '-combine {}'.format(args.combine)
    new_path = _output_path(args.outname or args.path, '_threshold_mask')
    fits.writeto(new_path, data, header, overwrite=args.overwrite,
                 output_verify='silentfix')


def _save_channel_mask(data, args):
    """Save the user-defined channel mask for inspection."""
    header = fits.getheader(args.path, copy=True)
    header['COMMENT'] = 'user-defined channel mask'
    header['COMMENT'] = 'made with bettermoments'
    header['COMMENT'] = '-lastchannel {}'.format(args.lastchannel)
    header['COMMENT'] = '-firstchannel {}'.format(args.firstchannel)
    new_path = _output_path(args.outname or args.path, '_channel_mask')
    fits.writeto(new_path, data, header, overwrite=args.overwrite,
                 output_verify='silentfix')


def _save_user_mask(data, args):
    """Save the user-defined velocity mask for inspection."""
    header = fits.getheader(args.path, copy=True)
    header['COMMENT'] = 'user-defined mask'
    header['COMMENT'] = 'made with bettermoments'
    header['COMMENT'] = '-mask {}'.format(args.mask)
    header['COMMENT'] = '-combine {}'.format(args.combine)
    new_path = _output_path(args.outname or args.path, '_user_mask')
    fits.writeto(new_path, data, header, overwrite=args.overwrite,
                 output_verify='silentfix')


def _save_model(model, args):
    """
    Same the reconstructed model as a FITS cube. The filename will replace the
    ``.fits`` extension with ``{method_name}_model.fits``.

    Args:
        model (array): Model cube to save.
        method (str): Name of the collapse method used, e.g., ``'gaussian'`` if
            ``collapse_gaussian`` was used.
    """
    header = fits.getheader(args.path, copy=True)
    header['COMMENT'] = 'model image from -method {}'.format(args.method)
    header['COMMENT'] = 'made with bettermoments'
    new_path = _output_path(args.outname or args.path, '_{}_model'.format(args.method))
    fits.writeto(new_path, model, header, overwrite=args.overwrite,
                 output_verify='silentfix')


def save_to_FITS(moments, method, path, outname=None, overwrite=True):
    """
    Save the returned fits from ``collapse_{method_name}`` as FITS cubes.
    The filenames will replace the ``.fits`` extension with ``_{param}.fits``.

    Args:
        moments (array): Array of moment values from one of the collapse
            methods.
        method (str): Name of the collapse method used, e.g., ``'zeroth'`` if
            ``collapse_zeroth`` was used.
        path (str): Path of the original data cube to grab header information.
        outname (str): Filename prefix for the saved images. Defaults to the
            path of the provided FITS file.
        overwrite (Optional[bool]): Whether to overwrite files.
    """
    from .methods import collapse_method_products
    moments = np.squeeze(moments)
    assert moments.ndim == 3, "Unexpected shape of `moments`."
    outputs = collapse_method_products(method=method).split(',')
    outputs = [output.strip() for output in outputs]
    assert len(outputs) == moments.shape[0], "Unexpected number of outputs."
    outname = outname or path
    bunits = _get_bunits(path)
    for moment, output in zip(moments, outputs):
        header = _write_header(path=path, bunit=bunits[output])
        fits.writeto(_output_path(outname, '_{}'.format(output)),
                     moment.astype(float), header, overwrite=overwrite,
                     output_verify='silentfix')
