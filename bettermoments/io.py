"""
All the functions to deal with data I/O.
"""

import scipy.constants as sc
from astropy.io import fits
import numpy as np


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
        data = data[int(stokes)]
    return np.where(np.isfinite(data), data, fill_value)


def _get_velax(path):
    """Read the velocity axis information."""
    return _read_velocity_axis(fits.getheader(path))


def _get_bunits(path):
    """Return the dictionary of units for each collapse_function result."""
    bunits = {}
    flux_unit = fits.getheader(path)['bunit']

    # method='first'

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


def _read_rest_frequency(header):
    """Read the rest frequency in [Hz]."""
    try:
        nu = header['restfreq']
    except KeyError:
        try:
            nu = header['restfrq']
        except KeyError:
            nu = header['crval3']
    return nu


def _read_velocity_axis(header):
    """Wrapper for _velocityaxis and _spectralaxis."""
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
    return header['crval3'] + specax * header['cdelt3']


def _collapse_beamtable(path):
    """Returns the median beam from the CASA beam table if present."""
    header = fits.getheader(path)
    if header.get('CASAMBM', False):
        try:
            beam = fits.open(path)[1].data
            beam = np.max([b[:3] for b in beam.view()], axis=0)
            return beam[0] / 3600., beam[1] / 3600., beam[2]
        except IndexError:
            print('WARNING: No beam table found despite CASAMBM flag.')
            return abs(header['cdelt1']), abs(header['cdelt2']), 0.0
    try:
        return header['bmaj'], header['bmin'], header['bpa']
    except KeyError:
        return abs(header['cdelt1']), abs(header['cdelt2']), 0.0


def _get_pix_per_beam(path):
    """Returns the number of pixels per beam FWHM."""
    bmaj, _, _ = _collapse_beamtable(path)
    return bmaj / abs(fits.getheader(path)['cdelt1'])


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
        new_header['BUNIT'] = header['BUNIT']
    for i in [1, 2]:
        for val in ['NAXIS', 'CTYPE', 'CRVAL', 'CDELT', 'CRPIX', 'CUNIT']:
            key = '%s%d' % (val, i)
            if key in header.keys():
                new_header[key] = header[key]
    try:
        new_header['RESTFRQ'] = header['RESTFRQ']
    except KeyError:
        try:
            new_header['RESTFREQ'] = header['RESTFREQ']
        except KeyError:
            new_header['RESTFREQ'] = 0.0
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
    try:
        new_header['RADESYS'] = header['RADSYS']
    except KeyError:
        pass
    if 'EQUINOX' in new_header.keys() and 'RADSYS' in new_header.keys():
        print("WARNING: Both 'EQUINOX' and 'RADSYS' found in header.")

    new_header['COMMENT'] = 'made with bettermoments'
    return new_header


def _save_smoothed_data(data, args):
    """Save the smoothed data for inspection."""
    header = fits.getheader(args.path, copy=True)
    header['COMMENT'] = 'smoothed data used for moment map creation'
    header['COMMENT'] = 'made with bettermoments'
    header['COMMENT'] = '-smooth {}'.format(args.smooth)
    header['COMMENT'] = '-polyorder {}'.format(args.polyorder)
    new_path = args.path.replace('.fits', '_smoothed_data.fits')
    fits.writeto(new_path, data, header, overwrite=args.nooverwrite,
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
    new_path = args.path.replace('.fits', '_mask.fits')
    fits.writeto(new_path, data, header, overwrite=args.nooverwrite,
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
    new_path = args.path.replace('.fits', '_channel_count.fits')
    fits.writeto(new_path, data, header, overwrite=args.nooverwrite,
                 output_verify='silentfix')


def _save_threshold_mask(data, args):
    """Save the smoothed data for inspection."""
    header = fits.getheader(args.path, copy=True)
    header['COMMENT'] = 'user-defined threshold mask'
    header['COMMENT'] = 'made with bettermoments'
    header['COMMENT'] = '-clip {}'.format(args.clip)
    header['COMMENT'] = '-smooththreshold {}'.format(args.smooththreshold)
    header['COMMENT'] = '-combine {}'.format(args.combine)
    new_path = args.path.replace('.fits', '_threshold_mask.fits')
    fits.writeto(new_path, data, header, overwrite=args.nooverwrite,
                 output_verify='silentfix')


def _save_channel_mask(data, args):
    """Save the user-defined channel mask for inspection."""
    header = fits.getheader(args.path, copy=True)
    header['COMMENT'] = 'user-defined channel mask'
    header['COMMENT'] = 'made with bettermoments'
    header['COMMENT'] = '-lastchannel {}'.format(args.lastchannel)
    header['COMMENT'] = '-firstchannel {}'.format(args.firstchannel)
    new_path = args.path.replace('.fits', '_channel_mask.fits')
    fits.writeto(new_path, data, header, overwrite=args.nooverwrite,
                 output_verify='silentfix')


def _save_user_mask(data, args):
    """Save the user-defined velocity mask for inspection."""
    header = fits.getheader(args.path, copy=True)
    header['COMMENT'] = 'user-defined mask'
    header['COMMENT'] = 'made with bettermoments'
    header['COMMENT'] = '-mask {}'.format(args.mask)
    header['COMMENT'] = '-combine {}'.format(args.combine)
    new_path = args.path.replace('.fits', '_user_mask.fits')
    fits.writeto(new_path, data, header, overwrite=args.nooverwrite,
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
    new_path = args.path.replace('.fits', '_{}_model.fits'.format(args.method))
    fits.writeto(new_path, model, header, overwrite=args.nooverwrite,
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
    for moment, output in zip(moments, outputs):
        header = _write_header(path=path, bunit=_get_bunits(path)[output])
        fits.writeto(outname.replace('.fits', '') + '_{}.fits'.format(output),
                     moment.astype(float), header, overwrite=overwrite,
                     output_verify='silentfix')
