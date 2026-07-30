"""
Tests for bettermoments.io: safe output paths, header propagation and unit
handling.
"""

import numpy as np
import pytest
from astropy.io import fits

from bettermoments.io import (_output_path, _read_velocity_axis,
                              _write_header, load_cube, save_to_FITS)
from .conftest import VELAX, NV, make_cube_data, write_cube


# -- Output paths (C1) -- #

def test_output_path_basic():
    assert _output_path('/a/cube.fits', '_M0') == '/a/cube_M0.fits'


def test_output_path_case_insensitive_extensions():
    assert _output_path('/a/cube.FITS', '_M0') == '/a/cube_M0.fits'
    assert _output_path('/a/cube.fit', '_M0') == '/a/cube_M0.fits'


def test_output_path_fits_in_directory_name():
    out = _output_path('/data/run.fits.bak/cube.fits', '_M0')
    assert out == '/data/run.fits.bak/cube_M0.fits'


def test_output_path_never_returns_input():
    with pytest.raises(ValueError):
        _output_path('/a/cube.fits', '')


def test_output_path_bare_prefix():
    assert _output_path('/a/prefix', '_M0') == '/a/prefix_M0.fits'


# -- Cube loading -- #

def test_load_cube(cube_path):
    data, velax = load_cube(cube_path)
    assert data.shape == (NV,) + data.shape[1:]
    assert np.all(np.isfinite(data))  # NaNs filled with zeros
    assert np.allclose(velax, VELAX, rtol=1e-5)


def test_velocity_axis_kms_conversion():
    """CUNIT3 in km/s must be converted to m/s (M7)."""
    header = fits.Header()
    header['NAXIS3'] = NV
    header['CTYPE3'] = 'VELO-LSR'
    header['CRVAL3'] = VELAX[0] / 1e3
    header['CDELT3'] = np.diff(VELAX).mean() / 1e3
    header['CRPIX3'] = 1.0
    header['CUNIT3'] = 'km/s'
    assert np.allclose(_read_velocity_axis(header), VELAX, rtol=1e-6)


def test_velocity_axis_frequency_ghz():
    """A GHz frequency axis must give the same velocities as the Hz one."""
    nu0 = 230.538e9
    freqs = nu0 * (1.0 - VELAX / 2.99792458e8)
    header = fits.Header()
    header['NAXIS3'] = NV
    header['CTYPE3'] = 'FREQ'
    header['CRVAL3'] = freqs[0]
    header['CDELT3'] = np.diff(freqs).mean()
    header['CRPIX3'] = 1.0
    header['CUNIT3'] = 'Hz'
    header['RESTFRQ'] = nu0
    velax_hz = _read_velocity_axis(header)
    header['CRVAL3'] = freqs[0] / 1e9
    header['CDELT3'] = np.diff(freqs).mean() / 1e9
    header['CUNIT3'] = 'GHz'
    velax_ghz = _read_velocity_axis(header)
    assert np.allclose(velax_hz, velax_ghz, rtol=1e-8)


# -- Header propagation (M5, M6, m8) -- #

def test_write_header_propagates_wcs_keys(cube_path):
    header = _write_header(cube_path, bunit='m/s')
    assert header['RADESYS'] == 'ICRS'
    assert header['SPECSYS'] == 'LSRK'
    assert header['PC1_1'] == 1.0
    assert header['PC2_2'] == 1.0
    assert header['RESTFRQ'] == pytest.approx(230.538e9)
    assert header['BUNIT'] == 'm/s'


def test_write_header_legacy_radecsys(tmp_path):
    data = make_cube_data(pad_nan=False)
    path = write_cube(tmp_path / 'cube.fits', data)
    with fits.open(path, mode='update') as hdul:
        del hdul[0].header['RADESYS']
        hdul[0].header['RADECSYS'] = 'FK5'
    header = _write_header(path, bunit='m/s')
    assert header['RADESYS'] == 'FK5'


def test_write_header_no_restfreq_not_invented(tmp_path):
    data = make_cube_data(pad_nan=False)
    path = write_cube(tmp_path / 'cube.fits', data)
    with fits.open(path, mode='update') as hdul:
        del hdul[0].header['RESTFRQ']
    header = _write_header(path, bunit='m/s')
    assert 'RESTFRQ' not in header
    assert 'RESTFREQ' not in header


# -- save_to_FITS -- #

def test_save_to_fits_roundtrip(cube_path, velax):
    from bettermoments.methods import collapse_zeroth
    data, velax = load_cube(cube_path)
    moments = collapse_zeroth(velax, data, 0.05)
    save_to_FITS(np.array(moments), 'zeroth', cube_path)
    for product in ['M0', 'dM0']:
        out = cube_path.replace('.fits', '_{}.fits'.format(product))
        saved = fits.getdata(out)
        assert saved.shape == data.shape[1:]
        assert fits.getheader(out)['BUNIT'] == 'Jy/beam m/s'


def test_save_to_fits_respects_outname(cube_path, tmp_path):
    from bettermoments.methods import collapse_zeroth
    data, velax = load_cube(cube_path)
    moments = collapse_zeroth(velax, data, 0.05)
    outname = str(tmp_path / 'custom')
    save_to_FITS(np.array(moments), 'zeroth', cube_path, outname=outname)
    assert (tmp_path / 'custom_M0.fits').exists()
    assert (tmp_path / 'custom_dM0.fits').exists()
