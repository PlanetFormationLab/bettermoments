"""
Shared fixtures: small synthetic FITS cubes with known line properties.
"""

import numpy as np
import pytest
from astropy.io import fits


NV, NY, NX = 60, 12, 16
V0, DV, FNU = 2000.0, 400.0, 1.0
RMS = 0.05
VELAX = np.linspace(0.0, 4000.0, NV)


def make_cube_data(seed=0, pad_nan=True):
    """A Gaussian line plus noise, optionally NaN-padded like a CASA cube."""
    rng = np.random.default_rng(seed)
    line = FNU * np.exp(-((VELAX[:, None, None] - V0) / DV)**2)
    taper = np.exp(-0.5 * (np.hypot(*np.meshgrid(
        np.arange(NX) - NX / 2, np.arange(NY) - NY / 2)) / 4.0)**2)
    data = line * taper[None] + rng.normal(0.0, RMS, (NV, NY, NX))
    if pad_nan:
        data[:, :2, :] = np.nan
        data[:, :, :2] = np.nan
    return data


def write_cube(path, data):
    header = fits.Header()
    header['BUNIT'] = 'Jy/beam'
    header['BMAJ'] = 1e-4
    header['BMIN'] = 1e-4
    header['BPA'] = 0.0
    header['CTYPE1'] = 'RA---SIN'
    header['CRVAL1'] = 45.0
    header['CDELT1'] = -2.8e-5
    header['CRPIX1'] = NX / 2
    header['CUNIT1'] = 'deg'
    header['CTYPE2'] = 'DEC--SIN'
    header['CRVAL2'] = 30.0
    header['CDELT2'] = 2.8e-5
    header['CRPIX2'] = NY / 2
    header['CUNIT2'] = 'deg'
    header['CTYPE3'] = 'VELO-LSR'
    header['CRVAL3'] = VELAX[0]
    header['CDELT3'] = np.diff(VELAX).mean()
    header['CRPIX3'] = 1.0
    header['CUNIT3'] = 'm/s'
    header['RESTFRQ'] = 230.538e9
    header['SPECSYS'] = 'LSRK'
    header['RADESYS'] = 'ICRS'
    header['PC1_1'] = 1.0
    header['PC2_2'] = 1.0
    fits.writeto(str(path), data.astype(np.float32), header, overwrite=True)
    return str(path)


@pytest.fixture()
def cube_path(tmp_path):
    """Path to a synthetic FITS cube with NaN padding."""
    return write_cube(tmp_path / 'cube.fits', make_cube_data())


@pytest.fixture()
def velax():
    return VELAX.copy()


@pytest.fixture()
def clean_data():
    """Noise-free masked line data (zeros off-line, like CLI-masked data)."""
    line = FNU * np.exp(-((VELAX[:, None, None] - V0) / DV)**2)
    data = np.where(line > 1e-3 * FNU, line, 0.0)
    return np.broadcast_to(data, (NV, NY, NX)).copy()


@pytest.fixture()
def noisy_data():
    return make_cube_data(seed=1, pad_nan=False)
