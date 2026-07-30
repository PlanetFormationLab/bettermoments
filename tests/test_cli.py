"""
End-to-end tests of the command-line interface.
"""

import sys

import numpy as np
import pytest
from astropy.io import fits

from bettermoments.collapse_cube import main


def run_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, 'argv', ['bettermoments'] + list(args))
    main()


def test_cli_zeroth(cube_path, monkeypatch):
    run_cli(monkeypatch, cube_path, '-method', 'zeroth', '--silent')
    M0 = fits.getdata(cube_path.replace('.fits', '_M0.fits'))
    assert np.nanmax(M0) > 0
    header = fits.getheader(cube_path.replace('.fits', '_M0.fits'))
    assert header['BUNIT'] == 'Jy/beam m/s'
    assert header['RADESYS'] == 'ICRS'


def test_cli_quadratic_with_acf(cube_path, monkeypatch, capsys):
    # A generous -rms keeps enough off-source pixels for the ACF estimate on
    # this small synthetic cube.
    run_cli(monkeypatch, cube_path, '-method', 'quadratic', '--acf',
            '-rms', '0.5')
    out = capsys.readouterr().out
    assert 'Estimating spectral noise ACF' in out
    v0 = fits.getdata(cube_path.replace('.fits', '_v0.fits'))
    assert np.any(np.isfinite(v0))


def test_cli_never_overwrites_input(cube_path, monkeypatch):
    """A .FITS extension must not lead to the input being overwritten (C1)."""
    import shutil
    upper = cube_path[:-10] + 'cube2.FITS'
    shutil.copy(cube_path, upper)
    before = fits.getdata(upper).copy()
    run_cli(monkeypatch, upper, '-method', 'zeroth', '--silent',
            '--returnmask')
    assert np.array_equal(fits.getdata(upper), before, equal_nan=True)


def test_cli_acf_unsupported_method_raises(cube_path, monkeypatch):
    with pytest.raises(ValueError, match='not supported'):
        run_cli(monkeypatch, cube_path, '-method', 'percentiles', '--acf')


def test_cli_unknown_method_rejected_early(cube_path, monkeypatch):
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, cube_path, '-method', 'quadractic')


def test_cli_import_does_not_disable_warnings():
    import warnings
    import bettermoments  # noqa: F401
    assert not any(f[0] == 'ignore' and f[2] is Warning and f[1] is None
                   for f in warnings.filters[:1])
