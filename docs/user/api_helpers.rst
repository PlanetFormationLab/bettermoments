Helper Functions
================

Functions for data I/O, masking and pre-processing that are useful when
building custom workflows outside of the command-line interface.

Data I/O
--------

.. autofunction:: bettermoments.io.load_cube

.. autofunction:: bettermoments.io.save_to_FITS

Pre-processing & Masking
------------------------

.. autofunction:: bettermoments.collapse_cube.estimate_RMS

.. autofunction:: bettermoments.collapse_cube.smooth_data

.. autofunction:: bettermoments.collapse_cube.get_channel_mask

.. autofunction:: bettermoments.collapse_cube.get_user_mask

.. autofunction:: bettermoments.collapse_cube.get_threshold_mask

.. autofunction:: bettermoments.collapse_cube.get_combined_mask

Spectral Noise Correlation
--------------------------

Tools for handling spectrally correlated noise (e.g.\ from Hanning smoothing
in the imaging pipeline). These underpin the ``--acf`` command-line flag and
the ``acf=`` keyword on the ``collapse_*`` functions, and can also be used
directly when building custom workflows.

.. autofunction:: bettermoments.collapse_cube.estimate_spectral_acf

.. autofunction:: bettermoments.collapse_cube.build_spectral_covariance

Utilities
---------

.. autofunction:: bettermoments.methods.check_finite_errors
