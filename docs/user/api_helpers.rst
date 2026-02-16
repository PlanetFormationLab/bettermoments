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

Utilities
---------

.. autofunction:: bettermoments.methods.check_finite_errors
