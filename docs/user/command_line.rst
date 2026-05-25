.. command_line

Command Line Interface
======================

This is the preferred way to interact with ``bettermoments``. If the install
has gone successfully, you should be able to run ``bettermoments`` from any
directory using,

.. code-block:: bash

    bettermoments path/to/cube.fits

Which, by default, will use the :func:`collapse_quadratic`
function to calculate line center and line peak maps. This will automatically
extract the data array and spectral axis from the cube and provide them to the
appropriate functions.

.. warning::

    The command line interface will automatically overwrite any files with the
    same name. Make sure that you move or rename old files which you want to
    keep or use the ``--nooverwrite`` flag.

Different Methods
-----------------

To change the method applied to collapse the cube, use the :code:`-method [name]`
flag, where the names of the functions are found in the `API`.
For example, to calculate the zeroth moment map,

.. code-block:: bash

    bettermoments path/to/cube.fits -method zeroth

which will produce a ``*M0.fits`` file with the uncertainties in ``*dM0.fits``.

Smoothing
---------

It is sometimes useful to smooth along the spectral axis of the data prior to
calculating the requested moment map. While this can remove high frequency noise
and usually lead to a better determination of the desired statistic, any level
of smoothing will reduce peak values of a spectrum, so any statistic based on
the absolute value of the data will be under-estimated.

Smoothing is achieved with the :code:`-smooth [window]` flag, where the window
size as the number of channels is given. By default this is a top hat function
which is applied along the spectral axis prior to any other calculations
(including the estimation of the RMS).

It is possible to request a Savitzky-Golay filter using the additional
:code:`-polyorder [order]` flag which denotes the order of the polynomal used
in the filter. Note that this needs to be greater than 1, but also two less than
the window size.

.. code-block:: bash

    bettermoments path/to/cube/fits -smooth 5

will smooth the data with a top-hat kernel with a width of 5 channels while

.. code-block:: bash

    bettermoments path/to/cube/fits -smooth 5 -polyorder 2

will smooth the data with Savitzky-Golay filter with a window size of 5 channels
and use a polynomial of order 2.

Masking
-------

When making a moment map it is often useful, sometimes necessary, to mask the
data to reduce the noise in the resulting image. There are a couple of different
options in ``bettermoments`` to do this.

Channel Selection
^^^^^^^^^^^^^^^^^

The most straight forward is a simple channel selection using the ``-firstchannel [chan]``
and ``-lastchannel [chan]`` arguments. By default these span the entire cube range.
For example,

.. code-block:: bash

    bettermoments path/to/cube.fits -method zeroth -firstchannel 5 -lastchannel 10

would create a zeroth moment map using the channels 5 to 10 inclusively.

Threshold Clipping
^^^^^^^^^^^^^^^^^^

One of the most common approaches is to apply a 'sigma clip', essentially
masking any pixels below some user-specified threshold, usual in untis of the
background RMS. In ``bettermoments`` this is applied with the ``-clip [value]``
argument. For example,

.. code-block:: bash

    bettermoments path/to/cube.fits -method zeroth -clip 2

would calculate a zeroth moment map out of all the pixels which have an absolute
value of greater than or equal to ``2 * RMS``. The background RMS is automatically
calculated using the central 50% of the pixels in the first and last 5 channels.
The number of channels used for this estimation can be changed with the
``-noisechannels`` argument. Rather than calculating the RMS automatically, you
can specify their own value with the ``-rms`` argument. Note that internally
the RMS is assumed to be homogeneous, both spatially and spectrally.

If you want include asymmetric bounds you can include two ``-clip`` values. For
example,

.. code-block:: bash

    bettermoments path/to/cube.fits -method zeroth -clip -3 2

would mask out all pixel values between ``-3 * RMS`` and ``2 * RMS``.

A threshold mask like the above can sometimes leave sharp boundaries if you have
large spatial gradients in the intensity. To counter this it is possible to
convolve the threshold mask with a 2D Gaussian kernel to smooth these edges
with the ``-smooththreshold [width]`` argument where the width is given in units
of the beam FWHM (or pixel scale if a beam isn't provided). Internally this will
make a copy of the data, convolve with the appropriate kernel, then generate
a boolean mask where the convolved map meets the specified ``-clip`` criteria.

.. warning::

    If you choose to smooth the threshold map, remember that the RMS in this
    image will be reduced due to the smoothing. The automatic calculation of
    the RMS is done before the smoothing of the map so it will be appropriate
    to provide a user-specified one with ``-rms [value]``.

User-Defined Masks
^^^^^^^^^^^^^^^^^^

Sometimes you may want to include a user-defined mask, such at the CLEAN mask
used when imaging interferometric data. As long as the mask has the same shape
as the data in the image cube you can include this with,

.. code-block:: bash

    bettermoments path/to/cube.fits -mask path/to/mask.fits

Combing Masks
^^^^^^^^^^^^^

If you've specified both a user-defined mask and provided a ``clip`` value then
``bettermoments`` will combine the two masks by default using ``AND``. If you
would rather choose a less conservative ``OR`` combination then you can include
the ``-combine or`` argument.

Returning Masks
^^^^^^^^^^^^^^^

It is often useful to have a copy of the mask used to generate the moment map
such that you can overplot it in channel maps to help make sense of what you're
seeing. To do this, use the ``--returnmask`` flag.

Correlated Noise (ACF)
----------------------

By default ``bettermoments`` assumes that the noise in adjacent spectral
channels is independent. For real interferometric cubes this is rarely true:
the imaging pipeline typically applies some form of spectral smoothing
(e.g. Hanning) which correlates the noise between neighbouring channels. When
that happens the diagonal-only uncertainty under-estimates the true error,
because variance from correlated channels accumulates rather than averages
down.

The ``--acf`` flag accounts for this by estimating the normalised spectral
autocorrelation function (ACF) of the noise from off-source pixels and
propagating the uncertainty through the full Toeplitz covariance
:math:`C_{ij} = \sigma^2 \, \rho(|i - j|)` rather than just its diagonal,

.. code-block:: bash

    bettermoments path/to/cube.fits -method zeroth --acf

The ACF is estimated from pixels whose peak absolute intensity across the
spectral axis is below ``2 * rms`` — i.e. clearly off-source sightlines — and
is truncated at the first lag that falls inside the white-noise band
:math:`\pm 2 / \sqrt{N_{\rm chan}}`. The estimated ACF and the corresponding
variance inflation :math:`S = 1 + 2 \sum_{\tau > 0} \rho(\tau)` are printed to
the terminal so you can sanity-check the result.

The ``--acf`` flag is currently supported by ``zeroth``, ``first``,
``second``, ``quadratic``, ``width``, ``gaussian``, ``gaussthick``,
``gausshermite``, and ``doublegauss``. Only the uncertainty maps change; the
moment values themselves are unaffected.

.. note::

    ACF estimation needs enough off-source spectra to give a stable estimate.
    For cubes where the line fills most of the spatial extent, the
    automatically-selected sample may be small — inspect the printed ACF and,
    if needed, estimate it externally with
    :func:`bettermoments.estimate_spectral_acf` on a more appropriate region.

Help
----

For help with the exact command line options, use

.. code-block:: bash

    bettermoments --help
