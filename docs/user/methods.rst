.. module:: bettermoments.methods
.. methods

API
===

Here we describe all the functions used to collapse the spectral cube which are
typically called by the command line interface. However, importing these into
your workflow may be useful.

In general, for the generated moment maps, ``MX``, where ``X`` is an integer
denotes a statistical moment. For the non-traditional methods, ``v0``, ``dV``
and ``Fnu`` represent the line center, width and peak, respectively.

.. note::
    The convolution for ``smooththreshold`` is currently experimental and is
    work in progress. If things look suspicious, please raise an issue.



Moment Maps
-----------

Implementation of traditional moment-map methods. See the `CASA documentation
<https://casa.nrao.edu/docs/CasaRef/image.moments.html>`_ for more information.

.. autofunction:: bettermoments.methods.collapse_zeroth

.. autofunction:: bettermoments.methods.collapse_first

.. autofunction:: bettermoments.methods.collapse_second

.. autofunction:: bettermoments.methods.collapse_eighth

.. autofunction:: bettermoments.methods.collapse_ninth

.. autofunction:: bettermoments.methods.collapse_maximum


Non-Traditional Methods
-----------------------

.. autofunction:: bettermoments.methods.collapse_quadratic

.. autofunction:: bettermoments.methods.collapse_width

.. autofunction:: bettermoments.methods.collapse_percentiles


(Higher Order) Gaussian Fits
----------------------------

.. autofunction:: bettermoments.methods.collapse_gaussian

.. autofunction:: bettermoments.methods.collapse_gaussthick

.. autofunction:: bettermoments.methods.collapse_gausshermite

.. autofunction:: bettermoments.methods.collapse_doublegauss

.. autofunction:: bettermoments.methods.collapse_analytical
