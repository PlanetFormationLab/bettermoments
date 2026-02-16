bettermoments
=============

   .. image:: _static/TWHya.png
      :width: 435px
      :height: 435px
      :scale: 100 %
      :alt: TWHya
      :align: center


``bettermoments`` creates moment maps of spectral line data and their
associated uncertainties. The command-line interface makes it as seamless as
possible to make all the traditional moment maps, in addition other, oftentimes
more useful, maps. In addition to the many traditional statistical moments,
``bettermoments`` contains many alternative ways collapse the cube.

The ``quadratic`` method is described in detail in `Teague & Foreman-Mackey
(2018)`_ while the uncertainties associated with more typical moment maps can
be found in `Teague (2019)`_.

Installation
------------

Installation is as easy as cloning the git repository then move to the home
directory,

.. code-block:: bash

    pip install bettermoments

and you're done!

.. toctree::
   :maxdepth: 2
   :caption: Contents

   user/methods
   user/api_helpers
   user/api_profiles
   user/command_line
   tutorials/Cookbook_1
   user/citations

.. _Teague (2019): https://iopscience.iop.org/article/10.3847/2515-5172/ab2125
.. _Teague & Foreman-Mackey (2018): https://iopscience.iop.org/article/10.3847/2515-5172/aae265
