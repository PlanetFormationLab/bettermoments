# Frequently Asked Questions


## _What methods are available?_

`bettermoments` has multiple methods for collapsing your datacube. Thus far the available options are:

#### Intensity

* **zeroth** - Simply integrated along the spectrum to return the total integrated intensity. More commonly known as the zeroth moment map.

Note that both **quadratic** and **maximum** also return the line peak if you want a radial brightness temperature profile.

#### Velocity / Line Center

* **quadratic** - As described in [Teague & Foreman-Mackey (2018)](http://iopscience.iop.org/article/10.3847/2515-5172/aae265/meta), this method fits a quadratic curve to the pixel of peak intensity and its two neighbouring pixels. This results in a line center and line peak.

* **maximum** - Similar to the **quadratic** method, but only uses the pixel of peak intensity. This limits precision of the line center to the velocity resolution of the data.

* **first** - The intensity weighted average velocity, or more commonly known as the first moment. This is very susceptable to noise in the spectrum so often requires clipping or masking the data to get nice results.

#### Velocity Dispersion

* **width** - An alternative to the typical second moment map. A rescaled ratio of the integrated intensity, calculated with **maximum**, divided by the line peak calculated by **quadratic**. For an intrinsic Gaussian line, this is equal to the Doppler width.

#### Analytical Fits

* _Coming Soon_

For more information on any of these functions, use the help: `bettermoments.collapse_cube.collapse_<method>?`.


## _What are the output files?_

Depending on the method you chose to collapse, differnent files will be returned. In general `_I0` is the total intensity, `_Fnu` is the peak flux density, `_v0` is the line center and `_dV` is the line width or velocity dispersion. There are also associated uncertainties, for example `_dv0` is the uncertainty on the line center.


## _Should I be smoothing my data beforehand?_

[Vakili & Hogg (2016)](https://arxiv.org/abs/1610.05873) showed that to get the most precise measure of a centroid, convolution with a Gaussian kernel with a width equal to the intrinsic width of the line is required. If this is invoked with the `-linewidth` argument in **quadratic**, note that the line peak will suffer. Depending on the SNR of the data, you may be able to get away without this convolution.

There is also a [Jupyter Notebook](https://github.com/PlanetFormationLab/bettermoments/blob/master/docs/notebooks/DetermineOptimalResolution.ipynb) which you can use to estimate the best resampling rate (for example using `mstransform` in `CASA` for radio data) to get the most precise line centroid.

## _Which method is most appropriate for my data?_

It depends on what you want to show. If you're worried that your line profile is complex, chances are that describing it with a single statistic is not the best thing to do and fitting a more appropriate analytical profile would be your best bet.


## _How does the quadratic method work?_

Rather than fitting the full line profile with an analytical expression, we fit a quadratic to the the peak pixel and its two adjacent pixels. For a single component, this has been shown in [Vakili & Hogg (2016)](https://arxiv.org/abs/1610.05873) to provide comparable precision to fitting the full spectrum with the true underlying profile. Thus, in the extremely likely scenario where the underlying profile is not known, this method will out perform more common methods, such as the intensity weighted average. This approach, including the correction to conserve flux, is also extensively discussed in Appendix C of [Courteau (1997)](https://arxiv.org/pdf/astro-ph/9709201.pdf).

We note that there are alternative implemenations of this method available, such as [NEMO](https://github.com/teuben/nemo).
