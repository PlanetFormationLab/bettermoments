# bettermoments

<p align='center'>
  <br/>
  <img src="https://github.com/PlanetFormationLab/bettermoments/blob/master/docs/_static/TWHya.png" width="435" height="435"><br/>
  <br>
  Measuring precise line-of-sight velocities from Doppler shifted lines is essential<br/>
  when looking for small scale deviations indicative of, for example, embedded planets.<br/>Do that with <b>bettermoments</b>.
  <br><br>
  <a href="https://doi.org/10.5281/zenodo.1419754"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.1419754.svg" alt="DOI"></a>
  <a href="http://ascl.net/1901.009"><img src="https://img.shields.io/badge/ascl-1901.009-blue.svg?colorB=262255" alt="ascl:1901.009" /></a>
  <a href='https://bettermoments.readthedocs.io/en/latest/?badge=latest'>
      <img src='https://readthedocs.org/projects/bettermoments/badge/?version=latest' alt='Documentation Status' />
  </a>
  <br><br><br><br>
</p>


## Installation

To install, we recommend using pip:

```bash
pip install bettermoments
```

## Usage

The easiest way to interface with ``bettermoments`` is through the command line. To use the default quadratic method simply use,

```bash
bettermoments path/to/cube.fits
```

while various other methods, discussed further in the documentation, can be accessed via the ``-method`` flag:

```bash
bettermoments path/to/cube.fits -method zeroth
```

A quick

```bash
bettermoments --help
```

will give a quick overview of what options there are.

## Documentation

More information can be found in the [documentation](https://bettermoments.readthedocs.io/en/latest/?badge=latest).
