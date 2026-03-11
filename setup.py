#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import setuptools

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md")) as f:
    long_description = f.read()

setuptools.setup(
    name="bettermoments",
    version="1.9.4",
    author="Richard Teague & Daniel Foreman-Mackey",
    author_email='rteague@mit.edu',
    packages=["bettermoments"],
    url="https://github.com/PlanetFormationLab/bettermoments",
    license="LICENSE.md",
    description=("Robust moment map making."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy",
        "astropy",
        "argparse",
        "tqdm",
        "emcee>=3",
        "zeus-mcmc",
        ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'bettermoments = bettermoments.collapse_cube:main',
        ],
    }
)
