# polBpy
Dispersion Analysis of Dust-Polarimetric Data and Magnetic Field (**B**) strength.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11414008.svg)](https://doi.org/10.5281/zenodo.11414008)

## Description
***polBpy*** is an open-source library for the analysis of dust polarimetric data (Stokes *I, Q, U* maps; polarization angle $\phi$ and fraction *p*) with the purpose of studying the magneto-turbulent state of the gas and determining the magnitude of the plane-of-sky (POS) component of the magnetic field $B_{\rm POS}$.

The development of this library was supported by the NASA/SOFIA Archival Research Program (Grant #09_0537, USRA for Villanova University).

## Features

This library is divided into two main modules:

1. The *dispersion* module which contains routines for performing polarization-angle dispersion calculations. Capabilities include:
  - Calculating the two-point dispersion (structure) function (Houde+09).
  - Calculating the autocorrelation function (Houde+09).
  - MCMC fitting and parameter determination (Guerra+21).
  - Local dispersion ($\mathcal{S}$; TBI).
2. The *DCF* module contains functions for calculating $B_{POS}$ using multiple David-Chandrasekhar-Fermi (DCF) approximations:
 - Classical (Davis52,Chandrasekhar-Fermi53).
 - Compressional (Skalidis+21)
 - Large-scale flow, shear flow (Lopez-Rodriguez+21,Guerra+23) 

Both the *dispersion* and *DCF* modules have the capabilities to perform corresponding analysis on a pixel-to-pixel basis resulting in maps of magnetoturbulent quantities and POS magnetic field strength.

## Installation
(Always download and install the latest tagged version!)

```
pip install git+https://github.com/jorgueagui/polBpy.git
```

or

```
git clone --depth 1 --branch v0.1.2 https://github.com/jorgueagui/polBpy.git
cd polBpy
pip install . 
```

### Required packages
(setup.py will install if not present)
- Numpy
- Scipy
- Astropy
- Matplotlib
- Emcee
- Joblib
- George
- Uncertainties
- Corner

## Tutorials
A series of Jupyter notebooks can be found [here.](https://github.com/jorgueagui/polBpy/tree/main/tutorials) They show examples of basic and advanced usage of this library.
- Tutorial I: Calculation of single-value $B_{POS}$ using all DCF approximations.
- Tutorial II: Calculation of range-values and maps of $B_{\rm POS}$ using all DCF approximations.
- Tutorial III: Example of dispersion analysis for a single region.
- Tutorial IV: Example of pixel-to-pixel dispersion analysis.
- Tutorial V: Map making of $B_{\rm POS}$ using DCF.

## References

A list of relevant references can be found [here.](https://github.com/jorgueagui/polBpy/blob/main/refs.md)

## Licensing

The use of this library is regulated by the MIT license. Details of this license can be found [here.](https://github.com/jorgueagui/polBpy/blob/main/License.md)

## Citation

If you use this package for a publication, please cite it as: *polBpy: a Python package for the analysis of dust polarimetric observations*, Jordan A. Guerra, David T. Chuss, Dylan Paré. DOI: 10.5281/zenodo.11414008. 2024.

## Other Links

- SOFIA/HAWC+ Archive at IPAC: [SOFIA/IRSA.](https://irsa.ipac.caltech.edu/applications/sofia/?__action=layout.showDropDown&)

## Version History
- 0.1.0, June 16, 2024. First version.
- 0.1.1, June 23, 2024. Minor patches in Tutorial V.
- 0.1.2, October 12, 2024. Add the option to output the MCMC samples to mcmc_fit. Updated Tutorial III.

## Issues
For any issues found with this package, please report to jordan.guerra [at] gmail.com.
