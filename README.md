# polBpy
Dispersion Analysis of Dust-Polarimetric Data and Magnetic Field (**B**) strength.

## Description
***polBpy*** is an open-source library for the analysis of dust polarimetric data (Stokes *I, Q, U* maps; polarization angle $\phi$ and fraction *p*) with the purpose of studying the magneto-turbulent state of the gas and determining the magnitude of the plane-of-sky (POS) component of the magnetic field $B_{POS}$.

The developement of this library was supported by the NASA/SOFIA Archival Research Program (Grant # USRA for Villanova University).

## Features

This libraryis divided into two main modules:

1. The *dispersion* module which contains routines for performing polarization-angle dispersion calculations. Capabilities inlcude:
  - Calculating the two-point dispersion (structure) function (Houde+09).
  - Calculating the autocorrelation function (Houde+09).
  - MCMC fitting and parameter determination (Guerra+21).
  - Local dispersion ($\mathcal{S}$; TBI).
2. The *DCF* module contains fucntions for calculating $B_{POS}$ using multiple David-Chandrasekhar-Fermi (DCF) approximations:
 - Classical (Davis52,Chandrasekhar-Fermi53).
 - Compressional (Skadilis+21)
 - Large-scale flow, shear flow (Lopez-Rodriguez+21,Guerra+23) 

Both the *dispersion* and *DCF* modules have the capabilities for applying the analysis on a pixel-to-pixel basis resulting in maps of variables.

## Installation
(TBF)
### Required packages
- Numpy
- Scipy
- Astropy
- Matplotlib
- emcee

## Tutorials
A series of Jupyter notebooks can be found [here.](https://github.com/jorgueagui/polBpy/tree/fbe89ea5aa79fb70be8148f458581906c2cc6af3/tutorials) They show examples of basic and advanced usage of this library.
- Tutotial I: Calculation of single-value $B_{POS}$ using all DCF approximations.
- Tutorial II: Calculation of range-values and maps of $B_{POS}$ using all DCF approximations.
- Tutorial III: Example of dispersion analysis for a single region.
- Tutorial IV: Example of pixel-to-pixel dispersion analysis.

## References

A list of relevant references can be found [here.](https://github.com/jorgueagui/polBpy/blob/9039d4af5d25c49130bf51be7fe0ce363424edcc/refs.md)

## Licensing

The use of this library is regulated by the XX lincense. Details of this license can be found [here.](https://github.com/jorgueagui/polBpy/blob/3d172ed7b52df5684b0ec117958172eb6f4c8679/License.md)

## Other Links

- SOFIA/HAWC+ Archive at IPAC ()
