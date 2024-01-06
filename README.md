# polBpy
Dispersion Analysis of Dust-Polarimetric Data and Magnetic Field (**B**) strength.

## Description
***polBpy*** is an open-source library for the analysis of dust polarimetric data (Stokes *I, Q, U* maps; polarization angle $\phi$ and fraction *p*) with the purpose of studying the magneto-turbulent state of the gas.
The developement of this library was supported by the NASA/SOFIA Archival Research Program (Grant # USRA for Villanova University).

## Features
This library includes routines for peforming/calculating/studying:
1. The polarization-angle dispersion:
  - Calculate the two-point dispersion function (Houde+09).
  - Local dispersion (S; TBD).
  - MCMC fitting and parameter determination.
2. The plane-of-sky (POS) **B**-field strength using multiple David-Chandrasekhar-Fermi (DCF) approximations:
 - Classical;
 - Compressional;
 - Shear flow;

Also inlcude routines to implement the dispersion analysis over regions and create maps of **B** values.

## Installation
(TBF)
### Required packages
- Numpy
- Scipy
- Astropy
- Matplotlib
- emcee
- 

## Tutorials
A series of Jupyter notebooks can be found here. They show examples of basic and advanced usage of this library.

## References

A list of relevant references can be found [here](https://github.com/jorgueagui/polBpy/blob/9039d4af5d25c49130bf51be7fe0ce363424edcc/refs.md)

## Licensing

## Other Links
