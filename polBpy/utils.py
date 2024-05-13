#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 18:14:38 2023

@author: jguerraa
"""

import numpy as np
from astropy.convolution import convolve, convolve_fft
from astropy.convolution import Gaussian2DKernel
from scipy import ndimage

# Some functions

def sigma_convolve(x,sigma,osigma,pixsize,fwhm=False):
    #
    if fwhm == False:
        fact = np.sqrt(8*np.log(2))
    else:
        fact = 1.0
    #
    sigma = np.sqrt((sigma/fact)**2 - (osigma/fact)**2)
    sigma /= pixsize
    kernel = Gaussian2DKernel(sigma)#,x_size=41,y_size=41)
    #
    return convolve(x,kernel)

#===================================================================
# Get PSD 1D (total radial power spectrum)
#===================================================================
def GetPSD1D(psd2D):
    h  = psd2D.shape[0]
    w  = psd2D.shape[1]
    wc = w//2
    hc = h//2

    # create an array of integer radial distances from the center
    Y, X = np.ogrid[0:h, 0:w]
    r    = np.hypot(X - wc, Y - hc).astype(np.int)

    # SUM all psd2D pixels with label 'r' for 0<=r<=wc
    # NOTE: this will miss power contributions in 'corners' r>wc
    psd1D = ndimage.mean(psd2D, r, index=np.arange(0, wc))
    spds1D = ndimage.standard_deviation(psd2D, r, index=np.arange(0, wc))

    return psd1D,spds1D,r
#===================================================================

# FIND THE WIDTH AT HALF MAX
def HWHM(X,Y,S):
    half_max = 0.499
    d = np.sign(half_max - Y[0:-1]) - np.sign(half_max - Y[1:])
    m1 = np.where(d < 0)
    mm = m1[0]
    y0 = Y[mm]
    sy0 = S[mm]
    y1 = Y[mm+1]
    sy1 = S[mm+1]
    x0 = X[mm]
    x1 = X[mm+1]
    hwhm = x0 + (0.5-y0)*(x1-x0)/(y1-y0)
    one = (0.5-y0)*(x1-x0)/(y1-y0)**2 - (x1-x0)/(y1-y0)
    two = (0.5-y0)*(x1-x0)/(y1-y0)**2
    shwhm = np.sqrt((one*sy0)**2 + (two*sy1)**2)
    return hwhm, shwhm

import sys

def update_progress(progress):
    # update_progress() : Displays or updates a console progress bar
    ## Accepts a float between 0 and 1. Any int will be converted to a float.
    ## A value under 0 represents a 'halt'.
    ## A value at 1 or bigger represents 100%
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2} \n".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()
    
def rms_val(vec):
    rms = np.sqrt(np.nanmean(vec**2))
    return rms
    
def rms_var(vec):
    rms = (2.*np.std(vec)**4)/(len(vec)-1)
    return np.sqrt(rms)
    
    
    