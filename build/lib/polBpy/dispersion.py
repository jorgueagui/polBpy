#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 14:33:06 2023

@author: jguerraa
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from polBpy import utils
import time
import sys

def autocorrelation(polflux,polflux_err,pixsize=False,mask=False,plots=False,hwhm=False):
    
    # This function calculates the 1D isotropic autoccorrelation of
    # the polarized flux in the emitting volume
    #
    if type(mask) != np.ndarray:
        mask = 1.0
        # all pixels considered -- not recommended.
        
    polflux *= mask
    # MAKE NaNs INTO ZEROS
    # Autocorrelation function fails if NaNs are in array
    polflux[~np.isfinite(polflux)] = 0.0
    
    if plots == True:
        # PLOT THE POL FLUX MAP USED
        plt.figure(0)
        plt.imshow(polflux,origin='lower')
        plt.title('Map of Polarized Flux')
    #
    # CALCULATE THE NORMALIZED 2D AUTO-CORRELATION 
    crr = signal.correlate2d(polflux,polflux,mode='full')
    crr = crr/np.nanmax(crr)

    # CALCULATE THE 1D AUTO-CORRELATION
    autocorr, sautocorr,r = utils.GetPSD1D(crr)
    #
    if pixsize == False:
        pixsize = 1.
        dvals = pixsize*np.arange(len(autocorr)) # l values in pixel #
        units = '[pixels]'
        
    else:
        dvals = pixsize*np.arange(len(autocorr))/60. # l values in arcmin
        units = '[arcmin]'
    
    if plots == True:
        # VISUALIZE THE 1D AUTO-CORRELATION FUNCTION
        plt.figure(1,figsize=(5,5))
        plt.errorbar(dvals,autocorr,yerr=sautocorr,fmt='b.-')
        plt.xlim([0.,np.nanmax(dvals)])
        plt.ylim([0.,1.0])
        plt.xlabel(r' $\ell$ '+units)
        plt.ylabel(r'Norm. Autocorr.')
        plt.title('Isotropic Autocorrelation Function')
        plt.axhline(y=0.5,color='r')

    if hwhm == True:
        res = utils.HWHM(dvals,autocorr, sautocorr)
    else:
        res = (autocorr,sautocorr,dvals)
    #
    return res

def structure_function(phi,phierr,x1,x2,pixsize,beam,verb=True):
    #
    start = time.time()
    
    # Calculate the total number of pairs inside the ROI
    N=len(phi)
    pairs=int(N*(N-1)/2)
    if verb:
        print('Number of vector pairs = ',pairs)
    #
    Dphi = np.array([])
    dist = np.array([])
    sig = np.array([])
    cosDphi = np.array([])
    sinDphi = np.array([])
    
    if verb:
        print("Calculating Pairs:")
    for i in range(1,N-1):
        
        #calculate angle difference
        diff = phi - np.roll(phi,-i)
        
        #check to see if 0<Dphi<90
        m1 = np.where(np.abs(diff) > 90)
        diff[m1] = 180. -np.abs(diff[m1])
        #
        Dphi = np.concatenate( (Dphi, np.deg2rad(diff[:-i]) ))
        cosDphi = np.concatenate( (cosDphi,np.cos( np.deg2rad(diff[:-i]) )))
        sinDphi = np.concatenate( (sinDphi,np.sin( np.deg2rad(diff[:-i]) )))
        
        #find the distance between the two points
        x1_1 = x1
        x1_2 = np.roll(x1,-i)
        x2_1 = x2
        x2_2 = np.roll(x2,-i)
        sep = np.sqrt( (x1_1 - x1_2)**2 + (x2_1 - x2_2)**2 )
        dist = np.concatenate((dist,sep[:-i]))
        
        #propagate the error in Dphi
        sig1 = np.deg2rad(np.sqrt(phierr**2 + np.roll(phierr,-i)**2 - 2*phierr*np.roll(phierr,-i)*np.exp(-sep*sep/(4*beam*beam)) ) )
        sig = np.concatenate((sig,sig1[:-i]))
        if verb:
            utils.update_progress((i)/N)
        
    #find max separation
    maxsep = np.nanmax(dist)
    if verb:
        print("Max sep",maxsep)
    Nbins = int(np.ceil(maxsep/pixsize)) #total number of bins.

    #Define arrays for final products
    Dphisum = np.zeros(Nbins) #RMS cosDphi for all bins
    cosDphisum = np.zeros(Nbins) #RMS Dphi for all bins
    sinDphisum = np.zeros(Nbins) #RMS sinDphi for all bins
    sigma2sum = np.zeros(Nbins) #rms sig^2Dphi for all bins
    sigma4sum = np.zeros(Nbins) #rms sig^2Dphi for all bins
    dvals = np.zeros(Nbins) #center values for each bin.
    dispsum_c = np.zeros(Nbins) #Dphi corrected for instrument sigma
    errors_c = np.zeros(Nbins)
    
    #Run through bins once and scrape the items necessary
    if verb:
        print("\n Binning:")
    #
    for i in range(0, Nbins):
        #
        # Values of \ell are defined as integer values of pixsize.
        dvals[i]=i*pixsize
        
        # Dphi(\ell) values for the bin \ell corresponds to pairs with
        # \ell - 0.5*pixsize < \ell < \ell + 0.5*pixsize
        minval=dvals[i] - 0.5*pixsize
        if minval < 0.:
            minval = 0.
        maxval=dvals[i] + 0.5*pixsize
        
        # Find the pairs in between the values
        mm = np.where( (dist <= maxval) & (dist > minval) )
        
        # RMS values for each Dphi term are calculated
        Dphisum[i] = utils.rms_var(Dphi[mm])
        cosDphisum[i] = utils.rms_val(cosDphi[mm])        
        sinDphisum[i] = utils.rms_val(sinDphi[mm]) 
        sigma2sum[i] = utils.rms_val(sig[mm]**2) 
        sigma4sum[i] = utils.rms_val(sig[mm]**4)
        
        # The dispersion function must be corrected to account for uncertainties in
        # polarization angles (phierr)
        dispsum_c[i]= cosDphisum[i]/(1. - 0.5*sigma2sum[i])
        errors_c[i]= np.sqrt( (sinDphisum[i]**2)*sigma2sum[i] + (sinDphisum[i]**2)*Dphisum[i] + \
                             0.75*(cosDphisum[i]**2)*((sigma2sum[i] + Dphisum[i])**2) - \
                                 (sinDphisum[i]**2)*((sigma2sum[i] + Dphisum[i])**2) )/np.sqrt(len(mm[0])) 
        if verb:
            utils.update_progress(i/Nbins)
    #
    end = time.time()
    if verb:
        print("Elapsed = ", end - start)

    return (dispsum_c,dvals,errors_c)

def dispersion_function(phi,phierr,pixsize,beam=0.0,fwhm=True,mask=False,verb=True):
    #
    #This function calculates the structure function of a set of data according 
    #Hildebrand et al. (2009) Errors are propagated according to standard error
    #propagation
    # phi is an array of polarization angles, phierr is the corresponding array of uncertainties

    if beam == 0.0:
        print("Nonzero value of beam size must be provided")
        sys.exit()
    else:
        #Transform the beam FWHM value to sigma value
        beam /= 2.355
    
    # Create position arrays in arcsec
    sz = phi.shape
    xpix = sz[0]
    ypix = sz[1]
    x = np.arange(xpix,dtype=float)
    y = np.arange(ypix,dtype=float)
    x1, x2 = np.meshgrid(x,y,indexing='ij')
    x1 *= pixsize
    x2 *= pixsize
    
    # Prepare data arrays
    # Setting pixels for which mask = 0 to NaN exclude them from calculations
    mask[mask == 0.0] = np.nan
    phi *= mask
    phierr *= mask
    x1 *= mask
    x2 *= mask
    
    # Put arrays into 1D
    phi = phi[np.isfinite(phi)].ravel()
    phierr = phierr[np.isfinite(phierr)].ravel()
    x1 = x1[np.isfinite(x1)].ravel()
    x2 = x2[np.isfinite(x2)].ravel()
    
    # Call the structure function routine
    disp_c,dvals,errors_c = structure_function(phi,phierr,x1,x2,pixsize,beam,verb=verb)
    
    # Outputs 
    disp_funct = 1.0 - disp_c
    lvec = dvals**2 #arcsec^2
    disp_funct_err = errors_c
    
    return (lvec,disp_funct,disp_funct_err)

#
def dispersion_function_map(phi,phierr,pixsize,beam=0.,w=0,mask=False,verb=True):
    #
    #This function calculates the dispersion function for an entire region using a moving kernel 
    # approximation. This routine calls dispersion_function for each valid pixel in the input array.
    
    if beam == 0.0:
        print("Nonzero value of beam size must be provided")
        sys.exit()
    else:
        #Transform the beam FWHM value to sigma value
        beam /= 2.355
        
    # Create position arrays in arcsec
    sz = phi.shape
    xpix = sz[0]
    ypix = sz[1]
    x = np.arange(xpix,dtype=float)
    y = np.arange(ypix,dtype=float)
    x1, x2 = np.meshgrid(x,y,indexing='ij')
    x1 *= pixsize
    x2 *= pixsize
    
    # Prepare data arrays
    mask[mask == 0.0] = np.nan
    phi *= mask
    phierr *= mask
    x1 *= mask
    x2 *= mask
    
    # w value is the radius, wsize is the diameter
    wsize = 2*w+1
    print('Analysis window size = ', wsize,'x',wsize)
    disp_funct_map = np.zeros((xpix,ypix,2*w+1))
    lvec_map = np.zeros((xpix,ypix,2*w+1))
    sigma_err_map = np.zeros((xpix,ypix,2*w+1))
    
    # CREATE CIRCULAR MASK
    cmask = np.empty((wsize,wsize))
    cmask[:] = np.nan
    xn,yn = np.ogrid[-w:w+1,-w:w+1]
    circ = xn*xn + yn*yn <= w*w
    cmask[circ] = 1.0
    
    # Loop over each pixel in the image
    for i in range(w,xpix-w):
        #
        for j in range(w,ypix-w):
            #
            if ~np.isfinite(mask[i,j]):
                print('Skipping Pixel =', i,j)
                continue
            else:
                #
                if verb:
                    print('Calculating Function for Pixel = ', i, j)
                #
                ii1 = i - w
                ii2 = i + (w+1)
                if ii1 < 0:
                    ii1 = 0
                if ii2 > xpix:
                    ii2 = xpix
                # Y
                jj1 = j - w
                jj2 = j + (w+1)
                if jj1 < 0:
                    jj1 = 0
                if jj2 > ypix:
                    jj2 = ypix
                #
                local_x1 = x1[ii1:ii2,jj1:jj2]*cmask
                local_x2 = x2[ii1:ii2,jj1:jj2]*cmask
                local_phi = phi[ii1:ii2,jj1:jj2]*cmask
                local_phierr = phierr[ii1:ii2,jj1:jj2]*cmask
                
                #
                local_phi = local_phi[np.isfinite(local_phi)].ravel()
                local_phierr = local_phierr[np.isfinite(local_phierr)].ravel()
                local_x1 = local_x1[np.isfinite(local_x1)].ravel()
                local_x2 = local_x2[np.isfinite(local_x2)].ravel()
                
                if len(local_phi) > w:
                    # 
                    disp_c,dvals,errors_c = structure_function(local_phi,local_phierr,local_x1,local_x2,pixsize,beam,verb=verb)
                    #
                    disp_funct = 1.0 - disp_c
                    lvec = dvals**2 #arcsec^2
                    disp_funct_err = errors_c
                    #
                    disp_funct_map[i,j,0:len(disp_funct)] = disp_funct
                    lvec_map[i,j,0:len(disp_funct)] = lvec
                    sigma_err_map[i,j,0:len(disp_funct)] = disp_funct_err
                    #utils.update_progress(i*j/npixels)
        
                else:
                    print('Not enough pixel locations to calculate structure function. Increase w value')
    
    return (lvec_map,disp_funct_map,sigma_err_map)