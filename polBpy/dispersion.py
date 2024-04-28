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

def autocorrelation(polflux,polflux_err,pixsize=1.0,mask=False,plots=False,hwhm=False):
    # This function calculates the 1D isotropic autoccorrelation of
    # the polarized flux in the emitting volume
    #
    polflux *= mask
    # MAKE NaNs INTO ZEROS
    #polflux[~polflux.mask] = np.nan
    polflux[~np.isfinite(polflux)] = 0.0
    
    if plots == True:
        # PLOT THE POL FLUX MAP USED
        plt.figure(0)
        plt.imshow(polflux,origin='lower')
        plt.title('Map of Polarized Flux')
    #
    # CALCULATE TEH 2D AUTO-CORRELATION 
    crr = signal.correlate2d(polflux,polflux,mode='full')
    crr = crr/np.nanmax(crr)

    # CALCULATE THE 1D AUTO-CORRELATION
    autocorr, sautocorr,r = utils.GetPSD1D(crr)
    dvals = pixsize*np.arange(len(autocorr)) # l VALUES
    
    if plots == True:
        # VISUALIZE THE 1D AUTO-CORRELATION FUNCTION
        plt.figure(1,figsize=(5,5))
        plt.errorbar(dvals/60.,autocorr,yerr=sautocorr,fmt='b.-')
        plt.xlim([0.,np.nanmax(dvals/60.)])
        plt.ylim([0.,1.0])
        plt.xlabel(r' $\ell$ [arcmin]')
        plt.ylabel(r'Norm. Autocorr.')
        plt.title('Isotropic Autocorrelation Function')
        plt.axhline(y=0.5,color='r')

    if hwhm == True:
        res = utils.HWHM(dvals/60.,autocorr, sautocorr)
    else:
        res = (autocorr,sautocorr,dvals)
    #
    return res

def structure_function(phi,phierr,x1,x2,pixsize,beam):
    #
    start = time.time()
    # Loop over the total number of points for the first pair
    N=len(phi)
    pairs=int(N*(N-1)/2)
    print('Number of vector pairs = ',pairs)
    #
    Dphi = np.array([])
    dist = np.array([])
    sig = np.array([])
    cosDphi = np.array([])
    sinDphi = np.array([])
    
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
        utils.update_progress((i)/N)
        
    #find max separation
    maxsep = np.nanmax(dist)
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
        utils.update_progress(i/Nbins)
    #
    end = time.time()
    print("Elapsed = ", end - start)

    return (dispsum_c,dvals,errors_c)

def dispersion_function(phi,phierr,pixsize,beam=0.0,fwhm=True,mask=False):
    #
    #This function calculates the structure function of a set of data according 
    #Hildebrand et al. (2009) Errors are propagated according to standard error
    #propagation
    # phi is an array of polarization angles, phierr is the corresponding array of uncertainties
    #C is a SkyCoord type argument for the location of each vector
    #C1.separation(C2)
    #pixsize is the pixelization of the map in arcseconds. This is the fundamental unit of the bin for the SF.
    #C = wcs.utils.skycoord_to_pixel(C,w2)
    #ra = pixsize*np.round(np.array(C[0]))
    #dec = pixsize*np.round(np.array(C[1]))
    if beam == 0.0:
        print("Nonzero value of beam size must be provided")
        exit
   
    #Transform the beam FWHM value to sigma value
    beam /= 2.355
    
    # Create position arrays in arcsec
    sz = phi.shape
    xpix = sz[0]
    ypix = sz[1]
    x = np.arange(xpix,dtype=float)
    y = np.arange(ypix,dtype=float)
    x1, x2 = np.meshgrid(x,y)
    x1 *= pixsize
    x2 *= pixsize
    
    # Prepare data arrays
    mask[mask == 0.0] = np.nan
    phi *= mask#np.ma.masked_array(phi,mask=mask)
    phierr *= mask#np.ma.masked_array(phierr,mask=mask)
    x1 *= mask#np.ma.masked_array(x1,mask=mask)
    x2 *= mask#np.ma.masked_array(x2,mask=mask)
    
    #
    phi = phi[np.isfinite(phi)].ravel()
    phierr = phierr[np.isfinite(phierr)].ravel()
    x1 = x1[np.isfinite(x1)].ravel()
    x2 = x2[np.isfinite(x2)].ravel()
    
    disp_c,dvals,errors_c = structure_function(phi,phierr,x1,x2,pixsize,beam)
    
    # Outputs 
    disp_funct = 1.0 - disp_c
    lvec = dvals**2 #arcsec^2
    disp_funct_err = errors_c
    
    return (lvec,disp_funct,disp_funct_err)

#
def dispersion_function_map(phi,phierr,pixsize,beam=0.,w=0,mask=False):
    #
    #This function calculates the structure function of a set of data according 
    #Hildebrand et al. (2009) Errors are propagated according to standard error
    #propagation
    # phi is an array of polarization angles, phierr is the corresponding array of uncertainties
    #C is a SkyCoord type argument for the location of each vector
    #C1.separation(C2)
    #pixsize is the pixelization of the map in arcseconds. This is the fundamental unit of the bin for the SF.
    #C = wcs.utils.skycoord_to_pixel(C,w2)
    #ra = pixsize*np.round(np.array(C[0]))
    #dec = pixsize*np.round(np.array(C[1]))
    if beam == 0.0:
        print("Nonzero value of beam size must be provided")
        exit
        
    # Create position arrays in arcsec
    sz = phi.shape
    xpix = sz[0]
    ypix = sz[1]
    x = np.arange(xpix)
    y = np.arange(ypix)
    x1, x2 = pixsize*np.meshgrid(x,y)
    
    # Prepare data arrays
    phi = np.ma.masked_array(phi,mask=mask)
    phierr = np.ma.masked_array(phierr,mask=mask)
    x1 = np.ma.masked_array(x1,mask=mask)
    x2 = np.ma.masked_array(x2,mask=mask)
    
    # w value is the radius, wsize is the diameter
    wsize = 2*w+1
    print('Analysis window size = ', wsize,'x',wsize)
    disp_funct_map = np.zeros((ypix,xpix,2*w+1))
    lvec_map = np.zeros((ypix,xpix,2*w+1))
    sigma_err_map = np.zeros((ypix,xpix,2*w+1))
    
    #
    # CREATE CIRCULAR MASK
    cmask = np.empty((wsize,wsize))
    cmask[:] = np.nan
    xn,yn = np.ogrid[-w:w+1,-w:w+1]
    circ = xn*xn + yn*yn <= w*w
    cmask[circ] = 1.0
    
    #
    for i in range(w,xpix-w):#w,xpix-w-2,1):#0,xpix-1,2*w+1):
        #
        #
        for j in range(w,ypix-w):
            #
            print('Pixels = ', j, i)
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
            local_x1 = x1[jj1:jj2,ii1:ii2]*cmask
            local_x2 = x2[jj1:jj2,ii1:ii2]*cmask
            local_phi = phi[jj1:jj2,ii1:ii2]*cmask
            local_phierr = phierr[jj1:jj2,ii1:ii2]*cmask
            
            # 
            local_phi = local_phi[local_phi.mask == True].ravel()
            local_phierr = local_phierr[local_phierr.mask == True].ravel()
            local_x1 = local_x1[local_x1.mask == True].ravel()
            local_x2 = local_x2[local_x2.mask == True].ravel()
            
            if len(local_phi) > w:
                # 
                disp_c,dvals,errors_c = structure_function(phi,phierr,x1,x2,pixsize,beam)
                #
                disp_funct = 1.0 - disp_c
                lvec = dvals**2 #arcsec^2
                disp_funct_err = errors_c
                #
                disp_funct_map[j,i,0:len(disp_funct)] = disp_funct
                lvec_map[j,i,0:len(disp_funct)] = lvec
                sigma_err_map[j,i,0:len(disp_funct)] = disp_funct_err
    
            else:
                print('Not enough pixel locations to calculate structure function. Increase w value')
    
    return (lvec_map,disp_funct_map,sigma_err_map)