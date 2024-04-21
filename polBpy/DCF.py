#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 20:19:48 2023

@author: jguerraa
"""

import numpy as np
from uncertainties import ufloat, unumpy
from uncertainties.umath import * 
import scipy
from polBpy import utils
import random

#Defining some constants
mu = 2.8 # H mean molecular weight
mH = 1.6737236E-24 # mass of H molecule [g]
kB = 1.0
#

def dcf_classical(den,vel,disp,rho=False,eta=1.0,uden=0.,uvel=0.,udisp=0,cdepth=0.,ucdepth=0.0):
    #
    # Create uncertainties objects for calculations
    den = ufloat(den,uden)
    vel = ufloat(vel,uvel)
    disp = ufloat(disp,udisp)
    #
    # If rho = True, den is interpretyed as mass density
    if rho == True:
        rho = 1.0*den
    else:
    # If rho = False, den is interpreted as colunm density
    # and cloud's depth is needed
        try:
            cdepth = ufloat(cdepth,ucdepth)
            rho = mu*mH*(den/cdepth)
        except ValueError:
            print('Value of cloud depth [cm] is needed')
            
    # Transforming velocity values to [cm/s]
    vel *= 1.0E+5
    # Calculating DCf value
    dcf_val_ = (4*np.pi*rho)**0.5
    dcf_val_ *= (vel/disp)
    dcf_val_ *= eta # 
    dcf_val_ *= 1.0E+6 # B-strength in micro Gauss
    #
    dcf_val = (dcf_val_.nominal_value,dcf_val_.std_dev)
    #
    return dcf_val

def dcf_compressional(den,vel,disp,rho=False,uden=0.,uvel=0.,udisp=0,cdepth=0.,ucdepth=0.0):
    # This function calculates the POS B-field strength according to the compressional
    # turbulence approximation by Skadilis et. al. (2021). This expressions requires exactly the same
    # values as the Classical DCF.
    #
    # Create uncertainties objects for calculations
    den = ufloat(den,uden)
    vel = ufloat(vel,uvel)
    disp = ufloat(disp,udisp)
    #
    # If rho = True, den is interpretyed as mass density
    if rho == True:
        rho = 1.0*den
    else:
    # If rho = False, den is interpreted as colunm density
    # and cloud's depth is needed
        try:
            cdepth = ufloat(cdepth,ucdepth)
            rho = mu*mH*(den/cdepth)
        except ValueError:
            print('Value of cloud depth [cm] is needed')
            
    # Transforming velocity values to [cm/s]
    vel *= 1.0E+5
    # Calculating DCf value
    dcf_val_ = (2*np.pi*rho)**0.5
    dcf_val_ *= (vel/disp**0.5)
    dcf_val_ *= 1.0E+6 # B-strength in micro Gauss
    #
    dcf_val = (dcf_val_.nominal_value,dcf_val_.std_dev)
    #
    return dcf_val

def dcf_ls_flow(den,vel,disp,flow,rho=False,eta=1.0,uden=0.,uvel=0.,udisp=0,cdepth=0.,ucdepth=0.0,uflow=0.0):
    # This function calculates the POS B-field strength according to the modified
    # large-scale flow approximation by Lopez-Rodriguez et. al. (2021). This expressions requires values of the large-scale
    # flow and its laplacian (evaluated over the same scale as depth), besides the typical values 
    # as the Classical DCF.
    #
    # Check that disp != vel/flow
    if disp == (vel/flow):
        print('This DCF approximation is not valid for disp = vel/flow')
        exit
    
    # Create uncertainties objects for calculations
    den = ufloat(den,uden)
    vel = ufloat(vel,uvel)
    disp = ufloat(disp,udisp)
    flow = ufloat(flow,uflow)
    #
    # If rho = True, den is interpretyed as mass density
    if rho == True:
        rho = 1.0*den
    else:
    # If rho = False, den is interpreted as colunm density
    # and cloud's depth is needed
        try:
            cdepth = ufloat(cdepth,ucdepth)
            rho = mu*mH*(den/cdepth)
        except ValueError:
            print('Value of cloud depth [cm] is needed')
            
    # Transforming velocity values to [cm/s]
    vel *= 1.0E+5
    flow *= 1.0E+5
    # Calculating DCF value
    dcf_val_ = (4*np.pi*rho)**0.5
    dcf_val_ *= (vel/disp)
    dcf_val_ *= eta # 
    dcf_val_ *= ((1.-disp*(flow/vel))**2)**0.5
    dcf_val_ *= 1.0E+6 # B-strength in micro Gauss
    #
    dcf_val = (dcf_val_.nominal_value,dcf_val_.std_dev)
    #
    return dcf_val


def dcf_shear_flow(den,vel,disp,flow,flow_lap,rho=False,uden=0.,uvel=0.,udisp=0,cdepth=0.,ucdepth=0.0,uflow=0.0,uflow_lap=0.0):
    # This function calculates the POS B-field strength according to the modified
    # shear-flow approximation by Guerra et. al. (2023). This expressions requires values of the large-scale
    # flow and its laplacian (evaluated over the same scale as depth), besides the typical values 
    # as the Classical DCF.
    #
    # Create uncertainties objects for calculations
    den = ufloat(den,uden)
    vel = ufloat(vel,uvel)
    disp = ufloat(disp,udisp)
    flow = ufloat(flow,uflow)
    flow_lap = ufloat(flow_lap,uflow_lap)
    #
    # If rho = True, den is interpretyed as mass density
    if rho == True:
        rho = 1.0*den
    else:
    # If rho = False, den is interpreted as colunm density
    # and cloud's depth is needed
        try:
            cdepth = ufloat(cdepth,ucdepth)
            rho = mu*mH*(den/cdepth)
        except ValueError:
            print('Value of cloud depth [cm] is needed')
            
    # Transforming velocity values to [cm/s]
    vel *= 1.0E+5
    # Let us calculate the term containing the algular dispersion
    disp_f = (1.-(flow/vel)*disp)/( (1. - (flow/vel)*disp)*(disp**2) + (disp/vel)*flow_lap )

    # Calculating DCF value
    dcf_val_ = (4*np.pi*rho)**0.5
    dcf_val_ *= vel*(np.abs(disp_f))**0.5
    dcf_val_ *= 1.0E+6 # B-strength in micro Gauss
    #
    dcf_val = (dcf_val_.nominal_value,dcf_val_.std_dev)
    #
    return dcf_val

def map_comb(m_den,m_vel,m_disp,m_flow=0.0,m_flow_lap=0.0,eta=1.0,rho=False,dcftype='class',
             m_uden=0.0,m_uvel=0.0,m_udisp=0.0,m_cdepth=0.0,m_ucdepth=0.0,m_uflow=0.0,m_uflow_lap=0.0):
    #
    # This routine calculates Bpos accordding to a DCF approximation
    # using maps assumed to have the same resolution.
    #
    # Create uncertainties arrays for calculations
    try:
        # If all the uncertainties are array-like
        m_den = unumpy.uarray(m_den,m_uden)
        m_vel = unumpy.uarray(m_vel,m_uvel)
        m_disp = unumpy.uarray(m_disp,m_udisp)
    except:
        # If the uncertainties are single values
        m_uden = np.full(m_den.shape,m_uden,dtype=float)
        m_uvel = np.full(m_vel.shape,m_uvel,dtype=float)
        m_udisp = np.full(m_disp.shape,m_udisp,dtype=float)
        m_den = unumpy.uarray(m_den,m_uden) 
        m_vel = unumpy.uarray(m_vel,m_uvel)
        m_disp = unumpy.uarray(m_disp,m_udisp)
    #
    # If rho = True, den is interpretyed as mass density
    if rho == True:
        m_rho = 1.0*m_den
    else:
        # If rho = False, den is interpreted as colunm density
        # and cloud's depth is needed
        try:
            if type(m_cdepth) == np.ndarray:
                # If depth and its uncertainty are arrays
                m_cdepth = unumpy.uarray(m_cdepth,m_ucdepth)
            else:
                # If depth and its uncertainty are single values
                m_cdepth = np.full(m_den.shape,m_cdepth,dtype=float)
                m_ucdepth = np.full(m_den.shape,m_ucdepth,dtype=float)
                m_cdepth = unumpy.uarray(m_cdepth,m_ucdepth)
                #
                m_rho = mu*mH*(m_den/m_cdepth)
        except ValueError:
            print('Value(s) of cloud depth [cm] is/are needed')
    # Transforming velocity values to [cm/s]
    
    if dcftype == 'class':
        #
        m_vel *= 1.0E+5
        # Calculating DCF value
        dcf_map_ = (4*np.pi*m_rho)**0.5
        dcf_map_ *= (m_vel/m_disp)
        dcf_map_ *= eta
        dcf_map_ *= 1.0E+6 # B-strength in micro Gauss
        #
        dcf_map = (unumpy.nominal_values(dcf_map_), unumpy.std_devs(dcf_map_))
        
    if dcftype == 'ls-flow':
        #
        try:
            # If all the uncertainties are array-like
            m_flow = unumpy.uarray(m_flow,m_uflow)

        except:
            # If the uncertainties are single values
            m_uflow = np.full(m_den.shape,m_uflow,dtype=float)
            m_flow = unumpy.uarray(m_flow,m_uflow)
        
        # Transforming velocity values to [cm/s]
        m_vel *= 1.0E+5
        m_flow *= 1.0E+5
        # Calculating DCF value
        dcf_map_ = (4*np.pi*m_rho)**0.5
        dcf_map_ *= (m_vel/m_disp)
        dcf_map_ *= eta # 
        dcf_map_ *= ((1.-m_disp*(m_flow/m_vel))**2)**0.5
        dcf_map_ *= 1.0E+6 # B-strength in micro Gauss
        #
        dcf_map = (unumpy.nominal_values(dcf_map_), unumpy.std_devs(dcf_map_))
        
    if dcftype == 'shear-flow':
        #
        try:
            # If all the uncertainties are array-like
            m_flow = unumpy.uarray(m_flow,m_uflow)
            m_flow_lap = unumpy.uarray(m_flow_lap,m_uflow_lap)

        except:
            # If the uncertainties are single values
            m_uflow = np.full(m_den.shape,m_uflow,dtype=float)
            m_uflow_lap = np.full(m_den.shape,m_uflow_lap,dtype=float)
            m_flow = unumpy.uarray(m_flow,m_uflow)
            m_flow_lap = unumpy.uarray(m_flow_lap,m_uflow_lap)
        
        # Transforming velocity values to [cm/s]
        m_vel *= 1.0E+5
        m_flow *= 1.0E+5
        m_flow_lap *= 1.0E+5
        m_disp_f = (1.-(m_flow/m_vel)*m_disp)/( (1. - (m_flow/m_vel)*m_disp)*(m_disp**2) + (m_disp/m_vel)*m_flow_lap )
    
        # Calculating DCF value
        dcf_map_ = (4*np.pi*m_rho)**0.5
        dcf_map_ *= m_vel*(np.abs(m_disp_f))**0.5
        dcf_map_ *= 1.0E+6 # B-strength in micro Gauss
        #
        dcf_map = (unumpy.nominal_values(dcf_map_), unumpy.std_devs(dcf_map_))
    
    return dcf_map

def dcf_map(m_den,m_vel,m_disp,pixsize,rho=False,m_uden=0.0,m_uvel=0.0,m_udisp=0.0,m_cdepth=0.0,m_ucdepth=0.0,res_den=0.0,res_vel=0.0,res_disp=0.0):
    #
    # If all images have the same resolution
    if ( res_den == res_vel and res_vel == res_disp and res_disp == res_den):
        #
        res_map = map_comb(m_den,m_vel,m_disp,pixsize,rho=rho,m_uden=0.0,m_uvel=0.0,m_udisp=0.0,m_cdepth=0.0,m_ucdepth=0.0)
        
    else:
        # If the resolution of the input images (arrays) are not the same
        # we smooth them to a common resolution -- the lowest one.
        res_ = [res_den,res_vel,res_disp]
        img = [m_den,m_vel,m_disp]
        u_img = [m_uden,m_uvel,m_udisp]
        #
        for i in range(len(res_)):
            #
            img[i] = utils.sigma_convolve(img[i],np.nanmax(res_),res_[i],pixsize)
            try:
                u_img[i] = utils.sigma_convolve(u_img[i],np.nanmax(res_),res_[i],pixsize)
            except:
                continue
        #
        # After smoothing the images, we call the map-combination function
        res_map = map_comb(m_den,m_vel,m_disp,pixsize,rho=rho,m_uden=0.0,m_uvel=0.0,m_udisp=0.0,m_cdepth=0.0,m_ucdepth=0.0)
        
    return res_map

def dcf_range(m_den,m_vel,m_disp,m_flow=False,m_flow_lap=False,eta=1.0,rho=False,dcftype='class',
             m_uden=0.0,m_uvel=0.0,m_udisp=0.0,m_cdepth=0.0,m_ucdepth=0.0,m_uflow=0.0,m_uflow_lap=0.0):
    #
    # This function calculates Bpos values according to a DCF approximation, when some of the variables are maps
    # but not all. In such case, the spatial distribution will not be correct, instead percentiles 5,50 (median), 
    # and 95 of the distribution will be provided.
    
    # Checking which variable is a single value and create arrays with it.
    types = [type(m_den),type(m_vel),type(m_disp),type(m_flow),type(m_flow_lap)]
    var_names = ['Column density','Velocity dispersion','Pol Angle dispersion','Large Scale Flow','Large Scale Flow Laplacian']
    #print(types)
    n_float = len(np.where(types == float))
    if n_float == len(types):
        print('Not all variables can be single values. Use dcf_classical or a similar single-value routine.')
        quit()
        
    if n_float == 4 or n_float == 3 or n_float == 2 or n_float == 1:
        #
        for i,j in enumerate(types):
            if j == float:
                print('Variable: ',var_names[i],'is a single value. Creating an array with this value.')
                if i == 0:
                    m_den = np.full_like(m_vel,m_den)
                    m_uden = np.full_like(m_vel,m_uden)
                if i == 1:
                    m_vel = np.full_like(m_den,m_vel)
                    m_uvel = np.full_like(m_den,m_uvel)
                if i == 2:
                    m_disp = np.full_like(m_den,m_disp)
                    m_udisp = np.full_like(m_den,m_udisp)
                if i == 3:
                    m_flow = np.full_like(m_den,m_flow)
                    m_uflow = np.full_like(m_den,m_uflow)
                if i == 4:
                    m_flow = np.full_like(m_den,m_flow_lap)
                    m_uflow = np.full_like(m_den,m_uflow_lap)
            else:
                continue
    
    if n_float == 0:
        print('All varibales are arrays. Please use dcf_map instead.')
    
    # Calculating the Bpos array
    res_map = map_comb(m_den,m_vel,m_disp,m_flow=m_flow,m_flow_lap=m_flow_lap,dcftype=dcftype,rho=rho,m_uden=m_uden,m_uvel=m_uvel,
                       m_udisp=m_udisp,m_cdepth=m_cdepth,m_ucdepth=m_ucdepth,m_uflow=m_uflow,m_uflow_lap=m_uflow_lap)
    Bpos = res_map[0]
    Bpos_u = res_map[1]
    
    #If all elements in Bpos_s are zero, percentiles are reported directly without uncertainties.
    if np.all(Bpos_u == 0.) == True:
        #
        Bpos[Bpos == 0.] = np.nan
        range_dcf = [np.nanpercentile(Bpos,5),np.nanpercentile(Bpos,50),np.nanpercentile(Bpos,95)]
        
    else:
        
        #number of "dice rolls"
        niter = 1000
        #make some dummy arrays to hold the realization
        Bpos[Bpos == 0.] = np.nan
        Bpos_u[Bpos_u == 0.] = np.nan
        #n = len(Bpos.flatten())
        #indexes = np.arange(0,n,1)
        b0p = Bpos.copy()
        b0p = b0p.flatten()
        sb0p = Bpos_u.copy()
        sb0p = sb0p.flatten()
        #
        b0p_median = []
        b0p_5 = []
        b0p_95 = []

        #
        for i in range(0,niter):
            #
            temp = random.gauss(b0p,sb0p)
            b0p_median.append(np.nanmedian(temp))
            b0p_5.append(np.nanpercentile(temp,5))
            b0p_95.append(np.nanpercentile(temp,95))
    
        x = np.median(b0p_median)
        sx = np.std(b0p_median)
    
        x1 = np.median(b0p_5)
        sx1 = np.std(b0p_5)
    
        x2 = np.median(b0p_95)
        sx2 = np.std(b0p_95)
    
        range_dcf = [(x1,sx1),(x,sx),(x2,sx2)]
    
    return range_dcf