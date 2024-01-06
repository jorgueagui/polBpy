#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 19:09:09 2023

@author: jguerraa
"""

from george.modeling import Model
import numpy as np
import george
import emcee
from scipy.stats import spearmanr

def chisqr(obs,mod,err):
    
    # Reduced Chi^2 parameter
    chisqr = 0.0
    for i,j,k in zip(obs,mod,err):
        #
        chisqr += ((i-j)**2)/(k**2)
        #
    chisqr = chisqr/(len(obs)-3)
    return chisqr  
#
def model_funct(x,a,d,f,beam=0.0):
    #
    dem = d**2 + 2*beam**2
    term = 1./(1. + (dem/(np.sqrt(2*np.pi)*(d**3)))*f)
    return term*( 1-np.exp(-x/(2*dem)) ) + a*x

def mcmc_fit(disp_funct,lvec,sigma_err,lmin=False,lmax=False,beam=0.0,a2=0.1,delta=0.1,f=1.0,bnds=False,num=500,fixed_delta=False):
    #
    class PolynomialModel(Model):
           
        parameter_names = ("a", "d", "f")

        def get_value(self, t):
            t = t.flatten()
            dem = self.d**2 + 2*beam**2
            term = 1./(1.+ (dem/(np.sqrt(2*np.pi)*(self.d**3)))*self.f)
            return term*( 1-np.exp(-t/(2*dem)) ) + self.a*t 
    
#-----------------------------------------------------------------------------------------------------------------------
    print('Entering the Fitting function')
    
    # lmax (in arcsec) defines the range for which the dispersion function is valid
    # I.e., the linear range in l^2 
    if lmax != False:
        print("Maximum value of l is neede for fitting.")
        exit
    else:
        #Determine the array element closer to lmax
        m = np.where(np.sqrt(lvec) < lmax)
        disp_funct = disp_funct[:m[0][0]]
        sigma_err = sigma_err[:m[0][0]]
        lvec = lvec[:m[0][0]]
    
    #If lmin is not False, it set the min value for the range to fit
    if lmin != False:
        m = np.where(np.sqrt(lvec) > lmin)
        disp_funct = disp_funct[m[0][0]:]
        sigma_err = sigma_err[m[0][0]:]
        lvec = lvec[m[0][0]:]
    
    #Initializing parameters
    chi2 = 1.0
    truth = dict(a=a2,d=delta,f=f)
    
    # Guess for parameter fitting
    kwargs = dict(**truth)
    
    # If you want to impose bounds to the parameters to fit, create a structure in the 
    # form shown below.
    if bnds == False:
        kwargs["bounds"] = dict(a=(0,np.inf),d=(0., np.inf),f=(0, np.inf))
    else:
        kwargs["bounds"] = bnds
        
    # Evaluate the model with the guess parameters
    mean_model = PolynomialModel(**kwargs)
    model1 = george.GP(mean=mean_model)
    
    # You can freeze any parameter but delta is the most common one.
    if fixed_delta == True:
        model1.freeze_parameter('mean:d')
            
    # Model 1 is the Gaussian process model with the initial guess
    model1.compute(lvec, sigma_err)
        
    #Define the prior probability
    def lnprob(p):
        model1.set_parameter_vector(p)
        return model1.log_likelihood(disp_funct, quiet=True) + model1.log_prior()
    
    # Setup the MCMC runs
    # Number of walker by default is 500 but it is control by parameter "num"
    initial = model1.get_parameter_vector()
    ndim, nwalkers = len(initial), num
    p0 = initial + 1e-8 * np.random.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
        
    #Running burn-in
    p0, _, _ = sampler.run_mcmc(p0, num)
    sampler.reset()
        
    #"Running production
    sampler.run_mcmc(p0, num);
    samples = sampler.flatchain
    
    # Calculate median values from posterior distributions
    try:
       	aa = np.median(samples[:,0])
       	dd = np.median(samples[:,1])
       	nn = np.median(samples[:,2])
    except:
        aa = np.median(samples[:,0])
        dd = delta
        nn = np.median(samples[:,1])

    # Evaluate the polynomial model with the best-fit parameters
    f_function = model_funct(lvec,aa,dd,nn)
    # Calculate the Chi^2 value for this fit and the dispersion function
    chi2 = chisqr(disp_funct,f_function,sigma_err)
    
    # Use the Chi^2 value to inflate the errors and run the MCM fit once more
    print('Inflating Errors by =',chi2)
    sigma_err *= np.sqrt(chi2)
    
    truth = dict(a=aa, d=dd, f=nn)
    kwargs = dict(**truth)
        
    if bnds == False:
        kwargs["bounds"] = dict(a=(0,np.inf),d=(0., np.inf),f=(0, np.inf))
    else:
        kwargs["bounds"] = bnds
    
    # Same steps as in the first MCMC run
    mean_model = PolynomialModel(**kwargs)
    model1 = george.GP(mean=mean_model)
    
    if fixed_delta == True:
    		model1.freeze_parameter('mean:d')
        
    model1.compute(lvec, sigma_err)
        
    initial = model1.get_parameter_vector()
    ndim, nwalkers = len(initial), num
    p0 = initial + 1e-8 * np.random.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
        
    #Running burn-in
    p0, _, _ = sampler.run_mcmc(p0, num)
    sampler.reset()
        
    #Running production
    sampler.run_mcmc(p0, num);
    #
    samples = sampler.flatchain
    
    # Calculate final values for parameters
    try:
       	aa = np.median(samples[:,0])
       	dd = np.median(samples[:,1])
       	nn = np.median(samples[:,2])
    except:
        aa = np.median(samples[:,0])
        dd = delta
        nn = np.median(samples[:,1])
        
    # Calculate quality metrics for fit
    f_function = model_funct(lvec,aa,dd,nn)
    chi2 = chisqr(disp_funct,f_function,sigma_err)
    rho = spearmanr(disp_funct,f_function)[0]
    
    # Create structure with final values
    params = dict(a=aa, d=dd, f=nn, chi=np.sqrt(chi2),rho=rho)
    #
    return params

from joblib import Parallel, delayed

def mcmc_fit_map(disp_func,l,beam=0.0):
    
    # DEFINING THE FUNCTION TO FIT THE DISPERSION FUNCTION
    # HOUDE ET AL 2011.
    
    #sz = data['disp_funct_map'].shape
    #disp_funct_map = np.resize(data['disp_funct_map'],(sz[0]*sz[1],sz[2]))
    #lvec_map = np.resize(data['lvec_map'],(sz[0]*sz[1],sz[2]))
    #sigma_err_map = np.resize(data['sigma_err_map'],(sz[0]*sz[1],sz[2]))    
    #
    #plt.figure(0)
    #btb0_map = np.empty((sz[1]*sz[0]))
    #delta_map = np.empty((sz[1]*sz[0]))
    #a0_map = np.empty((sz[1]*sz[0]))
    #
    
    #import multiprocessing
    #pool = mp.Pool(processes=4)
    #samples = 0.0
    #
    return 1.
