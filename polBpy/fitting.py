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
    #
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
    beam /= 2.355
    dem = d**2 + 2*beam**2
    term = 1./(1. + (dem/(np.sqrt(2*np.pi)*(d**3)))*f)
    return term*( 1-np.exp(-x/(2*dem)) ) + a*x

# Class containing the two-scale function for the MCMC solver
class PolynomialModel(Model):
       
    parameter_names = ("a", "d", "f", "beam")

    def get_value(self, t):
        t = t.flatten()
        dem = self.d**2 + 2*(self.beam/2.355)**2
        term = 1./(1.+ (dem/(np.sqrt(2*np.pi)*(self.d**3)))*self.f)
        return term*( 1-np.exp(-t/(2*dem)) ) + self.a*t 

 
def mcmc_fit(disp_funct,lvec,sigma_err,lmin=False,lmax=False,beam=0.0,a2=0.1,delta=1.0,f=1.0,bnds=False,num=500,fixed_delta=False,verb=True):
# MCMC solver for one dispersion function. This function is called iteratively by mcmc_fit_map.
#-----------------------------------------------------------------------------------------------------------------------
    if verb:
        print('Entering the Fitting function')
    
    # Trimming the arrays because the first element is NaN
    disp_funct = disp_funct[1:]
    lvec = lvec[1:]
    sigma_err = sigma_err[1:]
    # lmax (in arcsec) defines the range for which the dispersion function is valid
    # I.e., the linear range in l^2 
    if lmax == False:
        if verb:
            print("Caution! Fitting the dispersion function with the max range of \ell values")
        
    if lmax != False:
        #Determine the array element closer to lmax
        m = np.where(lvec < lmax)[0]
        disp_funct = disp_funct[m]
        sigma_err = sigma_err[m]
        lvec = lvec[m]
    
    #If lmin is not False, it set the min value for the range to fit
    if lmin != False:
        m = np.where(lvec > lmin)[0]
        disp_funct = disp_funct[m]
        sigma_err = sigma_err[m]
        lvec = lvec[m]
    
    #Initializing fitting function parameters
    truth = dict(a=a2,d=delta,f=f,beam=beam)
    
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
    model1.freeze_parameter('mean:beam')
    
    # You can freeze any parameter but delta is the most common one. This is controlled by the keyword fixed_delta
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
    
    if verb:
        print("First run -- Uninflatted errors...")
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
    f_function = model_funct(lvec,aa,dd,nn,beam=beam)
    # Calculate the Chi^2 value for this fit and the dispersion function
    chi2 = chisqr(disp_funct,f_function,sigma_err)
    
    # Use the Chi^2 value to inflate the errors and run the MCM fit once more
    if verb:
        print("Second run -- Inflating errors by Chi value = ",np.sqrt(chi2))
    
    sigma_err = np.sqrt(chi2)*sigma_err.copy()
    
    truth = dict(a=aa, d=dd, f=nn, beam=beam)
    kwargs = dict(**truth)
        
    if bnds == False:
        kwargs["bounds"] = dict(a=(0,np.inf),d=(0., np.inf),f=(0, np.inf))
    else:
        kwargs["bounds"] = bnds
    
    # Same steps as in the first MCMC run
    mean_model = PolynomialModel(**kwargs)
    model1 = george.GP(mean=mean_model)
    model1.freeze_parameter('mean:beam')
    
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
    f_function = model_funct(lvec,aa,dd,nn,beam=beam)
    # Evaluate Spearman's rho value, since fitting function is not linear.
    rho = spearmanr(disp_funct,f_function)[0]
    
    # Create structure with final values
    params = dict(a=aa, d=dd, f=nn, chi=np.sqrt(chi2),rho=rho)
    
    if verb:
        print("Done with MCMC fitting.")
    #
    return params


def mcmc_fit_map(disp_funct_map,lvec_map,sigma_err_map,wsize,pixsize,lmin=False,lmax=False,beam=0.0,a2=0.1,
                 delta=1.0,f=1.0,bnds=False,num=500,fixed_delta=False,verb=False,n_cores=False):
    
    # This function loops around all pixels in an image for fitting the pixel dispersion function.
    # The function mcmc_fit is called iteratively, so the process can be parallelized.
    
    sz = disp_funct_map.shape
    disp_funct_map = np.resize(disp_funct_map,(sz[0]*sz[1],sz[2]))
    lvec_map = np.resize(lvec_map,(sz[0]*sz[1],sz[2]))
    sigma_err_map = np.resize(sigma_err_map,(sz[0]*sz[1],sz[2]))    
    #
    
    import os
    from joblib import Parallel, delayed
    
    # By default, restrict the bound for delta to twice the size of the kernel (max physical scale considered)
    if bnds == False:
        bnds = dict(a=(0,np.inf),d=(0., 2.*wsize*pixsize),f=(0, np.inf))
    else:
        bnds = bnds
    #
    def mcmc_loop(i):
        # Main loop
        global res, lvec, disp_funct, sigma_err
        #
        disp_funct = disp_funct_map[i,1:-1]
        lvec = lvec_map[i,1:-1]
        sigma_err = sigma_err_map[i,1:-1]
        
        # Important that the dispersion function is valid
        if not all(disp_funct == 0.):
            res = mcmc_fit(disp_funct,lvec,sigma_err,lmin=lmin,lmax=lmax,beam=beam,a2=a2,delta=delta,f=f,bnds=bnds,num=num,fixed_delta=False,verb=verb)
        else:
            res = dict(a=np.NaN, d=np.NaN, f=np.NaN,chi=np.NaN,rho=np.NaN)
        
        return res
    
    inputs = range(sz[0]*sz[1])
    
    # Define the number of cores to use during the parallelization
    if n_cores == False:
        try:
            num_cores = os.cpu_count()
            num_cores_avail = int(num_cores/2) # Default is half the available cores
            
        except:
            num_cores_avail = 1 # If there are no multiple cores
    else:
        num_cores_avail = n_cores # Defined by user
    
    print('Running MCMC fitting in %s cores'%num_cores_avail)   
    #
    results = Parallel(n_jobs=num_cores_avail)(delayed(mcmc_loop)(i) for i in inputs)
    #
    fratio = [i['f'] for i in results]
    delta = [i['d'] for i in results]
    a2 = [i['a'] for i in results]
    chi = [i['chi'] for i in results]
    rho = [i['rho'] for i in results]
    
    #Putting resulting maps in the original image's shape
    fratio = np.reshape(fratio,(sz[0],sz[1]))
    delta = np.reshape(delta,(sz[0],sz[1]))
    a2 = np.reshape(a2,(sz[0],sz[1]))
    chi = np.reshape(chi,(sz[0],sz[1]))
    rho = np.reshape(rho,(sz[0],sz[1]))
    #
    return dict(a=a2, d=delta, f=fratio, chi=chi, rho=rho)
