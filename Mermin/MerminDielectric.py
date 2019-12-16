#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:22:36 2019

@author: tommy

Numerically calculates the Mermin dielectric function.
The integrals are converted to differential equation to be solved numerically.

This code follows the work by David Perkins, Andre Souza, Didier Saumon, 
and Charles Starrett as produced in the 2013 Final Reports from the Los 
Alamos National Laboratory Computational Physics Student Summer Workshop, 
in the section titled "Modeling X-Ray Thomson Scattering Spectra 
of Warm Dense Matter".
"""


import numpy as np
from scipy.integrate import odeint

def realintegrand(y, p, k, omega, kBT, mu, nu, delta):
    """
    The integrand present in the formula for the real part of the general
    RPA dielectric function.
    
    Parameters:
    ___________
    
    
    Returns:
    ________
    """
    
    # delta will help with avoiding singularities if the real part of nu is 0.
    nureal = nu.real
    if nureal**2 < delta:
        nureal = 1/2. * delta**(1/2.)
    
    # variables to avoid verbose lines later on.
    pp = (k**2 + 2*(omega-nu.imag) + 2*p*k)**2 + (2*nureal)**2
    pm = (k**2 + 2*(omega-nu.imag) - 2*p*k)**2 + (2*nureal)**2
    mp = (k**2 - 2*(omega-nu.imag) + 2*p*k)**2 + (2*nureal)**2
    mm = (k**2 - 2*(omega-nu.imag) - 2*p*k)**2 + (2*nureal)**2
    
    logpart = np.log(np.sqrt(pp/pm)) + np.log(np.sqrt(mp/mm))
    
    FD = 1/(1+np.exp((p**2/2 - mu)/kBT))
    
    return logpart * FD * p

def imagintegrand(y, p, k, omega, kBT, mu, nu):
    """
    The integrand present in the formula for the imaginary part of the general
    RPA dielectric function.
    
    Parameters:
    ___________
    
    
    Returns:
    ________
    """
    
    # variables to avoid verbose lines later on.
    pp = k**2 + 2*(omega-nu.imag) + 2*p*k
    pm = k**2 + 2*(omega-nu.imag) - 2*p*k
    mp = k**2 - 2*(omega-nu.imag) + 2*p*k
    mm = k**2 - 2*(omega-nu.imag) - 2*p*k
    
    arctanpart = np.arctan2(2.*nu.real, pp) - np.arctan2(2.*nu.real, pm) \
               + np.arctan2(-2.*nu.real, mp) - np.arctan2(-2.*nu.real, mm)
        
    FD = 1/(1+np.exp((p**2/2 - mu)/kBT))
    
    return arctanpart * FD * p  
    #return  arctanpart

def generalRPAdielectric(k, omega, kBT, mu, nu):
    """
    Numerically calculates the dielectric function  in Random Phase 
    Approximation (RPA), epsilon_{RPA}(k, omega + i*nu). This function is 
    labelled general becuase the frequency argument is made complex to account
    for collisions due to ions. This alone is not a correct expression for the
    dielectric function, and is used in calculating the Mermin dielectric 
    function.
    
    Parameters:
    ___________
    
    Returns:
    ________
    """
    
    y0 = 0
    p = np.linspace(0, 10, 50)
    
    delta = 10**-7
    
    realODEsolve = odeint(realintegrand, y0, p, args = (k, omega, kBT, mu, nu,
                                                        delta))
    
    # a small nu causes some problems when integrating the imaginary part of
    # the dielectric. When nu is small, the integrand is like a modulated 
    # step function between p1 and p2.
    if abs(nu.real) < 10**-5:
        p1 = abs(k**2-2*omega)/(2*k)
        p2 = (k**2 + 2*omega)/(2*k)
        if p1 > p2:
            tmp = p1
            p1 = p2
            p2 = tmp
        p = np.linspace(p1, p2, 50)
        
    imagODEsolve = odeint(imagintegrand, y0, p, args = (k, omega, kBT, mu, nu))

    return complex(1 + 2 / np.pi / k**3 * realODEsolve[-1],
                   2 / np.pi / k**3 * imagODEsolve[-1])

def MerminDielectric(k, omega, kBT, mu, nu):
    """
    Numerically calculates the Mermin dielectric, which builds upon the RPA
    dielectric function by taking into account electron collisions with ions.
    
    Parameters:
    ___________
    
    Returns:
    ________
    """
    
    RPAcomplexfreq = generalRPAdielectric(k, omega, kBT, mu, nu)
    RPAzerofreq    = generalRPAdielectric(k, 0., kBT, mu, 0.)
    
    numerator   = (omega + 1j*nu)*(RPAcomplexfreq - 1)
    denominator = omega + 1j*nu * (RPAcomplexfreq - 1)/(RPAzerofreq - 1)
    
    return 1 + numerator/denominator

def ELF(k, omega, kBT, mu, nu):
    """
    Electron Loss Function, related to the amount of energy dissapated in the 
    system.
    """
    
    eps = MerminDielectric(k, omega, kBT, mu, nu)
    return eps.imag/(eps.real**2 + eps.imag**2)

# Tests
if __name__=='__main__':
    import matplotlib.pyplot as plt    

    k = 0.5
    mu = 0.279
    kbT = 1/27.2114
    # Using nu = 0 does not work so well!
    nu = 0
    
    
    # Initial tests for imaginary part
    '''
    w = 1.5
    p = np.linspace(0, 5, 200)
    y0 = 0
    intp = odeint(imagintegrand, y0, p, args=(k, w, kbT, mu, nu))
    intp = imagintegrand(0, p, k, w, kbT, mu, nu)
    #integrandargs = lambda p, y : imagintegrand(p, y, k, w, kbT, mu, nu)
    #intp = solve_ivp(integrandargs, [0., 10.], [y0], max_step=0.1)
    
    plt.plot(p, 2/np.pi/k**3 * intp, label="w = {}".format(w))
    plt.legend()
    plt.show
    '''
    w = np.linspace(0, 4, 200)
    import time
    start = time.time()
    elf = np.asarray([ELF(k, x, kbT, mu, nu) for x in w])
    print("time = {}".format(time.time()-start))
    plt.plot(w, elf, label='RPA')
    plt.legend()
    plt.show()

