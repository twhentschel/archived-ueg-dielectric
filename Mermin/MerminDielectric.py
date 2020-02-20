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
from scipy.integrate import solve_ivp
def realintegrand(p, y, k, omega, nu, kBT, mu, delta):
    """
    The integrand present in the formula for the real part of the general
    RPA dielectric function.
    
    Parameters:
    ___________
    p: scalar
        The integration variable, which is also the momemtum of the electronic
        state.
    y: function
        Only necessary for the ODE code, has no meaning here.
    k: scalar
        The change of momentum for an incident photon with momentum k0 
        scattering to a state with momentum k1: k = |k1-k0|, in a.u.
    omega: scalar
        The change of energy for an incident photon with energy w0 
        scattering to a state with energy w1: w = w0-w1, in a.u.
    kBT: scalar
        Thermal energy (kb - Boltzmann's constant, T is temperature) in a.u.
    mu: scalar
        Chemical potential in a.u.    
    nu: scalar
        Collision frequency in a.u.         
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

def imagintegrand(p, y, k, omega, nu, kBT, mu):
    """
    The integrand present in the formula for the imaginary part of the general
    RPA dielectric function.
    
    Parameters:
    ___________
    p: scalar
        The integration variable, which is also the momemtum of the electronic
        state.
    y: function
        Only necessary for the ODE code, has no meaning here.
    k: scalar
        The change of momentum for an incident photon with momentum k0 
        scattering to a state with momentum k1: k = |k1-k0|, in a.u.
    omega: scalar
        The change of energy for an incident photon with energy w0 
        scattering to a state with energy w1: w = w0-w1, in a.u.
    kBT: scalar
        Thermal energy (kb - Boltzmann's constant, T is temperature) in a.u.
    mu: scalar
        Chemical potential in a.u.   
    nu: scalar
        Collision frequency in a.u. 
    
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

def generalRPAdielectric(k, omega, nu, kBT, mu):
    """
    Numerically calculates the dielectric function  in Random Phase 
    Approximation (RPA), epsilon_{RPA}(k, omega + i*nu). This function is 
    labelled general becuase the frequency argument is made complex to account
    for collisions due to ions. This alone is not a correct expression for the
    dielectric function, and is used in calculating the Mermin dielectric 
    function.
    
    Parameters:
    ___________
    k: scalar
        The change of momentum for an incident photon with momentum k0 
        scattering to a state with momentum k1: k = |k1-k0|, in a.u.
    omega: scalar
        The change of energy for an incident photon with energy w0 
        scattering to a state with energy w1: w = w0-w1, in a.u.
    kBT: scalar
        Thermal energy (kb - Boltzmann's constant, T is temperature) in a.u.
    mu: scalar
        Chemical potential in a.u.
    nu: scalar
        Collision frequency in a.u. 
        
    Returns:
    ________
    """
    
    y0 = [0]
    # Limits of integration - 10 sufficiently acts like infinity in this 
    # problem.
    plim = (0, 10)
    # Change the tolerance - matches scipy.integrate.odeint
    tol = 1.49012e-8
    
    delta = 10**-7
    realintargs = lambda p, y: realintegrand(p, y, k, omega, nu, kBT, mu, 
                                             delta)
    realODEsolve = solve_ivp(realintargs, plim, y0, method='LSODA',
                             rtol=tol, atol=tol)
    
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
        plim = (p1, p2)

    imagintargs = lambda p, y: imagintegrand(p, y, k, omega, nu, kBT, mu)
    
    imagODEsolve = solve_ivp(imagintargs, plim, y0, method='LSODA',
                             rtol=tol, atol=tol)
    
    return complex(1 + 2 / np.pi / k**3 * realODEsolve.y[0][-1],
                   2 / np.pi / k**3 * imagODEsolve.y[0][-1])

def generalMermin(epsilon, k, omega, nu, *args):
    """
    Numerically calculates the Mermin dielectric function. This adds some ionic
    structure to the dielectric function passed through epsilon. Typically this
    will be the RPA dielectric function, but we also want to allow for a 
    general dielectric functions.
    
    Parameters:
    ___________
    epsilon: function
        dielectric function that we want to add ionic information to. The 
        argument structure must be epsilon(k, omega, nu, args) and args must
        be ordered properly.
    k: scalar
        The change of momentum for an incident photon with momentum k0 
        scattering to a state with momentum k1: k = |k1-k0|, in a.u.
    omega: scalar
        The change of energy for an incident photon with energy w0 
        scattering to a state with energy w1: w = w0-w1, in a.u.
    nu: scalar
        Collision frequency in a.u. 
    args: tupple
        Additional arguments (temperature, chemical potential, ...)
    """
    
    epsnonzerofreq = epsilon(k, omega, nu, *args)
    epszerofreq    = epsilon(k, 0, 0, *args)
    
    numerator   = (omega + 1j*nu)*(epsnonzerofreq - 1)
    denominator = omega + 1j*nu * (epsnonzerofreq - 1)/(epszerofreq - 1)
    
    # if this case is not handled seperately, it can return a bad answer when
    # omega == 0.
    if abs(nu) == 0:
        return epsnonzerofreq
    
    return 1 + numerator/denominator
    
def MerminDielectric(k, omega, nu, kBT, mu):
    """
    Numerically calculates the Mermin dielectric, which builds upon the RPA
    dielectric function by taking into account electron collisions with ions.
    
    Parameters:
    ___________
    k: scalar
        The change of momentum for an incident photon with momentum k0 
        scattering to a state with momentum k1: k = |k1-k0|, in a.u.
    omega: scalar
        The change of energy for an incident photon with energy w0 
        scattering to a state with energy w1: w = w0-w1, in a.u.
    kBT: scalar
        Thermal energy (kb - Boltzmann's constant, T is temperature) in a.u.
    mu: scalar
        Chemical potential in a.u.
    nu: scalar
        Collision frequency in a.u. 
        
    Returns:
    ________
    """
    
    return generalMermin(generalRPAdielectric, k, omega, nu, kBT, mu)

def ELF(k, omega, nu, kBT, mu):
    """
    Electron Loss Function, related to the amount of energy dissapated in the 
    system.
    """
    
    eps = MerminDielectric(k, omega, nu, kBT, mu)
    return eps.imag/(eps.real**2 + eps.imag**2)

# Tests
if __name__=='__main__':
    import matplotlib.pyplot as plt    

    k = 0.246
    mu = 0.126
    kbT = 6/27.2114
    
    nu = 0
    
    
    # Initial tests for imaginary part
    '''
    w = 0
    p = (0, 10)
    y0 = [0]
    delta = 10**-10
    
    imagintargs = lambda p, y: imagintegrand(p, y, k, w, kbT, mu, nu)
    intpivp = solve_ivp(imagintargs, p, y0, method='LSODA',
                        rtol=1.49012e-8, atol=1.49012e-8,
                        max_step=1)
    p = np.linspace(0, 10, 100)
    intpode = odeint(imagintegrand, y0[0], p, tfirst=True,
                     args=(k, w, kbT, mu, nu))
    #intp = imagintegrand(0, p, k, w, kbT, mu, nu)
    #integrandargs = lambda p, y : imagintegrand(p, y, k, w, kbT, mu, nu)
    #intp = solve_ivp(integrandargs, [0., 10.], [y0], max_step=0.1)
    
    plt.plot(intpivp.t, intpivp.y[0], label="ivp")
    plt.plot(p, intpode, label="ode")
    
    plt.legend()
    plt.show
    '''
    w = np.linspace(0, 4, 200)

    import time
    start = time.time()
    eps = np.asarray([MerminDielectric(k, x, nu, kbT, mu) for x in w])
    print("time = {}".format(time.time()-start))
    plt.plot(w, eps.real, label='RPA')
    plt.legend()
    plt.show()

