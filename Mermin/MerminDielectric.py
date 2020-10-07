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

def realintegrand(p, k, omega, nu, kBT, mu, delta):
    """
    The integrand present in the formula for the real part of the general
    RPA dielectric function.
    
    Parameters:
    ___________
    p: scalar
        The integration variable, which is also the momemtum of the electronic
        state.
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
    # Modifying delta is probably unnecessary and in fact, I don't think 
    # including it as a input parameter is even useful because its impact on
    # the code are convoluted.
    deltamod = delta**(1/2.)
    
    
    # variables to avoid verbose lines later on.
    pp = (k**2 + 2*(omega-nu.imag) + 2*p*k)**2 + (2*nu.real + deltamod)**2
    pm = (k**2 + 2*(omega-nu.imag) - 2*p*k)**2 + (2*nu.real + deltamod)**2
    mp = (k**2 - 2*(omega-nu.imag) + 2*p*k)**2 + (2*nu.real + deltamod)**2
    mm = (k**2 - 2*(omega-nu.imag) - 2*p*k)**2 + (2*nu.real + deltamod)**2
    
    logpart = np.log(np.sqrt(pp/pm)) + np.log(np.sqrt(mp/mm))
    
    FD = 1/(1+np.exp((p**2/2 - mu)/kBT))
    
    return logpart * FD * p

def imagintegrand(p, k, omega, nu, kBT, mu):
    """
    The integrand present in the formula for the imaginary part of the general
    RPA dielectric function.
    
    Parameters:
    ___________
    p: scalar
        The integration variable, which is also the momemtum of the electronic
        state.
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
    
    # Integral for real part of the dielectric function
    delta = 10**-7
    realintargs = lambda p, y: realintegrand(p, k, omega, nu, kBT, mu, 
                                             delta)
    realODEsolve = solve_ivp(realintargs, plim, y0, method='LSODA',
                             vectorized=True, rtol=tol, atol=tol)
    
    # Integral for the imag part of the dielectric function
    
    # a small nu causes some problems when integrating the imaginary part of
    # the dielectric. When nu is small, the integrand is like a modulated 
    # step function between p1 and p2.
    if abs(nu) < 10**-5:
        p1 = abs(k**2-2*omega)/(2*k)
        p2 = (k**2 + 2*omega)/(2*k)
        if p1 > p2:
            tmp = p1
            p1 = p2
            p2 = tmp
        plim = (p1, p2)
    
    p1 = abs(k**2-2*omega)/(2*k)
    p2 = (k**2 + 2*omega)/(2*k)
    
    plims = np.sort([0, p1, p2, 10])
    plims = plims[plims <= 10]
    
    imagintargs = lambda p, y: imagintegrand(p, k, omega, nu, kBT, mu)
    
    imagODEsolve = sum([solve_ivp(imagintargs, (plims[i-1], plims[i]), y0,
                                  #method='LSODA', rtol=tol, atol=tol,
                                  rtol=tol, atol=tol,
                                  vectorized=True).y[0][-1] \
                        for i in range(1, len(plims))])

    return complex(1 + 2 / np.pi / k**3 * realODEsolve.y[0][-1],
                   2 / np.pi / k**3 * imagODEsolve)

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
    args: tuple
        Additional arguments (temperature, chemical potential, ...). Must be 
        same order as in the epsilon() function.
    """
    
    epsnonzerofreq = epsilon(k, omega, nu, *args)
    epszerofreq    = epsilon(k, 0, 0, *args)
    
    # If both nu is zero, expect epsnonzerofreq. But if omega also equals zero,
    # this code fails. Add a little delta to omega to avoid this.
    delta = 1e-10
    numerator   = ((omega + delta) + 1j*nu)*(epsnonzerofreq - 1)
    denominator = (omega+delta) \
                  + 1j*nu * (epsnonzerofreq - 1)/(epszerofreq - 1)
    
    
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

    # k = 10
    # mu = 0.305
    # kbT = 6/27.2114
    
    # nu = 0
    
    
    # # Initial tests for imaginary part
    
    # w = 1
    # p = (0, 10)
    # y0 = [0]
    # delta = 10**-10
    
    # imagintargs = lambda p, y: imagintegrand(p, k, w, kbT, mu, nu)
    # intpivp = solve_ivp(imagintargs, p, y0, method='LSODA',
    #                     rtol=1.49012e-8, atol=1.49012e-8,
    #                     max_step=1)
    # p = np.linspace(0, 10, 100)

    # #intp = imagintegrand(0, p, k, w, kbT, mu, nu)
    # #integrandargs = lambda p, y : imagintegrand(p, k, w, kbT, mu, nu)
    # #intp = solve_ivp(integrandargs, [0., 10.], [y0], max_step=0.1)
    
    # plt.plot(intpivp.t, intpivp.y[0], label="ivp")
    
    # plt.legend()
    # plt.show
    # '''
    # w = np.linspace(0, 1, 200)

    # import time
    # start = time.time()
    # eps = np.asarray([MerminDielectric(k, x, nu, kbT, mu) for x in w])
    # print("time = {}".format(time.time()-start))
    # plt.plot(w, eps.imag, label='RPA')
    # plt.legend()
    # plt.show()
    # '''
    k = 0.45
    T = 6/27.2114
    mu = 0.305
    w = np.linspace(0, 35/27.2114, 150)
    eps = np.asarray([MerminDielectric(k, x, 0, T, mu) for x in w])
    plt.plot(w*27.2114, eps.real, color='black')
    plt.plot(w*27.2114, eps.imag, color='red', linestyle='--')
    plt.plot(w*27.2114, eps.imag/(eps.real**2 + eps.imag**2),
             color='royalblue', linestyle='-.')
    plt.xlabel(r'$\omega = \omega_s - \omega_0$')
    plt.xlim(0, 35)
    plt.ylim(-0.72, 7)
    plt.show()
