#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:54:36 2020

@author: tommy

Calculates the dielectric function with the help of a local field correction
(LFC). This improves upon the RPA by considering electron correlations.

"""

import Mermin.MerminDielectric as MD

def generalLFC(k, omega, nu, kBT, mu, G):
    """
    Numerically calculates the dielectric function with the LFC term (G). 
    This function is labelled general becuase the frequency argument is made 
    complex as epsilon_{LFC}(k, omega+i*nu, ...).
    
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
        Collision frequency, typically a function of omega but this function
        will only accept one value at a time. 
    G: scalar
        Local field correction term. Typically a function of k and omega, but
        only pass in values one at a time.

    """
    eps_val = MD.generalRPAdielectric(k, omega, nu, kBT, mu)
    return 1 - (1 - eps_val)/(1 + (1 - eps_val)*G)

def LFCdielectric(k, omega, kBT, mu, G):
    """
    Numerically calculates the dielectric function with the LFC term (G). The
    frequency input is considered real (i.e. nu = 0 in relation to the above
    function)
    
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
    G: scalar
        Local field correction term. Typically a function of k and omega, but
        only pass in values one at a time.

    """
    return generalLFC(k, omega, 0, kBT, mu, G)

def ELF(k, omega, kBT, mu, G):
    """
    Electron Loss Function, related to the amount of energy dissapated in the 
    system.
    """
    
    eps = LFCdielectric(k, omega, kBT, mu, G)
    return eps.imag/(eps.real**2 + eps.imag**2)