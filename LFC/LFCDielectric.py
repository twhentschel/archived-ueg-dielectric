#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:54:36 2020

@author: tommy

Calculates the dielectric function with the help of a local field correction
(LFC). This improves upon the RPA by considering electron correlations.

"""

from RPA.RPA_ODEsolve import RPAeps

def LFCDielectric(k, omega, kBT, mu, G):
    """
    Numerically calculates the dielectric function with the LFC term (G).
    
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
    RPAeps_val = RPAeps(k, omega, kBT, mu)
    return 1 - (1 - RPAeps)/(1 + (1 - RPAeps)*G)
