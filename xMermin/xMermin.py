#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:30:41 2020

@author: tommy
"""

"""
Calculates the dielectric function with the extended Mermin approach. This 
combines both the Mermin and the LFC methods. For a detailed discussion see
A Wierling 2009 J. Phys. A: Math. Theor. 42 214051 

"""

from LFC import LFCDielectric as lfc
from Mermin import MerminDielectric as md

def xmermin(k, omega, nu, kBT, mu, G):
    """
    Numerically calculates the dielectric function in the extended Mermin
    approach. This is a combination of the dielectric function from the
    Mermin approximation and the local field correction approach.
    
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
    
    return md.generalMermin(lfc.generalLFC, k, omega, nu, kBT, mu, G)

def ELF(k, omega, nu, kBT, mu, G):
    eps = xmermin(k, omega, nu, kBT, mu, G)
    return eps.imag/(eps.imag**2 + eps.real**2)