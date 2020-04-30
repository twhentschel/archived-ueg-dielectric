#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:18:04 2020

The electron loss function (ELF): Im(-1/\epsilon) is difficult to integrate
over as k -> 0. Perhaps we can fit the peak using a Lorentzian function. This
file aims to test the viability of this idea.

@author: tommy
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

import Mermin.MerminDielectric as MD

def lorentzian(x, x0, a, gamma):
    return a * gamma**2 / (gamma**2 + (x - x0)**2)

# Physical Parameters:
a0 = 0.529*10**-8 # Bohr radius, cm

TeV = 6 # Temperature, eV
ne_cgs = 1.8*10**23 # electron density, cm^-3

neau = ne_cgs * a0**3 # electron density, au
EFau =  0.5*(3*np.pi**2*neau)**(2/3)# Fermi energy, au
kFau = np.sqrt(2*EFau) # Fermi momentum/wavevector, au
T_au = TeV/27.2114 # au
wpau = np.sqrt(4*np.pi*neau)
# Needed to assume some mu. For this, we are assuming the material and then
# using a spreadsheet that Stephanie prepared to get mu
muau = 0.305 # mu for aluminum, with ne_cgs=1.8*10**23, T=6ev, Z*=3; au   
#

k = 0.2 # au
omega = np.linspace(0.607, 0.6076, 500)
RPAELF = [MD.ELF(k, x, 0, T_au, muau) for x in omega]

# initial parameter guesses
p0 = [0.6531, 1.8208e2, 1.317e-3]

# Fit

#params, pcov = opt.curve_fit(lorentzian, omega, RPAELF, p0)

# visualize

# ELF
plt.plot(omega, RPAELF, label='ELF')

# Lorentzian
#plt.plot(omega, lorentzian(omega, *params), label='Lorentzian')

plt.legend()
plt.show()


