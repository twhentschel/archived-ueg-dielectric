#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 12:49:57 2019

@author: tommy
"""

from Mermin import MerminDielectric as MD
from xMermin import xMermin as xM
import numpy as np
import matplotlib.pyplot as plt

######## Parameters ###############
a0 = 0.529*10**-8 # Bohr radius, cm
TeV = 6 # Temperature, eV
ne_cgs = 1.8*10**23 # electron density, cm^-3
neau = ne_cgs * a0**3 # electron density, au
EFau =  0.5*(3*np.pi**2*neau)**(2/3)# Fermi energy, au
kFau = np.sqrt(2*EFau) # Fermi momentum/wavevector, au
T_au = TeV/27.2114 # au
#kFau = kF_eV / 27.2114 # au
wpau = np.sqrt(4*np.pi*neau)

# Needed to assume some mu. For this, we are assuming the material and then
# using a spreadsheet that Stephanie prepared to get mu
muau = 0.305 # mu for aluminum, with ne_cgs=1.8*10**23, T=6ev, Z*=3; au



#########################################

k = 0.1

filename = "tests/Al_6_eV_vw.txt"
w, RenuT, RenuB, ImnuT, ImnuB = np.loadtxt(filename, skiprows = 1, unpack=True)
nu = 1j*ImnuB; nu += RenuB

w = w
# count = 0
# for k in (1, 0.5, 0.3, 0.1):
#     ELFmermin = np.asarray([MD.ELF(k, x, y, T_au, muau) for x,y in zip(w,nu)])
#     ELFrpa = np.asarray([MD.ELF(k, x, 0, T_au, muau) for x in w])
    
#     plt.plot(w/wpau, ELFmermin, label = "k = {} au".format(k), c="C" + str(count))
#     plt.plot(w/wpau, ELFrpa, linestyle='-.', c="C" + str(count))
#     count = count + 1

# plt.xlabel(r"$\omega/\omega_p$")
# plt.ylabel("ELF")
# plt.xlim(0, 3)
# plt.legend()
# plt.show()

eps = np.asarray([MD.MerminDielectric(k, x, y, T_au, muau) 
                  for x,y in zip(w,nu)])

#plt.plot(w, eps.real, label="real")
#plt.plot(w, eps.imag, label="imag")
#plt.legend()
plt.plot(w, w*eps.imag/4/np.pi)
plt.show()
