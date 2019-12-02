#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 12:49:57 2019

@author: tommy
"""
import os
os.chdir("/home/th584/Documents/projects/dielectric-function")

from Mermin import MerminDielectric as MD
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
muau = 0.305 # mu for aluminum, with ne_cgs=1.8*10**23, T=1ev, Z*=3; au



#########################################

k = kFau

filename = "tests/Al_6_eV_vw.txt"
w, RenuT, RenuB, ImnuT, ImnuB = np.loadtxt(filename, skiprows = 1, unpack=True)
nu = 1j*ImnuB; nu += RenuB
ELF = np.asarray([MD.MerminELF(k, x, T_au, muau, y) for x,y in zip(w,nu)])

plt.plot(w, ELF)
