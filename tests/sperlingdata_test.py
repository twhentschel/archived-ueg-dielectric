#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:29:24 2020

@author: tommy


Test file for comparing theory to warm dense aluminun - data from 
Sperling et al., PRL 2015
"""

import numpy as np
import matplotlib.pyplot as plt

import Mermin.MerminDielectric as md
import LFC.LFCDielectric as lfc
import xMermin.xMermin as xmd

# for PIMC UEG data
from keras import losses
from keras.models import Sequential
from keras import regularizers
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU


########################################################
""" Obtain collision frequencies from data file """

filename = "xMermin/tests/Al_6_eV_vw.txt"
w, RenuT, RenuB, ImnuT, ImnuB = np.loadtxt(filename, skiprows = 1, unpack=True)
# Scale the collision frequencies
RenuB = RenuB * RenuT[0]/RenuB[0]
ImnuB = ImnuB * RenuT[0]/RenuB[0]
nu = 1j*ImnuB; nu += RenuB

# plt.plot(w, RenuB, color="C0", label="Born (real)")
# plt.plot(w, ImnuB, color="C0", linestyle="--", label="Born (imag.)")
# plt.plot(w, RenuT, color="C3", label="T-matrix (real)")
# plt.plot(w, ImnuT, color="C3", linestyle="--", label="T-matrix (imag.)")
# plt.xlim(0, 70)
# plt.xlabel(r"$\omega$ (a.u.)", fontsize=14)
# plt.legend()
# plt.show()

########################################################

########################################################
""" Obtain Experimental data from Sperling et al."""
file_exp = "tests/Al_deconvolved.csv"
wexp, DSFexp = np.loadtxt(file_exp, unpack=True, delimiter=',')

########################################################

#########################################

"""
Obtain static local field correction from PIMC data of uniform electron gas.
"""
# Define LFC
LR = LeakyReLU()
LR.__name__ = 'relu'

# Define the Keras model

N_LAYER = 40
W_LAYER = 64

model = Sequential()
model.add(Dense(W_LAYER, input_dim=3, activation=LR))

REGULARIZATION_RATE = 0.0000006

for i in range(N_LAYER-1):
	model.add( Dense( W_LAYER, activation=LR, 
                  kernel_regularizer=regularizers.l2( REGULARIZATION_RATE)))

model.add(Dense(1, activation='linear'))

# Load the trained weights from hdf5 file
model.load_weights('LFC/tests/LFC.h5')

# Define simple wrapper function (x=q/q_F):

def GPIMC(q, ne, T):
    rs = (3/4/np.pi/ne)**(1/3)
    kF = (9.0*np.pi/4.0)**(1.0/3.0)/rs
    x = q/kF
    result = model.predict( np.array( [[x,rs,T]] ) )
    return result[0][0]

########################################################
    
###################################################
""" Parameters corresponding to Sperling data"""
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
########################################################

# Experimental data plot
plt.scatter(wexp, DSFexp, color='k', s=5, label='Sperling et al., 2015')

w0 = 7980. # eV
Ha = 27.2114 # Hartree
hbarc = 1240 # eV nm
k0 = 2.14 # au
theta = 24.

k = 2*k0*np.sin(theta*np.pi/180/2)
#k = 0.8898694380652493
# model plots
def DSF(k, w, T, ELF, ne):
    return 1/(1-np.exp(-w/T))*k**2/4/np.pi**2/ne * ELF
G = GPIMC(k, neau, T_au)
DSFRPA = []
DSFMD  = []
DSFLFC = []
DSFXMD = []
for x,y in zip(w, nu):
    DSFRPA.append(DSF(k, x, T_au, md.ELF(k, x, 0., T_au, muau), neau))
    DSFMD.append(DSF(k, x, T_au, md.ELF(k, x, y, T_au, muau), neau))
    #DSFLFC.append(DSF(k, x, T_au, lfc.ELF(k, x, T_au, muau, G), neau))
    #DSFXMD.append(DSF(k, x, T_au, xmd.ELF(k, x, y, T_au, muau, G), neau))

# We can normalize the dielectric function

plt.plot(w0 - w*Ha, DSFRPA/max(DSFRPA)*0.6, label="No interactions")
plt.plot(w0 - w*Ha, DSFMD/max(DSFMD)*0.6, label="e-i collisions from AA")
#plt.plot(w0 - w*Ha, DSFLFC/max(DSFLFC)*0.6, label="e-e correlations")
#plt.plot(w0 - w*Ha, DSFXMD/max(DSFXMD)*0.6, label="e-i + e-e")

plt.xlabel("Scattered Photon Energy (eV)")
plt.ylabel("Signal (arb. units)")
#plt.title((r"$E_0$ = {} eV, $\theta$ = {}$^o$, $T$ = {} eV, " +
#          r"$n_e$ = {} e/cc").format(w0, theta, TeV, ne_cgs))
plt.title("X-Ray Thompson Scattering from Al at 6 eV")

plt.xlim(7930, 8000)
plt.ylim(0, 1)

plt.legend()
plt.show()