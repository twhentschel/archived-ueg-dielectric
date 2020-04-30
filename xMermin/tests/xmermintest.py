#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 17:21:49 2020

@author: tommy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import xMermin.xMermin as xmd

# for PIMC UEG data
from keras import losses
from keras.models import Sequential
from keras import regularizers
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU

######## Parameters ###############
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



#################################################

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
    
########################################################
""" Obtain collision frequencies from data file """

filename = "xMermin/tests/Al_6_eV_vw.txt"
w, RenuT, RenuB, ImnuT, ImnuB = np.loadtxt(filename, skiprows = 1, unpack=True)
nu = 1j*ImnuB * RenuT[0]/RenuB[0]; nu += RenuB * RenuT[0]/RenuB[0]
#nu = 1j*ImnuT; nu += RenuT
########################################################


# Let's take a look
k = 0.01
#w = np.linspace(10**-4, 4*wpau)
G = GPIMC(k, neau, T_au)
#w=w[0:300]

# xmerm = [xmd.xmermin(k, x, y, T_au, muau, G).imag
#          for x,y in zip(w,nu)]
merm = [xmd.xmermin(k, x, y, T_au, muau, 0).imag
         for x,y in zip(w,nu)]
# lfc = [xmd.xmermin(k, x, 0, T_au, muau, G).imag
#          for x in w]
# rpa = [xmd.xmermin(k, x, 0, T_au, muau, 0).imag
#       for x in w]

#p1 = plt.plot(w, w*xmerm/4/np.pi, label="xMermin")

plt.semilogx(w, w*merm/4/np.pi, label="Mermin (Born)")
#plt.semilogx(w, w*lfc/4/np.pi, label="LFC")
#plt.semilogx(w, w*rpa/4/np.pi, label="RPA")
plt.semilogx(w, neau*nu[0] / (w**2 + nu[0]**2), label="Drude")

plt.xlabel(r'$\omega$ (a.u.)')
plt.ylabel(r'$\sigma_1 (\omega)$ (a.u.)')
plt.title(("Real conductivity for Al @ T = {} eV, " + 
          "ne = {} 1/cc").format(TeV, ne_cgs))
plt.legend()
plt.show()