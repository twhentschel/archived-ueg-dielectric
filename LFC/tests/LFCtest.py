#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:34:32 2020

@author: tommy
"""
import numpy as np
import matplotlib.pyplot as plt

from LFC.LFCDielectric import LFCDielectric

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
	model.add( Dense( W_LAYER, activation=LR, kernel_regularizer=regularizers.l2( REGULARIZATION_RATE ) ) )

model.add(Dense(1, activation='linear'))

# Load the trained weights from hdf5 file
model.load_weights('LFC/tests/LFC.h5')

# Define simple wrapper function (x=q/q_F):

def GPIMC(q, ne, T):
    rs = (3/4/np.pi/ne)**(1/3)
    kF = (9.0*np.pi/4.0)**(1.0/3.0)/rs
    x = q/kF
    result = model.predict( np.array( [[x,rs,theta]] ) )
    return result[0][0]


# Let's take a look
k = 1
theta = 24 # scattering angle
w = np.linspace(0, 4*wpau)

lfcepsreal = [LFCDielectric(k, x, T_au, muau, GPIMC(k, neau, T_au)).real 
              for x in w]
rpaepsreal = [LFCDielectric(k, x, T_au, muau, 0).real for x in w]

plt.plot(w, lfcepsreal)
plt.plot(w, rpaepsreal)
plt.show()