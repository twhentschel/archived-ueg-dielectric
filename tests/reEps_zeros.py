 # frequency zero of the real part of the dielectric function.
 
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
from WDM_codes.DielectricFunc.RPA import rpa_dielectric
#from WDM_codes.DielectricFunc.RPA import KK_rpaeps
from WDM_codes.DielectricFunc.RPA import RPA_ODEsolve

# Theoretical positions oWDM_codes.DielectricFunc.RPA importf the peak
def wp(n):
    # Basic plasma frequency, in a.u.
    return np.sqrt(4 * np.pi * n )

def BG_wp(n, k, kbT):
    """
    Bohm-Gross dispersion relation for the plasma frequency, in a.u.
    n - electron density (au)
    k - wavenumber (au)
    kbT - thermal energy (au)
    """
    return np.sqrt(wp(n)**2 + 3*kbT*k**2)
    
def modBG_wp(n, k, kbT):
    """
    Modified Bohm-Gross dispersion relation, given in Glezner & Redmer, Rev.
    Mod. Phys., 2009, Eqn (16).
    
    n - electron density (au)
    k - wavenumber (au)
    kbT - thermal energy (au)
    """  
    
    thermal_deBroglie = np.sqrt(2*np.pi/kbT)
    
    return np.sqrt(BG_wp(n, k, kbT)**2 + 3*kbT*k**2 \
                    * 0.088 * n * thermal_deBroglie**3 \
                    + (k**2/2)**2)

 
######## Parameters ###############
a0 = 0.529*10**-8 # Bohr radius, cm
TeV = 1 # Temperature, eV
ne_cgs = 1*10**23 # electron density, cm^-3
neau = ne_cgs * a0**3 # electron density, au
EFau =  0.5*(3*np.pi**2*neau)**(2/3)# Fermi energy, au
kFau = np.sqrt(2*EFau) # Fermi momentum/wavevector, au
T_au = TeV/27.2114 # au
#kFau = kF_eV / 27.2114 # au
wpau = np.sqrt(4*np.pi*neau)

# Needed to assume some mu. For this, we are assuming the material and then
# using a spreadsheet that Stephanie prepared to get mu
muau = 0.279 # mu for aluminum, with ne_cgs=10**23, T=1ev, Z*=3; au
#########################################

k = 0.1 #kFau

# To find the zero, we must look within some interval [a, b] where the f(a) is
# negative and the f(b) is positive.
a = wp(neau)
b = modBG_wp(neau, k, T_au)

def reeps(w):
    return  RPA_ODEsolve.re_RPAeps(k, w, T_au, muau)
    #return rpa_dielectric.rpa_eps(k, w, T_au, muau, 2000)[0]
    

print("a = {}, reeps(a) = {}".format(a, reeps(a)))
while reeps(a) > 0:
    dfdx = (reeps(b)-reeps(a))/(b-a)
    print(dfdx)
    x = -reeps(a) / dfdx
    a = a + x
print("a = {}, reeps(a) = {}".format(a, reeps(a)))    
print("b = {}, reeps(b) = {}".format(b, reeps(b)))
last = a
while reeps(b) < 0:
    # Calculate an approximate derivative of f at b
    dfdx = (reeps(b)-reeps(last))/(b-last)
    print("slope = {}".format(dfdx))
    # Pick x s.t. f(b+x) = 0
    x = - reeps(last) / dfdx  
    last = b 
    b = b + x       
print("b = {}, reeps(b) = {}".format(b, reeps(b)))   
 
numroot = opt.brentq(reeps, a, b)
analroot = b 
 
w = np.linspace(0, 1, 20)
re_eps = np.asarray([reeps(x) for x in w])
plt.plot(w, re_eps)
#plt.axvline(numroot, color='red')
#plt.axvline(wpau, color='green')
plt.axhline(0, color='k', linestyle='--')
plt.show()

# At what k does wp give the minimum?
'''
w=wpau
def reeps(k):
    return  RPA_ODEsolve.re_RPAeps(k, w, T_au, muau)
# From experience
b = 0.2
a = 0.1
numroot = opt.brentq(reeps, a, b)
'''