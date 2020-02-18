 # frequency zero of the real part of the dielectric function.
 
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
import Mermin.MerminDielectric as MD

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
TeV = 6 # Temperature, eV
ne_cgs = 1*10**23 # electron density, cm^-3
neau = ne_cgs * a0**3 # electron density, au
EFau =  0.5*(3*np.pi**2*neau)**(2/3)# Fermi energy, au
kFau = np.sqrt(2*EFau) # Fermi momentum/wavevector, au
T_au = TeV/27.2114 # au
#kFau = kF_eV / 27.2114 # au
wpau = np.sqrt(4*np.pi*neau)

# Needed to assume some mu. For this, we are assuming the material and then
# using a spreadsheet that Stephanie prepared to get mu
muau = 0.126 # mu for aluminum, with ne_cgs=10**23, T=1ev, Z*=3; au
#########################################

k = 1e-7#kFau

def reeps(w):
    return  MD.MerminDielectric(k, w, T_au, muau, 0).real

############## Bracketed Method ###################
# """
# This way of finding the zero currently only works if there is a zero.
# Additionally, determining the bracket is kind of reminiscent of the secant
# method, so why don't I just try using that instead?
# """
# # To find the zero, we must look within some interval [a, b] where the f(a) is
# # negative and the f(b) is positive.
# a = wp(neau)
# b = modBG_wp(neau, k, T_au)

# def reeps(w):
#     return  MD.MerminDielectric(k, w, T_au, muau, 0).real
#     #return rpa_dielectric.rpa_eps(k, w, T_au, muau, 2000)[0]
    

# print("a = {}, reeps(a) = {}".format(a, reeps(a)))
# while reeps(a) > 0:
#     dfdx = (reeps(b)-reeps(a))/(b-a)
#     print(dfdx)
#     x = -reeps(a) / dfdx
#     a = a + x
# print("a = {}, reeps(a) = {}".format(a, reeps(a)))    
# print("b = {}, reeps(b) = {}".format(b, reeps(b)))
# last = a
# while reeps(b) < 0:
#     # Calculate an approximate derivative of f at b
#     dfdx = (reeps(b)-reeps(last))/(b-last)
#     print("slope = {}".format(dfdx))
#     # Pick x s.t. f(b+x) = 0
#     x = - reeps(last) / dfdx  
#     last = b 
#     b = b + x       
# print("b = {}, reeps(b) = {}".format(b, reeps(b)))   
 
# numroot = opt.brentq(reeps, a, b)
# analroot = b 
 
# w = np.linspace(0, 1, 20)
# re_eps = np.asarray([reeps(x) for x in w])
# plt.plot(w, re_eps)
# #plt.axvline(numroot, color='red')
# #plt.axvline(wpau, color='green')
# plt.axhline(0, color='k', linestyle='--')
# plt.show()

# # At what k does wp give the minimum?
# '''
# w=wpau
# def reeps(k):
#     return  RPA_ODEsolve.re_RPAeps(k, w, T_au, muau)
# # From experience
# b = 0.2
# a = 0.1
# numroot = opt.brentq(reeps, a, b)
# '''

############# Secant Method #############
def invelf(w):
    return  -MD.ELF(k, w, T_au, muau, 0)

# Find minimum
omegamin = opt.minimize_scalar(reeps, bracket=(0, modBG_wp(neau, k, T_au)), 
                               method='brent')

omegamin2 = opt.minimize_scalar(invelf, bracket=(0, modBG_wp(neau, k, T_au)),
                                method='brent')

root = modBG_wp(neau, k, T_au)
if reeps(omegamin.x) < 0:
    print("reeps(omegamin) < 0")
    try:
        root = opt.newton(reeps, root) # tried timing, no big difference at 
                                    # least for this case (0.04s  vs 0.08 s)
    except RuntimeError:
        print("Switching from Newton's method to the safer bisection method" +
              " to find the maximum position of the ELF.")
        maxiter = 10
        upperlim = root
        while reeps(upperlim) < 0 and maxiter > 0:
            upperlim = upperlim + k
        root = opt.brentq(reeps, omegamin.x, upperlim) # does not break for really
                                                  # small k
    #root = opt.newton(reeps, root, maxiter=23, full_output=True, tol=1e-5)

#plt.axvline(omegamin.x, c='g')

########### Find out at which omega the ELF is roughly 0 ############


# k-integral step
kstep = 1e-4

#effectiveinf = opt.newton(elf, root + kstep, tol=1e-5)
#print((elf(root+kstep) - elf(root))/(kstep))
