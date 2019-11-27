from WDM_codes.DielectricFunc.RPA import rpa_dielectric 
from WDM_codes.DielectricFunc.RPA import RPADielectric
from WDM_codes.DielectricFunc.RPA import RPA_ODEsolve
from WDM_codes.StoppingNumb.integrate import compositequad
from WDM_codes.DielectricFunc.RPA.tests.integrateELF import omegaintegral

import numpy as np
import matplotlib.pyplot as plt

# Theoretical positions of the peak
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
                    
def ELFWidth(k, kbT, mu, tol=15):
    """
    The width of the ELF curve, assuming the peak is roughly around k**2/2 
    (works well for high k, when modBG_wp(...) ~ k**2/2). tol roughly defines
    a tolerance, such that exp[(mu - b(k,ELFWidth(...))**2/2)/kbT] = exp(-tol),
    where b(k, w) = (2*w - k**2)/(2*k). These are used in the imaginary part of
    the RPA dielectric function. The total accuracy will go something like
    ln(1+exp(-tol))/k**3
    
    lim is a string for 'upper' or 'lower' ELF edge.
    """
    gamma = np.sqrt(2*(tol*kbT + mu))

    return gamma * k
    

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

def kscatter(k0, theta):
    """
    Calculates the momentum transfer for small relatie momentum transfers 
    (|k0| >~ |ks|). k0 is in au, and theta is in degrees. returns 
    k = |ks - k0|.
    """
    return 2*k0*np.sin(theta*np.pi/180/2)

w = np.linspace(0, 2, 200) #np.linspace(0, 4*wpau, 1000)
k0 = 2.14 # au
theta = 24 # degrees
k = 0.45# kscatter(k0, theta)
#reeps, imeps = RPADielectric.rpa_eps(k, w, T_au, muau)
#reeps, imeps = rpa_dielectric.rpa_eps(k, w, T_au, muau, 2000)
'''
reeps, imeps = zip(*[rpa_dielectric.rpa_eps(x, w, T_au, muau, 2000) for x in k])
reeps = np.asarray(reeps)
imeps = np.asarray(imeps)
'''
'''
datfile = "WDM_codes\\RPA\\RPAfor_zeroT.txt"
w, reeps, imeps = np.loadtxt(datfile, unpack=True)
'''
# Low-Omega approximation to ELF
'''
reeps, imeps = rpa_dielectric.rpa_eps(k, 0, T_au, muau, 2000)
ELF = 2*w/k**3/reeps**2/(1+np.exp((-muau + k**2/8)/T_au))
plt.plot(w, ELF, label='low-omega approx.')
'''

# Low-k approximation to RE{eps}
'''
reeps, imeps = rpa_dielectric.rpa_eps(k, w, T_au, muau, 2000)
reeps = 1 - wp(neau)/w**2*(1+1/w**2*(3/5*(k*kFau)**2 + k**4/4))
plt.plot(w, imeps/(imeps**2 + reeps**2), label='low-k approx.')
'''
'''
plt.plot(w, w*imeps/(imeps**2+reeps**2), label="k={:.2f}".format(k))#, color="lightseagreen")
plt.plot(w, reeps, linestyle='-.')
plt.plot(w, imeps, linestyle='--')
'''
# Exact/numerical RPA epsilon

ELF = np.asarray([RPA_ODEsolve.ELF(k, x, T_au, muau) for x in w])
reeps = np.asarray([RPA_ODEsolve.re_RPAeps(k,x,T_au,muau) for x in w])
imeps = RPA_ODEsolve.im_RPAeps(k,w,T_au,muau)
#DSF = [1/(1-np.exp(-x/T_au)) * k**2/(4*np.pi**2 * neau) \
#       * RPA_ODEsolve.ELF(k, x, T_au, muau) for x in w]

#plt.plot(7980 - w*27.2114, DSF, label='RPA (no collisions)')
plt.plot(w*27.2114, ELF, linestyle='-.', linewidth=2)
plt.plot(w*27.2114, reeps, color='k', linewidth=2)
plt.plot(w*27.2114, imeps, color='red', linestyle='--', linewidth=2)
plt.xlabel(r"$\omega = \omega_0 - \omega_1$ (eV)")
plt.ylabel(r"$\varepsilon(k,\omega)$")
plt.ylim(-0.7,7)
plt.xlim(0,40)

# Zero temperature
'''
def kTF(n):
    EF = 0.5*(3*np.pi**2*n)**(2/3)# Fermi energy, au
    return 6 * np.pi * n / EF
    
def T0_reeps(k, w, n):
    EF =  0.5*(3*np.pi**2*n)**(2/3)# Fermi energy, au
    kF = np.sqrt(2*EF) # Fermi momentum/wavevector, au
    ek = k**2/2 # energy
    return 1 + 1/2 * kTF(n)**2/k**2 * (1 \
                            + 1/(2*kF*k**3) * (4*EF*ek - (ek + w)**2) \
                            * np.log(abs(ek + k*kF + w)/abs(ek - k*kF + w))\
                            + 1/(2*kF*k**3) * (4*EF*ek - (ek - w)**2) \
                            * np.log(abs(ek + k*kF - w)/abs(ek - k*kF - w)))
plt.plot(w, T0_reeps(k, w, neau), label='T=0')
plt.axhline(0, color='k', linestyle='--')
plt.axvline(wpau, color='red')
'''
# ELF
'''
plt.plot(w/wpau, imeps/(imeps**2 + reeps**2), label='orig')
'''
'''
#plt.plot(w, reeps, linestyle='--', label='orig')
#plt.plot(w, imeps, linestyle='--', label='orig')
#plt.plot(k, reeps_smallk(k, w, T_au, muau), label="small-k approx")
# Let's integrate this!
integ, wmax, left, right = omegaintegral(k, T_au, muau, neau)
plt.axvline(left, c='green')
plt.axvline(right, c='green')
plt.axvline(wmax, c='red')
print(integ)
'''

'''
plt.axvline(wp(neau)/wpau, c='green', label=r"$\omega_p$")
plt.axvline(BG_wp(neau, k, T_au)/wpau, c='orange', label=r"$\omega_{BG}$")
plt.axvline(modBG_wp(neau, k, T_au)/wpau, c='red', label=r"$\omega_{mBG}$")
plt.axvline(k**2/2/wpau, c='magenta', label=r"$\omega_{k>>1}$")
'''

'''
plt.axvline(wp(neau), c='green', label=r"$\omega_p$")
plt.axvline(BG_wp(neau, k, T_au), c='orange', label=r"$\omega_{BG}$")
plt.axvline(modBG_wp(neau, k, T_au), c='red', label=r"$\omega_{mBG}$")
plt.axvline(k**2/2, c='magenta', label=r"$\omega_{k>>1}$")
'''

# Tanh remapping
'''
u = np.linspace(0, 10, 1000)
k = 2
v = 1
wmax = modBG_wp(neau, k, T_au)
reeps, imeps = rpa_dielectric.rpa_eps(k, k*v * np.tanh(u), 
                                      T_au, muau, 2000)
plt.plot(u, k*v * np.tanh(u)/k*imeps/(imeps**2+reeps**2) \
         * k*v / np.cosh(u)**2, 
         label="k={:.2f}".format(k))

def usubintegrand(u):
    reeps, imeps = RPADielectric.rpa_eps(k, k*v * np.tanh(u), 
                                          T_au, muau)
    return k*v * np.tanh(u)*imeps/(imeps**2+reeps**2) \
            * k*v / np.cosh(u)**2

print(compositequad(usubintegrand, 0, 15, 1000, "trapezoidal"))

plt.xlabel(r"u")
'''
#plt.xlim(0, 4)
#plt.ylabel("Re{eps}")
#plt.xlabel("w (eV)")
#plt.ylim(-20, 20)
'''
plt.title("k = {:0.2f} eV; T={:f} eV; ne ={:.1e} e/cc; mu = {:.2f} eV".\
         format(k, TeV, ne_cgs, muau))
'''
#plt.title("k = {:0.2f} eV; T={} eV; mu = {:.2f} eV".\
#         format(k, TeV, muau))
plt.legend()
plt.show()
'''
# Stopping power
plt.figure(2)
datfile = "WDM_codes\\StoppingNumb\\stopping_for.txt"
vnorm, Snorm = np.loadtxt(datfile, unpack=True)
plt.plot(vnorm, Snorm, color = "k")
#plt.xlabel(r"$v/v_{th}$")
plt.xlim(0, 30)
#plt.ylabel(r"$S_e/S_0$")
plt.ylim(0, 0.4)
plt.tick_params(which='both', bottom=False, labelbottom=False, left=False,labelleft=False)
#plt.title("Stopping Power - Barriga-Carrasco")
plt.show()
'''