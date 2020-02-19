import scipy.integrate as int
import numpy as np
import matplotlib.pyplot as plt

k = 1
w = 0.1#0.43130892007934835
mu = 0.279
kbT = 1/27.2114
delta = 0.0000001

def integrand(y, p, k, w, T, mu, delta):
    FD = 1/(1+np.exp((p**2/2 - mu)/T))
    logpart = 0.5 * (np.log((k**2 + 2*p*k + 2*w)**2 + delta)  \
            - np.log((k**2 - 2*p*k + 2*w)**2 + delta) \
            + np.log((k**2 + 2*p*k - 2*w)**2 + delta) \
            - np.log((k**2 - 2*p*k - 2*w)**2 + delta))

                   
    return p*FD*logpart
    
# Initial tests
'''
p = np.linspace(0, 20, 100)
y0 = 0
intp = int.odeint(integrand, y0, p, args=(k, w, kbT, mu, delta))

plt.plot(p, 1+2/np.pi/k**3 *intp, label="w = {}".format(w))

'''

def re_RPAeps(k, w, T, mu):
    delta = 10**-7
    y0 = 0
    p = np.linspace(0, 10)
    
    intp = int.odeint(integrand, y0, p, args=(k, w, T, mu, delta))
    return 1 + 2/np.pi/k**3 * intp[-1]

def im_RPAeps(k, w, kbT, mu):
    """
    Imaginary part of the RPA dielectric function. This one is not as 
    complicated as the real part: it is only a function evaluation. 
    """ 
    def FDD(E, mu, T):
        """
        Fermi-Dirac Distribution.
        """
        return 1/(1+np.exp((E-mu)/T))

    a = abs(2*w - k**2)/(2*k)
    b = (2*w + k**2)/(2*k)
    return 2*kbT/k**3 * np.log(FDD(mu, b**2/2, kbT)/FDD(mu, a**2/2, kbT))

def ELF(k, w, kbT, mu):
    im = im_RPAeps(k, w, kbT, mu)
    re = re_RPAeps(k, w, kbT, mu)
    return im/(re**2 + im**2)

def RPAeps(k, w, kbT, mu):
    return complex(re_RPAeps(k, w, kbT, mu), im_RPAeps(k, w, kbT, mu))

if __name__=="__main__":
    w = np.linspace(0, 4, 200)
    import time
    start = time.time()
    reeps = np.asarray([re_RPAeps(k, x, kbT, mu) for x in w])
    print("time = {}".format(time.time()-start))
    imeps = im_RPAeps(k, w, kbT, mu)
    plt.plot(w, reeps, label='real')
    plt.plot(w, imeps, label='imaginary')
    #plt.plot(w, [ELF(k, x, kbT, mu) for x in w])
    #plt.legend()
    plt.show()
