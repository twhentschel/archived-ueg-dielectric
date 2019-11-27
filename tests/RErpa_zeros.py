import matplotlib.pyplot as plt
import numpy as np

# RPA dielectric parameters:
# angle = 20 * pi / 180 -> determines k = |k_f - k_i|, and k_i is the value
# we are choosing in this problem.
# kbT = 20/27.2114
# mu = -0.5 / 27.2114

# This data was when I was inputing k_0 and used the angle above.
k0 = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0.5, 0.1])
zeros1 = np.array([4.5, 4.,3.5,3.1,2.6,2.2,1.8,1.3,0.9, .43, .22, .04])

# for this data, I gave Delta k explicitly (everyone calls it just k!)
k = np.array([14,13,12,11,10,9,8,7,6,5, 4, 3, 2, 1, 0.5, 0.1, 0.01])
zeros = np.array([27.1,22.1,19.2,16.8,14.7,12.8,11.1,9.5,8.,6.5,5.2,3.8,2.5,
                  1.3,.63,.13,.013])

plt.scatter(k, zeros)
plt.plot(k,1.3*k, 'r', label="slope=1.3")
plt.plot(k, 1.3*k + (k/9)**5)
plt.xlabel(r"$k$ (eV)")
plt.ylabel(r"zeros of Re{$\epsilon(\omega, k)$} (eV)")
plt.legend()
plt.show()