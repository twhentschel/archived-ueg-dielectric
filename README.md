# dielectric-function
Methods to calculate a dielectric function for many-body systems, using different approximations and what not.

The initial work for this project relied on something called the Random Phase Approximation (RPA) to calculate the dielectric function. This is essentially a mean field approximation that turns a many-body problem into a solvable one-body problem. In this instance, the dielectric function becomes a complex quantity, where the real part is an integral that must be calculated numerically while the imaginary part has an analytic form [W. R. Johnson et al., PRE, 2012]. 

This is fine for the most part, but it removes any information of electron-electron correlations and electron-ion interactions. 

The Mermin dielectric ansatz is a formula that builds upon the RPA dielectric function to include information about the ionic strucutre. It does this by making the frequency argument complex by including an electron-ion collision frequency. This is accomplished in a not-so-straightforward manner to conserve local-electron number [N. D. Mermin, PRB, 1970]. This method introduces another variable, the collision frequency, which must be calculated somewhere else.

A Local Field Correction (LFC) can be used to improve the dielectric function to incorporate electron-electron correlation effects [M. D. Barriga-Carrasco, PRE, 2009]. Like the collision frequency, the LFC must be calculated elsewhere.
