# cc_finder
cc_finder is a parallelized code that finds the critical curves and caustics of micro-lensing star fields using the parametric representation 
of the critical curves from Witt (A&A, Vol. 236, p. 311, 1990). The code is written using NVIDIA's CUDA computing platform. Instead of a 
sequential method of finding the initial roots for phi=0 as described in Witt 1990 and Witt, Kayser, and Refsdal (A&A, Vol. 268, p. 501, 1993), 
we use a parallelized implementation of the Aberth-Ehrlich method, a cubically convergent algorithm that allows for simultaneous approximations 
of the roots of a polynomial (https://en.wikipedia.org/wiki/Aberth_method). The paper that inspired this work, which provides more detailed 
explanations of the mathematical and computational methods, is: Kahina Ghidouche, Abderrahmane Sider, Raphael Couturier, Christophe Guyeux. 
Efficient high degree polynomial root finding using GPU. Journal of Computational Science, 2017, 18, pp. 46 - 56.

Once initial roots are found for phi=pi, further roots for varying phase 0<=phi<=2\*pi are found in parallel as well, simultaneously 
traversing pi -> 2\*pi and pi -> 0. We directly use all the stars in a field, avoiding complex Taylor expansions as described in WKR 1993. 
It is possible that further speedups could be gained through such approximations. 

Values for kappa_tot, kappa_smooth, shear, theta_e, num_phi (number of steps to use for phi from 0->2pi), num_stars, random seed, star input file, 
output file type (.bin or .txt), and output filename prefix, can all be input through the command line.  

