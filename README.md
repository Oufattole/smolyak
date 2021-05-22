# Smolyak Cubature Benchmarking
## Matlab Code
This folder contains the implementation of smolyak cubature and the benchmarking code for Winckel's implementation of Smolyak against QMC Cubature and adaptive cubature.

Wincke's implementation of Smolyak with Clenshaw-Curtis quadrature is in the `MATLAB Smolyak\spquad.m` file. It integrates on the interval [-1,1] but for my paper I am integrating from [0,1] so I wrote `MATLAB Smolyak\boundary_point_transform.m` to handle that (Note that `spquad.m` has a parameter to set the boundary point, but this function does not work correctly). `spquad.m` only returns the points to evalute and the weights for each point, so I wrote `MATLAB Smolyak\smolyak_integrate.m` to calculate the actual integral. The files `MATLAB Smolyak\cont.m`, `MATLAB Smolyak\disc.m`, `MATLAB Smolyak\gaussian.m`, and `MATLAB Smolyak\oscil.m` represent the functions I am integrating over in my experiments, these are copied from the website sourced in my paper. The file that actually runs the numerical experiment code for the Smolyak implementation is `MATLAB Smolyak\benchmark_smolyak.m`. It creates random functions by randomly generating the a and u vectors that define the functions (as explained in my paper) and it calculates the integral using smolyak and stores:

number of point evaluation, the integral, a vector, u vector, function type (continuous, discontinuous, or ...)

It stores this data in a csv, and I make one csv for each dimension, so you can see there are four `Matlab Smolyak\XX_smolyak.csv` files, note that XX tells you the dimension of the cubature integrals in the file.
## Python Code
I copied these csv files to the folder `smolyak_data` and then you can run the file `plot_error_vs_evals.py`. It contains all the benchmarking code. It uses QMC or adaptive cubature to make a ground truth and then calculates the relative error vs point evaluation plots that it stores in the `figures` folder.

My implementation along with helper functions for QMC cubature and adaptive cubature that I wrote are in the `smolyak.py` file. My implementation is in the q() class. You can see my unit tests for testing that it is a correct implementation in the `test_smolyak.py` file. I basically made sure it's univariate quadrature rules are correct. I checked it integrates constant functions perfectly at all levels. I then made sure it is able to solve continuous functions to high precision at a high enough level and at various dimensions.

Note that I am using python 3.6 on a ubuntu machine (I could not get the code to run on my mac, it's difficult to set up python code that interacts with gcc or mpi4 on a mac). I ran the Matlab code on Windows 10 on the latest version of Matlab (R2021a).
