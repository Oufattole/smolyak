import numpy as np
# import quadpy

# scheme = quadpy.c1.clenshaw_curtis(5)

# print(scheme.points)
# print(scheme.weights)
n=3
j_i=3
term = 0
assert((n-1)%2==0)
for i in range(1, (n-3)//2+1):
    term += 1/(4*i**2-1)*np.cos(2 * np.pi * i * (j_i-1)/(n-1))
w = 2/(n-1) * (1-np.cos(np.pi * (j_i-1))/(n*(n-2)) - 2*term)
print(term)
print(w)