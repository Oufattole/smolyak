import numpy as np
import quadpy

scheme = quadpy.c1.clenshaw_curtis(5)

print(scheme.points)
print(scheme.weights)