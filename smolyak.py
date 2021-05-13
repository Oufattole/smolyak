import math
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

class Function(object):
    """
    Dimensions: d

    self.u is a list of length d where each element is chosen from [0, 1]
    
    self.a is a list of length d where each element is chosen from the real numbers
    """
    def __init__(self, a, u):
        assert(len(a)==len(u))
        self.a = np.array(a)
        self.u = np.array(u)
    def evaluate(self, x):
        raise NotImplementedError("evaluate function not implemented")
    def plot(self, name):
        granularity = 100
        a = self.a
        u = self.u
        d = len(self.a)
        if d != 2:
            raise NotImplementedError("Plotting only implemented for len(a)=2")
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        d = len(a)
        x = np.linspace(0, 1, granularity)
        y = np.linspace(0, 1, granularity)

        X, Y = np.meshgrid(x, y)
        Z = np.empty(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i,j] = self.evaluate([X[i,j], Y[i,j]])
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none')
        ax.set_title(name)
        plt.savefig(name)


class Continuous_Function(Function):
    def evaluate(self, x):
        a = self.a
        u = self.u
        d = len(self.a)
        assert(len(x)==d)
        sum = 0
        for i in range(d):
            xi = x[i]
            ai = a[i]
            ui = u[i]
            sum += ai*abs(xi-ui)
        return np.exp(-sum)
    def plot(self):
        super().plot("continuous")

class Gaussian_Function(Function):
    def evaluate(self, x):
        a = self.a
        u = self.u
        d = len(self.a)
        assert(len(x)==d)
        sum = 0
        for i in range(d):
            xi = x[i]
            ai = a[i]
            ui = u[i]
            sum += ai**2 * (xi-ui)**2
        return np.exp(-sum)
    def plot(self):
        super().plot("gaussian")

class Oscillatory_Function(Function):
    def evaluate(self, x):
        a = self.a
        u = self.u
        d = len(self.a)
        assert(len(x)==d)
        sum = 2 * np.pi * u[0]
        for i in range(d):
            xi = x[i]
            ai = a[i]
            ui = u[i]
            sum += ai*xi
        return np.cos(sum)
    def plot(self):
        super().plot("oscillatory")

class Discontinuous_Function(Function):
    def evaluate(self, x):
        x = np.array(x)
        a = self.a
        u = self.u
        d = len(self.a)
        assert(len(x)==d)
        sum = 0
        if np.any(x>u):
            return 0
        for i in range(d):
            xi = x[i]
            ai = a[i]
            ui = u[i]
            sum += ai*xi
        return np.exp(sum)
    def plot(self):
        super().plot("discontinuous")