import math
from scipy.special import comb
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from qmcpy import *
import cubature

class Function(object):
    """
    Dimensions: d

    self.u is a list of length d where each element is chosen from [0, 1]
    
    self.a is a list of length d where each element is chosen from the real numbers

    boundary_transform:
        these functions are [0,1], boundary transform maps the function to a [-1,1]
        function with the same integral
    """
    def __init__(self, a, u, boundary_transform = False):
        assert(len(a)==len(u))
        self.a = np.array(a)
        self.u = np.array(u)
        self.count=0
        self.record = False
        self.record_count = False
        self.boundary_transform = boundary_transform
    def reset_record_count(self):
        self.record_count = True
        self.count=0
    def get_count(self):
        return self.count
    def evaluate(self, x):
        if self.record:
            self.points.append(x)
        if self.record_count:
            self.count += 1
        if self.boundary_transform:
            d = self.dimension()
            xx = [each/2+.5 for each in x]
            value = self._evaluate(xx)
            return value / (2**(d))
        else:
            return self._evaluate(x)
    def _evaluate(self):
        raise NotImplementedError("evaluate function must be implemented by subclass")
    def dimension(self):
        return len(self.a)
    def begin_evaluation_count(self):
        self.count = 0
    def get_evaluation_count(self):
        return self.count
    def record_evaluations_to_plot(self):
        """
        We want to record the points we evaluated the integral at.
        """
        assert(len(self.a)==2)
        self.points = []
        self.record = True
    def plot_evaluated_points(self, name="eval_points.png", title="eval_points"):
        """
        We want to see in the 2D case what points were evaluated at to approximate the integral
        This is to present the sparsity/distribution of evaluations of different methods in the paper
        """
        assert(self.record)
        points = np.array(self.points)
        if self.boundary_transform:
            points = np.array([[p/2+.5 for p in point] for point in points])

        plt.plot(points.T[0], points.T[1], 'ro')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("$x_1$")
        plt.ylabel("$y_1$")
        plt.title(title)
        plt.savefig(name)
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
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_zlabel("$f(x_1,x_2)$")
        plt.savefig(name)


class Continuous_Function(Function):
    def _evaluate(self, x):
        self.count +=1
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
        super().plot("Continuous Function")
    @classmethod
    def name(cls):
        return "Continuous Function"

class Gaussian_Function(Function):
    def _evaluate(self, x):
        self.count +=1
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
        super().plot("Gaussian Function")
    @classmethod
    def name(cls):
        return "Gaussian Function"

class Oscillatory_Function(Function):
    def _evaluate(self, x):
        self.count +=1
        a = self.a
        u = self.u
        d = len(self.a)
        assert(len(x)==d)
        sum = 0
        term = 2 * np.pi * u[0]
        for i in range(d):
            xi = x[i]
            ai = a[i]
            ui = u[i]
            sum += ai*xi
        return np.cos(term + sum)
    def plot(self):
        super().plot("Oscillatory Function")
    @classmethod
    def name(cls):
        return "Oscillatory Function"

class Discontinuous_Function(Function):
    def _evaluate(self, x):
        self.count +=1
        x = np.array(x)
        a = self.a
        u = self.u
        d = len(self.a)
        assert(len(x)==d)
        sum = 0
        if x[0]>u[0] or x[1]>u[1]:
            return 0
        for i in range(d):
            xi = x[i]
            ai = a[i]
            ui = u[i]
            sum += ai*xi
        return np.exp(sum)
    def plot(self):
        super().plot("Discontinuous Function")
    @classmethod
    def name(cls):
        return "Discontinuous Function"
class Hyper_Plane(Function):
    def _evaluate(self, x):
        return self.a[0]

class q():
    def __init__(self,l,f):
        self.d = f.dimension()
        assert(0<=l)
        self.l = l + self.d #add l to d
        self.f = f 

    def get_k(self):
        def sum_k(length,d,k=[]):
            #k iterates thrgouh dimensional quadrature levels
            if len(k) == d:# and sum(k)>=max(d,self.l-d+1):
                yield k.copy()
            else:
                for i in range(1,length+1):
                    k.append(i)
                    yield from sum_k(length-i, d, k)
                    k.pop()
        yield from sum_k(self.l,self.d)
    def get_m(self,k_i):
        return 1 if k_i ==1 else 2**(k_i-1)+1
    def get_j(self, k):
        #j iterates through the number of points for quadrature levels defined by k
        d,l= self.d,self.l
        def sum_j(k,j=[]):
            if len(j) == d:
                yield j.copy()
            else:
                for j_i in range(1,self.get_m(k[0])+1):
                    j.append(j_i)
                    yield from sum_j(k[1:], j)
                    j.pop()
        yield from sum_j(k)
    def _cc_univariate_point_weight(self, j_i, k_i):
        if k_i == 1: #edge case
            return 0, 2 
        else:
            n = self.get_m(k_i)
            d=self.f.dimension()
            
            p = 0 if n==1 else np.cos(np.pi*(j_i-1)/(n-1))
            w=0
            j_i = min(n+1-j_i, j_i)#weights are symmetric
            if j_i == 1:
                w = 1/(n*(n-2)) #removed divided by 2
            else:
                term = 0
                assert((n-1)%2==0)
                for i in range(1, (n-3)//2+1):
                    term += 1/(4*i**2-1)*np.cos(2 * np.pi * i * (j_i-1)/(n-1))
                w = 2/(n-1) * (1-np.cos(np.pi * (j_i-1))/(n*(n-2)) - 2*term) # removed divided by 2
            return p,w

    def _cc_sparse_grid_point_weight(self, j, k):
        d,l= self.d,self.l
        weight = 1
        point = []
        for j_i, k_i in zip(j,k):
            p,w = self._cc_univariate_point_weight(j_i,k_i)
            point.append(p)
            weight *= w
        l_thesis = np.sum(k)
        k_thesis = l
        weight *= (-1)**(k_thesis-l_thesis) *comb(d-1, k_thesis-l_thesis) #multiply the normalizing coefficient
        return tuple(point), weight

    def cc_sparse_grid_point_weights(self):
        for k in self.get_k():
            for j in self.get_j(k):
                point, weight = self._cc_sparse_grid_point_weight(j,k)
                yield point, weight
    
    def integrate(self):
        f = self.f
        d = f.dimension()
        integrand = 0
        weights = 0
        for point, weight in self.cc_sparse_grid_point_weights():
            weights += weight
            integrand += weight * f.evaluate(point) #create custom memoization evaluate
        return integrand
            

def tensor_integrate(f):
    def yield_tensor(dim, points_per_dim, point=None):
        if dim == 0:
            yield np.array(point)
        else:
            if point is None:
                point = []
            for i in range(0, points_per_dim):
                point.append(i/points_per_dim)
                yield from yield_tensor(dim-1, points_per_dim, point=point)
                point.pop()
    d = f.dimension()
    points_per_dim = 5
    integral = 0
    for point in yield_tensor(d, points_per_dim):
        value = f.evaluate(point)
        for dim in range(d):
            value /= points_per_dim
        integral += value
    return integral
            

    input = []
    for i in range(d):
        input.append(d)
    


def adaptive_cubature(f,n_max=None,abs_err=None, transform_boundary=False, discontinuous=None):
    d = f.dimension()
    function = lambda x: f.evaluate(x)
    xmin = None
    xmax = None
    assert(not ((not (discontinuous is None)) and transform_boundary))
    if transform_boundary:
        xmin = [-1]*d
        xmax = [1]*d
    elif not(discontinuous is None):
        xmin = [0]*d
        xmax = list(discontinuous[0:2]) + [1]*(d-2)
        assert(len(xmin)==len(xmax))
    else:
        xmin = [0]*d
        xmax = [1]*d
    val = None
    err = None
    if n_max:
        val, err = cubature.cubature(function, d, 1, xmin, xmax, maxEval=n_max) # useful params: abserr=1e-08, relerr=1e-08, maxEval=0
    elif abs_err:
        val, err = cubature.cubature(function, d, 1, xmin, xmax, abserr=abs_err)
    else:
        val, err = cubature.cubature(function, d, 1, xmin, xmax)

    return val[0]

def mc_integrate(f, n_max=None, abs_err=None, rel_err = None):
    abs_tol = 1e-7
    d = f.dimension()
    solution = None
    data = None
    integral = CustomFun(
        true_measure = Uniform(Lattice(d)),
        g = lambda x: np.array([f.evaluate(x_i) for x_i in x]))
    solution, data = CubQMCLatticeG(integral).integrate() #stopping criterion, useful param: n_init=1024.0, n_max=1024
    #data has the error and number of steps (I think it is n_init or n_max, check this) 
    if n_max:
        solution, data = CubQMCLatticeG(integral, n_init = n_max, n_max=n_max).integrate() # useful params: abserr=1e-08, relerr=1e-08, maxEval=0
    elif abs_err:
        solution, data = CubQMCLatticeG(integral,abs_tol=abs_err).integrate()
    elif rel_err:
        solution, data = CubQMCLatticeG(integral,rel_tol=rel_err).integrate()
    else:
        solution, data = CubQMCLatticeG(integral).integrate()
    # print(data)
    # print(f"final solution:{solution}")
    return solution
