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
    """
    def __init__(self, a, u):
        assert(len(a)==len(u))
        self.a = np.array(a)
        self.u = np.array(u)
        self.count=0
        self.record = False
    def evaluate(self, x):
        raise NotImplementedError("evaluate function not implemented")
    def dimension(self):
        return len(self.a)
    def begin_evaluation_count(self):
        self.count = 0
    def get_evaluation_count(self):
        return self.count
    def record_evaluations(self):
        """
        We want to record the points we evaluated the integral at.
        """
        assert(len(self.a)==2)
        self.points = []
        self.record = True
    def plot_evaluated_points(self, name="eval_points.png"):
        """
        We want to see in the 2D case what points were evaluated at to approximate the integral
        This is to present the sparsity/distribution of evaluations of different methods in the paper
        """
        assert(self.record)
        points = np.array(self.points)

        plt.plot(points.T[0], points.T[1], 'ro')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title("test evaluation points")
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
        plt.savefig(name)


class Continuous_Function(Function):
    def evaluate(self, x):
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
        super().plot("continuous")

class Gaussian_Function(Function):
    def evaluate(self, x):
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
        super().plot("gaussian")

class Oscillatory_Function(Function):
    def evaluate(self, x):
        self.count +=1
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
        self.count +=1
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
class Hyper_Plane(Function):
    def evaluate(self, x):
        if self.record:
            self.points.append(x)
        return self.a[0]
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
def generate_k(dim, l, k_left=None):
    if k_left == None:
        k_left = l+dim-1
    for k_i in range(1,k_left+1):
        next_k_left = k_left-k_i
        next_dim = dim-1
        if next_k_left >= next_dim:
            if dim == 1:
                yield [k_i]
            else:
                for k in generate_k(next_dim, l, next_k_left):
                    yield [k_i]+k

def generate_midpoint_j(k):
    k_value = k[0]
    if len(k)==1:
        for i in range(1,k_value+1):
            yield [i]
    else:
        for i in range(1,k_value+1):
            for sub_j in generate_midpoint_j(k[1:]):
                yield [i] + sub_j

def generate_mid_point(j,k):
    """
    maps j,k to point on sparse matrix ([0,1]^d) with univariate cubature
    levels defined by k 
    """
    p = []
    for j_i, k_i in zip(j,k):
        N = k_i
        denominator = k_i+1
        numerator = j_i
        p.append(numerator/denominator)
    return tuple(p)

def generate_q(k, l, dim, q_left=None):
    if q_left is None:
        q_left = l + 2*dim - np.sum(k)-1
    for q_i in range(1, q_left+1):
        next_q_left = q_left-q_i
        next_dim = dim-1
        if next_q_left >= next_dim:
            if dim == 1:
                yield [q_i]
            else:
                for q in generate_q(k, l, next_dim, q_left=next_q_left):
                    yield [q_i] + q
# def get_midpoint_v(k_i, j_i, q_i):
#     if q_i == 1:
#         return get_univariate_midpoint_weight(k_i,j_i) #univarite rule for level k_i and index j_i
#     else:
#         return get_univariate_midpoint_weight(k_i,j_i)
def get_midpoint_weight(j,k,dim, l):
    w_k_j = 1
    # for q in generate_q(k,l,dim):
    #     product = 1
    #     for k_i, j_i, q_i in zip(k,j,q):
    #         product *= get_midpoint_v(k_i, j_i, q_i)
    #     w_k_j += product
    
    k_norm = sum(k)
    for k_i, j_i in zip(k,j):
        w = 1/k_i
        c = (-1)**(l-k_norm)
        # print("choose")
        # print(dim-1)
        # print(l-k_norm)
        # print(np.choose(dim-1, l-k_norm))
        
        combos = comb(dim-1, l-k_norm)
        w_k_j *= w*c * combos
    return w_k_j

    # w_k_j = 
    # for q in range(np.sum(k), 2*dim + l):
    # check = 
    # assert(weight == check)
    # return weight

def generate_midpoint_point_weight(dim, l):
    """ midpoint cubature rule
    Parameter:
        k:
            d-dimensional array where each entry represents the 
            number of points perform a rectangle univariate
            integral for
    Yields:
        2-tuples of point and weight for rectangle cubature for smolyak
    """
    for k in generate_k(dim, l):
        # print(f"k:{k}")
        for j in generate_midpoint_j(k):
            # print(f"j:{j}")
            point = generate_mid_point(j,k)
            weight = get_midpoint_weight(j,k, dim, l)
            yield point, weight

    
def generate_clenshaw_curtis_points(j,k):
    """
    Parameter:
        j, k
    Yields:
        point
    """
    point = []
    for j_i, k_i in zip(j,k):
        n = k_i
        coord = .5 if n==1 else np.cos(np.pi*(j_i-1)/(n-1))/2+.5
        point.append(coord)
    return tuple(point)

def get_clenshaw_curtis_weight(j,k, dim, l):
    return 1 # TODO

def generate_clenshaw_curtis_point_weight(d, l):
    for k in generate_k(d, l):
        # print(f"k:{k}")
        for j in generate_midpoint_j(k):
            # print(f"j:{j}")
            point = generate_clenshaw_curtis_points(j,k)
            weight = get_midpoint_weight(j,k, d, l)
            yield point, weight


def smolyak_integrate(f, l=5):
    """
    f:
        input function
    l:
        approximation level ~ correlates with number of points we select for sparse grid
    returns:
        smolyak estimated integral
    """
    evaluated = {}
    d = f.dimension()
    k = generate_k(d,l)
    integrand = 0
    for point, weight in generate_clenshaw_curtis_point_weight(d, l):
        if point not in evaluated:
            evaluated[point] = f.evaluate(point)
        integrand += weight * evaluated[point] #create custom memoization evaluate
    return integrand

class q():
    def __init__(self,l,f,cc=True):
        self.d = f.dimension()
        assert(self.d<=l)
        self.l = l 
        self.f = f 
        self.cc=cc
    def get_k(self):
        def sum_k(length,d,k=[]):
            #k iterates thrgouh dimensional quadrature levels
            if len(k) == d:
                yield k.copy()
            else:
                for i in range(1,length+1):
                    k.append(i)
                    yield from sum_k(length-i, d, k)
                    k.pop()
        yield from sum_k(self.l,self.d)
    def get_m(self,k_i):
        if self.cc:
            return 1 if k_i ==1 else 2**(k_i-1)+1
        raise NotImplementedError("add smolyak midpoint cubature")
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
        n = self.get_m(k_i)
        p = .5 if n==1 else np.cos(np.pi*(j_i-1)/(n-1))/2+.5
        w=0
        j_i = min(n+1-j_i, j_i)#weights are symmetric
        if j_i == 1:
            w = 1/(n*(n-2))/2 #divided by 2
        else:
            term = 0
            assert((n-1)%2==0)
            for i in range(1, (n-3)//2+1):
                term += 1/(4*i**2-1)*np.cos(2 * np.pi * i * (j_i-1)/(n-1))
            w = 1/(n-1) * (1-np.cos(np.pi * (j_i-1))/(n*(n-2)) - 2*term) #divided by 2
        return p,w

    def _cc_point_weight(self, j, k):
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

    def cc_point_weights(self):
        for k in self.get_k():
            for j in self.get_j(k):
                point, weight = self._cc_point_weight(j,k)
                yield point, weight
    
    def integrate(self,l=5):
        f = self.f
        self.l = l
        integrand = 0
        for point, weight in self.cc_point_weights():
            integrand += weight * f.evaluate(point) #create custom memoization evaluate
        return integrand


    
    
            

def tensor_integrate(f):
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
    


def adaptive_cubature(f,n_max=None,abs_err=None):
    d = f.dimension()
    function = lambda x: f.evaluate(x)
    xmin = [0]*d
    xmax = [1]*d
    val = None
    err = None
    if n_max:
        val, err = cubature.cubature(function, d, 1, xmin, xmax, n_max) # useful params: abserr=1e-08, relerr=1e-08, maxEval=0
    elif abs_err:
        val, err = cubature.cubature(function, d, 1, xmin, xmax, abserr=abs_err)
    else:
        val, err = cubature.cubature(function, d, 1, xmin, xmax)

    return val[0]

def mc_integrate(f, n_max=None, abs_err=None):
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
        solution, data = CubQMCLatticeG(integral, n_max=n_max).integrate() # useful params: abserr=1e-08, relerr=1e-08, maxEval=0
    elif abs_err:
        solution, data = CubQMCLatticeG(integral,abs_tol=abs_err).integrate()
    else:
        solution, data = CubQMCLatticeG(integral).integrate()
    return solution
