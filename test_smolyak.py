import numpy as np
import smolyak
import unittest
import quadpy


class TestSmolyak(unittest.TestCase):

    def test_continuous(self):
        a = [5,5]
        u = [.5,.5]
        f = smolyak.Continuous_Function(a,u)
        y = f.evaluate([.5,.5])
        self.assertAlmostEqual(y, 1.0)
    def test_discontinuous(self):
        a = [5,5]
        u = [.5,.5]
        f = smolyak.Discontinuous_Function(a,u)
        y = f.evaluate([.5,.5])
        self.assertAlmostEqual(y, np.exp(5))
    def test_boundary_transform(self):
        #check transformed functions have the same integral as the original function
        for d in range(1,4):
            a = [5.0]*d
            u = [.5]*d
            original_functions = [
                smolyak.Hyper_Plane(a,u),
                smolyak.Continuous_Function(a,u), 
                smolyak.Gaussian_Function(a,u),
                ]
            transformed_functions = [
                smolyak.Hyper_Plane(a,u,boundary_transform=True),
                smolyak.Continuous_Function(a,u,boundary_transform=True), 
                smolyak.Gaussian_Function(a,u,boundary_transform=True),
                ]
            for o_f, t_f in zip(original_functions, transformed_functions):
                
                original_integral = smolyak.adaptive_cubature(o_f, abs_err=1e-04)
                transformed_integral = smolyak.adaptive_cubature(t_f, abs_err=1e-04, transform_boundary=True)
                self.assertAlmostEqual(original_integral, transformed_integral, places=3)
    def test_q_k(self):
        thesis = False

        a = [2,2]
        u = [.5,.5]
        f = smolyak.Hyper_Plane(a,u)
        d = 2
        check_l_0=[[1,1]]
        check_l_1_thesis=[[1, 1], [1, 2], [2, 1]]
        check_l_1=[[1, 1], [1, 2], [2, 1]]
        
        check_l_2_thesis=[[1, 2], [1, 3], [2, 1], [2, 2], [3, 1]]
        check_l_2=[[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [3, 1]]

        def check(l,check):
            q = smolyak.q(l,f)
            k = [each for each in q.get_k()]
            dim = 2
            self.assertEqual(check, k)
        check(0, check_l_0)
        if thesis == True:
            check(1, check_l_1_thesis)
            check(2, check_l_2_thesis)
        else:
            check(1, check_l_1)
            check(2, check_l_2)
        
    def test_q_j_cc(self):
        #init function
        a = [2,2]
        u = [.5,.5]
        f = smolyak.Hyper_Plane(a,u)
        expected_0 ={(1, 1): [[1, 1]]}
        expected_1 = {(1, 1): [[1, 1]], (1, 2): [[1, 1], [1, 2], [1, 3]], (2, 1): [[1, 1], [2, 1], [3, 1]]}
        def check_j_cc(l, excpected):
            #init smolyak class
            q = smolyak.q(l,f)
            #init check smolyak function
            level = q.get_m(l)
            #check answers
            expected = {}
            for k in q.get_k():
                j = [each for each in q.get_j(k)]
                self.assertEqual(excpected[tuple(k)], j)
        check_j_cc(0,expected_0)
        check_j_cc(1,expected_1)

        

    def test_q_cc_univariate_base_case(self):
        l=1
        a = [2]
        u = [.5]
        f = smolyak.Hyper_Plane(a,u)
        q = smolyak.q(l,f)
        k = l
        weights = []
        points = []
        for j in range(1,k+1):
            p, w = q._cc_univariate_point_weight(j, k)
            weights.append(w)
            points.append(p)
        self.assertAlmostEqual(sum(weights), 2)
        self.assertEqual(len(points), 1)
        self.assertEqual(len(weights), 1)
        self.assertEqual(points[0], 0)
        self.assertEqual(weights[0], 2)
    def test_q_cc_univariate(self):
        def check_cc(l):
            a = [2]
            u = [.5]
            f = smolyak.Hyper_Plane(a,u)
            q = smolyak.q(l,f)
            k = l

            check_l = q.get_m(l)
            scheme = quadpy.c1.clenshaw_curtis(check_l)
            p_check = scheme.points
            w_check = scheme.weights
            
            self.assertAlmostEqual(sum(w_check), 2)
            
            weights = []
            points = []
            n= q.get_m(l)
            for j in range(1,n+1):
                p, w = q._cc_univariate_point_weight(j, k)
                weights.append(w)
                points.append(p)
            self.assertAlmostEqual(sum(weights), 2)
            self.assertEqual(weights, weights[::-1])
            self.assertEqual(len(points), len(p_check))
            for p_cc, p_check in zip(points, p_check[::-1]):
                self.assertAlmostEqual(p_cc, p_check)
            total = 0
            for w_cc, w_check in zip(weights, w_check):
                self.assertAlmostEqual(w_cc, w_check)
                total+=w_cc
        levels = [2,3,4,5,6,7]
        for l in levels:
            check_cc(l)
    def test_cc_sparse_grid(self):
        def check(l,d):
            a = [2]*d
            u = [.5]*d
            f = smolyak.Continuous_Function(a,u)
            q = smolyak.q(l,f)
            sum_weights=0
            for point, weight in q.cc_sparse_grid_point_weights():
                sum_weights += weight
            self.assertAlmostEqual(sum_weights,2**d)
        levels = [0,1,2]
        check(0,1)
        check(1,1)
        for l in levels:
            check(l,2)
        for l in levels[3:]:
            check(l,3)
    def test_cc_sparse_grid_rigorous_l1_d3(self):
        l=1
        d=3
        expected_weights = [4/3,4/3,4/3,0,4/3,4/3,4/3]
        expected_points = [[-1,0,0],[0,-1,0],[0,0,-1],[0,0,0],[0,0,1],[0,1,0],[1,0,0]]
        expected = {tuple(expected_points[i]):expected_weights[i] for i in range(len(expected_weights))}
        
        a = [2]*d
        u = [.5]*d
        f = smolyak.Continuous_Function(a,u)
        q = smolyak.q(l,f)
        points = []
        weights = []
        result = {}
        for point, weight in q.cc_sparse_grid_point_weights():
            point = [round(each) for each in point]
            points.append(tuple(point))
            result.setdefault(tuple(point), 0)
            result[tuple(point)] += weight
            weights.append(weight)
        for point in points:
            self.assertAlmostEqual(expected[point], result[point])
    def test_cc_sparse_grid_rigorous_l2_d2(self):
        l=2
        d=2
        expected_weights = [
            0.111111111111111,-0.0888888888888890,0.111111111111111,
            1.06666666666667,-0.0888888888888890,1.06666666666667,
            -0.355555555555555,1.06666666666667,-0.0888888888888890,
            1.06666666666667,0.111111111111111,-0.0888888888888890,0.111111111111111
        ]
        ep = [
            [-1, -1],
            [-1,	0],
            [-1,	1],
            [-0.707106781186548,	0],
            [0,	-1],
            [0,	-0.707106781186548],
            [0,	0],
            [0,	0.707106781186548],
            [0,	1],
            [0.707106781186548,	0],
            [1,	-1],
            [1,	0],
            [1,	1],
        ]
        expected_points = []
        for each in ep:
            expected_points.append(tuple([round(element, 4) for element in each])) 
        expected = {tuple(expected_points[i]):expected_weights[i] for i in range(len(expected_weights))}
        a = [2]*d
        u = [.5]*d
        f = smolyak.Continuous_Function(a,u)
        q = smolyak.q(l,f)
        points = set()
        weights = []
        result = {}
        for point, weight in q.cc_sparse_grid_point_weights():
            point = [round(each, 4) for each in point]
            points.add(tuple(point))
            result.setdefault(tuple(point), 0)
            result[tuple(point)] += weight
            weights.append(weight)
        for point in points:
            self.assertAlmostEqual(expected[point], result[point])
    def test_q_integrate_hyper(self):
        for d in range(1, 7):
            for l in range(0,4):
                a = [3] * d
                u = [.5]*d
                f = smolyak.Hyper_Plane(a,u,boundary_transform=True)
                q = smolyak.q(l,f)
                smolyak_integral = q.integrate()
                self.assertAlmostEqual(a[0], smolyak_integral, places=5)
    def test_q_integrate_cont(self):
        #this test case takes 90 seconds to run
        for d in range(1,5):
            print(d)
            func = smolyak.Gaussian_Function
            l = 9
            if d == 4:
                l=10
            a = [5] * d
            u = [.5] * d
            f = func(a,u)
            check = smolyak.adaptive_cubature(f, abs_err=1e-04)
            self.assertAlmostEqual(smolyak.mc_integrate(f, abs_err=1e-04), check, places=3)
            f = func(a,u,boundary_transform=True)
            cubature_check = smolyak.adaptive_cubature(f, abs_err=1e-04, transform_boundary=True)
            smolyak_integral = smolyak.q(l,f).integrate()

            self.assertAlmostEqual(check,cubature_check, places=3)
            
            self.assertAlmostEqual(check, smolyak_integral, places=3)

if __name__ == '__main__':
    unittest.main()