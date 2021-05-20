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
    def test_continuous(self):
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
                # smolyak.Oscillatory_Function(a,u),
                # smolyak.Discontinuous_Function(a,u),
                ]
            transformed_functions = [
                smolyak.Hyper_Plane(a,u,boundary_transform=True),
                smolyak.Continuous_Function(a,u,boundary_transform=True), 
                smolyak.Gaussian_Function(a,u,boundary_transform=True),
                # smolyak.Oscillatory_Function(a,u,boundary_transform=True),
                # smolyak.Discontinuous_Function(a,u,boundary_transform=True),
                ]
            for o_f, t_f in zip(original_functions, transformed_functions):
                
                original_integral = smolyak.adaptive_cubature(o_f, abs_err=1e-04)
                transformed_integral = smolyak.adaptive_cubature(t_f, abs_err=1e-04, transform_boundary=True)
                # print(f"function:{o_f}\ndim:{d}\nratio:{transformed_integral/original_integral}")
                self.assertAlmostEqual(original_integral, transformed_integral, places=3)
    def test_q_k(self):
        #update if you keep second boolean value from thesis
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
            # print(f"l:{l}\nk:{k},\ncheck:{check}")
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

        
    # def test_q_cc(self):
    #     a = [2,2]
    #     u = [.5,.5]
    #     f = smolyak.Hyper_Plane(a,u)
    #     d=2
    #     def plot(l):
    #         q = smolyak.q(l,f)
    #         f.record_evaluations_to_plot()
    #         smolyak_integral = q.integrate(l)
    #         f.plot_evaluated_points("eval_"+str(l)+".png")
    #     plot(2)
    #     plot(3)
    #     plot(4)
    #     plot(5)
    #     plot(6)
        #check this against fig 1
        #https://bfi.uchicago.edu/wp-content/uploads/Judd-Maliar-Valero.pdf
    def test_q_cc_uni_base_case(self):
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
        # print(points)
        # print(weights)
        self.assertEqual(len(points), 1)
        self.assertEqual(len(weights), 1)
        self.assertEqual(points[0], 0)
        self.assertEqual(weights[0], 2)
    def test_q_cc_uni(self):
        def check_cc(l):
            a = [2]
            u = [.5]
            f = smolyak.Hyper_Plane(a,u)
            #check
            q = smolyak.q(l,f)
            k = l

            check_l = q.get_m(l)
            scheme = quadpy.c1.clenshaw_curtis(check_l)
            p_check = scheme.points
            w_check = scheme.weights
            
            self.assertAlmostEqual(sum(w_check), 2)
            # print(p_check)
            # print(w_check)
            #my implementation
            
            weights = []
            points = []
            n= q.get_m(l)# j goes up to n
            for j in range(1,n+1):
                p, w = q._cc_univariate_point_weight(j, k)#j,k are the multi-set indexes
                weights.append(w)
                points.append(p)
            self.assertAlmostEqual(sum(weights), 2)
            # print(points)
            # print(weights)
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
            # print(f"level:{l}\ndim:{d}")
            f = smolyak.Continuous_Function(a,u)
            q = smolyak.q(l,f)
            sum_weights=0
            for point, weight in q.cc_sparse_grid_point_weights():
                sum_weights += weight
                # print(f"p:{point}\nw:{weight}")
            self.assertAlmostEqual(sum_weights,2**d)
        levels = [0,1,2]
        check(0,1)
        check(1,1)
        for l in levels:
            check(l,2)
        for l in levels[3:]:
            check(l,3)
    def test_cc_sparse_grid_rigorous_l1_d3(self):
        #my sparse grid repeats points but if you sum the weights for repeated points
        #you get the same sparse grid as the correct implementation
        #my implementation is correct still
        l=1
        d=3
        expected_weights = [4/3,4/3,4/3,0,4/3,4/3,4/3]
        expected_points = [[-1,0,0],[0,-1,0],[0,0,-1],[0,0,0],[0,0,1],[0,1,0],[1,0,0]]
        expected = {tuple(expected_points[i]):expected_weights[i] for i in range(len(expected_weights))}
        
        a = [2]*d
        u = [.5]*d
        # print(f"level:{l}\ndim:{d}")
        f = smolyak.Continuous_Function(a,u)
        q = smolyak.q(l,f)
        points = []
        weights = []
        result = {}
        for point, weight in q.cc_sparse_grid_point_weights():
            # print(point)
            point = [round(each) for each in point]
            points.append(tuple(point))
            result.setdefault(tuple(point), 0)
            result[tuple(point)] += weight
            weights.append(weight)
        for point in points:
            self.assertAlmostEqual(expected[point], result[point])
    def test_cc_sparse_grid_rigorous_l2_d2(self):
        #my sparse grid repeats points but if you sum the weights for repeated points
        #you get the same sparse grid as the correct implementation
        #my implementation is correct still
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
        # print(f"level:{l}\ndim:{d}")
        f = smolyak.Continuous_Function(a,u)
        q = smolyak.q(l,f)
        points = set()
        weights = []
        result = {}
        for point, weight in q.cc_sparse_grid_point_weights():
            # print(point)
            point = [round(each, 4) for each in point]
            points.add(tuple(point))
            result.setdefault(tuple(point), 0)
            result[tuple(point)] += weight
            weights.append(weight)
        print(len(result))
        print(len(expected))
        print(points)
        print(expected.keys())
        for point in points:
            self.assertAlmostEqual(expected[point], result[point])
    def test_q_integrate_hyper(self):# d,ration: 1:1------------2:2-------3:4------4:8
        for d in range(1, 5):
            for l in range(0,3):
                # print(f"l:{l}")
                # print(f"d:{d}")
                a = [3] * d
                u = [.5]*d
                f = smolyak.Hyper_Plane(a,u,boundary_transform=True)
                q = smolyak.q(l,f)
                smolyak_integral = q.integrate()
                # print(f"smolyak:{smolyak_integral}")
                self.assertAlmostEqual(a[0], smolyak_integral, places=5)
                # f = smolyak.Hyper_Plane(a,u)
                # q = smolyak.q(l,f)
                # cubature_integral = smolyak.adaptive_cubature(f)
                # self.assertAlmostEqual(cubature_integral, smolyak_integral, places=3)

                

    def test_q_integrate_cont(self):# d,ration: 1:1------------2:2-------3:4------4:8
        func = smolyak.Gaussian_Function
        d = 2
        l = 9
        # gt_levels = [0,1,2,3,4]
        # gt_expected = [0.4463, 0.4556]
        a = [5] * d
        u = [.5] * d
        f = func(a,u)
        check = smolyak.adaptive_cubature(f, abs_err=1e-04)
        self.assertAlmostEqual(smolyak.mc_integrate(f, abs_err=1e-04), check, places=3)
        # transformed_integral = smolyak.adaptive_cubature(t_f, abs_err=1e-04, transform_boundary=True)
        # print(f"function:{o_f}\ndim:{d}\nratio:{transformed_integral/original_integral}")
        f = func(a,u,boundary_transform=True)
        cubature_check = smolyak.adaptive_cubature(f, abs_err=1e-04, transform_boundary=True)
        smolyak_integral = smolyak.q(l,f).integrate()

        self.assertAlmostEqual(check,cubature_check, places=3)#0.1255
        
        self.assertAlmostEqual(check, smolyak_integral, places=3)
        print(check)

    # def test_q_integrate_2(self):#also a multiple of 4
    #     """
    #     (multiplying univariate weights by 2) no way these are correct
    #     (multiplying)
    #     """
    #     a = [5,5]
    #     u = [.5,.5]
    #     d=2
    #     l=10
    #     functions = [
    #         smolyak.Continuous_Function(a,u), 
    #         smolyak.Gaussian_Function(a,u),
    #         smolyak.Oscillatory_Function(a,u)
    #         ]
    #     for f in functions:
    #         q = smolyak.q(l,f)
    #         smolyak_integral = q.integrate(l)
    #         cubature_integral = smolyak.adaptive_cubature(f)
    #         self.assertAlmostEqual(cubature_integral, smolyak_integral, places=2)

    # def test_q_integrate_3(self):#solution is 2*d greater
    #     """
    #     (dividing univariate weights by 2 was the issue)
    #     """
    #     def check(a):
    #         a = a*3
    #         u = [.5]*3
    #         f = smolyak.Hyper_Plane(a,u)
    #         d=3
    #         l=8
    #         q = smolyak.q(l,f)
    #         smolyak_integral = q.integrate(l)
    #         self.assertAlmostEqual(a[0], smolyak_integral, places=2)
    #     for i in range(1,5):
    #         check([i])
    # def test_q_integrate_4(self):#solution is 2*d greater
    #     """
    #     possible issues
    #     cc weights comb (m vs k, what does thesis posit)
    #         check thesis followed correctly
    #             Might be minimum |k| in q.ket_k() is not correct (tried this, no change in output)
    #             check
    #     dividing weights by 2*d in cc_weights
    #         remove division (NOPE)
    #     increasing l
    #         Once I increase l enough, i get the right solution for hyper_plane
    #             maybe this will work for continuous_function?, but I need more speed, lets try
    #     boundary handled incorrectly (this is the issue)
    #         [-1,1]=> [0,1] isn't correct (Nope)
    #         maybe the grid weights don't add up to 1 (this must be it)
    #         try dividing by 2^(d-1), right now you are doing 2^(d)
    #     remove thesis lower bound on k
    #         make sure your code has the same weights and points as teh matlab code

    #     """
    #     def check(d):
    #         a = [1]*d
    #         u = [.5]*d
    #         f = smolyak.Hyper_Plane(a,u)
    #         l=10 # works for 12
    #         q = smolyak.q(l,f)
    #         smolyak_integral = q.integrate(l)
    #         print(d)
    #         self.assertAlmostEqual(a[0], smolyak_integral, places=2)
    #     for i in range(6,7):
    #         check(i)
    # def test_q_integrate_5(self):#solution is 2*d greater
    #     a = [2,2,2]
    #     u = [.5,.5,.5]
    #     d=3
    #     l=10 # works for 13
    #     functions = [
    #         smolyak.Continuous_Function(a,u), 
    #         smolyak.Gaussian_Function(a,u),
    #         smolyak.Oscillatory_Function(a,u)
    #         ]
    #     for f in functions:
    #         q = smolyak.q(l,f)
    #         smolyak_integral = q.integrate(l)
    #         cubature_integral = smolyak.adaptive_cubature(f)
    #         self.assertAlmostEqual(cubature_integral, smolyak_integral, places=2)
    #         q = smolyak.q(l,f)
    #         smolyak_integral = q.integrate(l)
    #         cubature_integral = smolyak.adaptive_cubature(f, abs_err=1e-04)
    #         mc_integral = smolyak.mc_integrate(f, abs_err=1e-04)
    #         self.assertAlmostEqual(cubature_integral, mc_integral, places=3)
    #         self.assertAlmostEqual(cubature_integral, smolyak_integral, places=3)
    # def test_q_integrate_6(self):#solution is 2*d greater
    #     """
    #     (dividing univariate weights by 2 was the issue)
    #     """
    #     a = [2,2,2,2,2,2,2]
    #     a= [each*5 for each in a]
    #     u = [.5,.5,.5,.5,.5,.5,.5]
    #     f = smolyak.Continuous_Function(a,u)
    #     l=10
    #     q = smolyak.q(l,f)
        
    #     import time

    #     start_time = time.time()
    #     smolyak_integral = q.integrate(l)
    #     print("--- %s smolyak ---" % (time.time() - start_time))

    #     start_time = time.time()
    #     cubature_integral = smolyak.adaptive_cubature(f, abs_err=1e-03)
    #     print("--- %s cubature ---" % (time.time() - start_time))

    #     start_time = time.time()
    #     mc_integral = smolyak.mc_integrate(f, abs_err=1e-03)
    #     print("--- %s mc ---" % (time.time() - start_time))
    #     self.assertAlmostEqual(cubature_integral, mc_integral, places=2)
    #     self.assertAlmostEqual(cubature_integral, smolyak_integral, places=2)


    # def test_isupper(self):
    #     self.assertTrue('FOO'.isupper())
    #     self.assertFalse('Foo'.isupper())
    #     self.assertEqual(s.split(), ['hello', 'world'])

if __name__ == '__main__':
    unittest.main()