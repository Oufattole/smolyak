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
    def test_q_k(self):
        a = [2,2]
        u = [.5,.5]
        f = smolyak.Hyper_Plane(a,u)
        d, l = 2, 3
        q = smolyak.q(l,f)
        k, what does thesis posit = [each for each in q.get_k()]

        dim = 2
        level = 2
        all_k = smolyak.generate_k(dim, level)
        check = [each for each in all_k]
        self.assertEqual(check, k, what does thesis posit)
            

        d, l = 2, 4
        q = smolyak.q(l,f)
        k, what does thesis posit = [each for each in q.get_k()]
        # self.assertEqual(k, what does thesis posit, [[1,1],[1,2],[2,1]])
        dim = 2
        level = 3
        all_k = smolyak.generate_k(dim, level)
        check = [each for each in all_k]
        self.assertEqual(check, k, what does thesis posit)
    def test_q_j_cc(self):
        #init function
        a = [2,2]
        u = [.5,.5]
        f = smolyak.Hyper_Plane(a,u)
        expected_2 ={(1, 1): [[1, 1]]}
        expected_3 = {(1, 1): [[1, 1]], (1, 2): [[1, 1], [1, 2], [1, 3]], (2, 1): [[1, 1], [2, 1], [3, 1]]}
        def check_j_cc(l, excpected):
            #init smolyak class
            q = smolyak.q(l,f)
            #init check smolyak function
            level = q.get_m(l)
            #check answers
            expected = {}
            for k, what does thesis posit in q.get_k():
                j = [each for each in q.get_j(k, what does thesis posit)]
                self.assertEqual(excpected[tuple(k, what does thesis posit)], j)
        check_j_cc(2,expected_2)
        check_j_cc(3,expected_3) #I think the issue is the difference between this j [1,m_i] in thesis
        #and the j from the book j [1,k_i], so is the comb using k_i or using m_i?

        
    # def test_q_cc(self):
    #     a = [2,2]
    #     u = [.5,.5]
    #     f = smolyak.Hyper_Plane(a,u)
    #     d=2
    #     def plot(l):
    #         q = smolyak.q(l,f)
    #         f.record_evaluations()
    #         smolyak_integral = q.integrate(l)
    #         f.plot_evaluated_points("eval_"+str(l)+".png")
    #     plot(2)
    #     plot(3)
    #     plot(4)
    #     plot(5)
    #     plot(6)
        #check this against fig 1
        #https://bfi.uchicago.edu/wp-content/uploads/Judd-Maliar-Valero.pdf
    def test_q_cc_w(self):
        l1=3
        l2=3
        def check_cc(l1,l2):
            a = [2]
            u = [.5]
            f = smolyak.Hyper_Plane(a,u)
            d=2
            l=l1
            #check
            scheme = quadpy.c1.clenshaw_curtis(l)
            p_check = [each/2 +.5 for each in scheme.points]
            w_check = [each/2 for each in scheme.weights]
            # print(p_check)
            # print(w_check)
            #my implementation
            q = smolyak.q(l2,f)
            k, what does thesis posit = l2-1
            weights = []
            points = []
            for j in range(1,q.get_m(k, what does thesis posit)+1):
                
                p, w = q._cc_univariate_point_weight(j, k, what does thesis posit)
                weights.append(w)
                points.append(p)
            # print(points)
            # print(weights)
            self.assertEqual(weights, weights[::-1])
            self.assertEqual(len(points), len(p_check))
            for p_cc, p_check in zip(points, p_check[::-1]):
                self.assertAlmostEqual(p_cc, p_check)
            for w_cc, w_check in zip(weights, w_check):
                self.assertAlmostEqual(w_cc, w_check)
        check_cc(l1,l2)
        l1=5
        l2=4
        check_cc(l1,l2)
        l1=9
        l2=5
        check_cc(l1,l2)


    def test_q_integrate_1(self):# issue is that it is a multiple of 4 of actual answer
        a = [5,5]
        u = [.5,.5]
        f = smolyak.Hyper_Plane(a,u)
        d=2
        l=10
        q = smolyak.q(l,f)
        smolyak_integral = q.integrate(l)
        self.assertAlmostEqual(5, smolyak_integral, places=2)

    def test_q_integrate_2(self):#also a multiple of 4
        """
        (multiplying univariate weights by 2) no way these are correct
        (multiplying)
        """
        a = [5,5]
        u = [.5,.5]
        d=2
        l=10
        f = smolyak.Continuous_Function(a,u)
        q = smolyak.q(l,f)
        smolyak_integral = q.integrate(l)
        cubature_integral = smolyak.adaptive_cubature(f)
        self.assertAlmostEqual(cubature_integral, smolyak_integral, places=2)

    def test_q_integrate_3(self):#solution is 2*d greater
        """
        (dividing univariate weights by 2 was the issue)
        """
        a = [2,2,2]
        a= [each*5 for each in a]
        u = [.5,.5,.5]
        f = smolyak.Hyper_Plane(a,u)
        d=3
        l=10
        q = smolyak.q(l,f)
        smolyak_integral = q.integrate(l)
        self.assertAlmostEqual(a[0], smolyak_integral, places=2)
    def test_q_integrate_4(self):#solution is 2*d greater
        """
        possible issues
        cc weights comb (m vs k, what does thesis posit)
        dividing weights by 2*d
        (dividing univariate weights by 2 was the issue)
        """
        a = [2,2,2,2,2,2,2]
        a= [each*5 for each in a]
        u = [.5,.5,.5,.5,.5,.5,.5]
        f = smolyak.Hyper_Plane(a,u)
        d=3
        l=8
        q = smolyak.q(l,f)
        smolyak_integral = q.integrate(l)
        self.assertAlmostEqual(a[0], smolyak_integral, places=2)
    def test_q_integrate_5(self):#solution is 2*d greater
        """
        (dividing univariate weights by 2 was the issue)
        """
        a = [2,2,2]
        a= [each*5 for each in a]
        u = [.5,.5,.5]
        f = smolyak.Continuous_Function(a,u)
        d=3
        l=8
        q = smolyak.q(l,f)
        smolyak_integral = q.integrate(l)
        cubature_integral = smolyak.adaptive_cubature(f, abs_err=1e-04)
        mc_integral = smolyak.mc_integrate(f, abs_err=1e-04)
        self.assertAlmostEqual(cubature_integral, mc_integral, places=3)
        self.assertAlmostEqual(cubature_integral, smolyak_integral, places=3)
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