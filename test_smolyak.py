import numpy as np
import smolyak
import unittest


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
    # def test_continuous_plot(self):
    #     a = [5,5]
    #     u = [.5,.5]
    #     f = smolyak.Continuous_Function(a,u)
    #     f.plot()
    #     f = smolyak.Gaussian_Function(a,u)
    #     f.plot()
    #     f = smolyak.Oscillatory_Function(a,u)
    #     f.plot()
    #     f = smolyak.Discontinuous_Function(a,u)
    #     f.plot()
    #     self.assertTrue(True)
    def test_smolyak_continuous(self):
        a = [5,5]
        u = [.5,.5]
        f = smolyak.Continuous_Function(a,u)
        smolyak_integral = smolyak.smolyak_integrate(f)
        # tensor_integral = smolyak.tensor_integrate(f)
        # mc_integral = smolyak.mc_integrate(f)
        cubature_integral = smolyak.adaptive_cubature(f)
        self.assertAlmostEqual(cubature_integral, smolyak_integral, places=2)
        # self.assertAlmostEqual(mc_integral, smolyak_integral)
        # self.assertAlmostEqual(mc_integral, tensor_integral, places=4)
    def test_generate_midpoint(self):
        dim = 2
        level = 3
        all_k = smolyak.generate_k(dim, level)
        count_k = 0
        count_j = 0
        max_k=1
        for k in all_k:#check k
            max_k = sum(k) if sum(k) > max_k else max_k
            count_k += 1
            self.assertTrue(sum(k)<=level+dim -1)
            self.assertTrue(dim<=sum(k))
            all_j = smolyak.generate_midpoint_j(k)
            for j in all_j:
                np_j = np.array(j)
                np_k = np.array(k)
                self.assertTrue(np.all(np_j<=np_k))
                self.assertTrue(np.all(np.ones(np_j.shape)<=np_j))
                count_j += 1
                p = smolyak.generate_mid_point(j,k)
                np_p = np.array(p)
                if j[0] == k[0] and j[0] == 1:
                    self.assertAlmostEqual(p[0], 1/k[0]/2)
                self.assertEqual(len(p), len(j))
                self.assertTrue(np.all(np_p<=np.ones(np_p.shape)))
                self.assertTrue(np.all(np.zeros(np_p.shape)<=np_p))
                q_max = level+2*dim - np.sum(np_k) - 1
                q_min = len(k)
                for q in smolyak.generate_q(k, level, dim):
                    np_q = np.array(q)
                    q_norm = np.sum(q)
                    self.assertTrue(np.sum(q) <= q_max)
                    self.assertTrue(np.all(np.ones(np_q.shape) <= np_q))
        # self.assertEqual(count_k, 3)
        # self.assertEqual(count_j, 5)
        self.assertEqual(max_k, level+dim-1)
    def test_point_weight(self):
        dim = 2
        l = 3
        for k in smolyak.generate_k(dim, l):
            for j in smolyak.generate_midpoint_j(k):
                point = smolyak.generate_mid_point(j,k)
                weight = smolyak.get_midpoint_weight(j,k, dim, l)
                # print()
                # print(f"k:{k}")
                # print(f"j:{j}")
                # print(f"p:{point}")
                # print(f"w:{weight}")
    def test_univariate_midpoint(self):
        k = [2]
        j = [1]
        point = smolyak.generate_mid_point(j,k)
        self.assertAlmostEqual(point[0], 1/3)
    def test_univariate_midpoint_integral(self):
        a = [2,2]
        u = [.5,.5]
        f = smolyak.Hyper_Plane(a,u)
        smolyak_integral = smolyak.smolyak_integrate(f, 2)
        self.assertAlmostEqual(2, smolyak_integral, places=2)
        smolyak_integral = smolyak.smolyak_integrate(f, 5)
        self.assertAlmostEqual(2, smolyak_integral, places=2)
        


        


    # def test_isupper(self):
    #     self.assertTrue('FOO'.isupper())
    #     self.assertFalse('Foo'.isupper())
    #     self.assertEqual(s.split(), ['hello', 'world'])

if __name__ == '__main__':
    unittest.main()