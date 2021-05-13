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
    def test_smolyak(self):
        


    # def test_isupper(self):
    #     self.assertTrue('FOO'.isupper())
    #     self.assertFalse('Foo'.isupper())
    #     self.assertEqual(s.split(), ['hello', 'world'])

if __name__ == '__main__':
    unittest.main()