import unittest
from mytorch.models import *
from mytorch.optimizers import *
from mytorch.parameter import *


class TestFunctions(unittest.TestCase):

    def setUp(self):
        self.pts = [[1, 2], [3, 4]]
        self.linear_1 = Linear(2, 1)
        self.linear_2 = Linear(2, 2)
        self.linear_3 = Linear(2, 3)
        self.x = Parameter(10)
        self.y = Parameter(10)
        self.z = 10
        self.a = self.x * self.y
        self.b = self.y * self.z
        self.c = self.a + self.b
        self.output = self.c.log()

    def test_topological_ordering(self):
        right_ordering = [id(self.output), id(self.c), id(self.b), id(self.a), id(self.y), id(self.x)]
        ordering = []

        for i in self.output.topological_ordering():
            ordering.append(id(i))

        self.assertEqual(ordering, right_ordering)

    def test_one_step_back(self):
        self.assertEqual(len(self.output.one_step_back(1)), 1)
        self.assertEqual(id(self.output.one_step_back(1)[0][0]), id(self.c))
        self.assertAlmostEqual(self.output.one_step_back(1)[0][1], 1 / 200)

        self.assertEqual(len(self.c.one_step_back(1 / 200)), 2)
        self.assertEqual(id(self.c.one_step_back(1 / 200)[0][0]), id(self.a))
        self.assertEqual(id(self.c.one_step_back(1 / 200)[1][0]), id(self.b))
        self.assertEqual(self.c.one_step_back(1 / 200)[0][1], 1 / 200)
        self.assertEqual(self.c.one_step_back(1 / 200)[1][1], 1 / 200)

    def test_forward(self):
        self.assertAlmostEqual(self.output.value, math.log(200))

    def test_backward(self):
        self.output.backward()

        self.assertAlmostEqual(self.x.derivative, 0.05)
        self.assertAlmostEqual(self.y.derivative, 0.1)

    def test_Linear(self):
        for pt in self.pts:
            self.assertEqual(len(self.linear_1(pt)), 1)
            self.assertEqual(len(self.linear_2(pt)), 2)
            self.assertEqual(len(self.linear_3(pt)), 3)

    def test_get_params(self):
        self.assertEqual(len(self.linear_1.get_params()), 3)
        self.assertEqual(len(self.linear_2.get_params()), 6)
        self.assertEqual(len(self.linear_3.get_params()), 9)

    def test_SGD(self):
        self.assertIsNone(self.x.derivative)
        self.x.derivative = 2
        lr = 0.01
        SGD([self.x], lr).step()
        self.assertEqual(self.x.value, 10 - lr * self.x.derivative)

    def test_Adam(self):
        self.assertIsNone(self.x.derivative)
        self.x.derivative = 2
        lr = 0.01

        Adam([self.x], lr).step()
        self.assertEqual(self.x.value, 10 - lr * 2 / (2 + 1e-8))

    def test_zero_grad(self):
        self.assertIsNone(self.x.derivative)
        self.assertIsNone(self.y.derivative)
        self.x.derivative = 2
        self.x.derivative = 4

        Adam([self.x, self.y]).zero_grad()

        self.assertIsNone(self.x.derivative)
        self.assertIsNone(self.y.derivative)


















if __name__ == '__main__':
    unittest.main()