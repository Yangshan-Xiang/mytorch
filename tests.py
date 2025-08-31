import unittest
import math

from mytorch.models import Linear
from mytorch.parameter import Parameter


class TestFunctions(unittest.TestCase):

    def setUp(self):
        self.pts = [[1, 2], [3, 4]]
        self.linear_1 = Linear(2, 1)
        self.linear_2 = Linear(2, 2)
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
        self.assertEqual(self.output.one_step_back(1)[0][1], 1 / 200)

        self.assertEqual(len(self.c.one_step_back(1 / 200)), 2)
        self.assertEqual(id(self.c.one_step_back(1 / 200)[0][0]), id(self.a))
        self.assertEqual(id(self.c.one_step_back(1 / 200)[1][0]), id(self.b))
        self.assertEqual(self.c.one_step_back(1 / 200)[0][1], 1 / 200)
        self.assertEqual(self.c.one_step_back(1 / 200)[1][1], 1 / 200)

    def test_forward(self):
        self.assertEqual(self.output, math.log(200))

    def test_backward(self):
        self.output.backward()

        self.assertEqual(self.x.derivative, 0.05)
        self.assertEqual(self.y.derivative, 0.1)

    def test_Linear(self):
        for pt in self.pts:
            self.assertEqual(len(self.linear_1(pt)), 1)
            self.assertEqual(len(self.linear_2(pt)), 2)










if __name__ == '__main__':
    unittest.main()