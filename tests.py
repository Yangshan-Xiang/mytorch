import unittest
import numpy as np
from models.HTModelFixed import HTModelFixed

class TestFunctions(unittest.TestCase):

    def test_HTModelFixed(self):
        self.assertEqual(HTModelFixed(4, 2, 10, [12, 24]).forward(np.random.randn(4, 2)).shape[0], 10)


if __name__ == '__main__':
    unittest.main()