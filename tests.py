import unittest
import torch
from models.HTModel import HTModel

class TestFunctions(unittest.TestCase):

    def test_HTModelFixed(self):
        self.assertEqual(HTModel(4, 3, 2, 10, [12, 24]).forward(torch.randn(64, 4, 3)).shape[0], 64)
        self.assertEqual(HTModel(4, 3, 2, 10, [12, 24]).forward(torch.randn(64, 4, 3)).shape[1], 10)


if __name__ == '__main__':
    unittest.main()