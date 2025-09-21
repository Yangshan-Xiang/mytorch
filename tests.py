import unittest

from mytorch.models import *
from mytorch.optimizers import *
from mytorch.parameter import *

from mytorch.tensor.arithmetic import *


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

        self.storage = [1, 2, 3, 4, 5, 6]
        self.shape = (2, 2) # Only need 4 elements in the storage
        self.stride = (1, 2) # Transposed
        self.offset = 2 # Starting from the 3rd element in the storage, skipping the first two
        self.tensor = Tensor(self.storage, self.shape, self.stride, self.offset) # Representing [[3, 5], [4, 6]]

        self.storage2 = [1, 2, 3, 4]
        self.shape2 = (2, 1, 2)
        self.stride2 = (2, 2, 1)
        self.offset2 = 0
        self.tensor2 = Tensor(self.storage2, self.shape2, self.stride2, self.offset2)

        self.storage3 = [1, 2, 3, 4, 5, 6]
        self.shape3 = (3, 2)
        self.stride3 = (1, 3)
        self.offset3 = 0
        self.tensor3 = Tensor(self.storage3, self.shape3, self.stride3, self.offset3)

        self.storage4 = [1, 2, 3, 4, 5, 6]
        self.shape4 = (3, 2)
        self.stride4 = (2, 1)
        self.offset4 = 0
        self.tensor4 = Tensor(self.storage4, self.shape4, self.stride4, self.offset4)




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

    def test_Adagrad(self):
        self.assertIsNone(self.x.derivative)
        self.x.derivative = 2
        lr = 0.1
        Adagrad([self.x], lr).step()
        self.assertEqual(self.x.value, 10 - lr * 2 / (2 + 1e-8)) # As you can see, it is the same as Adam


    def test_zero_grad(self):
        self.assertIsNone(self.x.derivative)
        self.assertIsNone(self.y.derivative)
        self.x.derivative = 2
        self.x.derivative = 4

        Adam([self.x, self.y]).zero_grad()

        self.assertIsNone(self.x.derivative)
        self.assertIsNone(self.y.derivative)

    def test_to_storage_idx(self):
        self.assertEqual(to_storage_idx((0, 0), self.stride, self.offset), 2)
        self.assertEqual(to_storage_idx((0, 1), self.stride, self.offset), 4)
        self.assertEqual(to_storage_idx((1, 0), self.stride, self.offset), 3)
        self.assertEqual(to_storage_idx((1, 1), self.stride, self.offset), 5)

    def test_to_tensor_idx(self):
        self.assertEqual(to_tensor_idx(0, self.shape), (0, 0))
        self.assertEqual(to_tensor_idx(1, self.shape), (0, 1))
        self.assertEqual(to_tensor_idx(2, self.shape), (1, 0))
        self.assertEqual(to_tensor_idx(3, self.shape), (1, 1))

    def test_from_broadcast_idx(self):
        self.assertEqual(from_broadcast_idx((0, 0, 0), self.shape), (0, 0))
        self.assertEqual(from_broadcast_idx((0, 0, 1), self.shape), (0, 1))
        self.assertEqual(from_broadcast_idx((0, 1, 0), self.shape), (1, 0))
        self.assertEqual(from_broadcast_idx((0, 1, 1), self.shape), (1, 1))
        self.assertEqual(from_broadcast_idx((1, 0, 0), self.shape), (0, 0))
        self.assertEqual(from_broadcast_idx((1, 0, 1), self.shape), (0, 1))
        self.assertEqual(from_broadcast_idx((1, 1, 0), self.shape), (1, 0))
        self.assertEqual(from_broadcast_idx((1, 1, 1), self.shape), (1, 1))

    def test_is_contiguous(self):
        self.assertEqual(self.tensor.is_contiguous(), False)
        self.assertEqual(self.tensor2.is_contiguous(), True)
        self.assertEqual(self.tensor3.is_contiguous(), False)
        self.assertEqual(self.tensor4.is_contiguous(), True)


    def test_contiguous_stride(self):
        self.assertEqual(self.tensor.contiguous_stride(), (2, 1))
        self.assertEqual(self.tensor2.contiguous_stride(), (2, 2, 1))
        self.assertEqual(self.tensor3.contiguous_stride(), (2, 1))
        self.assertEqual(self.tensor4.contiguous_stride(), (2, 1))

    def test_broadcastable(self):
        self.assertEqual(self.tensor.broadcastable(self. tensor2), (2, 2, 2))
        self.assertEqual(self.tensor.broadcastable(self.tensor3), False)
        self.assertEqual(self.tensor.broadcastable(self.tensor4), False)
        self.assertEqual(self.tensor2.broadcastable(self.tensor3), (2, 3, 2))
        self.assertEqual(self.tensor2.broadcastable(self.tensor4), (2, 3, 2))
        self.assertEqual(self.tensor3.broadcastable(self.tensor4), (3, 2))

    def test_tensor_map(self):
        output1 = tensor_map(neg)(self.tensor)
        storage1 = output1.storage
        shape1 = output1.shape
        stride1 = output1.stride
        offset1 = output1.offset

        output2 = tensor_map(neg)(self.tensor, (2, 2, 2))
        storage2 = output2.storage
        shape2 = output2.shape
        stride2 = output2.stride
        offset2 = output2.offset

        self.assertEqual(storage1, [-3, -5, -4, -6])
        self.assertEqual(shape1, (2, 2))
        self.assertEqual(stride1, (2, 1))
        self.assertEqual(offset1, 0)

        self.assertEqual(storage2, [-3, -5, -4, -6, -3, -5, -4, -6])
        self.assertEqual(shape2, (2, 2, 2))
        self.assertEqual(stride2, (4, 2, 1))
        self.assertEqual(offset2, 0)

    def test_tensor_zip(self):
        output1 = tensor_zip(add)(self.tensor, self.tensor2)
        storage1 = output1.storage
        shape1 = output1.shape
        stride1 = output1.stride
        offset1 = output1.offset

        self.assertEqual(storage1, [4, 7, 5, 8, 6, 9, 7, 10])
        self.assertEqual(shape1, (2, 2, 2))
        self.assertEqual(stride1, (4, 2, 1))
        self.assertEqual(offset1, 0)

        self.assertRaises(ValueError, tensor_zip(add), self.tensor, self.tensor3)
        self.assertRaises(ValueError, tensor_zip(add), self.tensor, self.tensor4)

        output2 = tensor_zip(add)(self.tensor3, self.tensor4)
        storage2 = output2.storage
        shape2 = output2.shape
        stride2 = output2.stride
        offset2 = output2.offset

        self.assertEqual(storage2, [2, 6, 5, 9, 8, 12])
        self.assertEqual(shape2, (3, 2))
        self.assertEqual(stride2, (2, 1))
        self.assertEqual(offset2, 0)

    def test_Neg(self):
        self.assertEqual((-self.tensor).storage, [-3, -5, -4, -6])
        self.assertEqual((-self.tensor).shape, (2, 2))
        self.assertEqual((-self.tensor).stride, (2, 1))

        self.assertEqual((-self.tensor2).storage, [-1, -2, -3, -4])
        self.assertEqual((-self.tensor2).shape, (2, 1, 2))
        self.assertEqual((-self.tensor2).stride, (2, 2, 1))

        self.assertEqual((-self.tensor3).storage, [-1, -4, -2, -5, -3, -6])
        self.assertEqual((-self.tensor3).shape, (3, 2))
        self.assertEqual((-self.tensor3).stride, (2, 1))

        self.assertEqual((-self.tensor4).storage, [-1, -2, -3, -4, -5, -6])
        self.assertEqual((-self.tensor4).shape, (3, 2))
        self.assertEqual((-self.tensor4).stride, (2, 1))







if __name__ == '__main__':
    unittest.main()
