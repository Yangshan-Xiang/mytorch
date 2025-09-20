from mytorch.tensor.parameter import *
from mytorch.tensor.operations import *
from mytorch.tensor.utils import *
from typing import Callable

def tensor_map(func) -> Callable:
    """
    A higher-order function which turns a float-to-float (or vector to vector) function into
    a tensor-to-tensor function.

    Args:
        func: A float-to-float (or vector-to-vector) function.

    Returns:
        A tensor-to-tensor function.
    """

    def _map(inp: Tensor, out_shape: tuple = None) -> Tensor:
        """
        The inner function which applies the given function on every element of the input tensor.
        It is also possible to specify the output shape to enable broadcasting.

        Args:
            inp (Tensor): The input tensor.
            out_shape (tuple): The shape of the output tensor after broadcasting.

        Returns:
            Tensor: The output tensor.

        """
        if not isinstance(inp, Tensor):
            raise TypeError(f"Expected Tensor, got {type(inp)} instead.")
        else:
            inp_storage, inp_shape, inp_stride, inp_offset = inp.info()
            inp_num = math.prod(inp_shape)  # The number of elements within the storage which are actually used

        if out_shape is None or (isinstance(out_shape, tuple) and inp_shape == out_shape):
            if len(inp_storage) == inp_num:
                # This indicates every element inside the storage is used, then we can just apply the function
                # on the whole storage numpy array, utilize the SIMD (single instruction, multiple data) to
                # speed up the computation
                out_storage = func(inp_storage)
                out = Tensor(out_storage, inp_shape, inp_stride)  # In this case, offset has to be zero
            else:
                # When some elements inside the storage are not used, we want to remove them as we don't want to feed
                # them into the function as well
                out_storage = array([0] * inp_num)

                for out_storage_idx in range(inp_num):
                    out_tensor_idx = to_tensor_idx(out_storage_idx, inp_shape)  # Ensure contiguous layout
                    inp_storage_idx = to_storage_idx(out_tensor_idx, inp_stride, inp_offset)
                    out_storage[out_storage_idx] = func(inp_storage[inp_storage_idx])  # No SIMD
                out = Tensor(out_storage, inp_shape)

        elif not isinstance(out_shape, tuple):
            raise TypeError(f"Expected tuple, got {type(out_shape)} instead.")
        elif len(inp_shape) > len(out_shape):
            raise ValueError("Invalid shapes for broadcasting.")
        else:
            for i in range(-1, -len(inp_shape) - 1, -1):
                if inp_shape[i] != 1 and inp_shape[i] != out_shape[i]:
                    raise ValueError("Invalid shapes for broadcasting.")
                else:
                    pass

            out_num = math.prod(out_shape)
            out_storage = array([0] * out_num)  # Given output storage might contain unused elements

            for out_storage_idx in range(out_num):
                out_tensor_idx = to_tensor_idx(out_storage_idx, out_shape)  # This ensures contiguous layout for output
                inp_tensor_idx = from_broadcast_idx(out_tensor_idx, out_shape, inp_shape)  # Broadcasting
                inp_storage_idx = to_storage_idx(inp_tensor_idx, inp_stride, inp_offset)
                out_storage[out_storage_idx] = func(inp_storage[inp_storage_idx])  # No SIMD
            out = Tensor(out_storage, out_shape)  # So we use the default contiguous stride for output

        return out

    return _map


class Neg:
    """
    Custom function for negation.
    """
    @staticmethod
    def forward(x: TensorParameter) -> TensorParameter:
        cache = ()
        history = History(Neg, cache, (x,))
        return TensorParameter(tensor_map(neg)(x.tensor), history)

    @staticmethod
    def backward(cache, grad: ndarray) -> tuple:
        return (-1 * grad,)

TensorParameter.__add__ =
TensorParameter.__radd__ =
TensorParameter.__sub__ =
TensorParameter.__rsub__ =
TensorParameter.__mul__ =
TensorParameter.__rmul__ =
TensorParameter.__truediv__ =
TensorParameter.__rtruediv__ =
TensorParameter.__neg__ = Neg.forward
TensorParameter.__eq__ =
TensorParameter.__gt__ =
TensorParameter.__lt__ =

