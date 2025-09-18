# Here we use the numpy array instead of the python built-in list to store our tensors to improve the performance.

from numpy import array, ndarray
from typing import Callable
import math
from mytorch.tensor.utils import *


class TensorCore:
    """
    To store the four essential characteristics of a tensor.

    Attributes:
        storage (numpy.ndarray): A 1-dimensional numpy array which stores the value of the tensor elements.
        shape (tuple): The shape of the tensor.
        stride (tuple): The stride of the tensor.
        offset (int): The offset of the tensor.
    """

    def __init__(self, storage, shape: tuple, stride: tuple = None, offset: int = 0):
        self.storage = storage if isinstance(storage, ndarray) else array(storage)

        if not isinstance(shape, tuple):
            raise TypeError(f"Shape must be a tuple, got {type(shape)} instead.")
        elif len(storage) < math.prod(shape):
            # As we can just use a part of the storage, offset is needed
            raise ValueError("Tensor of given shape requires more elements than its storage has.")
        elif not all(isinstance(s, int) for s in shape):
            raise TypeError("Shape can only contain integers.")
        elif not all(s > 0 for s in shape):
            raise ValueError("shape can only contain positive integers.")
        else:
            self.shape = shape

        if stride is None:
            # If stride is not given, we will use the contiguous stride
            self.stride = self.contiguous_stride()
        elif not isinstance(stride, tuple):
            raise TypeError(f"Stride must be a tuple, got {type(stride)} instead.")
        elif len(stride) != len(shape):
            raise ValueError("Invalid stride for shape.")
        elif not all(isinstance(s, int) for s in stride):
            raise TypeError("Stride can only contain integers.")
        elif not all(s > 0 for s in stride):
            raise ValueError("Stride can only contain positive integers.")
        elif (sum([math.prod(l) for l in list(zip([s - 1 for s in self.shape], [*stride]))])
              > math.prod(self.shape) - 1):
            raise ValueError("Invalid stride for shape.")
        else:
            self.stride = stride

        if not isinstance(offset, int):
            raise TypeError(f"Offset must be a int, got {type(offset)} instead.")
        elif offset < 0:
            raise ValueError("Offset must be a non-negative integer.")
        elif len(storage) < math.prod(shape) + offset:
            raise ValueError(f"Offset can't be bigger than {len(storage) - math.prod(shape)}.")
        else:
            self.offset = offset

    def info(self):
        return self.storage, self.shape, self.stride, self.offset

    def is_contiguous(self):
        return self.stride == self.contiguous_stride()

    def contiguous_stride(self):
        shape = self.shape
        stride = [1] * len(shape)
        for i, s in enumerate(tuple(reversed(shape[1:]))):
            stride[i + 1] = stride[i] * s
        return tuple(reversed(stride))

    def broadcastable(self, other: 'TensorCore') -> bool:
        """
        According to the broadcasting rules, we check whether the two tensors (cores) are broadcastable.
        """
        if not isinstance(other, TensorCore):
            raise TypeError(f"Expected TensorCore, got {type(other)} instead.")
        else:
            self_shape = self.shape
            other_shape = other.shape

        for i in range(-1, -min(len(self_shape), len(other_shape)) - 1, -1):
            if self_shape[i] != other_shape[i] and self_shape[i] != 1 and other_shape[i] != 1:
                # We achieve this by comparing the size of the corresponding dimension of two tensors.
                return False
            else:
                pass
        return True

    def __repr__(self):
        return f"TensorCore({self.storage}, {self.shape}, {self.stride}, {self.offset})"


class History:
    def __init__(self, generator, cache, parents):
        self.generator = generator
        self.cache = cache
        self.parents = parents

    def __repr__(self):
        return f"History({self.generator}, {self.cache}, {self.parents})"


class Tensor:
    """
    A multi-dimensional matrix.

    Attributes:
        core :

    """

    def __init__(self, core: TensorCore, history: History = None, gradient: ndarray = None):
        self.core = core
        self.history = history
        self.gradient = gradient

    def __add__(self, y) -> 'Tensor':
        return

    def __radd__(self, y) -> 'Tensor':
        return

    def __sub__(self, y) -> 'Tensor':
        return

    def __rsub__(self, y) -> 'Tensor':
        return

    def __mul__(self, y) -> 'Tensor':
        return

    def __rmul__(self, y) -> 'Tensor':
        return

    def __truediv__(self, y) -> 'Tensor':
        return

    def __rtruediv__(self, y) -> 'Tensor':
        return

    def __neg__(self) -> 'Tensor':
        return

    def __eq__(self, y) -> bool:
        return

    def __gt__(self, y) -> bool:
        return

    def __lt__(self, y) -> bool:
        return

    def __repr__(self):
        return f"Tensor(core={self.core}, history={self.history}, gradient={self.gradient})"


class Zeros(Tensor):
    def __init__(self, shape: tuple):
        storage = array((0,) * math.prod(shape))
        core = TensorCore(storage, shape)
        super().__init__(core)


class Ones(Tensor):
    def __init__(self, shape: tuple):
        storage = array((1,) * math.prod(shape))
        core = TensorCore(storage, shape)
        super().__init__(core)


def map(func) -> Callable:
    """
    A higher-order function which turns a float-to-float (or vector to vector) function into
    a tensor-to-tensor function. Here we only need the core instead of the whole tensor.

    Args:
        func: A float-to-float (or vector-to-vector) function.

    Returns:
        A tensor-to-tensor function.

    """

    def _map(inp: TensorCore, out_shape: tuple = None) -> TensorCore:
        """
        The inner function which applies the given function on every element of the input tensor (core).
        It is essential to also feed in our output tensor (core) to enable broadcasting, however,
        all we need is its shape.

        Args:
            inp (TensorCore): The input tensor.
            out_shape (tuple): The shape of the output tensor after broadcasting.

        Returns:
            TensorCore: The output tensor.

        """
        if not isinstance(inp, TensorCore):
            raise TypeError(f"Expected TensorCore, got {type(inp)} instead.")
        else:
            inp_storage, inp_shape, inp_stride, inp_offset = inp.info()
            inp_num = math.prod(inp_shape)  # The number of elements within the storage which are actually used

        if out_shape is None or (isinstance(out_shape, tuple) and inp_shape == out_shape):
            if len(inp_storage) == inp_num:
                # This indicates every element inside the storage is used, then we can just apply the function
                # on the whole storage numpy array, utilize the SIMD (single instruction, multiple data) to
                # speed up the computation
                out_storage = func(inp_storage)
                out = TensorCore(out_storage, inp_shape, inp_stride)  # In this case, offset has to be zero
            else:
                # When some elements inside the storage are not used, we want to remove them as we don't want to feed
                # them into the function as well
                out_storage = array([0] * inp_num)

                for out_storage_idx in range(inp_num):
                    out_tensor_idx = to_tensor_idx(out_storage_idx, inp_shape)  # Ensure contiguous layout
                    inp_storage_idx = to_storage_idx(out_tensor_idx, inp_stride, inp_offset)
                    out_storage[out_storage_idx] = func(inp_storage[inp_storage_idx])  # No SIMD
                out = TensorCore(out_storage, inp_shape)

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
            out = TensorCore(out_storage, out_shape)  # So we use the default contiguous stride for output

        return out

    return _map



