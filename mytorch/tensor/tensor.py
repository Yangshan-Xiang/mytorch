# Here we use the numpy array instead of the python built-in list to store our tensors to improve the performance.
import numpy as np
from numpy import array, ndarray
from typing import Callable
import math


class TensorCore:
    """
    To store the four essential characteristics of a tensor.

    Attributes:
        storage (numpy.ndarray): A 1-dimensional numpy array which stores the value of the tensor elements.
        shape (tuple): The shape of the tensor.
        stride (tuple): The stride of the tensor.
        offset (int): The offset of the tensor.
    """
    def __init__(self, storage, shape: tuple, stride: tuple=None, offset: int=0):
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
    """

    """
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
    def __init__(self, core: TensorCore, history: History=None, gradient: ndarray=None):
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


def to_storage_idx(tensor_idx: tuple, stride: tuple, offset: int):
    """
    Given the indices of an element within a tensor, the function returns the index of
    the element within the storage of the tensor.

    Args:
        tensor_idx (tuple): The index of the element within the tensor.
        stride (tuple): The stride of the tensor.
        offset (int): The offset of the tensor.

    Returns:
        int: The index of the element within the storage of the tensor.

    """
    storage_idx = 0

    for i in range(len(tensor_idx)):
        storage_idx += tensor_idx[i] * stride[i]

    return storage_idx + offset


def to_tensor_idx(storage_idx: int, shape: tuple):
    """
    Given the indices of an element within the storage of a tensor, the function returns the index of
    the element within the tensor based on contiguous layout, regardless of the stride.

    Args:
        storage_idx (int): The index of the element within the storage of the tensor.
        shape (tuple): The shape of the tensor.

    Returns:
        tuple: The index of the element within the tensor based on contiguous layout.

    """
    tensor_idx = [0] * len(shape)

    for i in range(len(shape) - 1, -1, -1):
        tensor_idx[i] = storage_idx % shape[i]
        storage_idx //= shape[i]

    return tuple(tensor_idx)


def from_broadcast_idx(broadcast_idx: tuple, broadcast_shape: tuple, shape: tuple):
    """


    Args:
        broadcast_idx:
        broadcast_shape:
        shape:

    Returns:

    """
    index = [0] * len(shape)
    for i in range(-1, -len(shape) - 1, -1):
        if shape[i] == 1:
            index[i] = 0
        else:
            index[i] = broadcast_idx[i]
    return tuple(index)


def mapping(func) -> Callable[[TensorCore, TensorCore], TensorCore]:
    """
    A higher-order function which turns a float-to-float (or vector to vector) function into
    a tensor-to-tensor function. Here we only need the core instead of the whole tensor.

    Args:
        func: A float-to-float (or vector-to-vector) function.

    Returns:
        A tensor-to-tensor function.

    """
    def _mapping(inp: TensorCore, out: TensorCore=None) -> TensorCore:
        # It is essential to feed in our output tensor core as input to enable broadcasting
        if not isinstance(inp, TensorCore):
            raise TypeError(f"Expected TensorCore, got {type(inp)} instead.")
        else:
            inp_storage, inp_shape, inp_stride, inp_offset = inp.info()


        if out is None or (isinstance(out, TensorCore) and inp.shape == out.shape):
            if len(inp_storage) == math.prod(inp_shape):
                out_storage = func(inp_storage) # Single instruction, multiple data (SIMD), faster computation.
                out = TensorCore(out_storage, inp_shape, inp_stride)
            else:
                out_storage = array([0] * math.prod(inp_shape))
                for out_storage_idx in range(math.prod(inp_shape)):
                    out_tensor_idx = to_tensor_idx(out_storage_idx, inp_shape)
                    inp_storage_idx = to_storage_idx(out_tensor_idx, inp_stride, inp_offset)
                    out_storage[out_storage_idx] = func(inp_storage[inp_storage_idx])
                out = TensorCore(out_storage, inp_shape)

        elif not isinstance(out, TensorCore):
            raise TypeError(f"Expected TensorCore, got {type(out)} instead.")
        elif len(inp.shape) > len(out.shape):
            raise ValueError("Invalid shapes for broadcasting.")
        else:
            for i in range(-1, -len(inp.shape) - 1, -1):
                if inp.shape[i] != 1 and inp.shape[i] != out.shape[i]:
                    raise ValueError("Invalid shapes for broadcasting.")
                else:
                    pass

            _, out_shape, _, _ = out.info() # Only need the shape
            out_storage = array([0] * math.prod(out_shape)) # Given output storage might contain unused elements

            for out_storage_idx in range(math.prod(out_shape)):
                out_tensor_idx = to_tensor_idx(out_storage_idx, out_shape) # This ensures contiguous layout for output
                inp_tensor_idx = from_broadcast_idx(out_tensor_idx, out_shape, inp_shape)
                inp_storage_idx = to_storage_idx(inp_tensor_idx, inp_stride, inp_offset)
                out_storage[out_storage_idx] = func(inp_storage[inp_storage_idx])
            out = TensorCore(out_storage, out_shape) # So we use the default contiguous stride for output

        return out

    return _mapping


def neg(x: float) -> float:
    return -x









