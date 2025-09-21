from numpy import array, ndarray # We use numpy array instead of python list for our tensor storage to improve performance.
import math
from typing import Union
from mytorch.tensor.utils import *


class History:
    """
    In order to store all the necessary history information of a parameter which will later
    be used in the backward pass.

    Attributes:
        generator: The function which generates the value of the parameter.
        cache: Save computed values during the forward pass to avoid unnecessary computation in the backward pass.
        parents (tuple): The input values for the generator.

    """
    def __init__(self, generator, cache, parents: tuple):
        self.generator = generator
        self.cache = cache
        self.parents = parents

    def __repr__(self):
        return f"History(generator={self.generator}, parents={self.parents})"


class Tensor:
    """
    To store the four essential characteristics of a tensor.

    Attributes:
        storage (ndarray): A 1-dimensional numpy array which stores the value of the tensor elements.
        shape (tuple): The shape of the tensor.
        stride (tuple): The stride of the tensor.
        offset (int): The offset of the tensor.
    """

    def __init__(self,
                 storage,
                 shape: tuple,
                 stride: tuple = None,
                 offset: int = 0,
                 history: History = None):
        self.storage = array(storage)

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

        self.history = history
        self.gradient = None

    def is_constant(self):
        """
        Checks if the tensor is constant, if it is, then we don't need to compute the gradient w.r.t. it.
        Be aware that we didn't use 'requires_grad' here like what PyTorch did, so make sure that you have
        initialized the history of the tensor with 'History()' when you want the gradient w.r.t. it to be
        computed.
        """

        return self.history is None

    def is_leaf(self):
        """
        Checks if the tensor is a leaf node, as we only need the gradients w.r.t. the leaf nodes.
        """
        return self.history is not None and self.history.parents is None

    def core(self):
        return self.storage, self.shape, self.stride, self.offset

    def numpy(self) -> ndarray:
        """
        Returns the tensor as a numpy array.
        """
        storage = self.storage
        shape = self.shape
        stride = self.stride
        offset = self.offset

        num = math.prod(shape)
        out = array([0] * num).reshape(shape)
        for idx in range(num):
            out_tensor_idx = to_tensor_idx(idx, shape)
            in_storage_idx = to_storage_idx(out_tensor_idx, stride, offset)
            out[out_tensor_idx] = storage[in_storage_idx]
        return out

    def is_contiguous(self):
        """
        Checks if the tensor is contiguous.
        """
        return self.stride == self.contiguous_stride()

    def contiguous_stride(self):
        """
        Returns the contiguous stride of the tensor.
        """
        shape = self.shape
        stride = [1] * len(shape)
        for i, s in enumerate(tuple(reversed(shape[1:]))):
            stride[i + 1] = stride[i] * s
        return tuple(reversed(stride))

    def broadcastable(self, other: 'Tensor') -> Union[bool, tuple]:
        """
        According to the broadcasting rules, we check whether the two tensors are broadcastable,
        if they aren't, returns False, if they are, returns the shape of the broadcast tensor.

        Args:
            self (Tensor): A Tensor object.
            other (Tensor): The other tensor.

        Returns:
            Union[bool, tuple]: False if the inputs are not broadcastable, the shape of the broadcast tensor otherwise.
        """

        if not isinstance(other, Tensor):
            raise TypeError(f"Expected Tensor, got {type(other)} instead.")
        else:
            self_shape = self.shape
            other_shape = other.shape
            if len(self_shape) > len(other_shape):
                broadcast_shape = list(self_shape)
            else:
                broadcast_shape = list(other_shape)

        for i in range(-1, -min(len(self_shape), len(other_shape)) - 1, -1):
            if self_shape[i] != other_shape[i]:
                if self_shape[i] != 1:
                    if other_shape[i] != 1:
                        return False
                    else:
                        broadcast_shape[i] = self_shape[i]
                else:
                    broadcast_shape[i] = other_shape[i]
            else:
                broadcast_shape[i] = self_shape[i]

        return tuple(broadcast_shape)

    def __repr__(self):
        if self.history is None:
            return (f"Tensor("
                    f"storage={self.storage}, "
                    f"shape={self.shape}, "
                    f"stride={self.stride}, "
                    f"offset={self.offset})")
        else:
            return (f"Tensor("
                    f"storage={self.storage}, "
                    f"shape={self.shape}, "
                    f"stride={self.stride}, "
                    f"offset={self.offset}), "
                    f"history={self.history}")








