from numpy import array, ndarray # We use numpy array instead of python list for our tensor storage to improve performance.
import math
from mytorch.functions import History


class Tensor:
    """
    To store the four essential characteristics of a tensor.

    Attributes:
        storage (ndarray): A 1-dimensional numpy array which stores the value of the tensor elements.
        shape (tuple): The shape of the tensor.
        stride (tuple): The stride of the tensor.
        offset (int): The offset of the tensor.
    """

    def __init__(self, storage, shape: tuple, stride: tuple = None, offset: int = 0):
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

    def broadcastable(self, other: 'Tensor') -> bool:
        """
        According to the broadcasting rules, we check whether the two tensors are broadcastable.
        """
        if not isinstance(other, Tensor):
            raise TypeError(f"Expected Tensor, got {type(other)} instead.")
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
        return f"Tensor(storage={self.storage}, shape={self.shape}, stride={self.stride}, offset={self.offset})"




class TensorParameter:
    """
    A multi-dimensional matrix.

    Attributes:



    """

    def __init__(self, tensor: Tensor, history: History = None):
        self.tensor = tensor
        self.history = history
        self.gradient = None

    def __add__(self, y) -> 'TensorParameter':
        return

    def __radd__(self, y) -> 'TensorParameter':
        return

    def __sub__(self, y) -> 'TensorParameter':
        return

    def __rsub__(self, y) -> 'TensorParameter':
        return

    def __mul__(self, y) -> 'TensorParameter':
        return

    def __rmul__(self, y) -> 'TensorParameter':
        return

    def __truediv__(self, y) -> 'TensorParameter':
        return

    def __rtruediv__(self, y) -> 'TensorParameter':
        return

    def __neg__(self) -> 'TensorParameter':
        return

    def __eq__(self, y) -> bool:
        return

    def __gt__(self, y) -> bool:
        return

    def __lt__(self, y) -> bool:
        return

    def __repr__(self):
        return f"TensorParameter(tensor={self.tensor}, history={self.history}, gradient={self.gradient})"

    def is_leaf(self):
        return






