import math
from typing import Union
from mytorch.tensor.utils import *


class History:
    """
    Stores the computation history of a tensor parameter which will be used in the backward pass.

    Attributes:
        function: The function which is used.
        cache: Stores some useful values computed during forward pass to save computation.
        parents (tuple): The input values for the function.

    """
    def __init__(self, function = None, cache = None, parents: tuple = None):
        self.function = function
        self.cache = cache
        self.parents = parents

    def __repr__(self):
        if self.parents is None:
            return f"History()"
        else:
            return f"History(function={self.function}, parents={self.parents})"



class Tensor:
    """
    Stores the four essential features of a tensor, and its history and gradient if it is a tensor parameter.

    Attributes:
        storage (list): A 1-dimensional list which stores the value of the elements of the tensor.
        shape (tuple): The shape of the tensor.
        stride (tuple): The stride of the tensor.
        offset (int): The offset of the tensor.
        history (History): Stores the computation history of the tensor parameter.
    """

    def __init__(self,
                 storage: list,
                 shape: tuple = None,
                 stride: tuple = None,
                 offset: int = 0,
                 history: History = None):

        if not isinstance(storage, list):
            raise TypeError("Storage must be a list.")
        else:
            self.storage = storage

        if shape is None:
            self.shape = (len(storage),) # Default shape is a 1-d vector
        elif not isinstance(shape, tuple):
            raise TypeError(f"Shape must be a tuple, got {type(shape)} instead.")
        elif len(self.storage) < math.prod(shape):
            # As we can just use a part of the storage, offset is needed
            raise ValueError("Tensor of given shape requires more elements than its storage has.")
        elif not all(isinstance(s, int) for s in shape):
            raise TypeError("Shape can only contain integers.")
        elif not all(s > 0 for s in shape):
            raise ValueError("shape can only contain positive integers.")
        else:
            self.shape = shape

        if stride is None:
            # If stride is not given, we will use the corresponding contiguous stride
            self.stride = self.contiguous_stride()
        elif not isinstance(stride, tuple):
            raise TypeError(f"Stride must be a tuple, got {type(stride)} instead.")
        elif len(stride) != len(self.shape):
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
        elif len(self.storage) < math.prod(self.shape) + offset:
            raise ValueError(f"Offset can't be bigger than {len(self.storage) - math.prod(self.shape)}.")
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

    def to_constant(self):
        """
        Changes a tensor parameter into a constant tensor by simply removing its history.
        """
        return Tensor(self.storage, self.shape, self.stride, self.offset)

    def is_leaf(self):
        """
        Checks if the tensor parameter is a leaf parameter, as we only need to store the gradients
        w.r.t. leaf parameters.
        """

        return self.history is not None and self.history.parents is None

    def core(self):
        return self.storage, self.shape, self.stride, self.offset

    def is_contiguous(self):
        """
        Checks if the tensor's stride is contiguous.
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
            self (Tensor): The tensor itself.
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

    def reshape(self, *shape):
        pass
    def __getitem__(self):
        pass
    def __setitem__(self):
        pass

    def topological_sort(self) -> list:
        """
        Find the topological ordering of tensor parameters. Here we apply depth-first search
        with reverse post-ordering to linearly order the parameters. The result might
        not be unique, it ensures that a parent parameter will only be visited
        after all of its children have been visited to avoid redundant computations.
        """

        visited = [] # Avoid revisit
        ordered = []

        def dfs(x):
            if x.history: # We don't need constant tensors here
                if id(x) not in visited: # Use id() to identify different parameters
                    visited.append(id(x))

                    if x.history.parents:
                        for parent in x.history.parents:
                            dfs(parent) # Recursion
                        ordered.append(x)
                    else:
                        ordered.append(x)
                else:
                    pass
            else:
                pass

        dfs(self)
        ordered.reverse() # Reverse post-ordering
        return ordered

    def chain_rule(self, grad: 'Tensor') -> list:
        """
        Applies chain rule to compute and assign the derivatives of the child with respect to its parents.
        """

        h = self.history
        return list(zip(h.parents, h.function.backward(h.cache, grad))) # Pair the gradients with its parents

    def backward(self) -> None:
        """
        Applies the chain rule to compute and save the gradients w.r.t. the leaf tensor parameters.
        The saved gradients will then be used by optimizers to update themselves.
        """

        ordered = self.topological_sort() # No constant tensors already
        catalog = {id(x): Tensor([0]) for x in ordered} # To record computed gradients
        catalog[id(self)] = Tensor([1]) # Initialize the starting gradient as Tensor([1])

        for param in ordered:
            if param.history.parents: # Not a leaf parameter
                for parent, grad in param.chain_rule(catalog[id(param)]):
                    if parent.history:
                        catalog[id(parent)] += grad # Accumulate the gradients according to the chain rule
                    else:
                        pass
            else: # A leaf parameter
                if not param.gradient:
                    param.gradient = Tensor([0])
                else:
                    pass
                param.gradient += catalog[id(param)] # Enable it to accumulate derivatives,
                # useful in certain cases like multitask learning.

    def __repr__(self):
        if self.history is None:
            return (f"Tensor("
                    f"storage={[round(element, 4) for element in self.storage]}, "
                    f"shape={self.shape}, "
                    f"stride={self.stride}, "
                    f"offset={self.offset})")
        else:
            return (f"Tensor("
                    f"storage={[round(element, 4) for element in self.storage]}, "
                    f"shape={self.shape}, "
                    f"stride={self.stride}, "
                    f"offset={self.offset}, "
                    f"history={self.history})")


    # To avoid circular import, following methods are all defined in arithmetic.py
    def __add__(self, other):
        pass
    def __radd__(self, other):
        pass
    def __sub__(self, other):
        pass
    def __rsub__(self, other):
        pass
    def __mul__(self, other):
        pass
    def __rmul__(self, other):
        pass
    def __truediv__(self, other):
        pass
    def __rtruediv__(self, other):
        pass
    def __eq__(self, other):
        pass
    def __gt__(self, other):
        pass
    def __lt__(self, other):
        pass
    def __neg__(self):
        pass
    def __pow__(self):
        pass
    def softmax(self, dim: int):
        pass
    def log(self):
        pass
    def exp(self):
        pass
    def relu(self):
        pass
    def rcp(self):
        pass











