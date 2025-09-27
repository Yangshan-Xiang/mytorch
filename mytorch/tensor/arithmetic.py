import itertools

from mytorch.tensor.tensor import *
from mytorch.tensor.operations import *
from mytorch.tensor.utils import *
from typing import Callable

def tensor_map(func) -> Callable:
    """
    A higher-order function which turns a float-to-float function into
    a tensor-to-tensor function.

    Args:
        func: A float-to-float function.

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
            inp_storage, inp_shape, inp_stride, inp_offset = inp.core()
            inp_num = math.prod(inp_shape)

        if out_shape is None or (isinstance(out_shape, tuple) and inp_shape == out_shape): # Without broadcasting
            out_storage = [None] * inp_num
            for out_storage_idx in range(inp_num):
                out_tensor_idx = to_tensor_idx(out_storage_idx, inp_shape) # Ensures contiguous layout for output
                inp_storage_idx = to_storage_idx(out_tensor_idx, inp_stride, inp_offset)
                out_storage[out_storage_idx] = func(inp_storage[inp_storage_idx])
            out = Tensor(out_storage, inp_shape)

        elif not isinstance(out_shape, tuple):
            raise TypeError(f"Expected tuple, got {type(out_shape)} instead.")
        elif len(inp_shape) > len(out_shape):
            raise ValueError("Invalid shapes for broadcasting.")
        else: # With broadcasting
            for i in range(-1, -len(inp_shape) - 1, -1):
                if inp_shape[i] != 1 and inp_shape[i] != out_shape[i]:
                    raise ValueError("Invalid shapes for broadcasting.")
                else:
                    pass
            out_num = math.prod(out_shape)
            out_storage = [None] * out_num
            for out_storage_idx in range(out_num):
                out_tensor_idx = to_tensor_idx(out_storage_idx, out_shape)  # Ensures contiguous layout for output
                inp_tensor_idx = from_broadcast_idx(out_tensor_idx, inp_shape)  # Broadcasting?????????
                inp_storage_idx = to_storage_idx(inp_tensor_idx, inp_stride, inp_offset)
                out_storage[out_storage_idx] = func(inp_storage[inp_storage_idx])
            out = Tensor(out_storage, out_shape)  # So we use the default contiguous stride for output

        return out

    return _map

def tensor_zip(func) -> Callable:
    """

    """
    def _zip(x: Tensor, y: Tensor) -> Tensor:
        """

        """
        if not isinstance(x, Tensor):
            raise TypeError(f"The 1st argument is expected to be Tensor, got {type(x)} instead.")
        elif not isinstance(y, Tensor):
            raise TypeError(f"The 2nd argument is expected to be Tensor, got {type(y)} instead.")
        else:
            x_storage, x_shape, x_stride, x_offset = x.core()
            y_storage, y_shape, y_stride, y_offset = y.core()

        if x_shape == y_shape: # Without broadcasting
            num = math.prod(x_shape)
            out_storage = [None] * num
            for out_storage_idx in range(num):
                out_tensor_idx = to_tensor_idx(out_storage_idx, x_shape)

                x_storage_idx = to_storage_idx(out_tensor_idx, x_stride, x_offset)
                y_storage_idx = to_storage_idx(out_tensor_idx, y_stride, y_offset)

                out_storage[out_storage_idx] = func(x_storage[x_storage_idx], y_storage[y_storage_idx])
            out = Tensor(out_storage, x_shape)
        else: # With broadcasting
            if x.broadcastable(y) is False:
                raise ValueError(f"Tensors with shapes {x_shape} and {y_shape} are not broadcastable.")
            else:
                out_shape = x.broadcastable(y)
                out_num = math.prod(out_shape)
                out_storage = [None] * out_num

            for out_storage_idx in range(out_num):
                out_tensor_idx = to_tensor_idx(out_storage_idx, out_shape)

                x_tensor_idx = from_broadcast_idx(out_tensor_idx, x_shape)
                y_tensor_idx = from_broadcast_idx(out_tensor_idx, y_shape)

                x_storage_idx = to_storage_idx(x_tensor_idx, x_stride, x_offset)
                y_storage_idx = to_storage_idx(y_tensor_idx, y_stride, y_offset)

                out_storage[out_storage_idx] = func(x_storage[x_storage_idx], y_storage[y_storage_idx])
            out = Tensor(out_storage, out_shape)

        return out

    return _zip


def tensor_reduce(func) -> Callable:
    """

    """
    def _reduce(x: Tensor, dim: int, keep_dim: bool = True) -> Tensor:
        """

        """
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected Tensor, got {type(x)} instead.")
        else:
            x_storage, x_shape, x_stride, x_offset = x.core()
        out_shape = list(x_shape)
        out_shape[dim] = 1
        out_shape = tuple(out_shape)
        num = math.prod(out_shape)
        out_storage = [None] * num
        for out_storage_idx in range(num):
            out_tensor_idx = to_tensor_idx(out_storage_idx, out_shape)
            for i in range(x_shape[dim]):
                x_tensor_idx = list(out_tensor_idx)
                x_tensor_idx[dim] = i
                x_tensor_idx = tuple(x_tensor_idx)
                x_storage_idx = to_storage_idx(x_tensor_idx, x_stride, x_offset)
                if out_storage[out_storage_idx] is None:
                    out_storage[out_storage_idx] = x_storage[x_storage_idx]
                else:
                    out_storage[out_storage_idx] = func(out_storage[out_storage_idx], x_storage[x_storage_idx])
        if keep_dim:
            return Tensor(out_storage, out_shape)
        else:
            if len(out_shape) == 1:
                return Tensor(out_storage, out_shape)
            else:
                return Tensor(out_storage, out_shape[:dim] + out_shape[dim + 1:])

    return _reduce


class Neg:
    @staticmethod
    def forward(x: Tensor) -> Tensor:
        neg_x = tensor_map(neg)(x)
        if x.history:
            neg_x.history = History(Neg, (), (x,))
        else:
            pass
        return neg_x

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        # We tend to use numpy array for tensors who don't need history
        return (-grad,)


class Exp:
    @staticmethod
    def forward(x: Tensor) -> Tensor:
        exp_x = tensor_map(exp)(x)
        if x.history:
            exp_x.history = History(Exp, exp_x.constant(), (x,)) # Tensors in cache should be constant
        else:
            pass
        return exp_x

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        exp_x = cache
        return (exp_x * grad,)


class Add:
    @staticmethod
    def forward(x: Tensor, y: Tensor) -> Tensor:
        add_x_y = tensor_zip(add)(x, y)
        if x.history or y.history:
            add_x_y.history = History(Add, (), (x, y))
        else:
            pass
        return add_x_y

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        return grad, grad


class Sub:
    @staticmethod
    def forward(x: Tensor, y: Tensor) -> Tensor:
        sub_x_y = tensor_zip(sub)(x, y)
        if x.history or y.history:
            sub_x_y.history = History(Sub, (), (x, y))
        else:
            pass
        return sub_x_y

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        return grad, -grad


class Mul:
    @staticmethod
    def forward(x: Tensor, y: Tensor) -> Tensor:
        mul_x_y = tensor_zip(mul)(x, y)
        if x.history or y.history:
            mul_x_y.history = History(Mul, (x.constant(), y.constant()), (x, y))
        else:
            pass
        return mul_x_y

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        x, y = cache
        return y * grad, x * grad


class Div:
    @staticmethod
    def forward(x: Tensor, y: Tensor) -> Tensor:
        div_x_y = tensor_zip(div)(x, y)
        if x.history or y.history:
            cache = (x.constant(), y.constant())
            div_x_y.history = History(Div, cache, (x, y))
        else:
            pass
        return div_x_y

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        x, y = cache
        return grad / y, -grad * x / y ** Tensor([2])


class Rcp:
    @staticmethod
    def forward(x: Tensor) -> Tensor:
        rcp_x = tensor_map(div)(x)
        if x.history:
            rcp_x.history = History(Rcp, rcp_x.constant(), (x,))
        else:
            pass
        return rcp_x

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        rcp_x = cache
        return (-grad * rcp_x ** Tensor([2]),)


class Log:
    @staticmethod
    def forward(x: Tensor) -> Tensor:
        log_x = tensor_map(log)(x)
        if x.history:
            cache = x.constant()
            log_x.history = History(Log, cache, (x,))
        else:
            pass
        return log_x

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        x = cache
        return (grad / x,)


class Relu:
    @staticmethod
    def forward(x: Tensor) -> Tensor:
        relu_x = tensor_map(relu)(x)
        if x.history:
            relu_x.history = History(Relu, x.constant(), (x,))
        else:
            pass
        return relu_x

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        x = cache
        return (grad * (x > Tensor([0])),)

class Softmax:
    """

    """
    @staticmethod
    def forward(x: Tensor, dim: int) -> Tensor:
        exp_x = tensor_map(exp)(x)
        sum_exp_x = tensor_reduce(add)(exp_x, dim)
        softmax_x = exp_x / sum_exp_x
        if x.history:
            softmax_x.history = History(Softmax, (softmax_x.constant(), dim), (x,))
        else:
            pass
        return softmax_x

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        """

        """
        softmax_x, dim = cache

        return (softmax_x * (grad - tensor_reduce(add)(grad * softmax_x, dim)),)


class Sigmoid:
    @staticmethod
    def forward(x: Tensor) -> Tensor:
        sigmoid_x = tensor_map(sigmoid)(x)
        if x.history:
            sigmoid_x.history = History(Sigmoid, sigmoid_x.constant(), (x,))
        else:
            pass
        return sigmoid_x

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        sigmoid_x = cache
        return (sigmoid_x * (Tensor([1]) - sigmoid_x) * grad,)

class Pow:
    @staticmethod
    def forward(x: Tensor, y: Tensor) -> Tensor:
        pow_x_y = tensor_zip(pow)(x, y)
        if x.history or y.history:
            pow_x_y.history = History(Pow, (x.constant(), y.constant()), (x, y))
        else:
            pass
        return pow_x_y

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        x, y = cache
        return grad * y * x ** (y - Tensor([1])), x.log() * x ** y

class Reshape:
    @staticmethod
    def forward(x: Tensor, *shape) -> Tensor:
        if not x.is_contiguous():
            x.to_contiguous() # New storage
        else: # Same storage
            pass
        x_shape = x.shape
        if math.prod(x_shape) != math.prod(shape):
            raise ValueError(f"Can't reshape to {shape}.")
        else:
            reshape_x = Tensor(x.storage,
                               shape,
                               None,
                               x.offset)
        if x.history:
            reshape_x.history = History(Reshape, x_shape, (x,))
        else:
            pass
        return reshape_x

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        x_shape = cache
        return (grad.reshape(*x_shape),)

class Permute:
    @staticmethod
    def forward(x: Tensor, *dims) -> Tensor:
        _, x_shape, x_stride, _ = x.core()
        if (any(d < 0 for d in dims) or
                len(x.shape) - 1 < max(dims) or
                len(x.shape) != len(dims) or
                len(dims) != len(set(dims))):
            raise ValueError("Invalid dimensions.")
        else: # Same storage
            permute_x = Tensor(x.storage,
                               tuple(x_shape[i] for i in dims),
                               tuple(x_stride[i] for i in dims),
                               x.offset)
        if x.history:
            permute_x.history = History(Permute, dims, (x,))
        else:
            pass
        return permute_x

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        dims = cache
        return (grad.permute(*dims),)

class Sum:
    @staticmethod
    def forward(x: Tensor, dim: int, keep_dim: bool = True) -> Tensor:
        sum_x = tensor_reduce(add)(x, dim, keep_dim)

        if x.history:
            sum_x.history = History(Sum, (x.shape, dim), (x,))
        else:
            pass
        return sum_x

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        out_shape, dim = cache
        if len(grad.shape) != len(out_shape):
            grad.shape = grad.shape[:dim] + (1,) + grad.shape[dim:]
            grad.stride = contiguous_stride(grad.shape)
        else:
            pass

        return (Ones(out_shape) * grad,)


class Prod:
    @staticmethod
    def forward(x: Tensor, dim: int, keep_dim: bool = True) -> Tensor:
        prod_x = tensor_reduce(mul)(x, dim, keep_dim)

        if x.history:
            prod_x.history = History(Prod, (x.constant(), prod_x.constant(), dim), (x,))
        else:
            pass
        return prod_x

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        x, prod_x, dim = cache
        if len(grad.shape) != len(x.shape):
            grad.shape = grad.shape[:dim] + (1,) + grad.shape[dim:]
            grad.stride = contiguous_stride(grad.shape)

            prod_x.shape = prod_x.shape[:dim] + (1,) + prod_x.shape[dim:]
            prod_x.stride = contiguous_stride(prod_x.shape)
        else:
            pass

        return (grad * prod_x / x,)


class MatMul:
    @staticmethod
    def forward(x: Tensor, y: Tensor) -> Tensor:
        """

        Args:

        Returns:

        """

        if not isinstance(y, Tensor):
            raise TypeError(f"Expected Tensor, got {type(y).__name__} instead.")

        x_storage, x_shape, x_stride, x_offset = x.core()
        y_storage, y_shape, y_stride, y_offset = y.core()
        x_remove = False # Check whether we need to remove imaginary dimension
        y_remove = False
        if len(x_shape) == 1: # Vector
            x_remove = True
            x_stride = x_shape + x_stride
            x_shape = (1,) + x_shape # Convert to a matrix with an extra imaginary dimension of size 1
        if len(y_shape) == 1:
            y_remove = True
            y_shape = y_shape + (1,)
            y_stride = (1,) + y_stride

        # The last two dimensions represent matrix, any extra dimensions ahead represent batch
        if x_shape[-1] != y_shape[-2]:
            raise ValueError(f"Shapes {x_shape[-2:]} and {y_shape[-2:]} are not compatible for matrix multiplication.")
        else:
            batch_shape = broadcastable(x_shape[:-2], y_shape[:-2]) # Batch shape can be broadcast
            if batch_shape is False:
                raise ValueError(f"Batch shapes are not broadcastable.")
            else:
                out_shape = batch_shape + (x_shape[-2], y_shape[-1])
                num = math.prod(out_shape)
                out_storage = [0] * num
                for out_storage_idx in range(num):
                    out_tensor_idx = to_tensor_idx(out_storage_idx, out_shape)
                    for i in range(x_shape[-1]):
                        x_tensor_idx = out_tensor_idx[:-1] + (i,)
                        x_tensor_idx = from_broadcast_idx(x_tensor_idx, x_shape)

                        y_tensor_idx = out_tensor_idx[:-2] + (i,) + out_tensor_idx[-1:]
                        y_tensor_idx = from_broadcast_idx(y_tensor_idx, y_shape)

                        x_storage_idx = to_storage_idx(x_tensor_idx, x_stride, x_offset)
                        y_storage_idx = to_storage_idx(y_tensor_idx, y_stride, y_offset)
                        out_storage[out_storage_idx] += x_storage[x_storage_idx] * y_storage[y_storage_idx]
                if x_remove:
                    out_shape = out_shape[:-2] + out_shape[-1:]
                if y_remove:
                    out_shape = out_shape[:-1]
                if len(out_shape) == 0:
                    out_shape = (1,) # Keep the shape (1, ) for scalar instead of ()
                out = Tensor(out_storage, out_shape)

                if x.history or y.history:
                    out.history = History(MatMul, (x.constant(), y.constant()), (x, y))
                else:
                    pass
                return out

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        """

        """

        x, y = cache
        x_shape = x.shape # Remember the original shape to adjust the shape of the output gradient,
        # as they must align with each other
        y_shape = y.shape
        x_remove = False
        y_remove = False

        if len(x.shape) == 1 and len(y.shape) != 1:
            x_remove = True

            x.stride = x.shape + x.stride
            x.shape = (1,) + x.shape

            grad.stride = grad.stride[:-1] + (grad.stride[-1],) + grad.stride[-1:]
            grad.shape = grad.shape[:-1] + (1,) + grad.shape[-1:]
            grad.stride = contiguous_stride(grad.shape)

        if len(y.shape) == 1 and len(x.shape) != 1:
            y_remove = True

            y.shape = y.shape + (1,)
            y.stride = (1,) + y.stride

            grad.shape = grad.shape + (1,)
            grad.stride = grad.stride + (1,)

        if len(y.shape) == len(x.shape) == 1:
            x_remove = True
            y_remove = True

            x.stride = x.shape + x.stride
            x.shape = (1,) + x.shape

            y.shape = y.shape + (1,)
            y.stride = (1,) + y.stride

            grad.shape = (1, 1)
            grad.stride = (1, 1)

        def transpose(t: Tensor):
            t_shape = list(t.shape)
            t_stride = list(t.stride)
            t_shape[-1], t_shape[-2] = t_shape[-2], t_shape[-1]
            t_stride[-1], t_stride[-2] = t_stride[-2], t_stride[-1]
            t.shape = tuple(t_shape)
            t.stride = tuple(t_stride)

        transpose(x)
        transpose(y)

        x_grad = grad @ y
        y_grad = x @ grad

        if x_remove:
            x_grad = Tensor(x_grad.storage, x_grad.shape[:-2] + x_grad.shape[-1:])
        if y_remove:
            y_grad = Tensor(y_grad.storage, y_grad.shape[:-1])

        def align(gradient: Tensor, shape: tuple) -> Tensor:
            """
            As we allow for batched matrix multiplication, the shape of the output gradient might not
            match the shape of the respective tensor parameter, so this function is used to align
            their shape by sum over the broadcast dimensions of the output gradient.

            Args:
                gradient (Tensor): The gradient of the tensor parameter.
                shape (tuple): The shape of the tensor parameter which is also the target shape of our gradient.

            Returns:
                Tensor: The gradient whose shape is aligned.

            """
            grad_shape = gradient.shape
            if grad_shape != shape:
                for dim in range(-1, -len(shape) - 1, -1):
                    if grad_shape[dim] != shape[dim]:
                        gradient = gradient.sum(dim)
                if len(grad_shape) != len(shape):
                    for _ in range(len(grad_shape) - len(shape)):
                        gradient = gradient.sum(0, keep_dim=False)
            return gradient

        return align(x_grad, x_shape), align(y_grad, y_shape)





def _eq(x: Tensor, y: Tensor) -> Tensor:
    if x.shape != y.shape:
        raise ValueError(f"Input tensors must have the same shape, got {x.shape} & {y.shape} instead.")
    else:
        pass
    return tensor_zip(eq)(x, y)

def _lt(x: Tensor, y: Tensor) -> Tensor:
    if x.shape != y.shape:
        raise ValueError(f"Input tensors must have the same shape, got {x.shape} & {y.shape} instead.")
    else:
        pass
    return tensor_zip(lt)(x, y)

def _gt(x: Tensor, y: Tensor) -> Tensor:
    if x.shape != y.shape:
        raise ValueError(f"Input tensors must have the same shape, got {x.shape} & {y.shape} instead.")
    else:
        pass
    return tensor_zip(gt)(x, y)




Tensor.__neg__ = Neg.forward
Tensor.__add__ = Add.forward
Tensor.__radd__ = Add.forward
Tensor.__sub__ = Sub.forward
Tensor.__rsub__ = lambda x, y: Sub.forward(y, x)

Tensor.__mul__ = Mul.forward
Tensor.__rmul__ = Mul.forward
Tensor.__truediv__ = Div.forward
Tensor.__rtruediv__ = lambda x, y: Div.forward(y, x)
Tensor.__eq__ = _eq
Tensor.__gt__ = _gt
Tensor.__lt__ = _lt
Tensor.__pow__ = Pow.forward
Tensor.__rpow__ = lambda x, y: Pow.forward(y, x)
Tensor.__matmul__ = MatMul.forward
Tensor.sum = Sum.forward
Tensor.prod = Prod.forward
Tensor.kron = NotImplemented
Tensor.log = Log.forward
Tensor.softmax = Softmax.forward
Tensor.sigmoid = Sigmoid.forward
Tensor.exp = Exp.forward
Tensor.relu = Relu.forward
Tensor.rcp = Rcp.forward
Tensor.reshape = Reshape.forward
Tensor.permute = Permute.forward










