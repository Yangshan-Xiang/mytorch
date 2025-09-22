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
    def _zip(x: Tensor, y: Tensor) -> Tensor: # We now only consider both inputs are Tensor without extra broadcasting
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


def tensor_contract(func) -> Callable:
    """

    """
    def _contract(x: Tensor, dim: int) -> Tensor: # Without extra broadcasting
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
        return Tensor(out_storage, out_shape)
    return _contract


class Neg:
    @staticmethod
    def forward(x: Tensor) -> Tensor:
        neg_x = tensor_map(neg)(x)
        if x.history is not None:
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
        if x.history is not None:
            exp_x.history = History(Exp, exp_x, (x,))
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
        if x.history is not None:
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
        if x.history is not None:
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
        if x.history is not None:
            mul_x_y.history = History(Sub, (x, y), (x, y))
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
        if x.history is not None:
            cache = (x, y)
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
        if x.history is not None:
            rcp_x.history = History(Rcp, rcp_x, (x,))
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
        if x.history is not None:
            cache = x
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
        if x.history is not None:
            relu_x.history = History(Relu, x, (x,))
        else:
            pass
        return relu_x

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        x = cache
        return (grad * (x > Tensor([0])),)

class Softmax:
    @staticmethod
    def forward(x: Tensor, dim: int) -> Tensor:
        exp_x = tensor_map(exp)(x)
        sum_exp_x = tensor_contract(add)(exp_x, dim)
        softmax_x = exp_x / sum_exp_x
        if x.history is not None:
            softmax_x.history = History(Softmax, softmax_x, (x,))
        else:
            pass
        return softmax_x

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        softmax_x = cache
        return (softmax_x * (Tensor([1]) - softmax_x) * grad,)


class Pow:
    @staticmethod
    def forward(x: Tensor, y: Tensor) -> Tensor:
        pow_x_y = tensor_zip(pow)(x, y)
        if x.history is not None:
            pow_x_y.history = History(Div, (x, y), (x, y))
        else:
            pass
        return pow_x_y

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        x, y = cache
        return grad * y * x ** (y - Tensor([1])), x.log() * x ** y


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
Tensor.log = Log.forward
Tensor.softmax = Softmax.forward
Tensor.exp = Exp.forward
Tensor.relu = Relu.forward
Tensor.rcp = Rcp.forward

