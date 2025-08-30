
class Optimizer:
    """
    The fundamental construction for an optimizer, will be inherited by actual optimizers.
    """
    def __init__(self, params: list, lr: float, maximize: bool, weight_decay: float):
        self.params = params
        self.lr = lr
        self.maximize = maximize
        self.weight_decay = weight_decay

    def zero_grad(self, none_grad: bool=True) -> None:
        for param in self.params:
            if none_grad:
                param.derivative = None
            else:
                param.derivative = 0


class SGD(Optimizer):
    """
    The Stochastic Gradient Descent optimizer.

    """
    def __init__(self, params: list, lr: float=0.001, maximize: bool=False, weight_decay: float=0,
                 momentum: float=0, dampening: float=0, nesterov: bool=False):
        super().__init__(params, lr, maximize, weight_decay)
        self.momentum = momentum
        self.dampening = dampening
        self.nesterov = nesterov
        self.buffer = [None] * len(params) # Store necessary values when SGD needs momentum

    def step(self) -> None:
        for i, param in enumerate(self.params):
            self.buffer = param.derivative
            if self.maximize:
                grad = -1 * param.derivative
            else:
                grad = param.derivative
            if self.weight_decay != 0:
                grad += self.weight_decay * param.value
            if self.momentum != 0:
                if self.buffer[i] is None:
                    self.buffer[i] = grad
                else:
                    self.buffer[i] = self.momentum * self.buffer[i] + (1 - self.dampening) * grad
                if self.nesterov:
                    grad += self.momentum * self.buffer[i]
                else:
                    grad = self.buffer[i]

            param.value -= self.lr * grad

class Adam(Optimizer):
    """
    The Adam optimizer.
    """



















