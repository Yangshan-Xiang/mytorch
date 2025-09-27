from mytorch.tensor.tensor import *
import random


class Parameter(Tensor):
    """
    To register a tensor, so it can be recognized as a model parameter.
    """
    def __init__(self, tensor: Tensor):
        super().__init__(tensor.storage, tensor.shape, tensor.stride, tensor.offset)
        self.history = History() # Need history by default

class Model:
    """
    Includes some basic methods for building models.
    """
    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return f"Model()"

    def forward(self, x):
        raise NotImplementedError("The forward method should be implemented within its subclass.")

    def get_params(self) -> list:
        """
        As models can have not only their own parameters but also submodels, this method can help us get
        all the parameters in one clean list which will be fed into our optimizers.
        """

        params = []
        for attr in self.__dict__.items():
            if isinstance(attr[1], Parameter):
                params.append(attr[1])
            elif isinstance(attr[1], Model):
                params.extend(attr[1].get_params())
            else:
                pass
        return params

class Linear(Model):
    """
    The linear model.
    """
    def __init__(self, in_size, out_size, needs_bias: bool = True):
        """
        Specify the input and output size of the linear model. The bias term is optional.
        """
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.needs_bias = needs_bias


        self.weight = Parameter(Tensor([random.random()] * in_size * out_size, (in_size, out_size)))

        if needs_bias:
            self.bias = Parameter(Tensor([random.random()] * out_size))
        else:
            self.bias = None

    def __repr__(self):
        return f"Linear({self.in_size}, {self.out_size})"

    def forward(self, x: Tensor) -> Tensor:
        """
        The forward pass of the linear model.
        """
        if x.shape[-1] != self.in_size:
            raise ValueError("The size of the last dimension of the input must equal "
                             "the input size of the linear model.")

        return x @ self.weight + self.bias








class MLP(Model):
    """
    The multi-layer perceptron which is a stack of linear layers with nonlinear activation functions in between.
    """
    def __init__(self, d_hidden: int=1, n_hidden: int=1, needs_bias: bool = True):
        super().__init__()
        self.in_size = 2
        self.d_hidden = d_hidden
        self.out_size = 1

        self.in_layer = Linear(2, d_hidden, needs_bias)
        self.hidden_layers = Models()
        for i in range(n_hidden):
            self.hidden_layers.append(Linear(d_hidden, d_hidden, needs_bias))
        self.out_layer = Linear(d_hidden, 1, needs_bias)

    def forward(self, x: list) -> list:
        out = [o.relu() for o in self.in_layer(x)]
        out = [o.relu() for o in self.hidden_layers(out)]

        return self.out_layer(out)

