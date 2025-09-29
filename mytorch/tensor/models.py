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
            elif isinstance(attr[1], Models):
                for model in attr[1]:
                    params.extend(model.get_params())
            else:
                pass
        return params

class Models:
    """

    """
    def __init__(self, *models):
        for model in models:
            if not isinstance(model, (Model, Models)):
                raise AssertionError(f"Expected Model or Models, got {type(model).__name__} instead.")
        self.models = list(models)

    def __call__(self, x):
        """

        """
        for model in self.models:
            x = model(x)
        return x

    def __getitem__(self, index):
        return self.models[index]

    def append(self, x) -> None:
        self.models = Models(*self.models, x).models

    def __repr__(self):
        return f"Models({self.models})"


class Linear(Model):
    """
    The linear model.
    """
    def __init__(self, inp_size, out_size, needs_bias: bool = True):
        """
        Specify the input and output size of the linear model. The bias term is optional.
        """
        super().__init__()
        self.inp_size = inp_size
        self.out_size = out_size
        self.needs_bias = needs_bias

        self.weight = Parameter(Tensor([random.random()] * inp_size * out_size, (inp_size, out_size)))
        if needs_bias:
            self.bias = Parameter(Tensor([random.random()] * out_size))
        else:
            self.bias = None

    def __repr__(self):
        return f"Linear({self.inp_size}, {self.out_size}, needs_bias={self.needs_bias})"

    def forward(self, x: Tensor) -> Tensor:
        """
        The forward pass of the linear model.
        """

        if x.shape[-1] != self.inp_size:
            raise ValueError("The size of the last dimension of the input must equal to "
                             "the input size of the linear model.")

        return x @ self.weight + self.bias

class MLP(Model):
    """
    The multi-layer perceptron which is a stack of linear layers with nonlinear activation functions in between.
    """
    def __init__(self, inp_size, out_size):
        super().__init__()
        self.inp_size = inp_size
        self.out_size = out_size

        self.layers = Models()
        self.layers.append(Linear(inp_size, 8)) # The input layer
        for i in range(4):
            self.layers.append(Linear(8, 8)) # The hidden layers
        self.layers.append(Linear(8, out_size)) # The output layer

    def forward(self, x: list) -> list:
        for layer in self.layers:
            x = layer(x)
            x = x.relu()

        return x

