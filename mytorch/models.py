from mytorch.optimizers import *
from mytorch.parameter import *
import random


class Model:
    """
    Includes some basic methods for building models.
    """
    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return f"Model()"

    def forward(self, x):
        raise NotImplementedError("The forward method is not implemented.")

    def get_params(self) -> list:
        """
        As models can have submodels, this method can help us get all the Parameter instances in one clean list
        which will then be feed into our optimizer.
        """
        params = []
        for attr in self.__dict__.items():
            if isinstance(attr[1], Parameters):
                params.extend(attr[1].list())

            elif isinstance(attr[1], Model):
                params.extend(attr[1].get_params())

            elif isinstance(attr[1], Models):
                for model in attr[1].list():
                    params.extend(model.get_params())
            else:
                pass
        return params


class Models(Collection):
    """
    To store a collection of models, so that they can act like one single model.
    """
    def __init__(self, *args):
        super().__init__(*args)

        for arg in self.list():
            if not isinstance(arg, Model):
                raise TypeError("Arguments must be Model instances or (nested) lists of Model instances.")

    def __call__(self, x):
        for layer in self.list():
            x = layer(x)
        return x

    def __repr__(self):
        if len(self.args) == 1:
            return f"Models({self.args[0]})"
        else:
            return f"Models{self.args}"

    def append(self, x) -> None:
        self.args = Models(*self.args, x).args




class Linear(Model):
    """
    The linear model.
    """
    def __init__(self, d_in, d_out, needs_bias: bool = True):
        """
        Specify the input and output dimension of the linear model. The bias term is optional.
        """
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.needs_bias = needs_bias

        self.weights = []
        self.biases = []
        self.params = Parameters(self.weights, self.biases)

        for i in range(d_out):
            row = []
            for j in range(d_in):
                row.append(Parameter(random.random()))
            self.weights.append(row)  # Structure the parameters to make the computation in forward pass easier

        if needs_bias:
            for i in range(d_out):
                self.biases.append(Parameter(random.random()))
        else:
            self.biases = None

    def __repr__(self):
        return f"Linear({self.d_in}, {self.d_out})"


    def forward(self, x: list) -> list:
        """
        The forward pass of the linear model.
        """
        if len(x) != self.d_in:
            raise ValueError("The dimension of the input data must equal the input dimension of the linear model.")

        y = []
        for i in range(self.d_out):
            if self.needs_bias:
                out = self.weights[i][0] * x[0] + self.biases[i]
            else:
                out = self.weights[i][0] * x[0]

            for j in range(1, self.d_in):
                out += self.weights[i][j] * x[j]
            y.append(out)
        return y


class MLP(Model):
    """
    The multi-layer perceptron which is a stack of linear layers with nonlinear activation functions in between.
    """
    def __init__(self, d_hidden: int=1, n_hidden: int=1, needs_bias: bool = True):
        super().__init__()
        self.d_in = 2
        self.d_hidden = d_hidden
        self.d_out = 1

        self.in_layer = Linear(2, d_hidden, needs_bias)
        self.hidden_layers = Models()
        for i in range(n_hidden):
            self.hidden_layers.append(Linear(d_hidden, d_hidden, needs_bias))
        self.out_layer = Linear(d_hidden, 1, needs_bias)

    def forward(self, x: list) -> list:
        out = [o.relu() for o in self.in_layer(x)]
        out = [o.relu() for o in self.hidden_layers(out)]

        return self.out_layer(out)


