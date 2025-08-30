from mytorch.datasets import *
from mytorch.optimizers import *
from mytorch.parameter import *
import random


class Module:
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError("The forward method is not implemented.")

    def get_params(self): # Return all parameters of the model as a list
        params = []
        for attr in self.__dict__.items():
            if attr[0] == 'params':
                params.extend(attr[1])
            elif isinstance(attr[1], Module):
                params.extend(attr[1].get_params())
            else:
                pass
        return params


class LinearLayer(Module):
    def __init__(self, d_in, d_out, bias: bool = True):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.bias = bias

        self.weights = []
        self.biases = []

        self.params = []

        for i in range(d_out):
            row = []
            for j in range(d_in):
                p = Parameter(random.random())
                row.append(p)
                self.params.append(p)
            self.weights.append(row)  # Structure the parameters to make the computation in forward pass easier

        if bias:
            for i in range(d_out):
                p = Parameter(random.random())
                self.params.append(p)
                self.biases.append(p)
        else:
            self.biases = None

    def forward(self, x: list) -> list:
        if len(x) != self.d_in:
            raise ValueError("The length of the input data must equal the input dimension of the linear layer.")

        y = []
        for i in range(self.d_out):
            if self.bias:
                out = self.weights[i][0] * x[0] + self.biases[i]
            else:
                out = self.weights[i][0] * x[0]

            for j in range(1, self.d_in):
                out += self.weights[i][j] * x[j]
            y.append(out)
        return y


