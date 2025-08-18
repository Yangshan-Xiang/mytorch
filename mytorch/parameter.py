from mytorch.myfunctions import *


class Parameter:
    """
    A custom class of numbers in order to keep track of their computation histories.

    Attributes:
        value : The value of the parameter.


        derivative : The derivative of the generator with respect to the parameter.

    """

    def __init__(self, value: float, history = None):

        self.value = value
        self.history = history
        self.derivative = None

    def __add__(self, y):
        return Parameter(* Add.forward(self, y))

    def __radd__(self, y):
        return self + y

    def __sub__(self, y):
        return Parameter(* Sub.forward(self, y))

    def __rsub__(self, y):
        return -(self - y)

    def __mul__(self, y):
        return Parameter(* Mul.forward(self, y))

    def __rmul__(self, y):
        return self * y

    def __truediv__(self, y):
        return Parameter(* Div.forward(self, y))

    def __rtruediv__(self, y):
        return (self / y).rcp()

    def __neg__(self):
        return Parameter(* Neg.forward(self))

    def __eq__(self, y):
        y_val = getattr(y, "value", y)
        return self.value == y_val

    def __gt__(self, y):
        y_val = getattr(y, "value", y)
        return self.value > y

    def __lt__(self, y):
        y_val = getattr(y, "value", y)
        return self.value < y

    def __repr__(self):
        return f"Parameter({self.value})"

    def log(self):
        return Parameter(* Log.forward(self))

    def sigmoid(self):
        return Parameter(* Sigmoid.forward(self))

    def relu(self):
        return Parameter(* ReLU.forward(self))

    def exp(self):
        return Parameter(* Exp.forward(self))

    def rcp(self):
        return Parameter(* Rcp.forward(self))


























