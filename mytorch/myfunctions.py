import math


class History:
    """
    In order to store all the necessary history information of a parameter which will later
    be used in the backward pass.

    Attributes:
        generator : The function which generates the value of the parameter.
        cache : Save computed values during the forward pass to avoid unnecessary computation in the backward pass.
        parents : The input values for the generator.

    """
    def __init__(self, generator = None, cache = None, parents = None):
        self.generator = generator
        self.cache = cache
        self.parents = parents

    def __repr__(self):
        return f"History(generator={self.generator}, parents={self.parents})"


class Mul:
    """
    Custom function for multiplication which is also able to go backwards. We here
    use the class object instead of function object to bring forward and backward
    pass together to make the codes more concise.
    """
    @staticmethod
    def forward(x, y) -> tuple:
        x_val = x.value # Since this class is used to override method in class Parameter,
        # the first input is always a Parameter object.
        y_val = getattr(y, "value", y)

        cache = x_val, y_val # The value of x and y are needed to compute the derivatives in backward pass
        history = History(Mul, cache, (x, y)) # Collect history information made up of generator, memo and parents

        return x_val * y_val, history

    @staticmethod
    def backward(cache, d: float) -> tuple:
        x_val, y_val = cache
        return y_val * d, x_val * d # The order can not be reversed


class Div:
    """
    Custom function for division.
    """
    @staticmethod
    def forward(x, y) -> tuple:
        x_val = x.value
        y_val = getattr(y, "value", y)

        cache = x_val, y_val
        history = History(Div, cache, (x, y))

        return x_val / y_val, history

    @staticmethod
    def backward(cache, d: float) -> tuple:
        x_val, y_val = cache
        return d / y_val, -d * x_val / y_val ** 2


class Add:
    """
    Custom function for addition.
    """
    @staticmethod
    def forward(x, y) -> tuple:
        x_val = x.value
        y_val = getattr(y, "value", y)

        cache = () # Nothing needs to be saved for the backward pass of addition
        history = History(Add, cache, (x, y))

        return x_val + y_val, history

    @staticmethod
    def backward(cache, d: float) -> tuple:
        return d, d


class Sub:
    """
    Custom function for subtraction.
    """
    @staticmethod
    def forward(x, y) -> tuple:
        x_val = x.value
        y_val = getattr(y, "value", y)

        cache = () # Nothing needs to be saved for the backward pass of addition
        history = History(Sub, cache, (x, y))

        return x_val - y_val, history

    @staticmethod
    def backward(cache, d: float) -> tuple:
        return d, -d


class Rcp:
    """
    Custom function for reciprocal.
    """
    @staticmethod
    def forward(x) -> tuple:
        x_val = x.value

        cache = x_val
        history = History(Rcp, cache, x)

        return 1 / x_val, history

    @staticmethod
    def backward(cache, d: float) -> float:
        x_val = cache
        return -d / x_val ** 2


class Log:
    """
    Custom function for logarithm.
    """
    @staticmethod
    def forward(x) -> tuple:
        x_val = x.value

        cache = x_val
        history = History(Log, cache, x)

        return math.log(x_val), history

    @staticmethod
    def backward(cache, d: float) -> float:
        x_val = cache
        return d / x_val


class Exp:
    """
    Custom exponential function.
    """
    @staticmethod
    def forward(x) -> tuple:
        exp_x = math.exp(x.value)

        cache = exp_x
        # We store exp(x) instead of x because only exp(x) is required during the backward pass.
        # This avoids redundant computation.
        history = History(Exp, cache, x)

        return exp_x, history

    @staticmethod
    def backward(cache, d: float) -> float:
        exp_x = cache
        return exp_x * d


class Neg:
    """
    Custom function for negation.
    """
    @staticmethod
    def forward(x) -> tuple:
        x_val = x.value

        cache = ()
        history = History(Neg, cache, x)

        return -x_val, history

    @staticmethod
    def backward(cache, d: float) -> float:
        return -1 * d


class Sigmoid:
    """
    Custom Sigmoid function.
    """
    @staticmethod
    def forward(x) -> tuple:
        sigmoid_x = 1 / (1 + math.exp(-x.value))

        cache = sigmoid_x
        history = History(Sigmoid, cache, x)

        return sigmoid_x, history

    @staticmethod
    def backward(cache, d: float) -> float:
        sigmoid_x = cache
        return sigmoid_x * (1 - sigmoid_x) * d


class ReLU:
    """
    Custom ReLU function.
    """
    @staticmethod
    def forward(x) -> tuple:
        x_val = x.value

        cache = x_val
        history = History(ReLU, cache, x)

        return max(0.0, x_val), history

    @staticmethod
    def backward(cache, d: float) -> float:
        x_val = cache
        return d * (x_val > 0)


















































