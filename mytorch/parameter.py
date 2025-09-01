from mytorch.functions import *


class Parameter:
    """
    A custom class of numbers in order to keep track of their computation histories.

    Attributes:
        value : The value of the parameter.
        history : The history of the parameter represented by a class object with three attributes.
        derivative : The derivative with respect to leaf parameter which is indispensable for updating itself.
    """

    def __init__(self, value: float, history=None):

        self.value = value
        self.history = history
        self.derivative = None

    def __add__(self, y) -> 'Parameter':
        return Parameter(*Add.forward(self, y))

    def __radd__(self, y) -> 'Parameter':
        return self + y

    def __sub__(self, y) -> 'Parameter':
        return Parameter(*Sub.forward(self, y))

    def __rsub__(self, y) -> 'Parameter':
        return -(self - y)

    def __mul__(self, y) -> 'Parameter':
        return Parameter(*Mul.forward(self, y))

    def __rmul__(self, y) -> 'Parameter':
        return self * y

    def __truediv__(self, y) -> 'Parameter':
        return Parameter(*Div.forward(self, y))

    def __rtruediv__(self, y) -> 'Parameter':
        return (self / y).rcp()

    def __neg__(self) -> 'Parameter':
        return Parameter(*Neg.forward(self))

    def __eq__(self, y) -> bool:
        y_val = getattr(y, "value", y)
        return self.value == y_val

    def __gt__(self, y) -> bool:
        y_val = getattr(y, "value", y)
        return self.value > y_val

    def __lt__(self, y) -> bool:
        y_val = getattr(y, "value", y)
        return self.value < y_val

    def __repr__(self) -> str:
        if isinstance(self.value, float):
            return f"Parameter({self.value:.3f})"
        else:
            return f"Parameter({self.value})"

    def log(self) -> 'Parameter':
        return Parameter(*Log.forward(self))

    def sigmoid(self) -> 'Parameter':
        return Parameter(*Sigmoid.forward(self))

    def relu(self) -> 'Parameter':
        return Parameter(*ReLU.forward(self))

    def exp(self) -> 'Parameter':
        return Parameter(*Exp.forward(self))

    def rcp(self) -> 'Parameter':
        return Parameter(*Rcp.forward(self))

    def is_leaf(self) -> bool:
        return self.history is None

    def topological_ordering(self) -> list:
        """
        Find the topological ordering of the parameters. Here we apply depth-first search
        with reverse post-ordering to linearly order the parameters. The ordering might
        not be unique, it ensures that a parent parameter will only be visited
        after all of its children has been visited to avoid redundant computations.

        """

        visited = [] # Avoid revisit
        ordered = []

        def dfs(x):
            if isinstance(x, Parameter): # We don't need constants
                if id(x) not in visited: # Use id() to differentiate between two Parameters with the same value
                    visited.append(id(x))

                    if x.history is not None:
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

    def one_step_back(self, d: float) -> list:
        """
        Move one step back from children to their parents. Apply chain rule to compute
        and assign the derivatives of the child with respect to its parents.
        """

        if self.history is None:
            raise AssertionError("Leaf parameter has no history, no way to step back.")

        h = self.history

        return list(zip(h.parents, h.generator.backward(h.cache, d))) # Pair the derivatives with its parents

    def backward(self) -> None:
        """
        In the backward pass, we apply the chain rule for multivariate functions to compute
        and save the derivatives of the leaf parameters. The derivatives will then be used to
        update their values.

        """
        ordered = self.topological_ordering()
        catalog = {id(x): 0 for x in ordered}
        catalog[id(self)] = 1 # Initialize the starting derivative as 1, which is just the derivative of the
        # first parameter on the ordered list with respect to itself

        for param in ordered:
            if param.history is not None:
                for parent, d in param.one_step_back(catalog[id(param)]):
                    if isinstance(parent, Parameter):
                        catalog[id(parent)] += d # Accumulate the derivatives according to the chain rule
                        # for multivariate functions
                    else:
                        pass
            else:
                if param.derivative is None:
                    param.derivative = 0
                else:
                    pass
                param.derivative += catalog[id(param)] # Enable it to accumulate derivatives,
                # useful in certain cases like multitask learning.


class Collection:
    """
    In order to store and process a collection of Parameter instances or Model instances.
    """
    def __init__(self, *args):
        self.args = args

    def __getitem__(self, index):
        return self.args[index]

    def list(self):
        """
        As the input arguments can contain nested list of instances, we can flatten them into one clean list.
        """
        flattened = []
        def extract(obj):
            if isinstance(obj, list):
                for element in obj:
                    extract(element)
            else:
                flattened.append(obj)

        for arg in self.args:
            extract(arg)

        return flattened


class Parameters(Collection):
    """
    To store Parameter instances, so that they can be recognized and processed during training.
    """
    def __init__(self, *args):
        super().__init__(*args)

        for arg in self.list():
            if not isinstance(arg, Parameter):
                raise TypeError("Arguments must be Parameter instances or (nested) lists of Parameter instances.")

    def __repr__(self):
        if len(self.args) == 1:
            return f"Parameters({self.args[0]})"
        else:
            return f"Parameters{self.args}"

    def append(self, x) -> None:
        self.args = Parameters(*self.args, x).args





