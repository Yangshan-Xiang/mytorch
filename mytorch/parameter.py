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
        return f"Parameter({self.value:.3f})"

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
                param.derivative += catalog[id(param)] # Enable it to accumulate derivatives, useful in certain cases.



