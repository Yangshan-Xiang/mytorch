import math


eps = 1e-12 # For numerical stability

def add(x: float, y: float):
    return x + y

def sub(x: float, y: float):
    return x - y

def mul(x: float, y: float):
    return x * y

def div(x: float, y: float):
    return x / (y + eps)

def neg(x: float) -> float:
    return -x

def exp(x: float) -> float:
    return math.exp(x)

def log(x: float) -> float:
    return math.log(x + eps)

def sqrt(x: float) -> float:
    return math.sqrt(x + eps)

def relu(x: float) -> float:
    return max(0.0, x)

def sigmoid(x: float) -> float:
    if x >= 0: # For numerical stability
        return 1 / (1 + math.exp(-x))
    else:
        return math.exp(x) / (1 + math.exp(x))

def rcp(x: float) -> float:
    return 1 / (x + eps)

def eq(x: float, y: float) -> bool:
    return x == y

def gt(x: float, y: float) -> bool:
    return x > y

def ge(x: float, y: float) -> bool:
    return x >= y

def lt(x: float, y: float) -> bool:
    return x < y

def le(x: float, y: float) -> bool:
    return x <= y

def maximum(x: float, y: float) -> float:
    return max(x, y)





