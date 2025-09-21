from numpy import exp as _exp
from numpy import log as _log
from numpy import maximum as _max


eps = 1e-12 # For numerical stability

def add(x: float, y: float):
    return x + y

def sub(x: float, y: float):
    return x - y

def mul(x: float, y: float):
    return x * y

def div(x: float, y: float):
    return x / y

def neg(x: float) -> float:
    return -x

def exp(x: float) -> float:
    return _exp(x)

def log(x: float) -> float:
    return _log(x + eps)

def relu(x: float) -> float:
    return _max(0.0, x)

def rcp(x: float) -> float:
    return 1 / x

