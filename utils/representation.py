import numpy as np

def gaussian(x, sigma):
    return np.exp(- np.linalg.norm(x) ** 2 / 2 * sigma ** 2)