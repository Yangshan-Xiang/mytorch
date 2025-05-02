import numpy as np
from utils.representation import *

class HT:
    """
    Initialize the Hierarchical Tucker (HT) model presented in section 3.2 of the paper.

    Args:

    Returns:


    """
    def __init__(self, n, m, y, ranks):
        self.N = n
        self.M = m
        self.Y = y
        self.ranks = ranks
        self.L = int(np.log2(n))
        self.params = {'level_0': np.random.randn(n, ranks[0], m)}
        for l in range(1, self.L):
            self.params[f'level_l'] = np.random.randn(n // (2**l), ranks[l], ranks[l-1])
        self.params['final'] = np.random.randn(y, ranks[-1])

    def forward(self, x):
        """
        Forward pass of the HT model.

        Args:

        Returns:

        """
        fx = np.zeros((self.N, self.M))
        theta =  np.random.uniform(0, 1, self.M)
        for i in range(self.N):
            for j in range(self.M):
                fx[i, j] = gaussian(x[i], theta[j])









