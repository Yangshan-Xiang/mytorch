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


        self.params = {'level_0': np.random.randn(n, ranks[0], m)} # First level parameters

        for l in range(1, self.L):
            self.params[f'level_{l}'] = np.random.randn(n // (2**l), ranks[l], ranks[l-1]) # Intermediate level parameters.

        self.params['final'] = np.random.randn(y, ranks[-1]) # Final parameters, n // (2**l) equals to 1 when l = np.log2(n).

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

        phi = {0: self.params['level_0']}
        for l in range(1, self.L):
            num_nodes = self.N // 2 ** l
            phi[l] = np.zeros((num_nodes, self.ranks[l], self.M ** 2 ** l))
            for j in range(num_nodes):
                for gamma in range(self.ranks[l]):
                    sum_term = 0
                    for alpha in range(self.ranks[l-1]):
                        sum_term += self.params[f'level_{l}'][j, gamma, alpha] * np.kron(phi[l-1][2j, alpha], phi[l-1][2j+1, alpha])
                        # The result of the kronecker product is flattened, so all phi tensors are stored in 1-D vectors.

                    phi[l][j, gamma] = sum_term


















