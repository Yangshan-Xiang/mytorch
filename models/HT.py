import numpy as np

class HTModel:
    def __init__(self, N, M, Y, ranks):
        """
        Initialize the HT model

        Args:
            N: Order of the tensor (number of input vectors)
            M: Dimension of each mode (number of representation functions)
            Y: Number of output classes
            ranks: List of ranks [r0, r1, ..., rL-1] where L = log2(N)
        """
        self.N = N
        self.M = M
        self.Y = Y
        self.L = int(np.log2(N))
        self.ranks = ranks

        # Initialize parameters
        self.params = {}

        # Level 0 parameters (a^0,j,γ ∈ ℝ^M)
        self.params['level_0'] = np.random.randn(N, ranks[0], M)

        # Intermediate level parameters (a^l,j,γ ∈ ℝ^r_{l-1})
        for l in range(1, self.L):
            self.params[f'level_{l}'] = np.random.randn(N // (2 ** l), ranks[l], ranks[l - 1])

        # Final level parameters (a^L,y ∈ ℝ^r_{L-1})
        self.params['final'] = np.random.randn(Y, ranks[-1])

    def forward(self, X):
        """
        Forward pass of the HT model

        Args:
            X: Input tensor of shape (N, s) where s is the input dimension

        Returns:
            Output scores of shape (Y,)
        """

        F = X  # Assuming X is already the (N, M) representation

        # Level 0: ϕ^0,j,γ = a^0,j,γ (already initialized)
        phi = {}
        phi[0] = self.params['level_0']  # Shape (N, r0, M)

        # Recursive computation through levels
        for l in range(1, self.L):
            num_nodes = self.N // (2 ** l)
            phi[l] = np.zeros((num_nodes, self.ranks[l], self.M ** (2 ** l)))

            for j in range(num_nodes):
                for gamma in range(self.ranks[l]):
                    left = phi[l - 1][2 * j]
                    right = phi[l - 1][2 * j + 1]

                    # Tensor product and sum
                    sum_term = 0
                    for alpha in range(self.ranks[l - 1]):
                        tensor_prod = np.kron(left[alpha], right[alpha])
                        sum_term += self.params[f'level_{l}'][j, gamma, alpha] * tensor_prod

                    phi[l][j, gamma] = sum_term

        # Final computation of A^y
        output_scores = np.zeros(self.Y)

        for y in range(self.Y):

            left = phi[self.L - 1][0]
            right = phi[self.L - 1][1]

            sum_term = 0
            for alpha in range(self.ranks[-1]):
                tensor_prod = np.kron(left[alpha], right[alpha])
                sum_term += self.params['final'][y, alpha] * tensor_prod

            A_y = sum_term.reshape(tuple([self.M] * self.N))

            indices = np.indices(tuple([self.M] * self.N))

            product_terms = np.ones_like(A_y)
            for i in range(self.N):

                product_terms *= F[i, indices[i]]

            output_scores[y] = np.sum(A_y * product_terms)

        return output_scores

    def get_num_params(self):
        """Compute the total number of parameters in the model"""
        total = 0

        # Level 0 parameters
        total += self.N * self.ranks[0] * self.M

        # Intermediate level parameters
        for l in range(1, self.L):
            total += (self.N // (2**l)) * self.ranks[l] * self.ranks[l-1]

        # Final level parameters
        total += self.Y * self.ranks[-1]

        return total




