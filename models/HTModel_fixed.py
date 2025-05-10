import numpy as np

# The key to implement this model is how to index and structure all the parameters such that they can
# be computed recursively
class HTModel_fixed():
    # HT model with FIXED and randomly initialized parameters
    def __init__(self, N:int, M:int, ranks:list, Y:int):
        # The notations are consistent with the notations used in the original paper
        if (not isinstance(N, int) or not isinstance(M, int) or not isinstance(Y, int)
                or not all(isinstance(r, int) for r in ranks)):
            raise TypeError("N, M, ranks and Y must be integers.")
        if N < 1 or M < 1 or Y < 1 or all(r < 1 for r in ranks):
            raise ValueError("N, M, ranks and Y must be positive.")
        if (N & (N - 1)) != 0:
            raise ValueError("N must be the power of 2.")
        if len(ranks) != np.log2(N):
            raise ValueError(f"The number of ranks must equal to {int(np.log2(N))}.")

        self.N = N # The number of input data, the number of vectorized patches of one input image, need to be
        # the power of 2 for simplicity as described in the paper
        self.M = M # The number of representation functions, the number of input channels
        self.ranks = ranks # A list of ranks of the HT decomposition, the number of channels in each feature map
        self.Y = Y # The number of output classes, the number of channels in the output layer
        self.L = int(np.log2(N)) # The number of hidden-layers, exclude the representation layer and output layer

        # Define the parameters
        self.params = [np.random.randn(N, ranks[0], M)] # The parameters of the 1st hidden-layer
        for l in range(1, self.L):
            self.params.append(np.random.randn(N // 2 ** l, ranks[l], ranks[l-1])) # The parameters of the
            # rest hidden-layers
        self.params.append(np.random.randn(Y, ranks[-1])) # The parameters of the output layer.

    def forward(self, X): # X is the input instance which has the size of (N, s)
        # Representation layer, will convert the shape of X into (N, M)
        weights = np.random.randn(X.shape[-1], self.M)
        F = X @ weights

        # The task here is to compute the phi tensors hierarchically according to the equation (4) in the paper
        phis = [self.params[0]] # phi_0 tensor is just the stack of parameter vectors in the 1st hidden-layer
        for l in range (1, self.L):
            phi = np.zeros((self.N // 2 ** l, self.ranks[l], self.M ** 2 ** l)) # Compute the intermediate level
            # phi tensors hierarchically
            # Compute the phi_l tensor by computing each phi_l_j_gamma recursively
            for j in range(self.N // 2 ** l):
                for gamma in range(self.ranks[l]):
                    for alpha in range(self.ranks[l-1]):
                        # This line of code EXACTLY matches the expression of phi_l_j_gamma depicted in
                        # the equation (4)
                        phi[j, gamma] += (np.kron(phis[l-1][2 * j, alpha], phis[l-1][2 * j + 1, alpha]) *
                                          self.params[l][j, gamma, alpha])
            phis.append(phi)
        # Compute the final coefficient tensor A_y using the phi tensors
        A = np.zeros((self.Y, * [self.M] * self.N)) # Evert A_y tensor has the shape of M ^ N
        for y in range(self.Y):
            for alpha in range(self.ranks[-1]):
                # Again this EXACTLY matches the expression of A_y depicted in the equation (4)
                A[y] += (np.kron(phis[-1][0, alpha], phis[-1][1, alpha]).reshape(* [self.M] * self.N)
                         * self.params[-1][y, alpha]) # Reshaped to M ^ N

        # Compute the score function h_y(X) depicted in function (2) in the paper
        h = np.zeros(self.Y)

        # Prepare the factor which can be directly multiplied with tensor A_y
        factor = np.ones(A[0].shape)
        for i in range(self.N):
            factor *= F[i, np.indices(A[0].shape)[i]]

        for y in range(self.Y):
            h[y] = np.sum(A[y] * factor) # Multiply A_y with the factor and sum up all the elements

        return h