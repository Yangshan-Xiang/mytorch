import numpy as np

class HTModel():
    def __init__(self, N, M, ranks, Y):
        # The notations are consistent with the notations used in the original paper
        if not isinstance(N, int) or not isinstance(M, int) or not isinstance(Y, int) or not (isinstance(r, int) for r in ranks):
            raise TypeError("N, M, ranks and Y must be integers.")
        if N < 1 or M < 1 or Y < 1 or (r < 1 for r in ranks):
            raise ValueError("N, M, ranks and Y must be positive.")
        if (N & (N - 1)) != 0:
            raise ValueError("N must be the power of 2.")
        if len(ranks) != np.log2(N):
            raise ValueError(f"The number of ranks must equal to {np.log2(N)}.")

        self.N = N # The number of input data, the number of vectorized patches of one input image, need to be
        # the power of 2 for simplicity as described in the paper
        self.M = M # The number of representation functions, the number of input channels
        self.ranks = ranks # The ranks of the HT decomposition, the number of channels in each feature map
        self.Y = Y # The number of output classes, the number of channels in the output layer
        self.L = np.log2(N) # The number of hidden-layers, exclude the representation layer and output layer

        # Define the parameters, they are randomly initialized and NOT trainable now!
        np.random.seed(42)
        self.params = [np.random.randn(N, ranks[0], M)] # The parameters of the 1st hidden-layer
        for l in range(1, self.L):
            self.params.append(np.random.randn(N // 2 ** l, ranks[l], ranks[l-1])) # The parameters of the rest hidden-layers
        self.params.append(np.random.randn(1, Y, ranks[-1])) # The parameters of the output layer.

    def forward(self, X):


