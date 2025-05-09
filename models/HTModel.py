class HTModel():
    def __init__(self, N, M, ranks, Y):
        self.N = N
        self.M = M
        self.ranks = ranks
        self.Y = Y

        # Define the parameters, randomly initialized and NOT trainable now!
