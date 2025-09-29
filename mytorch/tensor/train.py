from mytorch.tensor.arithmetic import *
from mytorch.tensor.models import *
from mytorch.tensor.optimizers import *


class Data:
    def __init__(self, pts: int):
        self.pts = pts

    def points(self):
        xs = []
        for i in range(self.pts):
            xs.append([random.random() - 0.5, random.random() - 0.5])
        return xs

    def diagonal(self):
        xs = self.points()
        ys = []
        for x in xs:
            if x[0] >= x[1]:
                ys.append(1)
            else:
                ys.append(0)
        return xs, ys

    def circle(self):
        xs = self.points()
        ys = []
        for x in xs:
            if x[0] ** 2 + x[1] ** 2 >= 0.2:
                ys.append(1)
            else:
                ys.append(0)
        return xs, ys


def train(model: str, optim: str):
    pts = 100
    epochs = 500
    if model == 'Linear':
        model = Linear(2, 1)
    elif model == 'MLP':
        model = MLP(2, 1)
    else:
        raise ValueError(f'Model {model} is not supported')

    if optim == 'Adam':
        optimizer = Adam(model.get_params(), lr = 0.001) # Adam optimizer is much better and more stable than SGD optimizer
    elif optim == 'Adagrad':
        optimizer = Adagrad(model.get_params(), lr = 0.1)
    elif optim == 'SGD':
        optimizer = SGD(model.get_params(), lr = 0.001)
    else:
        raise ValueError(f'Optimizer {optim} is not supported')

    for epoch in range(1, epochs + 1):
        xs, ys = Data(pts).diagonal()
        ys = Tensor(ys, (pts, 1)) # (100, 1)
        xs = [i for x in xs for i in x]
        xs = Tensor(xs, (pts, 2)) # (100, 2)

        prob = model(xs).sigmoid() # (100, 2) @ (2, 1) = (100, 1)
        loss = - (ys * prob.log() + (Tensor([1]) - ys) * (Tensor([1]) - prob).log()) # (100, 1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = prob > Tensor([0.5]) # (100, 1)
        correct = pred == ys # (100, 1)
        num_correct = correct.sum(0)

        if epoch % 10 == 0:
            print(num_correct / Tensor([pts]))


if __name__ == "__main__":
    train('Linear', 'SGD')
