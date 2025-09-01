from mytorch.models import *
from mytorch.optimizers import *
import random


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


def train():
    pts = 100
    lr = 0.001
    epochs = 500
    model = MLP(8, 1)
    optimizer = Adam(model.get_params(), lr) # Adam optimizer is much better and more stable than SGD optimizer

    for epoch in range(1, epochs + 1):
        correct = 0
        total_loss = 0
        xs, ys = Data(pts).circle()

        for i in range(len(xs)):
            prob = model(xs[i])[0].sigmoid()
            y = ys[i]
            if y == 1 and prob.value >= 0.5:
                correct += 1
            else:
                pass
            if y == 0 and prob.value <= 0.5:
                correct += 1
            else:
                pass

            loss = - (y * prob.log() + (1 - y) * (1 - prob).log())
            total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f'epoch: {epoch}/{epochs}, accuracy: {correct * 100 / pts :.2f}%, loss: {total_loss.value / pts ** 2:.4f}')


if __name__ == "__main__":
    train()
