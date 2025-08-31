from mytorch.models import *
from mytorch.optimizers import *


def train():

    lr = 0.001
    epochs = 500
    model = MLP()
    optimizer = SGD(model.get_params(), lr)

    for epoch in range(1, epochs + 1):
        correct = 0
        total_loss = 0
        def data(n):
            pts = [] # Points
            for x_1 in range(n):
                for x_2 in range(n):
                    pts.append([x_1, x_2])
            for pt in pts:
                if pt[0] == pt[1]:
                    pts.remove(pt)

            label = [] # True labels
            for pt in pts:
                if pt[0] > pt[1]:
                    label.append(0)
                else:
                    label.append(1)
            return pts, label

        n = 10
        xs, ys = data(n)
        for i in range(len(xs)):
            pred = model(xs[i])[0].sigmoid()
            y = ys[i]

            if y == 1 and pred.value >= 0.5:
                correct += 1
            else:
                pass
            if y == 0 and pred.value <= 0.5:
                correct += 1
            else:
                pass

            loss = - (y * pred + (1 - y) * (1 - pred))
            total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0 or epoch == epochs:
            print('epoch:', epoch, f'accuracy: {correct * 100 / n ** 2:.2f}%', f'loss: {total_loss.value / n ** 2:.3f}')


if __name__ == "__main__":
    train()
