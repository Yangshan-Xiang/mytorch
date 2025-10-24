import random
import numpy as np
from mytorch.tensor.loss import *
from mytorch.tensor.arithmetic import *
from mytorch.tensor.models import *
from mytorch.tensor.optimizers import *
import matplotlib.pyplot as plt

random.seed(42)

class Data:
    def __init__(self, pts: int):
        self.pts = pts

    def points(self):
        xs = []
        for i in range(self.pts):
            xs.append([random.uniform(-1, 1), random.uniform(-1, 1)])
        return xs

    def diagonal(self):
        xs = self.points()
        ys = []
        for x in xs:
            if x[0] >= x[1]:
                ys.append(0)
            else:
                ys.append(1)
        return xs, ys

    def circle(self):
        xs = self.points()
        ys = []
        for x in xs:
            if x[0] ** 2 + x[1] ** 2 >= 0.5:
                ys.append(1)
            else:
                ys.append(0)
        return xs, ys
    def spiral(self, noise: float = 0.02, rotations: int = 2):
        n_class = self.pts // 2
        xs, ys = [], []

        for i in range(n_class):
            r = i / n_class
            theta = rotations * math.pi * r
            x1 = r * math.cos(theta) + random.gauss(0, noise)
            y1 = r * math.sin(theta) + random.gauss(0, noise)
            xs.append([x1, y1])
            ys.append(0)

        for i in range(n_class):
            r = i / n_class
            theta = rotations * math.pi * r + math.pi
            x2 = r * math.cos(theta) + random.gauss(0, noise)
            y2 = r * math.sin(theta) + random.gauss(0, noise)
            xs.append([x2, y2])
            ys.append(1)

        return xs, ys

def train(model, dataset: str, optim: str, epochs: int, lr: float):
    pts = 100

    # ----- optimizer -----
    if optim == 'Adam':
        optimizer = Adam(model.get_params(), lr)
    elif optim == 'Adagrad':
        optimizer = Adagrad(model.get_params(), lr)
    elif optim == 'SGD':
        optimizer = SGD(model.get_params(), lr)
    else:
        raise ValueError(f'Optimizer {optim} is not supported')

    # ----- dataset -----
    if dataset == 'circle':
        xs, ys = Data(pts).circle()
    elif dataset == 'diagonal':
        xs, ys = Data(pts).diagonal()
    elif dataset == 'spiral':
        xs, ys = Data(pts).spiral()
    else:
        raise ValueError(f'Dataset {dataset} is not supported')

    ys = Tensor(ys, (pts, 1))
    xs = [i for x in xs for i in x]
    xs = Tensor(xs, (pts, 2))

    # store metrics
    losses, accs = [], []

    # ----- training -----
    for epoch in range(1, epochs + 1):
        prob = model(xs).sigmoid()
        loss = BCELoss('mean')(prob, ys)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = prob > Tensor([0.5])
        correct = pred == ys
        num_correct = correct.sum(0)
        acc = (num_correct / pts).storage[0] * 100

        losses.append(loss.storage[0])
        accs.append(acc)

        if epoch % 50 == 0:
            print(f"epoch: {epoch}/{epochs}, acc: {acc:.2f}%, loss: {loss.storage[0]:.4f}")

    # ----- test -----
    if dataset == 'circle':
        xs_test, ys_test = Data(pts).circle()
    elif dataset == 'diagonal':
        xs_test, ys_test = Data(pts).diagonal()
    elif dataset == 'spiral':
        xs_test, ys_test = Data(pts).spiral()
    else:
        raise ValueError(f'Dataset {dataset} is not supported')
    ys_test = Tensor(ys_test, (pts, 1))
    xs_test = [i for x in xs_test for i in x]
    xs_test = Tensor(xs_test, (pts, 2))

    prob = model(xs_test).sigmoid()
    pred = prob > Tensor([0.5])
    correct = pred == ys_test
    num_correct = correct.sum(0)
    print(f"test acc = {(num_correct / pts).storage[0] * 100 :.2f}%")
    return accs, losses

def plots(accs, losses, dataset: str, model, optim, epochs):
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    # Left: training curves
    ax1 = axes[0]
    ax1.plot(range(1, epochs + 1), losses, color='k', label='Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.tick_params(axis='y')
    ax1.set_title(f'Loss Curves ({optim})')

    ax2 = axes[1]
    ax2.plot(range(1, epochs + 1), accs, color='k', label='Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.tick_params(axis='y')
    ax2.set_title(f'Accuracy Curves ({optim})')



    # Right: decision boundary (as contour map)
    ax = axes[2]

    # generate meshgrid for visualization
    grid_size = 200
    x1 = np.linspace(-1, 1, grid_size)
    x2 = np.linspace(-1, 1, grid_size)
    X1, X2 = np.meshgrid(x1, x2)
    grid = np.stack([X1.ravel(), X2.ravel()], axis=1)

    # convert to Tensor and get model outputs
    grid_tensor = Tensor(grid.flatten().tolist(), (grid.shape[0], 2))
    preds = model(grid_tensor).sigmoid()
    Z = np.array(preds.storage).reshape(grid_size, grid_size)
    Z_binary = (Z > 0.5).astype(int)

    # plot filled contour (probability map)
    ax.contourf(X1, X2, Z_binary, levels=[0, 0.5, 1], colors=['deepskyblue', 'lightcoral'])

    pts = 100
    if dataset == 'circle':
        X, Y = Data(pts).circle()
    elif dataset == 'diagonal':
        X, Y = Data(pts).diagonal()
    elif dataset == 'spiral':
        X, Y = Data(pts).spiral()
    else:
        raise ValueError(f'Dataset {dataset} is not supported')

    X0 = [x for x, y in zip(X, Y) if y == 0]
    X1p = [x for x, y in zip(X, Y) if y == 1]
    ax.scatter([x[0] for x in X0], [x[1] for x in X0], color='blue', label='Class 0', edgecolor='k')
    ax.scatter([x[0] for x in X1p], [x[1] for x in X1p], color='red', label='Class 1', edgecolor='k')

    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title('Decision Boundary (Contour Map)')
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))
    ax.axis('equal')
    ax.grid(True)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    model = MLP(2, 10, 1)
    optim = 'SGD'
    epochs = 500
    dataset = 'diagonal'
    accs, losses = train(model, dataset, optim, epochs, 0.01)
    plots(accs, losses, dataset, model, optim, epochs)
