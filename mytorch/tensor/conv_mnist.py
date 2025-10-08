from torchvision import datasets, transforms

from torch.utils.data import DataLoader
from mytorch.tensor.tensor import *
import random
from mytorch.tensor.loss import *
from mytorch.tensor.arithmetic import *
from mytorch.tensor.models import *
from mytorch.tensor.optimizers import *


def mnist(batch_size: int = 64, train: bool = True, download: bool = True):
    dataset = datasets.MNIST(root='./data', train=train, download=download, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=batch_size)

    batched = []
    for images, labels in dataloader:
        batched.append([Tensor(images.flatten().numpy().tolist(), (images.shape[0], 1, 28, 28)),
                        Tensor(labels.numpy().tolist())])

    return batched




def mnist_train(model, optim: str):
    epochs = 5
    batch_size = 64

    if optim == 'Adam':
        optimizer = Adam(model.get_params(), lr = 0.001) # Adam optimizer is much better and more stable than SGD optimizer
    elif optim == 'Adagrad':
        optimizer = Adagrad(model.get_params(), lr = 0.1)
    elif optim == 'SGD':
        optimizer = SGD(model.get_params(), lr=0.01)
    else:
        raise ValueError(f'Optimizer {optim} is not supported')

    batched = mnist(batch_size=batch_size, train=True)

    for epoch in range(1, epochs + 1):
        for images, labels in batched:
            prob = model(images).reshape(batch_size, 10)
            prob = prob.softmax(-1)
            loss = CrossEntropyLoss('mean')(prob, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, pred = prob.max(1, keepdim=False)
            correct = pred == labels
            num_correct = correct.sum(0) # type: ignore

            print(f"epoch: {epoch}/{epochs}, "
                  f"acc: {(num_correct / images.shape[0]).storage[0] * 100 :.2f}%, "
                  f"loss: {loss.storage[0] :.4f}")

if __name__ == "__main__":
    mnist_train(Conv2d(1, 10, 28), 'Adagrad')