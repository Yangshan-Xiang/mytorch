import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import pathlib
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import sampler


data_dir = pathlib.Path('data/')
mnist = datasets.MNIST(data_dir, download=True, train=True, transform=transforms.ToTensor())

# X_sample, y_sample = mnist[0]
#
# print(mnist[0][0].shape)
# plt.imshow(X_sample, cmap='gray')
# plt.title(f'Label: {y_sample}')
# plt.show()

# DO NOT CHANGE
# tmp_dataloader = torch.utils.data.DataLoader(mnist, batch_size=len(mnist), shuffle=True)

 # TODO calculate the mean and standard deviation of MNIST train dataset
mean = mnist.data.float().mean() / 255
std = mnist.data.float().std() / 255
# print('mean =',mean)
# print('std =', std)

# DO NOT CHANGE
mnist_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean,), (std,))])


# TODO download the dataset for training and testing, with normalization transform
mnist_train = datasets.MNIST(data_dir, train=True, transform=mnist_transforms,download=True)
mnist_test = datasets.MNIST(data_dir, train=False, transform=mnist_transforms,download=True)

# print(len(mnist_train), len(mnist_test))
# TODO split the train dataset in mnist_train and mnist_val
train_set, val_set = torch.utils.data.random_split(mnist_train, [54000, 6000],
                                                   generator=torch.Generator().manual_seed(1))
# print(len(val_set))
batch_size = 256

# TODO create dataloader for training, validation and test
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

# # TODO display an element of the train_dataloader
# x, y = next(iter(train_dataloader))
# plt.title(f'Label: {y[0]}')
# plt.imshow(x[0].squeeze(), cmap="gray")
# plt.show()

epochs = 1
input_dim = 28 * 28
output_dim = 10
lr = 0.001

class LogisticRegression(torch.nn.Module):
    """
    Logistic regression model inherits the torch.nn.Module
    which is the base class for all neural network modules.
    """
    def __init__(self, input_dim, output_dim):
        """ Initializes internal Module state. """
        super(LogisticRegression, self).__init__()
        # TODO define linear layer for the model
        self.linear = torch.nn.Linear(input_dim, output_dim)


    def forward(self, x):
        """ Defines the computation performed at every call. """
        # What are the dimensions of your input layer?
        # TODO flatten the input to a suitable size for the initial layer
        x = torch.flatten(x, start_dim = 1)
        # TODO run the data through the layer
        outputs = torch.softmax(self.linear(x), dim=1)
        return outputs






