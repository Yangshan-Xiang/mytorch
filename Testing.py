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

# DO NOT CHANGE
data_dir = pathlib.Path('data/')
mnist = datasets.MNIST(data_dir, download=True, train=True)