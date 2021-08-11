from operator import index
import pandas as pd
import numpy as np
import os
import torch
from torch import nn

# import train/test datasets
train = pd.read_csv('train.csv', index_col=False)
test = pd.read_csv('test.csv', index_col=False)

# split to X and y
X_train = train.drop(columns='Label')
y_train = train['Label']

X_test = test.drop(columns='Label')
y_test = test['Label']
