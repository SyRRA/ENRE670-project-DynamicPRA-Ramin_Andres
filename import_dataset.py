import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, BatchNormalization, MaxPooling2D
import time
from keras.utils import np_utils
import itertools
import os
#%%
# open dataset
Folder = 'DATASET_FINAL'
path = os.getcwd()+'\\'+Folder
file = 'crusher_.npz'
dataset = np.load(path+'\\'+file)

x_train = np.asarray(dataset['x_train'])
y_train = np.asarray(dataset['y_train'])

x_test = np.asarray(dataset['x_test'])
y_test = np.asarray(dataset['y_test'])