''' Data exploration - ENRE670 Project '''
#################################################
''' import libraries '''

import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import scipy.sparse as sp

import time

#%% 
# open dataset
Folder = 'CrusherSystem_dataset'
path = os.getcwd() + '\\' + Folder
files = os.listdir(path)

#%%
# dataset to df
start_time = time.time()
dataset_xlsx_all = [pd.read_excel(path + '\\' + file, sheet_name = 0) for file in files[:-1]]
print("--- %s seconds ---" % (time.time() - start_time))

#%%
# separate X and Y
df_x = dataset_xlsx_all[:-1]
df_y = dataset_xlsx_all[-1]

#%%
#################################################
''' preprocessing X data '''

# create a dictionary for x with keys: date_i, date_f, sensor_data
df_x_dict = {'date_i': [], 'date_f': [], 'sensor_data': []}

#%%
df_x_dict.update({'date_i':df_x[0].iloc[0,1], 'date_f':df_x[0].iloc[1,1], 'sensor_data':df_x[0].iloc[5:,2:]})
#%%

# append dict in a list
df_x_list = [{'date_i':df_x[i].iloc[0,1], 'date_f':df_x[i].iloc[1,1], 'sensor_data':df_x[i].iloc[5:,2:]} for i in range(len(df_x))]

