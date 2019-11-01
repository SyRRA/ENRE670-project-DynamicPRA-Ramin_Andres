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
import datetime
import time

#%%
def isnumber(x):
    try:
        float(x)
        return True
    except:
        return False
    
#%% 
# open dataset
Folder = 'CrusherSystem_dataset'
path = os.getcwd()+'\\'+Folder
files = os.listdir(path)

#%%
# dataset to df
start_time = time.time()
dataset_xlsx_all = [pd.read_excel(path+'\\'+file, sheet_name=0, header=None) for file in files]
print("--- %s seconds ---" % (time.time() - start_time))

#%%
# separate X and Y
df_x = dataset_xlsx_all[:-1]
df_y = dataset_xlsx_all[-1]

#%%
#################################################
''' preprocessing X data '''

# create a dictionary for x with keys: date_i, date_f, sensor_data

df_x_list = [{'date_i':df_x[i].iloc[0,1], 'date_f':df_x[i].iloc[1,1], 'sensor_data':df_x[i].iloc[6:,2:]} for i in range(len(df_x))]

for dic in df_x_list:
    dic['sensor_data'].columns = np.arange(len(dic['sensor_data'].columns))
    
#%%
# clean the data

for dic in df_x_list:
    dic['sensor_data'].iloc[:,1:] = dic['sensor_data'].iloc[:,1:][dic['sensor_data'].iloc[:,1:].applymap(isnumber)]
    dic['sensor_data'].iloc[:,0].replace(' ', np.nan, inplace=True)
    dic['sensor_data'] = dic['sensor_data'].dropna(subset=[0])
    dic['sensor_data'].reset_index(drop=True, inplace=True)
    
#%%
# concatenate
    
df_x_con_list = [dic['sensor_data'] for dic in df_x_list]

#%%
df_x_concat = pd.concat(df_x_con_list)
df_x_clean = df_x_concat.drop_duplicates(subset=df_x_concat.columns[0])
#df_x_clean = pd.to_numeric(df_x_clean, errors='coerce')
#%%
sensor_info = [pd.to_numeric(df_x_clean[i],errors='coerce').describe() for i in range(1,len(df_x_clean.columns))]
#%%
nans_info = [np.multiply(100,np.divide(pd.to_numeric(df_x_clean[i],errors='coerce').isnull().sum(),len(df_x_clean[i]))) for i in range(1,len(df_x_clean.columns))]

#%%
df_x_clean[1].plot(xticks = np.arange(0, len(df_x_clean[1]), step=20))

