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
# open dataset
Folder = 'pkl datasets'
path = os.getcwd()+'\\'+Folder
files = os.listdir(path)

#%%
# dataset to df
start_time = time.time()
dataset_pkl_all = [pd.read_pickle(path+'\\'+file) for file in files]
print("--- %s seconds ---" % (time.time() - start_time))
#%%
# Open datasets - timestamp is used in both x and y variables!
# Crusher
crusher_ = dataset_pkl_all[1]
crusher_x = dataset_pkl_all[1].iloc[:,:-1]
crusher_y = dataset_pkl_all[1].iloc[:,[0,-1]]
# Filter
filter_ = dataset_pkl_all[4]
filter_x = dataset_pkl_all[4].iloc[:,:-1]
filter_y = dataset_pkl_all[4].iloc[:,[0,-1]]
# Belt
belt_ = dataset_pkl_all[0]
belt_x = dataset_pkl_all[0].iloc[:,:-1]
belt_y = dataset_pkl_all[0].iloc[:,[0,-1]]
# Feeder 1
feed1_ = dataset_pkl_all[2]
feed1_x = dataset_pkl_all[2].iloc[:,:-1]
feed1_y = dataset_pkl_all[2].iloc[:,[0,-1]]
# Feeder 2
feed2_ = dataset_pkl_all[3]
feed2_x = dataset_pkl_all[3].iloc[:,:-1]
feed2_y = dataset_pkl_all[3].iloc[:,[0,-1]]

#%%
failure = filter_.loc[filter_['Detencion'] == 1]
good = filter_.loc[filter_['Detencion'] == 0]
#%%
# All data
#fig,ax = plt.plot()
plt.figure()
plt.hist(filter_[14].values, cumulative=True, density=True, bins=500)
plt.hist(good[14].values, cumulative=True, density=True, bins=500, color = 'g',alpha=1)
plt.hist(failure[14].values, cumulative=True, density=True, bins=500, color = 'r',alpha=0.3)

#plt.xticks(np.arange(0,51,2))

#%%
li = [crusher_y, filter_y, belt_y, feed1_y, feed2_y]
num = 0
for i in li:
    num += i['Detencion'].sum()
    
total_failures = [df['Detencion'].sum() for df in li]

#%%
def plot_cdf(x,col):

    x = x.iloc[:,1:]
    fail = x.loc[x[col]==1]
    good = x.loc[x[col]==0]
    fig, ax = plt.subplots(len(x.columns)-1,1,squeeze=True)
    for i in range(len(x.columns)-1):
        ax[i].hist(x.iloc[:,i].values, cumulative=True, density=True, bins=500)
        ax[i].hist(good.iloc[:,i].values, cumulative=True, density=True, bins=500, color='g', alpha=0.8)
        ax[i].hist(fail.iloc[:,i].values, cumulative=True, density=True, bins=500, color='r', alpha=0.3)
        ax[i].set_ylabel('sensor '+str(x.columns[i]))
        
    #fig.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    plt.show()
    
    return print(len(fail)), print(x.isna().sum()), len(x)
    
    
    
#%%
# crusher must be done in two setps crusher.iloc[:,:6] , crusher.iloc[:,4:]
plot_cdf(feed1_,'Detencion')
    
    
    
    
    
    
    