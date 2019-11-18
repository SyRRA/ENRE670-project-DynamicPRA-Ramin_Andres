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
crusher_x = dataset_pkl_all[1].iloc[:,:-2]
crusher_y = dataset_pkl_all[1].iloc[:,[0,-2]]
# Filter
filter_ = dataset_pkl_all[4]
filter_x = dataset_pkl_all[4].iloc[:,:-2]
filter_y = dataset_pkl_all[4].iloc[:,[0,-2]]
# Belt
belt_ = dataset_pkl_all[0]
belt_x = dataset_pkl_all[0].iloc[:,:-2]
belt_y = dataset_pkl_all[0].iloc[:,[0,-2]]
# Feeder 1
feed1_ = dataset_pkl_all[2]
feed1_x = dataset_pkl_all[2].iloc[:,:-2]
feed1_y = dataset_pkl_all[2].iloc[:,[0,-2]]
# Feeder 2
feed2_ = dataset_pkl_all[3]
feed2_x = dataset_pkl_all[3].iloc[:,:-2]
feed2_y = dataset_pkl_all[3].iloc[:,[0,-2]]
# Maintenance logs
maint_logs = dataset_pkl_all[5]
#%% Max frame size
#arr = maint_logs['Mantencion'].values

#Aa = (filter_['Time']-filter_['Time'].shift()).fillna(0)
#Aaa = Aa.apply(lambda x: x  / np.timedelta64(1,'m')).astype('int64') % (24*60)

#%% Total number of failures
li = [crusher_y, filter_y, belt_y, feed1_y, feed2_y]
num = 0
for i in li:
    num += i['Detencion'].sum()
    
total_failures = [df['Detencion'].sum() for df in li]

#%%
crusher_time = crusher_['Time']
crusher_delta = crusher_time.diff().fillna(0)
delta = crusher_delta.apply(lambda x: x  / np.timedelta64(1,'m')).astype('int64') % (24*60)
#%%
new_time_df = pd.DataFrame({'Time':crusher_time,'delta':delta}).reset_index()
frames_out = []
frames_in = []
for i in range(len(new_time_df)-1):
    if (new_time_df['delta'][i+1]-new_time_df['delta'][i]) == 0:
        frames_in.append(new_time_df['Time'][i])
    else:
        j=0
        frames_out.append(pd.Series(frames_in, name=j))
        frames_in = []
        j+=1
# sum(len(li) for li in frames_out2)
#%%
frames_out2 = [ix for ix in frames_out if len(ix)!=0]
#%%
frames_out2[0] = pd.date_range(new_time_df['Time'][0], frames_out2[0], freq='2min', closed=None)


################################################
################################################ until here everything is OK
#%%
# Generating TIME Windows - OK
crusher_tw = [crusher_.set_index('Time').loc[frames] for frames in frames_out2]
filter_tw = [filter_.set_index('Time').loc[frames] for frames in frames_out2]
belt_tw = [belt_.set_index('Time').loc[frames] for frames in frames_out2]
feed1_tw = [feed1_.set_index('Time').loc[frames] for frames in frames_out2]
feed2_tw = [feed2_.set_index('Time').loc[frames] for frames in frames_out2]

#%% amount of NaNs with failure label - Being developed
AaA = crusher_.dropna()
AaAa = crusher_.drop(AaA.index)
print(AaAa['Detencion'].sum())

################################################
################################################ from here everything is OK    - data description and plots
#%%
def plot_cdf(x,col,component='insert component name'):
    
    if component == 'Filter':
        fail = filter_.loc[filter_['Detencion'] == 1]
        good = filter_.loc[filter_['Detencion'] == 0]
        plt.figure()
        plt.hist(filter_[14].values, cumulative=True, density=True, bins=500)
        plt.hist(good[14].values, cumulative=True, density=True, bins=500, color = 'g',alpha=1)
        plt.hist(fail[14].values, cumulative=True, density=True, bins=500, color = 'r',alpha=0.3)
        plt.ylabel('sensor '+str(filter_.columns[1]))
        plt.title(component)
        
    else:       
        x = x.iloc[:,1:]
        fail = x.loc[x[col]==1]
        good = x.loc[x[col]==0]
        fig, ax = plt.subplots(len(x.columns)-2,1,squeeze=True)
        for i in range(len(x.columns)-2):
            ax[i].hist(x.iloc[:,i].values, cumulative=True, density=True, bins=500)
            ax[i].hist(good.iloc[:,i].values, cumulative=True, density=True, bins=500, color='g', alpha=0.8)
            ax[i].hist(fail.iloc[:,i].values, cumulative=True, density=True, bins=500, color='r', alpha=0.3)
            ax[i].set_ylabel('sensor '+str(x.columns[i]))
            
        #fig.tight_layout()
        plt.subplots_adjust(hspace=0.5)
        plt.show()
        fig.suptitle(component)
    return print(len(fail)), print(100*np.divide(x.isna().sum(),len(x))), len(x)
    
    
    
#%%
# crusher must be done in two setps crusher.iloc[:,:6] , crusher.iloc[:,4:]
plot_cdf(crusher_,'Detencion','Crusher')

#%%
# Define label thresholds
#from collections import defaultdict
# Mean and std for failure data and normal data
def mean_std_comp(x):
    x = x.iloc[:,1:]
    fail = x.loc[x['Detencion']==1]
    ms_dict = {'sensor':[],'mean':[],'std':[],'p25,50,75,90,98':[]}
    ms_dict_fail = {'sensor':[],'mean':[],'std':[],'p25,50,75,90,98':[]}
    for i in range(len(x.columns)-2):            
        ms_dict['sensor'].append(x.columns[i])
        ms_dict['mean'].append(x.iloc[:,i].mean(skipna=True)) 
        ms_dict['std'].append(x.iloc[:,i].std(skipna=True))
        ms_dict['p25,50,75,90,98'].append(x.iloc[:,i].quantile([.25,.5,.75,.9,.98]))
        
        ms_dict_fail['sensor'].append(fail.columns[i])
        ms_dict_fail['mean'].append(fail.iloc[:,i].mean(skipna=True)) 
        ms_dict_fail['std'].append(fail.iloc[:,i].std(skipna=True))
        ms_dict_fail['p25,50,75,90,98'].append(fail.iloc[:,i].quantile([.25,.5,.75,.9,.98]))

    return ms_dict, ms_dict_fail
#%%
filter_ms, filter_ms_fail = mean_std_comp(filter_)

#%%
# Plot specific sensors
plt.figure()
a = filter_[14] #.interpolate(method='linear', axis=0).ffill()
a.plot(style='o', ms=2)
a.isna().sum()#plot()
#feed1_['Detencion'].replace(1,10).plot(c='r')
#crusher_['Detencion'].replace(1,10).plot(c='r')
filter_['Detencion'].replace(1,10).plot(c='r')


#%%      
plt.figure()
a = belt_[17] #.interpolate(method='linear', axis=0).ffill()
a.plot(style='o', ms=2)
a.isna().sum()#plot()
#feed1_['Detencion'].replace(1,10).plot(c='r')
#crusher_['Detencion'].replace(1,10).plot(c='r')
filter_['Detencion'].replace(1,10).plot(c='r')
belt_['Detencion'].replace(1,10).plot(c='g')


    
    
    
    
    
    