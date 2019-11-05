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
df_x = dataset_xlsx_all[1:]
#%%
df_y = dataset_xlsx_all[0] # This is failure data for 2018 -- The is the original excel is dataset_xlsx_all[-1]
df_y.columns = df_y.iloc[0]
df_y = df_y[1:]

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
#df_x_clean = df_x_concat
#%%
df_x_clean.to_pickle('X_data_pkl.pkl')

#%%
#################################################
''' information from the dataset - relevant to cleaning '''
#%% 
# Sensor information
sensor_info = [pd.to_numeric(df_x_clean[i],errors='coerce').describe() for i in range(1,len(df_x_clean.columns))]
nans_info = [np.multiply(100,np.divide(pd.to_numeric(df_x_clean[i],errors='coerce').isnull().sum(),len(df_x_clean[i]))) for i in range(1,len(df_x_clean.columns))]
#%%
# clean sensors 4,8 and 9
df_x_clean.drop(columns=[5,9,10,12,13], inplace=True)
#%%
# Separate data by component - Components: crusher, filter, Belt, feed1, feed2

# Crusher
df_x_crusher = df_x_clean.iloc[:,:9].reset_index(drop=True)
df_x_crusher.rename(columns = {df_x_crusher.columns[0]:'Time'}, inplace=True)
df_x_crusher['Time'] = pd.to_datetime(df_x_crusher['Time'],infer_datetime_format=True)

# Filter
df_x_filter = df_x_clean.iloc[:,[0,9]].reset_index(drop=True).rename(columns = {'0':'Time'})
df_x_filter.rename(columns = {df_x_filter.columns[0]:'Time'}, inplace=True)
df_x_filter['Time'] = pd.to_datetime(df_x_filter['Time'],infer_datetime_format=True)

# Belt
df_x_belt = df_x_clean.iloc[:,[0,10,11,12]].reset_index(drop=True).rename(columns = {'0':'Time'})
df_x_belt.rename(columns = {df_x_belt.columns[0]:'Time'}, inplace=True)
df_x_belt['Time'] = pd.to_datetime(df_x_belt['Time'],infer_datetime_format=True)

# Feeder_1
df_x_feed1 = df_x_clean.iloc[:,[0,13,14]].reset_index(drop=True).rename(columns = {'0':'Time'})
df_x_feed1.rename(columns = {df_x_feed1.columns[0]:'Time'}, inplace=True)
df_x_feed1['Time'] = pd.to_datetime(df_x_feed1['Time'],infer_datetime_format=True)

# Feeder_2
df_x_feed2 = df_x_clean.iloc[:,[0,15,16]].reset_index(drop=True).rename(columns = {'0':'Time'})
df_x_feed2.rename(columns = {df_x_feed2.columns[0]:'Time'}, inplace=True)
df_x_feed2['Time'] = pd.to_datetime(df_x_feed2['Time'],infer_datetime_format=True)

#%%
#################################################
''' preprocessing Y data '''

# group by component 
timestamp_y = df_x_clean.iloc[:,0]
timestamp_y = timestamp_y.to_frame()
timestamp_y.columns = ['Time']
timestamp_y = pd.to_datetime(timestamp_y.iloc[:,0],infer_datetime_format=True)
components_list = [df_x_crusher, df_x_filter, df_x_belt, df_x_feed1, df_x_feed2]

# Component label data
#%%
# floor to the minute
df_y.iloc[:,5] = pd.to_datetime(df_y.iloc[:,5],infer_datetime_format=True)
df_y.iloc[:,6] = pd.to_datetime(df_y.iloc[:,6],infer_datetime_format=True)
df_y.iloc[:,5] = pd.Series(df_y.iloc[:,5]).dt.floor('min')
df_y.iloc[:,6] = pd.Series(df_y.iloc[:,6]).dt.floor('min')
#%%
df_y.iloc[:,5] = df_y.iloc[:,5].dt.floor('120s')
df_y.iloc[:,6] = df_y.iloc[:,6].dt.ceil('120s')

#%%
# components grouping
grouped_components = df_y.groupby('Equipo')
#grouped_components.groups.keys()
df_y_components_dic = {'crusher':grouped_components.get_group('140-CR-004').iloc[:,[2,5,6]], 'filter':grouped_components.get_group('140-SN-003').iloc[:,[2,5,6]],
                       'belt':grouped_components.get_group('130-CV-004').iloc[:,[2,5,6]], 'feed1':grouped_components.get_group('130-FE-006').iloc[:,[2,5,6]],
                       'feed2':grouped_components.get_group('130-FE-007').iloc[:,[2,5,6]]}



#%%
# create arranges for labeling 
df_y_comp_labels = [pd.merge(timestamp_y, pd.concat([pd.DataFrame({'Time': pd.date_range(row.iloc[1], row.iloc[2], freq='120s'),
                                  'Detencion': row.iloc[0]}, columns=['Time', 'Detencion']) for i, row in df_y_components_dic[comp].iterrows()], 
    ignore_index=True), on='Time', how='left') for comp in df_y_components_dic]
    
#%%
# create labels ---- 0: normal, 1: failure
df_y_comp_just_labels = [comp['Detencion'].replace(['Falla', 'FALLA'],[1,1]).fillna(0) for comp in df_y_comp_labels]

#%%
for i in range(len(df_y_comp_labels)):
    df_y_comp_labels[i]['Detencion'] = df_y_comp_just_labels[i]
                   

#%%
#################################################
''' Save data for each component as a .pkl with datetimes()'''

# Crusher
df_crusher = df_x_crusher.merge(df_y_comp_labels[0], on='Time', how='left')
df_crusher.to_pickle('df_crusher.pkl')               
# Filter
df_filter = df_x_filter.merge(df_y_comp_labels[1], on='Time', how='left')
df_filter.to_pickle('df_filter.pkl')
# Belt
df_belt = df_x_belt.merge(df_y_comp_labels[2], on='Time', how='left')
df_belt.to_pickle('df_belt.pkl')
# Feeder 1
df_feed1 = df_x_feed1.merge(df_y_comp_labels[3], on='Time', how='left')
df_feed1.to_pickle('df_feed1.pkl')
# Feeder 2
df_feed2 = df_x_feed2.merge(df_y_comp_labels[4], on='Time', how='left')
df_feed2.to_pickle('df_feed2.pkl')                  

#%%
#%% 
#################################################
# NOTES
#################################################

# LOOK FOR A DELTA BEFORE AND AFTER THE PROPSED LABEL TO IDENTIFY ANOMALIES (BEGGENING AND INERTIA)
# FILL NaNs IN THE SENSORS (OR CUT THEM OFF) - DECIDE

    



