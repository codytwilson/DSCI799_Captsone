# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 11:08:04 2023

@author: cwilson
"""

import matplotlib.pyplot as plt
from get_data_for_modelling import dataset_dict_nonml
from get_data_for_modelling import init_dataset_dict
import torch
from LSTM_model import Model as Model_LSTM
from iTransformer import Model as Model_iTransformer
from PatchTST import Model as Model_PatchTST
from TimesNet import Model as Model_Timesnet
from sample_random_prediction_plotter_functions import give_configs, fit_moving_average_model
from model_train_vali_pred import predict
import numpy as np
import pandas as pd
from statsmodels.iolib.smpickle import load_pickle

shop_converter = {'CSF':'Shop A','CSM':'Shop B','FED':'Shop C'}

batch_size = 10
sample_test_index = 3900
data_dict_ml,  enc_count = init_dataset_dict('M', batch_size)
data_dict_nonml = dataset_dict_nonml()

file_iTransformer = 'myiTransformer_final.pth'
file_PatchTST = 'myPatchTST_final.pth'
file_TimesNet = 'myTimesNet_final.pth'
file_LSTM = 'LSTM_final_hn_neg1.pth'
file_ARIMA = '(2, 2, 2).pk1'
file_ARIMAX = 'x(2, 2, 2).pk1'
file_MA_windowsize= 672

results_iTransformer = 'iTransformer_202311121024.csv'
results_PatchTST = 'PatchTST_202311112005.csv'
results_TimesNet = 'TimesNet_202311171325.csv'
results_LSTM = 'LSTM_202311271835.csv'
results_ARIMA = 'arima_202311291119.csv'
results_ARIMAX = 'arimaX_202311291159.csv'
results_MA = 'ma_202311291319.csv'



model_meta = {}
model_meta['iTransformer'] = {'file':file_iTransformer,'results':results_iTransformer,
                              'model':Model_iTransformer}
model_meta['PatchTST'] = {'file':file_PatchTST,'results':results_PatchTST,
                              'model':Model_PatchTST}
model_meta['TimesNet'] = {'file':file_TimesNet,'results':results_TimesNet,
                              'model':Model_Timesnet}
model_meta['LSTM'] = {'file':file_LSTM,'results':results_LSTM,
                              'model':Model_LSTM}
model_meta['ARIMA'] = {'file':file_ARIMA,'results':results_ARIMA}
model_meta['ARIMAX'] = {'file':file_ARIMAX,'results':results_ARIMAX}
model_meta['MA'] = {'windowsize':file_MA_windowsize,'results':results_MA}

deep_learners = ['iTransformer','PatchTST','TimesNet','LSTM']
arimas = ['ARIMA','ARIMAX']

for k in model_meta.keys():
    print(f'Working on model: {k}')
    if k in deep_learners:
        configs = give_configs(k, enc_count, batch_size=batch_size)
        model = model_meta[k]['model'](configs)
        
        
        model_meta[k]['plotter_' + str(sample_test_index)] = {}
        for shop in data_dict_ml.keys():
        # shop = 'CSM'
            test_data = data_dict_ml[shop]['test']
            yhat, true = predict(configs = configs,
                                  dataset = test_data['dataset'],
                                  model = model,
                                  index = sample_test_index,
                                  return_y_only=False)
            ytrue = true[:,-1]
            
            
            _, previous = predict(configs = configs,
                                  dataset = test_data['dataset'],
                                  model = model,
                                  index = sample_test_index - 672,
                                  return_y_only=False)
            
            # index, y
            predicted = np.concatenate([true[:, 0].round().reshape(-1,1), yhat], axis=1)
            
            previous_plotter = np.concatenate([previous[:,0].round().reshape(-1,1), previous[:,-1].reshape(-1,1)], axis=1)
            
            plotter = np.concatenate([previous_plotter, predicted], axis=0)
            
            model_meta[k]['plotter_' + str(sample_test_index)][shop] = plotter
            
            
            
            
    if k in arimas:
        # print('arima')
        model_meta[k]['plotter_' + str(sample_test_index)] = {}
        
        # shop = 'CSM'
        for shop in data_dict_ml.keys():
                
            test_data = data_dict_nonml[shop]['test']
            scaler = data_dict_nonml[shop]['scaler']
            test_data_unscaled = scaler.inverse_transform(test_data)
        
            start = model_meta['iTransformer']['plotter_' + str(sample_test_index)][shop][0,0]
                
            start_idx = np.where(test_data_unscaled[:,0] == start)[0][0]
            to_pred = test_data[start_idx:start_idx+672]
            predder = test_data[start_idx+672:start_idx+672+672]
            

            file = shop + '_' + model_meta[k]['file']
            
            model = load_pickle('.\\saved_models\\' + file)
            
            yhat = model.get_forecast(steps=672, exog=to_pred[:,:-1]).predicted_mean
            
            predder[:,-1] = yhat
            
            predder = scaler.inverse_transform(predder)
            
            predicted_plotter = np.concatenate([predder[:,0].reshape(-1,1), predder[:,-1].reshape(-1,1)], axis=1)
            
            previous_plotter = np.concatenate([to_pred[:,0].reshape(-1,1), to_pred[:,-1].reshape(-1,1)], axis=1)
            
            plotter = np.concatenate([previous_plotter, predicted_plotter], axis=0)
            
            model_meta[k]['plotter_' + str(sample_test_index)][shop] = plotter
            
        
    if k == 'MA':
        model_meta[k]['plotter_' + str(sample_test_index)] = {}
        
        for shop in data_dict_ml.keys():
    
            window = model_meta['MA']['windowsize']
            test_data = data_dict_nonml[shop]['test']
            scaler = data_dict_nonml[shop]['scaler']
            test_data_unscaled = scaler.inverse_transform(test_data)
            
            
            start = model_meta['iTransformer']['plotter_' + str(sample_test_index)][shop][0,0]
    
            start_idx = np.where(test_data_unscaled[:,0] == start)[0][0]
            
            to_pred = test_data_unscaled[start_idx:]
            
            
            previous = test_data_unscaled[start_idx:start_idx+672]
            
            previous_plotter = np.concatenate([previous[:,0].reshape(-1,1), previous[:,-1].reshape(-1,1)], axis=1)
            
            
            yhat = fit_moving_average_model(to_pred[:,-1], window_size=window)
            
            predicted = test_data_unscaled[start_idx+672:start_idx+672+672,0]
            
            predicted_plotter = np.concatenate([predicted.reshape(-1,1), yhat[:672].reshape(-1,1)], axis=1)
    
            plotter = np.concatenate([previous_plotter, predicted_plotter], axis=0)        
            
            model_meta[k]['plotter_' + str(sample_test_index)][shop] = plotter
            
    

shops = ['CSF','CSM','FED']

df_dict = dict([(shop, pd.DataFrame(columns=model_meta.keys())) for shop in shops])




for k in model_meta.keys():
    print(k)
    
    for shop in shops:
        results = model_meta[k]['plotter_' + str(sample_test_index)][shop]
        idx = results[:,0].round().astype(int)
        yhat = results[:,-1]
        # Ensure that the DataFrame has the required columns

        # Iterate through the results and update the DataFrame
        for i, index in enumerate(idx):
            if index not in df_dict[shop].index:
                df_dict[shop].loc[index] = [None] * len(model_meta.keys())
    
            # Update the value at the specified index and column
        df_dict[shop].loc[idx, k] = yhat



for k in df_dict.keys():
    
    dated = pd.read_csv('.\\data\\data_for_ml_' + k + '.csv')
    
    df = df_dict[k]
    
    added_date = pd.merge(df, dated, left_index=True, right_on='Unnamed: 0', how='left')
    
    added_date = added_date.set_index('date')

    added_date = added_date[list(model_meta.keys()) + ['Worth']]
    
    added_date = added_date.round(decimals=3)
    
    added_date = added_date.iloc[:-1]
    
    added_date.index = pd.to_datetime(added_date.index)

    df_dict[k] = added_date.copy()
    
    

#%%

resample_hours = 12
resample_str = str(resample_hours) + 'H'


shop = 'CSM'
for shop in df_dict.keys():
    df = df_dict[shop]
    
    df = df.resample(resample_str).sum()
    
    df_yhat = df.iloc[672//resample_hours:,]
    df_yhat = df_yhat.drop(columns=['Worth'])
    
    
    plt.title(shop_converter[shop])
    plt.plot(df['Worth'], color='black')
    plt.plot(df_yhat, label=df_yhat.columns)
    plt.legend()
    plt.xticks(rotation=45)
    plt.ylabel('Worth')
    plt.ylim(bottom=0)
    plt.show()










