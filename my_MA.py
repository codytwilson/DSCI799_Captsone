# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:24:55 2023

@author: cwilson
"""

import pandas as pd
import numpy as np
from get_data_for_modelling import dataset_dict_nonml
import torch
from torch import nn
from sklearn.metrics import mean_squared_error
from model_train_vali_pred import return_pooled_for_loss
import datetime

data_dict = dataset_dict_nonml()




def fit_moving_average_model(train_y, window_size):
 
    
    y = np.roll(train_y, -672)
    ma = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
    ma = np.roll(ma, 672)
    ma[window_size:] = ma[window_size]
    
    
    return ma

def weighted_mse(y_true, y_pred):
    # convert to tensor to use existing framework
    y_true = torch.Tensor(y_true)
    y_pred = torch.Tensor(y_pred)
    
    # get pooled variants
    y_biweekly, y_weekly, y_daily, y_4hour = return_pooled_for_loss(y_true, batch_reshape=False)
    yhat_biweekly, yhat_weekly, yhat_daily, yhat_4hour = return_pooled_for_loss(y_pred, batch_reshape=False)
    
    # calcualte mse 
    mse_biweekly = mean_squared_error(yhat_biweekly, y_biweekly)
    mse_weekly = mean_squared_error(yhat_weekly, y_weekly)
    mse_daily = mean_squared_error(yhat_daily, y_daily)
    mse_4hour = mean_squared_error(yhat_4hour, y_4hour)

    # Weighted MSE
    weighted_mse = mse_biweekly * 0.4 + mse_weekly * 0.3 + mse_daily * 0.2 + mse_4hour * 0.1

    return weighted_mse



losses_train, losses_val, losses_test, ws = [], [], [], []


#%%

window_size = 24*7*4
i=0
for window_size in np.arange(24, 673, 24):
    print(f'Iteration {i} has window size = {window_size}')
    losses_train_shop, losses_val_shop, losses_test_shop = [], [], []

    
    for shop in data_dict.keys():
        print(f'\t***** Shop {shop} *****')
        train_x = data_dict[shop]['train'][:,:-1]
        train_y = data_dict[shop]['train'][:,-1]
        
        test_x = data_dict[shop]['test'][:,:-1]
        test_y = data_dict[shop]['test'][:,-1] 
    
        val_x = data_dict[shop]['val'][:,:-1]  
        val_y = data_dict[shop]['val'][:,-1] 
        
        
        reassmbled_y = np.concatenate([np.zeros(window_size), train_y, val_y, test_y])
        model_results = fit_moving_average_model(reassmbled_y, window_size=window_size)
        model_results = model_results[window_size-1:]    
            
        borders = data_dict[shop]['borders']
        train_yhat = model_results[borders[0][0]:borders[1][0]]
        val_yhat = model_results[borders[0][1]:borders[1][1]]
        test_yhat = model_results[borders[0][2]:borders[1][2]]
        
        
        train_loss = weighted_mse(train_y, train_yhat)
        test_loss = weighted_mse(test_y, test_yhat)
        val_loss = weighted_mse(val_y, val_yhat)
            
        print(f'/\/\/\/\/\/\/\/\ Train Loss:  {train_loss}')
        print(f'/\/\/\/\/\/\/\/\ Val Loss:    {val_loss}')
        print(f'/\/\/\/\/\/\/\/\ Test Loss:   {test_loss}')
        
        
        losses_train_shop.append(train_loss)
        losses_val_shop.append(val_loss)
        losses_test_shop.append(test_loss)
    
    # end of the shop loop
    
    epoch_train_loss = np.mean(losses_train_shop)
    epoch_val_loss = np.mean(losses_val_shop)
    epoch_test_loss = np.mean(losses_test_shop)
    
    
    print(f'\n----- {window_size} -----')
    print(f'Train Loss:  {epoch_train_loss}')
    print(f'Test Loss:   {epoch_test_loss}')                        
    print(f'Val Loss:    {epoch_val_loss}\n')
    
    losses_train.append(epoch_train_loss)
    losses_val.append(epoch_val_loss)
    losses_test.append(epoch_test_loss)   
    ws.append(window_size)
    
  

epoch_results = pd.DataFrame(columns=['Train','Test','Val','window_size'])
epoch_results['Train'] = losses_train
epoch_results['Test'] = losses_test
epoch_results['Val'] = losses_val
epoch_results['window_size'] = ws
now = datetime.datetime.now().strftime('%Y%m%d%H%M')
epoch_results.to_csv('ma' + '_' + now + '.csv')


