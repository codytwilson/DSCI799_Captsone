# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 10:05:22 2023

@author: cwilson
"""

import pandas as pd
import numpy as np
from get_data_for_modelling import dataset_dict_nonml
import torch
from torch import nn
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from model_train_vali_pred import return_pooled_for_loss
import datetime

data_dict = dataset_dict_nonml()

def fit_arima_model(train_x, train_y, order, use_x=False):
    if use_x:
        model = SARIMAX(train_y, exog=train_x, order=order)
    else:
        model = SARIMAX(train_y, order=order)
    results = model.fit(disp=False)
    return results

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


arima_order = [5, 1, 1]

losses_train, losses_val, losses_test, orders = [], [], [], []


#%%


i = 0
for p in np.arange(0,3):
    arima_order[0] = p
    for d in np.arange(0,3):
        arima_order[1] = d
        for q in np.arange(0,3):
            arima_order[2] = q
            
            
            print(f'Iteration {i} has (p,q,d)={arima_order}')
            losses_train_shop, losses_val_shop, losses_test_shop = [], [], []
            models={}
            for shop in data_dict.keys():
                print(f'\t***** Shop {shop} *****')
                train_x = data_dict[shop]['train'][:,:-1]
                train_y = data_dict[shop]['train'][:,-1]
                
                test_x = data_dict[shop]['test'][:,:-1]
                test_y = data_dict[shop]['test'][:,-1] 
    
                val_x = data_dict[shop]['val'][:,:-1]       
                val_y = data_dict[shop]['val'][:,-1] 
    
                
                model_results = fit_arima_model(train_x, train_y, tuple(arima_order), use_x=True)
                models[shop] = model_results
                train_yhat = model_results.get_forecast(steps=len(train_x), exog=train_x).predicted_mean
                test_yhat = model_results.get_forecast(steps=len(test_x), exog=test_x).predicted_mean
                val_yhat = model_results.get_forecast(steps=len(val_x), exog=val_x).predicted_mean
                
                train_loss = weighted_mse(train_y,train_yhat)
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
            
            
            print(f'\n----- {tuple(arima_order)} -----')
            print(f'Train Loss:  {epoch_train_loss}')
            print(f'Test Loss:   {epoch_test_loss}')                        
            print(f'Val Loss:    {epoch_val_loss}\n')
            
            losses_train.append(epoch_train_loss)
            losses_val.append(epoch_val_loss)
            losses_test.append(epoch_test_loss)   
            orders.append(str(tuple(arima_order)))
            
            
            if len(losses_val) == 1 or epoch_val_loss <= min(losses_val):
                print('Saving Model!')
                model_name = str(tuple(arima_order)) + '.pk1'
                if len(losses_val) > 1:
                    previous_best = min(losses_val[:-1])
                else:
                    previous_best = 'inf'
                print(f'Val Loss decrease from {previous_best} -> {epoch_val_loss}')
                
                ''' (0, 2, 2) '''
                for k in models.keys():
                    m = models[k]
                    m.save('.\\saved_models\\' + k + '_x' + model_name)
    
                
            

epoch_results = pd.DataFrame(columns=['Train','Test','Val','orders'])
epoch_results['Train'] = losses_train
epoch_results['Test'] = losses_test
epoch_results['Val'] = losses_val
epoch_results['orders'] = orders
now = datetime.datetime.now().strftime('%Y%m%d%H%M')
epoch_results.to_csv('arimaX' + '_' + now + '.csv')


