# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 11:36:37 2023

@author: cwilson
"""

from types import SimpleNamespace
import numpy as np


def give_configs(model_name, enc_count, device='cpu', batch_size=1):
    
    if model_name == 'iTransformer':
    
        configs = {'model':'iTransformer',
                   'task_name':'long_term_forecast',
                    'seq_len':24*7*4,
                    'label_len':0,
                    'pred_len':24*7*4,
                    'e_layers':2,
                    'd_layers':1,
                    'factor':3,
                    'enc_in':enc_count,
                    'dec_in':enc_count,
                    'c_out':enc_count,
                    'd_model':32,
                    'd_ff':32,
                    'num_kernels':6,
                    'embed':'fixed',
                    'freq':'h',
                    'dropout':0.1,
                    'output_attention':False,
                    'batch_size':batch_size,
                    'features':'M',
                    'n_heads':3,
                    'activation':'relu',
                    'device':device,
                    'patience':3}
    
    if model_name == 'LSTM':
        
        configs = {'model':'LSTM',
                   'task_name':'long_term_forecast',
                    'seq_len':24*7*4,
                    'label_len':0,
                    'pred_len':24*7*4,
                    'e_layers':2,
                    'd_layers':1,
                    'factor':3,
                    'enc_in':enc_count,
                    'dec_in':enc_count,
                    'c_out':enc_count,
                    'd_model':32,
                    'd_ff':32,
                    'num_kernels':6,
                    'embed':'fixed',
                    'freq':'h',
                    'dropout':0.1,
                    'output_attention':False,
                    'batch_size':batch_size,
                    'features':'M',
                    'n_heads':3,
                    'activation':'relu',
                    'device':device,
                    'patience':3,
                    'hidden_units':6}
        
    if model_name == 'PatchTST':
        
        configs = {'model':'PatchTST',
                   'task_name':'long_term_forecast',
                    'seq_len':24*7*4,
                    'label_len':0,
                    'pred_len':24*7*4,
                    'e_layers':2,
                    'd_layers':1,
                    'factor':3,
                    'enc_in':enc_count,
                    'dec_in':enc_count,
                    'c_out':enc_count,
                    'd_model':32,
                    'd_ff':32,
                    'num_kernels':6,
                    'embed':'fixed',
                    'freq':'h',
                    'dropout':0.1,
                    'output_attention':False,
                    'batch_size':batch_size,
                    'features':'M',
                    'n_heads':3,
                    'activation':'relu',
                    'device':device,
                    'patience':3}    
        
    if model_name == 'TimesNet':
        
        configs = {'model':'TimesNet',
                   'task_name':'long_term_forecast',
                    'seq_len':24*7*4,
                    'label_len':24*7*2,
                    'pred_len':24*7*4,
                    'e_layers':2,
                    'd_layers':1,
                    'factor':3,
                    'enc_in':enc_count,
                    'dec_in':enc_count,
                    'c_out':enc_count,
                    'd_model':32,
                    'd_ff':32,
                    'top_k':5,
                    'des':'Exp',
                    'itr':1,
                    'num_kernels':6,
                    'embed':'timeF',
                    'freq':'h',
                    'dropout':0.1,
                    'output_attention':False,
                    'batch_size':batch_size,
                    'features':'M',
                    'n_heads':3,
                    'activation':'relu',
                    'device':device,
                    'patience':3}
    
    return SimpleNamespace(**configs)

def fit_moving_average_model(train_y, window_size):
 
    
    y = np.roll(train_y, -672)
    ma = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
    ma = np.roll(ma, 672)
    ma[window_size:] = ma[window_size]
    
    
    return ma