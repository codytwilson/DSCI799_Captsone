# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 09:28:58 2023

@author: cwilson
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from data_loader import Dataset_Custom
from torch.utils.data import DataLoader

def yield_DatasetCustom(file, test_train_val, features='M'):
    '''
    it looks like features is either 'M' or 'S'
    'M': all columns in dataset exceot for date and target
    'S' only target
    '''
    root_path = '.\\data\\'
    # data_path = 'data_for_ml_CSF.csv'
    data_path = file
    sequence_length, label_length, forecast_length = 24*7*4, 24*7*2, 24*7*2
    target = 'Worth'
    # features = ['Quantity', 'Weight', 'OustandingWeight', 'Manhours', 'DayShift'] + ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    test = Dataset_Custom(root_path = root_path, 
                   data_path = data_path,
                   flag=test_train_val,
                   size = (sequence_length, label_length, forecast_length),
                   features = features,
                   target = target)

    return test



def init_dataset_dict(features, batch_size):
    data_dict = {}
    for shop in ['CSM','CSF','FED']:
        data_dict[shop] = {}
        file = 'data_for_ml_' + shop + '.csv'
        file_shape = pd.read_csv(".\\Data\\" + file).shape
        print(f'Shop: {shop}\n---------------')
        print(f'Size of input df: {file_shape}')
        for test_train_val in ['train','test','val']:
            data_dict[shop][test_train_val] = {}
            dataset = yield_DatasetCustom(file, test_train_val, features)
            data_dict[shop][test_train_val]['dataset'] = dataset
            if test_train_val == 'train':
                shuffle = False
            else:
                shuffle = False
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            data_dict[shop][test_train_val]['dataloader'] = dataloader
            print(f'{test_train_val}: {len(dataset)}')
            print(f'x: {dataset[0][0].shape}')
            print(f'y: {dataset[0][1].shape}')
            print(f'x_mark: {dataset[0][2].shape}')
            print(f'y_mark: {dataset[0][3].shape}')
            enc_count = dataset[0][0].shape[1] #???   
    
    if features == 'S':
        enc_count = 1
        
    return data_dict, enc_count



def dataset_dict_nonml():
    data_dict = {}
    for shop in ['CSM','CSF','FED']:
        data_dict[shop] = {}
        file = 'data_for_ml_' + shop + '.csv'
        file_shape = pd.read_csv(".\\Data\\" + file).shape
        print(f'Shop: {shop}\n---------------')
        print(f'Size of input df: {file_shape}')
        for test_train_val in ['train','test','val']:
            data_dict[shop][test_train_val] = {}
            
            dataset = yield_DatasetCustom(file, test_train_val, 'M')
            
            
            data_dict[shop][test_train_val] = dataset.get_scaled_data()
            print(f'{test_train_val}: {len(dataset)}')
            # print(f'x: {data[:,:-1].shape}')
            # print(f'y: {data[:,-1].shape}')     
            
        data_dict[shop]['scaler'] = dataset.scaler
        data_dict[shop]['borders'] = dataset.get_borders()
            
    return data_dict