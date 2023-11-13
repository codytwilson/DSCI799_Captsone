# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 08:53:16 2023

@author: cwilson
"""

''' general '''
import pandas as pd
import sys
import os 
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import numpy as np
import datetime
from types import SimpleNamespace
''' mine '''
from get_data_for_modelling import init_dataset_dict
from model_train_vali_pred import train, vali, predict
''' thank you time-series-library & the internet'''
from TimesNet import Model
from requisite.utils.tools import EarlyStopping, adjust_learning_rate, visual
from requisite.utils.metrics import metric


features = 'S'
batch_size = 10
data_dict,  enc_count = init_dataset_dict(features, batch_size)


if torch.cuda.is_available(): 
    device = "cuda:0" 
else: 
    device = "cpu" 



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
            'top_k':2,
            'des':'Exp',
            'itr':1,
            'num_kernels':6,
            'embed':'timeF',
            'freq':'h',
            'dropout':0.1,
            'output_attention':False,
            'batch_size':batch_size,
            'features':features,
            'n_heads':3,
            'activation':'relu',
            'device':device,
            'patience':3}
configs = SimpleNamespace(**configs)
model = Model(configs)

args = {'learning_rate':1e-5,
        'lradj':'type1'}
args = SimpleNamespace(**args)
learning_rate = 1e-5
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
print(model)

#%%


'''
past_results = pd.read_csv('.\iTransformer_202311101540.csv', index_col=0)
'''
losses_train, losses_val, losses_test, lr_ = [], [], [], []
train_epochs = 2
#%%


early_stopping = EarlyStopping(patience=configs.patience, verbose=True, model_name=configs.model)


for epoch in range(train_epochs):  
    print(f'=========== {configs.model} ===========')
    print(f'--------\nEpoch {epoch}\n--------')
    print('Learning Rate: {}'.format(optimizer.param_groups[0]['lr']), end = '\n\n')
    lr_.append(optimizer.param_groups[0]['lr'])
    
    losses_train_shop, losses_val_shop, losses_test_shop = [], [], []
    trainer_count = 0
    for shop in data_dict.keys():
        trainer_count += 1
        print(f'\t***** Shop {shop} *****')
        train_loss = train(configs = configs, 
                           dataloader = data_dict[shop]['train']['dataloader'], 
                           model = model, 
                           criterion = criterion,
                           optimizer = optimizer,
                           num_epochs=train_epochs, 
                           current_epoch=epoch, 
                           trainer_count=trainer_count)
        print(f'/\/\/\/\/\/\/\/\ Train Loss:  {train_loss}')
        
        val_loss = vali(configs = configs, 
                           dataloader = data_dict[shop]['val']['dataloader'], 
                           model = model, 
                           criterion = criterion)
        print(f'/\/\/\/\/\/\/\/\ Val Loss:    {val_loss}')
        
        test_loss = vali(configs = configs, 
                           dataloader = data_dict[shop]['test']['dataloader'], 
                           model = model, 
                           criterion = criterion)
        print(f'/\/\/\/\/\/\/\/\ Test Loss:   {test_loss}')
        
        
        losses_train_shop.append(train_loss)
        losses_val_shop.append(val_loss)
        losses_test_shop.append(test_loss)
        
        # end of the shop loop
        
    epoch_train_loss = np.mean(losses_train_shop)
    epoch_val_loss = np.mean(losses_val_shop)
    epoch_test_loss = np.mean(losses_test_shop)
    
    losses_train.append(epoch_train_loss)
    losses_val.append(epoch_val_loss)
    losses_test.append(epoch_test_loss)
    
    print("Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f}".format(epoch + 1,  epoch_train_loss, epoch_val_loss, epoch_test_loss))

  
    
    early_stopping(epoch_val_loss, model, '.\\saved_models\\')
    if early_stopping.early_stop:
        print("Early stopping")
        break
    
    adjust_learning_rate(optimizer, epoch + 1, args)
    
    
    
    
    


epoch_results = pd.DataFrame(columns=['Train','Test','Val','lr'])
epoch_results['Train'] = losses_train
epoch_results['Test'] = losses_test
epoch_results['Val'] = losses_val
epoch_results['lr'] = lr_
if 'past_results' in locals():
    epoch_results = pd.concat([past_results, epoch_results])
now = datetime.datetime.now().strftime('%Y%m%d%H%M')
epoch_results.to_csv(configs.model + '_' + now + '.csv')



'''

plotting here

'''
    
torch.save(model.state_dict(), '.\\saved_models\\' + configs.model + '_' + now + '.pth')

# saved_params = model.load_state_dict(torch.load('.\\saved_models\\myiTransformer.pth'))

val = data_dict['CSM']['val']
yhat, ytrue = predict(configs = configs, 
                   dataset = val['dataset'], 
                   model = model, 
                   index = len(val['dataset']) - 1)