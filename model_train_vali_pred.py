# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 14:05:44 2023

@author: cwilson
"""

import time
import torch
import numpy as np
from torch import nn



def return_pooled_for_loss(tensor):

    tensor_for_pool = tensor.reshape(tensor.shape[0], 1, -1)
    goal_biweekly = nn.MaxPool1d(kernel_size = 24 * 7 * 2) # 40%
    goal_weekly = nn.MaxPool1d(kernel_size= 24 * 7) # 30%
    goal_daily = nn.MaxPool1d(kernel_size = 24) # 20%
    goal_4hour = nn.MaxPool1d(kernel_size = 4) # 10%
    
    tensor_biweekly = goal_biweekly(tensor_for_pool).reshape(tensor.shape[0], -1, 1)
    tensor_weekly = goal_weekly(tensor_for_pool).reshape(tensor.shape[0], -1, 1)
    tensor_daily = goal_daily(tensor_for_pool).reshape(tensor.shape[0], -1, 1)
    tensor_4hour = goal_4hour(tensor_for_pool).reshape(tensor.shape[0], -1, 1)
    
    return tensor_biweekly, tensor_weekly, tensor_daily, tensor_4hour
        










def vali(configs, model, dataloader, criterion):
    total_loss = []
    model.eval()
    vali_iter = 0
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(dataloader):
            # print(i, batch_x.shape, batch_y.shape, batch_x_mark.shape, batch_y_mark.shape)
            # break            
            vali_iter += 1
            
            batch_x = batch_x.float().to(configs.device)
            batch_y = batch_y.float()
    
            batch_x_mark = batch_x_mark.float().to(configs.device)
            batch_y_mark = batch_y_mark.float().to(configs.device)
    
            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -configs.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :configs.label_len, :], dec_inp], dim=1).float().to(configs.device)
            # encoder - decoder
       
            if configs.output_attention:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = 0
            outputs = outputs[:, -configs.pred_len:, f_dim:]
            batch_y = batch_y[:, -configs.pred_len:, f_dim:].to(configs.device)
    
            # pred = outputs.detach().cpu()
            # true = batch_y.detach().cpu()
            
            y_biweekly, y_weekly, y_daily, y_4hour = return_pooled_for_loss(batch_y)
            yhat_biweekly, yhat_weekly, yhat_daily, yhat_4hour = return_pooled_for_loss(outputs)

            loss_biweekly = criterion(yhat_biweekly, y_biweekly)
            loss_weekly = criterion(yhat_weekly, y_weekly)
            loss_daily = criterion(yhat_daily, y_daily)
            loss_4hour = criterion(yhat_4hour, y_4hour)
            # weight the lossses
            loss = loss_biweekly * 0.4 + loss_weekly * 0.3 + loss_daily * 0.2 + loss_4hour * 0.1

        
            
            # loss = criterion(pred, true)
            total_loss.append(loss)
            

    total_loss = np.average(total_loss)
    model.train()
    return total_loss

    
    
    



def train(configs, dataloader, model, criterion, optimizer, num_epochs, current_epoch, trainer_count):
    
    train_steps = len(dataloader)
    time_now = time.time()
    iter_count = 0
    train_loss = []

    model.train()
    epoch_time = time.time()
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(dataloader):
        # print(i, batch_x.shape, batch_y.shape, batch_x_mark.shape, batch_y_mark.shape)
        # break
        iter_count += 1
        optimizer.zero_grad()
        batch_x = batch_x.float().to(configs.device)

        batch_y = batch_y.float().to(configs.device)
        batch_x_mark = batch_x_mark.float().to(configs.device)
        batch_y_mark = batch_y_mark.float().to(configs.device)

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -configs.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :configs.label_len, :], dec_inp], dim=1).float().to(configs.device)

        if configs.output_attention:
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        # this is to get the target from the outputs - if there is only one regressor (target) then just get 0: instead of -1:
        f_dim = -1 if batch_x.shape[-1] > 1 else 0
        outputs = outputs[:, -configs.pred_len:, f_dim:]
        batch_y = batch_y[:, -configs.pred_len:, f_dim:].to(configs.device)
        
        
        
        y_biweekly, y_weekly, y_daily, y_4hour = return_pooled_for_loss(batch_y)
        yhat_biweekly, yhat_weekly, yhat_daily, yhat_4hour = return_pooled_for_loss(outputs)

        loss_biweekly = criterion(yhat_biweekly, y_biweekly)
        loss_weekly = criterion(yhat_weekly, y_weekly)
        loss_daily = criterion(yhat_daily, y_daily)
        loss_4hour = criterion(yhat_4hour, y_4hour)
        # weight the lossses
        loss = loss_biweekly * 0.4 + loss_weekly * 0.3 + loss_daily * 0.2 + loss_4hour * 0.1
        
        # loss = criterion(outputs, batch_y)
        train_loss.append(loss.item())
        

        if (i + 1) % 100 == 0:
            print("\titers: {0}, epoch: {1}-{2} | loss: {3:.7f}".format(i + 1, current_epoch + 1, trainer_count, loss.item()))
            speed = (time.time() - time_now) / iter_count
            left_time = speed * ((num_epochs - current_epoch) * train_steps - i)
            print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
            iter_count = 0
            time_now = time.time()         



        loss.backward()
        optimizer.step()

    print("Epoch: {}-{} cost time: {}".format(current_epoch + 1, trainer_count, time.time() - epoch_time))
    train_loss = np.average(train_loss)
    
    return train_loss
    

def predict(configs, dataset, model, index):
    def this_reshape(a):
        return a.reshape(1, a.shape[0], a.shape[1])
    batch_x = dataset[index][0]
    batch_y = dataset[index][1]
    batch_x_mark = dataset[index][2]
    batch_y_mark = dataset[index][3]
    
    # make it so its a batch size of one (B, R, C)
    batch_x = torch.Tensor(this_reshape(batch_x)).float().to(configs.device)
    batch_y = torch.Tensor(this_reshape(batch_y)).float().to(configs.device)
    batch_x_mark = torch.Tensor(this_reshape(batch_x_mark)).float().to(configs.device)
    batch_y_mark = torch.Tensor(this_reshape(batch_y_mark)).float().to(configs.device)
    
    
    model.eval()
    with torch.no_grad():

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -configs.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :configs.label_len, :], dec_inp], dim=1).float().to(configs.device)

        if configs.output_attention:
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)   
        
        true_scale = dataset.inverse_transform(outputs[0])
    
    target = -1 if true_scale.shape[-1] > 1 else 0
    pred_y = true_scale[:,target:]
    # get the batch_y in the correct scale
    batch_y = dataset.inverse_transform(batch_y[0,:,:])
    true_y = batch_y[:, target:]
    
    
    return pred_y, true_y