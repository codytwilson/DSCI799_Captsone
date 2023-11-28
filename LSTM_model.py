# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 13:59:06 2023

@author: cwilson
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # d_model is the size of the linear input  
        
        self.num_layers = configs.e_layers
        self.hidden_units = configs.hidden_units
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        
        self.layer_norm1 = nn.LayerNorm(self.enc_in, elementwise_affine=True)
        
        self.lstm = nn.LSTM(
            input_size = self.enc_in,
            hidden_size = self.hidden_units,
            batch_first = True,
            num_layers = self.num_layers,
            bidirectional = True
        )
        
        dropout = nn.Dropout(configs.dropout)
        # normalize after dropout
        layer_norm2 = nn.LayerNorm( (self.hidden_units, ), elementwise_affine=True)
        # go from num vars to size d_model
        linear = nn.Linear(self.hidden_units, configs.d_model)
        # relu
        relu = nn.ReLU(configs.d_model)
        # predict
        projection = nn.Linear(configs.d_model, self.pred_len * self.enc_in, bias=True)
        
        self.projection_stack = nn.Sequential(
            dropout,
            layer_norm2,
            linear,
            relu,
            projection
            )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        
        # init = self.linear_init(x)
        
        batch_size = x_enc.shape[0]
        # print(f"x_enc.shape: {x_enc.shape}")
        # print(batch_size)
        h0 = torch.zeros(self.num_layers * 2,
                          batch_size,
                          self.hidden_units
                          ).requires_grad_()
        c0 = torch.zeros(self.num_layers * 2,
                          batch_size,
                          self.hidden_units
                          ).requires_grad_() 
        # print(f"h0.shape: {h0.shape}, c0.shape: {c0.shape}")
        x = self.layer_norm1(x_enc)
        # print(f"x.shape: {x.shape}")
        _, (hn, _) = self.lstm(x, (h0, c0))
        # print(f"hn.shape: {hn.shape}")
        
        
        
        out = self.projection_stack(hn[-1]).view(batch_size, -1, self.enc_in)
        '''
        
        this is the only change between the lstm models
        hn[0] is the hidden state of backwards direction 
        hn[-1] is the hidden state of forwards direction
        therefore this gets both 
        
        hn_concatenated = torch.cat((hn[0], hn[-1]), dim=0)

        out = self.projection_stack(hn_concatenated).view(batch_size, -1, self.enc_in)
        '''
        
        
        
        # print(f"out.shape: {out.shape}")
        # out_reshaped = out.reshape(x.shape[0], out.shape[0]//x.shape[0])
        # print(f"out_reshaped.shape: {out_reshaped.shape}")
        # print(out.shape, out_reshaped.shape)
        
        return out
    

