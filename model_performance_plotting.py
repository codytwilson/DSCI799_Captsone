# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 18:36:57 2023

@author: codyt
"""

import matplotlib.pyplot as plt
import pandas as pd


deepleaerning_csvs = ['iTransformer_202311121024.csv','PatchTST_202311112005.csv',
                      'TimesNet_202311171325.csv','LSTM_202311271835.csv']

csv_dict = {}
for csv in deepleaerning_csvs:
    model_name = csv.split('_')[0]
    csv_dict[model_name] = pd.read_csv('.\\' + csv)


for j in ['Train','Test','Val']:
    plt.title(j + ' Epoch Performance')
    
    for i in csv_dict.keys():
        df = csv_dict[i].iloc[:10]
        plt.plot(df[j], label=i)
    
    plt.legend()
    plt.ylabel('MSE')
    plt.xlabel('Training Epoch')
    plt.show()
    
#%%

arima_csv = ''


