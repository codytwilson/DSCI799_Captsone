# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 08:53:33 2023

@author: codyt
"""

import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss

colors = {'CSM':'tab:blue','CSF':'tab:orange','FED':'tab:green'}
regressors = ['Quantity','Weight','Worth','Manhours','OustandingWeight']



records_count = df.groupby('Shop').count()

def resample_original_df(resampling_string, method='mean', by_shop=True):
    df_rolling = df.set_index('Timestamp')
    df_rolling.index = pd.to_datetime(df_rolling.index)
    if by_shop:
        df_rolling = df_rolling.groupby('Shop').resample(resampling_string)
    else:
        df_rolling = df_rolling.resample(resampling_string)
        
        
    if method == 'mean':
        df_rolling = df_rolling.mean()
    elif method == 'sum':
        df_rolling = df_rolling.sum() / 3
    elif method == 'std':
        df_rolling = df_rolling.var()
        
    df_rolling = df_rolling.reset_index()   
    return df_rolling



def resample_original_df_another_way(resampling_string, window=8, method='mean', by_shop=True):
    df_rolling = df.set_index('Timestamp')
    df_rolling.index = pd.to_datetime(df_rolling.index)
    if by_shop:
        df_rolling = df_rolling.groupby('Shop').resample(resampling_string).sum()
    else:
        df_rolling = df_rolling.resample(resampling_string).sum() / 3

    if method == 'mean':
        df_rolling = df_rolling.rolling(window).mean()
    elif method == 'sum':
        df_rolling = df_rolling.rolling(window).sum()
    elif method == 'std':
        df_rolling = df_rolling.rolling(window).std()
    elif method == 'exponential weighting' or method == 'ewm':
        df_rolling = df_rolling.ewm(com=800, min_periods=2).mean()
        # df_rolling = df_rolling.ewm(span=100).mean()
    df_rolling = df_rolling.reset_index()   
    return df_rolling



def rolling_average_of_feature_by_shop(feature, resampling_string):

    df_rolling = resample_original_df(resampling_string)
    
    fig,axes = plt.subplots(nrows=len(colors.keys()), sharex=True, sharey=True)
    for k,shop in enumerate(colors.keys()):
        ax = axes[k]
        chunk = df_rolling[df_rolling['Shop'] == shop]
        ax.plot(chunk['Timestamp'], chunk[feature], color=colors[shop])
        ax.set_ylabel(shop)
        if k == 0:
            ax.set_title('Moving Average of ' + feature + ' at ' + resampling_string + ' interval')
    plt.xticks(rotation=45)
    
rolling_average_of_feature_by_shop('Worth', '100H') 
rolling_average_of_feature_by_shop('Manhours', '100H') 

def rolling_average_of_feature_by_feature(features, resampling_string, shop='CSM', xlim=None, method='mean'):      
    df_rolling = resample_original_df(resampling_string, method=method)
    chunk = df_rolling[df_rolling['Shop'] == shop] 
    if xlim != None:
        chunk = chunk[chunk['Timestamp'] >= xlim[0]]
        chunk = chunk[chunk['Timestamp'] <= xlim[1]]
    fig,axes = plt.subplots(nrows=len(features), sharex=True, sharey=False, figsize=(6,5))
    for k,feature in enumerate(features):
        if len(features) > 1:
            ax = axes[k]
        else:
            ax = axes
        ax.plot(chunk['Timestamp'], chunk[feature], color=colors[shop])
        # ax.set_ylabel(feature)
        ax.set_title(feature)
        # if k == 0:
        #     ax.set_title('Moving Average of ' + feature + ' at ' + resampling_string + ' interval')
    plt.xticks(rotation=45)

rolling_average_of_feature_by_feature(regressors,'100H', shop='CSM')
rolling_average_of_feature_by_feature(regressors,'100H', shop='CSM', xlim=(pd.Timestamp('2021-05-01'),pd.Timestamp('2022-04-01')))
rolling_average_of_feature_by_feature(['Worth'],'240H', shop='CSF', method='mean')
rolling_average_of_feature_by_feature(regressors,'100H', shop='FED')



#%% stationarity testing

df_stationary = resample_original_df('48H')

adfuller_results = pd.DataFrame(columns=regressors, index=colors.keys())

for shop in colors.keys():
    for r in regressors:
        results = adfuller(df_stationary[df_stationary['Shop'] == shop][r])
        # the p-value is item 1
        adfuller_results.loc[shop,r] = results[1]

ad_fuller_to_plot_cols = list(itertools.product(colors.keys(), regressors))
ad_fuller_to_plot_cols = [i[0] +'_'+i[1] for i in ad_fuller_to_plot_cols]
ad_fuller_to_plot_index = np.arange(2,121,2)
ad_fuller_to_plot = pd.DataFrame(columns=ad_fuller_to_plot_cols, index=ad_fuller_to_plot_index)
for timespan in ad_fuller_to_plot_index:
    df_stationary_testing = resample_original_df(str(timespan) + 'H')
    for r in regressors:
        for shop in colors.keys():
            results = adfuller(df_stationary_testing[df_stationary_testing['Shop'] == shop][r])
            ad_fuller_to_plot.loc[timespan, shop + '_' + r] = results[1]


fig,axes = plt.subplots(nrows=len(regressors), sharex=True)
for k,r in enumerate(regressors):
    ax = axes[k]
    # ax.set_ylim(0,0.05)
    cols = [i for i in ad_fuller_to_plot.columns if r == i[4:len(r)+4]]
    for col in cols:
        y = ad_fuller_to_plot[col]
        shop = col[:3]
        ax.plot(ad_fuller_to_plot.index, y, color=colors[shop])
    ax.set_title(r)
ax.set_xlabel('Number of Hours for Moving Average')

''' YOU NEED TO REWRITE THIS BC IT IS COPY PASTED '''
def test_stationarity(df_input, feature, window=100):
    
    timeseries = df_input[feature]
    timeseries = timeseries[~timeseries.isna()]
    rolmean = timeseries.rolling(window).mean()
    rolstd = timeseries.rolling(window).std()
    orig = plt.plot(timeseries.index, timeseries, color='blue', label='Original')
    mean = plt.plot(timeseries.index, rolmean, color='red', label='Rolling Mean')
    std = plt.plot(timeseries.index, rolstd, color='k', label='Rolling STD')
    plt.legend(loc='best')
    plt.title(f'{feature} with window wize of {window} UNITS')
    plt.ylabel(feature)
    plt.show(block=False)
    
    print('Dickey Fuller Resulters')
    adf_results = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(adf_results[:4], index=['T-statistic','p-value','# Lags','# Obs'])
    print(dfoutput)






#%%


window = 720
df_stationary_testing = resample_original_df_another_way('1H', window=window, method='ewm', by_shop=True)
df_stationary_testing = df_stationary_testing.set_index('Timestamp')
df_stationary_testing = df_stationary_testing[df_stationary_testing['Shop'] == 'CSM']
# df_stationary_testing = np.log(df_stationary_testing.iloc[:,1:])
test_stationarity(df_stationary_testing, 'Worth', window=window)
test_stationarity(df_stationary_testing, 'Weight', window=window)
test_stationarity(df_stationary_testing,'Manhours', window=window)
test_stationarity(df_stationary_testing, 'Quantity', window=window)
# test_stationarity(df_stationary_testing[df_stationary_testing['Shop'] == 'CSM']['OustandingWeight'], window=window)

df_stationary_testing = resample_original_df_another_way('1H', window=window, method='ewm', by_shop=True)
df_stationary_testing = df_stationary_testing.set_index('Timestamp')
df_stationary_testing = df_stationary_testing[df_stationary_testing['Shop'] == 'FED']
# df_stationary_testing = np.log(df_stationary_testing.iloc[:,1:])
test_stationarity(df_stationary_testing, 'Worth', window=window)
test_stationarity(df_stationary_testing, 'Weight', window=window)
test_stationarity(df_stationary_testing,'Manhours', window=window)
test_stationarity(df_stationary_testing, 'Quantity', window=window)

df_stationary_testing = resample_original_df_another_way('1H', window=window, method='ewm', by_shop=True)
df_stationary_testing = df_stationary_testing.set_index('Timestamp')
df_stationary_testing = df_stationary_testing[df_stationary_testing['Shop'] == 'CSF']
# df_stationary_testing = np.log(df_stationary_testing.iloc[:,1:])
test_stationarity(df_stationary_testing, 'Worth', window=window)
test_stationarity(df_stationary_testing, 'Weight', window=window)
test_stationarity(df_stationary_testing,'Manhours', window=window)
test_stationarity(df_stationary_testing, 'Quantity', window=window)
