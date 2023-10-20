# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 08:53:33 2023

@author: codyt
"""

import pandas as pd
import matplotlib.pyplot as plt
colors = {'CSM':'tab:blue','CSF':'tab:orange','FED':'tab:green'}


production_data = '.\\data\\output_production_data.csv'
timekeeping_data = '.\\data\\output_timekeeping.csv'

prod = pd.read_csv(production_data)
time = pd.read_csv(timekeeping_data)
time = time.rename(columns={'Unnamed: 0':'Timestamp'})
# time = time.set_index('Timestamp')
time_pivot = pd.melt(time, id_vars='Timestamp', value_vars=['CSF','CSM','FED'], var_name = 'Shop', value_name='Direct Hours')

df = pd.merge(prod, time_pivot, left_on=['shop','Timestamp'], right_on=['Shop','Timestamp'], how='outer', indicator=True)


df_right = df[df['_merge'] == 'right_only']
df_left = df[df['_merge'] == 'left_only']
df = df[df['_merge'] == 'both']


df_vars = ['Shop','Timestamp','Quantity','Weight','Earned Hours','Remaining lbs','Direct Hours']
df = df[df_vars]
df = df.rename(columns={'Earned Hours':'Worth','Direct Hours':'Manhours','Remaining lbs':'OustandingWeight'})

earliest = df[['Shop','Timestamp']].groupby('Shop').min()
latest = df[['Shop','Timestamp']].groupby('Shop').max()
df = df[(df['Timestamp'] >= earliest['Timestamp'].max()) & (df['Timestamp'] <= latest['Timestamp'].min())]
records_count = df.groupby('Shop').count()

def rolling_average_of_feature(feature, resampling_string):
    df_rolling = df.set_index('Timestamp')
    df_rolling.index = pd.to_datetime(df_rolling.index)
    df_rolling = df_rolling.groupby('Shop').resample(resampling_string).mean()
    df_rolling = df_rolling.reset_index()
    fig,axes = plt.subplots(nrows=len(colors.keys()), sharex=True, sharey=True)
    for k,shop in enumerate(colors.keys()):
        ax = axes[k]
        chunk = df_rolling[df_rolling['Shop'] == shop]
        ax.plot(chunk['Timestamp'], chunk[feature], color=colors[shop])
        ax.set_ylabel(shop)
        if k == 0:
            ax.set_title('Moving Average of ' + feature + ' at ' + resampling_string + ' interval')
    plt.xticks(rotation=45)
    
rolling_average_of_feature('Worth', '100H') 
rolling_average_of_feature('Manhours', '100H') 
