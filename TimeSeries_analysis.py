# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 08:53:33 2023

@author: codyt
"""

import pandas as pd

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
