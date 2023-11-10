# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 08:25:51 2023

@author: codyt
"""

import pandas as pd


def create_ml_data(by_shop=False):
    
    production_data = '.\\data\\output_production_data.csv'
    timekeeping_data = '.\\data\\output_timekeeping.csv'
    
    prod = pd.read_csv(production_data)
    time = pd.read_csv(timekeeping_data)
    
    # rename appropriately
    time = time.rename(columns={'Unnamed: 0':'Timestamp'})
    # got to rearrange this one so each timestamp/shop has a record for column = 'Direct Hours'
    time_pivot = pd.melt(time, id_vars='Timestamp', value_vars=['CSF','CSM','FED'], var_name = 'Shop', value_name='Direct Hours')
    # join them up
    df = pd.merge(prod, time_pivot, left_on=['shop','Timestamp'], right_on=['Shop','Timestamp'], how='outer', indicator=True)
    # make sure its datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # just a look to see whats in one/not the other 
    df_right = df[df['_merge'] == 'right_only']
    df_left = df[df['_merge'] == 'left_only']
    # only keep records where data is in both dfs
    df = df[df['_merge'] == 'both']
    
    
    df_vars = ['Shop','Timestamp','Quantity','Weight','Earned Hours','Remaining lbs','Direct Hours']
    df = df[df_vars]
    df = df.rename(columns={'Earned Hours':'Worth','Direct Hours':'Manhours','Remaining lbs':'OustandingWeight'})
    
    earliest = df[['Shop','Timestamp']].groupby('Shop').min()
    latest = df[['Shop','Timestamp']].groupby('Shop').max()
    df = df[(df['Timestamp'] >= earliest['Timestamp'].max()) & (df['Timestamp'] <= latest['Timestamp'].min())]
    df = df[df['Timestamp'] >= pd.Timestamp('2021-01-10')]
    
    df['Weekday'] = df['Timestamp'].dt.weekday
    df['DayShift'] = df['Timestamp'].apply(lambda x: 1 if (x.hour >= 6 and x.hour <= 17) else 0)
    
    # weekdays = {0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}
    # df['Weekday'] = df['Weekday'].apply(lambda x: weekdays[x])
    # weekdays_oh = pd.get_dummies(df['Weekday'])
    # df = df.drop(columns=['Weekday'])
    # df = pd.concat([df, weekdays_oh], axis=1)

    df = df.rename(columns={'Timestamp':'date'})
    
    df_shop_oh = pd.get_dummies(df['Shop'])
    df = pd.concat([df, df_shop_oh], axis=1)
    df = df.drop(columns='Shop')
    
    
    
    if by_shop:
        
        for shop in df_shop_oh.columns:
            df_out = df[df[shop] == True]
            df_out = df_out
            df_out.to_csv('.\\data\\data_for_ml_' + str(shop) +'.csv')
    else:
        df.to_csv('.\\data\\data_for_ml.csv')
    