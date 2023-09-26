# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 16:24:15 2023

@author: codyt
"""


import pandas as pd
import numpy as np
import datetime
from retrieve_dfs_from_csvs import load_production_worksheet_csv_to_df, get_job_data_from_production_worksheet, get_timeline_data_from_production_worksheet



pw = load_production_worksheet_csv_to_df()
# job_data.shape[0] == time_data.shape[0]
job_data = get_job_data_from_production_worksheet(pw)
time_data = get_timeline_data_from_production_worksheet(pw)

ratios_hourly = pd.read_csv('.\\ratios_hourly.csv')
ratios_daily = pd.read_csv('.\\ratios_daily.csv')



# we need to join the shops together
time_data = time_data.groupby(level=0).sum()
# swap columns and rows
time_data = time_data.transpose()
# set the index to be datetime incase it does weird shit for some reason
time_data.index = pd.to_datetime(time_data.index)
# going to limit our study to 2020 + 
time_data = time_data[time_data.index > datetime.datetime(2020,1,1)]
''' TESTING PURPOSES '''
# time_data = time_data[time_data.index > datetime.datetime(2023,8,1)]
# time_data = time_data[time_data.index < datetime.datetime(2023,9,15)]
''' TESTING PURPOSES '''
# we need a bigger index b/c we are using date as start of the week
hourly_index = pd.date_range(start=time_data.index.min(), end=time_data.index.max() + pd.Timedelta(days=7), freq='H')
# the resample only goes until the last timestamp in time_data
# the first ffill fills in everything from that resample
# then we set the index to be hourly_index, which properly gives us the last week
#   ex: time_data ends on 9/3/2023. resample gives until 9/3/2023, but hourly_index goes until 9/10/2023 00:00
# the last ffill is to fill in that last week that got added via reindex(hourly_index)
resampled_time_data = time_data.resample('H', label='right').ffill().reindex(hourly_index).ffill()


# Step 2: Transform the weekly hours distribution
# Calculate the distribution for each hour






def randomize_according_to_distribution(series):
    randomizer = []
    for ts in series.index:
        
        weekday = ts.weekday() + 1
        if weekday == 7:
            weekday = 0
        weekday_mean = ratios_daily.loc[weekday, 'mean']
        weekday_std = ratios_daily.loc[weekday, 'std']
        hour = ts.hour
        hour_mean = ratios_hourly.loc[hour, 'mean']
        hour_std = ratios_hourly.loc[hour, 'std']
        random_ratio_day = np.random.normal(loc = weekday_mean, scale = weekday_std)
        random_ratio_hour = np.random.normal(loc = hour_mean, scale = hour_std)
        randomizer.append(max(0, random_ratio_day * random_ratio_hour))

    return series * randomizer



resampled_time_data2 = resampled_time_data.copy()
resampled_time_data2['CSM'] = randomize_according_to_distribution(resampled_time_data2['CSM'])
resampled_time_data2['CSF'] = randomize_according_to_distribution(resampled_time_data2['CSF'])
resampled_time_data2['FED'] = randomize_according_to_distribution(resampled_time_data2['FED'])























