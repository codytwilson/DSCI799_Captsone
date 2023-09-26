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

# this is the equivalent of transposing b/c it has a multiindex with 3 levels - Shop, Job, Sequence
time_data = time_data.stack().unstack([0,1,2])
time_data.index = pd.to_datetime(time_data.index)

time_data = time_data[time_data.index > datetime.datetime(2023,9,1)]
time_data = time_data[time_data.index < datetime.datetime(2023,10,1)]

hourly_index = pd.date_range(start=time_data.index.min(), end=time_data.index.max() + pd.Timedelta(days=7), freq='H')
resampled_time_data = time_data.resample('H').ffill().reindex(hourly_index)

# Step 2: Transform the weekly hours distribution
# Calculate the distribution for each hour





distribution = []
for index, row in resampled_time_data.iterrows():
    weekday = index.weekday() + 1
    if weekday == 7:
        weekday = 0
    weekday_mean = ratios_daily.loc[weekday, 'mean']
    weekday_std = ratios_daily.loc[weekday, 'std']
    hour = index.hour
    hour_mean = ratios_hourly.loc[hour, 'mean']
    hour_std = ratios_hourly.loc[hour, 'std']
    
    
    if weekday == 0:
        week_of = index.date()
    else:
        week_of = index.date() - datetime.timedelta(days=weekday)
    
    weeks_total = time_data[time_data.index == pd.Timestamp(week_of)]
    
    # loc = mean, scale = std
    random_day = np.random.normal(loc = weekday_mean, scale = weekday_std)
    random_hour = np.random.normal(loc = hour_mean, scale = hour_std)
    
    random_row = weeks_total * random_day * random_hour
    
    # Generate a random sample based on the mean and std
    # random_sample = np.random.normal(loc=weekday_mean * hour_mean, scale=weekday_std * hour_std)
    
    # Ensure the value is non-negative
    random_row.index = index

    resampled_time_data

resampled_time_data['transformed_hours'] = distribution

# Resulting DataFrame with hourly datetime index and transformed hours
print(resampled_time_data)
