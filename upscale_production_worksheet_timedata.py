# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 16:24:15 2023

@author: codyt
"""


import pandas as pd
import numpy as np
import datetime
from retrieve_dfs_from_csvs import load_production_worksheet_csv_to_df, get_job_data_from_production_worksheet, get_timeline_data_from_production_worksheet
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
colors = {'CSM':'tab:blue','CSF':'tab:orange','FED':'tab:green'}


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
# going to limit our study to 2021 + 
time_data = time_data[time_data.index > datetime.datetime(2020,1,1)]
''' TESTING PURPOSES '''
# time_data = time_data[time_data.index > datetime.datetime(2023,8,1)]
# time_data = time_data[time_data.index < datetime.datetime(2023,9,15)]
''' TESTING PURPOSES '''


# TODO: outliers
time_data = time_data[time_data.index > datetime.datetime(2021,1,1)]
time_data = time_data[time_data.index < datetime.datetime(2023,10,1)]
time_data_independent = time_data.melt()['value']
time_data_boxplots = []
for shop in time_data.columns:
    chunk = time_data[shop][time_data[shop] > 0]
    time_data_boxplots.append(chunk)
time_data_boxplots.append(time_data_independent[time_data_independent > 0])

time_data_description = time_data.replace(0,np.nan).describe()
# TODO: Missing Data
time_data_description.loc['count0',:] = (time_data == 0).sum()
time_data_description.loc['under500',:] = ((time_data < 500) & (time_data > 0)).sum()
time_data_description.loc['95%'] = time_data.apply(np.quantile, args=([0.95]), axis=0)
time_data_description.loc['99%'] = time_data.apply(np.quantile, args=([0.99]), axis=0)


fig,ax = plt.subplots()
ax.boxplot(time_data_boxplots)
ax.set_xticks(ticks=range(1,len(time_data_boxplots)+1), labels=list(time_data.columns) + ['Combined'])
ax.set_ylim((0,3500))
ax.set_ylabel('Hours Worked')
ax.set_title('Weekly Hours Worked by Shop')



# TODO: infill missing and outlier

# let outliers be 25% to 99%
# Let missing values be zero
infill_value = time_data_independent[time_data_independent > 0].median()
upper_bound = np.quantile(time_data_independent[time_data_independent > 0], 0.99)
lower_bound = np.quantile(time_data_independent[time_data_independent > 0], 0.06)
time_data_infilled = time_data.copy()
time_data_infilled[time_data_infilled == 0] = infill_value
time_data_infilled[time_data_infilled < lower_bound] = infill_value
time_data_infilled[time_data_infilled > upper_bound] = infill_value






#%%

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



# we need a bigger index b/c we are using date as start of the week
hourly_index = pd.date_range(start=time_data.index.min(), end=time_data.index.max() + pd.Timedelta(days=7), freq='H')

time_data['CSM'] = time_data['CSM'].apply(lambda x: int(str(int(x))[:4]) if len(str(int(x))) > 4 else int(x))
time_data['CSF'] = time_data['CSF'].apply(lambda x: int(str(int(x))[:4]) if len(str(int(x))) > 4 else int(x))
time_data['FED'] = time_data['FED'].apply(lambda x: int(str(int(x))[:4]) if len(str(int(x))) > 4 else int(x))

# the resample only goes until the last timestamp in time_data
# the first ffill fills in everything from that resample
# then we set the index to be hourly_index, which properly gives us the last week
#   ex: time_data ends on 9/3/2023. resample gives until 9/3/2023, but hourly_index goes until 9/10/2023 00:00
# the last ffill is to fill in that last week that got added via reindex(hourly_index)
resampled_time_data = time_data.resample('H', label='right').ffill().reindex(hourly_index).ffill()



resampled_time_data2 = resampled_time_data.copy()
resampled_time_data2['CSM'] = randomize_according_to_distribution(resampled_time_data2['CSM'])
resampled_time_data2['CSF'] = randomize_according_to_distribution(resampled_time_data2['CSF'])
resampled_time_data2['FED'] = randomize_according_to_distribution(resampled_time_data2['FED'])




# this is to show that the randomized values are close to the actual weekly

resampled_time_data2_back = resampled_time_data2.resample('W', label='left').sum()
shops = time_data.columns



# fig, axes = plt.subplots(ncols=1, nrows=len(shops), sharex=True, sharey=True)
for k,shop in enumerate(shops):
    # ax = axes[k]
    fig, ax = plt.subplots()
    ax.plot(time_data.index, time_data[shop], marker='.', color=colors[shop])
    ax.plot(resampled_time_data2_back.index, resampled_time_data2_back[shop], marker='*', linewidth=0.0, color='k')
    
    to_plot = abs((time_data - resampled_time_data2_back) / time_data)
    to_plot = to_plot.fillna(0)
    to_plot = to_plot.replace((np.inf, -np.inf), 0)
    # ax.plot(to_plot.index, to_plot[shop], color=colors[shop])
    # ax.boxplot(to_plot[shop])
    ax.set_ylim((0,4000))
    # ax.set_ylim((0.001,0.5))
    # ax.set_yscale('log')
    
    ax.set_title(shop + ' Real vs Resampled')
    date_strings = time_data.index.strftime('%Y-Q%q')
    ax.set_xlim(time_data.iloc[0].name, time_data.iloc[-1].name)
    ax.legend(['Real','Resampled'])
plt.xticks(rotation=45)
plt.tight_layout()





to_plot = 100 * abs((time_data - resampled_time_data2_back) / time_data)
to_plot = to_plot.fillna(0)
to_plot = to_plot.replace((np.inf, -np.inf), 0)
boxplot_list = []
for shop in shops:
    
    boxplot_list.append(to_plot[shop].values.tolist())
    # ax.plot(to_plot.index, to_plot[shop], color=colors[shop])

boxplot_list.append(pd.melt(to_plot)['value'].values.tolist())

fig,ax = plt.subplots()
ax.boxplot(boxplot_list)
ax.set_ylim((0,10))
ax.set_xticks(ticks=range(1,len(shops)+2), labels=list(shops)+['Total'])
ax.set_ylabel('Absolute Percentage')
ax.set_title('Percentage Difference Between Original\nand Upscaled Weekly Hours')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())








#%% Output the hourly data source
def output_time_data():
    resampled_time_data2.to_csv('.\\data\\output_timekeeping.csv', index=True)
