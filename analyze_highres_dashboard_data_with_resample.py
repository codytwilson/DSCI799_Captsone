# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 14:03:21 2023

@author: cwilson
"""


import pandas as pd
import numpy as np
import datetime
from retrieve_dfs_from_csvs import load_and_combine_archive_csv_to_df, cleanup_archive_df
import matplotlib.pyplot as plt

highres = load_and_combine_archive_csv_to_df()
highres = cleanup_archive_df(highres)
highres = highres.reset_index(drop=True)
highres = highres.sort_values('Timestamp')

start_date_csm = datetime.datetime(2023,3,26,0,0,0)
start_date_fed = datetime.datetime(2023,5,21,0,0,0)
start_date_csf = datetime.datetime(2023,5,21,0,0,0)
highres = highres[((highres['StartOfWeek'] >= start_date_csm) & (highres['Shop'] == 'CSM')) 
                  | ((highres['StartOfWeek'] >= start_date_fed) & (highres['Shop'] == 'FED'))
                  | ((highres['StartOfWeek'] >= start_date_csf) & (highres['Shop'] == 'CSF'))]
remove_anyythin_in_this_hour = 4
highres = highres[highres['Timestamp'].dt.hour != remove_anyythin_in_this_hour]



weekdays = {0:'Sunday',
            1:'Monday',
            2:'Tuesday',
            3:'Wednesday',
            4:'Thursday',
            5:'Friday',
            6:'Saturday'}

# # befoer 6 am is nightshift
highres['NighShiftHoursMorning'] = highres["Timestamp"].dt.hour < 5
# # after 5 pm is nightshift
highres['NighShiftHoursNight'] = highres['Timestamp'].dt.hour >= 17

# originally monday is zero but we want sunday is zero
highres['weekday'] = highres['Timestamp'].dt.weekday + 1
# set any 7 to zero
highres['weekday'][highres['weekday'] == 7] = 0

# move the morningin night shift hours to the previous dau's hours
nightshift_backaDay = highres[highres['NighShiftHoursMorning']]['weekday'] - 1
# set the negatives to day 6 = saturday
nightshift_backaDay[nightshift_backaDay < 0] = 6
# now set those nightshift shifted ones into the main df
highres.loc[nightshift_backaDay.index, 'weekday'] = nightshift_backaDay



highres2 = highres.set_index('Timestamp')
hourly_highres = highres2.groupby('Shop').resample('H').max()
hourly_highres.index = hourly_highres.index.droplevel(0)
''' NEED TO COUNT THE NUMBER OF NULL HOURS '''

fig8, (ax80, ax81) = plt.subplots(nrows=2, ncols=1)
for shop in ['CSM','CSF','FED']:
    chunk = hourly_highres[hourly_highres['Shop'] == shop]
    x = chunk.index
    y = chunk['Earned Hours'] #/ chunk['Direct Hours']
    y2 = chunk['Earned Hours'].cumsum()
    ax80.scatter(x,y, s=0.6)
    ax81.scatter(x,y2, s=0.6)
    
ax80.legend(['CSM','CSF','FED'])
ax80.set_title('Earned Hours by Shop - Hourly Resampling')


def custom_resampler(array):
    if (array.index.hour < 5).any():
        return array.shift(1).max()
    else:
        return array.max()


daily_highres = highres2[['Shop','Direct Hours','Earned Hours','weekday','StartOfWeek']].groupby('Shop').resample('D').apply(custom_resampler)
daily_highres = daily_highres.reset_index(drop=False)
''' NEED TO COUNT THE NUMBER OF NULL DAYS '''
# this is to infill the weekday column
null_weekday = daily_highres[daily_highres['weekday'].isna()].copy()
null_weekday['weekday'] = null_weekday['Timestamp'].dt.weekday + 1
null_weekday['weekday'] = null_weekday['weekday'].replace(7,0)
daily_highres.loc[null_weekday.index, 'weekday'] = null_weekday['weekday']
# this is to infill the start of week column
null_startofweek = daily_highres[daily_highres['StartOfWeek'].isna()].copy()
null_startofweek_weekday0 = null_startofweek[null_startofweek['weekday'] == 0].copy()
null_startofweek_weekday0['StartOfWeek'] = null_startofweek_weekday0['Timestamp']
daily_highres.loc[null_startofweek_weekday0.index, 'StartOfWeek'] = null_startofweek_weekday0['StartOfWeek']
daily_highres['StartOfWeek'] = daily_highres['StartOfWeek'].ffill()


daily_highres = daily_highres.set_index('Timestamp')
# daily_highres['Direct Hours Interpolate'] = daily_highres['Direct Hours'].interpolate(method='time')

daily_highres['Earned Hours Interpolated'] = daily_highres['Earned Hours'].interpolate(method='time')



fig9, (ax90, ax91) = plt.subplots(nrows=2, ncols=1)
for shop in ['CSM','CSF','FED']:
    chunk = daily_highres[daily_highres['Shop'] == shop]
    x = chunk.index
    y = chunk['Earned Hours'] #/ chunk['Direct Hours']
    y2 = chunk['Earned Hours'].cumsum()
    ax90.plot(x,y)
    ax91.scatter(x,y2, s=0.6)
    
ax90.legend(['CSM','CSF','FED'])
ax90.set_title('Earned Hours by Shop - Daily Resampling')
























