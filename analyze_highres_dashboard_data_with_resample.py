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
from sklearn.linear_model import LinearRegression


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
# remove_anyythin_in_this_hour = 4
# highres = highres[highres['Timestamp'].dt.hour != remove_anyythin_in_this_hour]



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
highres['weekday'] = highres['weekday'].replace(7,0)

# move the morningin night shift hours to the previous dau's hours
nightshift_backaDay = highres[highres['NighShiftHoursMorning']]['weekday'] - 1
# set the negatives to day 6 = saturday
nightshift_backaDay[nightshift_backaDay < 0] = 6
# now set those nightshift shifted ones into the main df
highres.loc[nightshift_backaDay.index, 'weekday'] = nightshift_backaDay



highres2 = highres.set_index('Timestamp')
hourly_highres = highres2.groupby('Shop').resample('H').max()
hourly_highres.index = hourly_highres.index.droplevel(0)
hourly_highres = hourly_highres.reset_index(drop=False)

def figure_StartOfWeek(row):
    if row['Timestamp'].dayofweek == 6 & row['Timestamp'].hour < 5:
        return row['Timestamp'].date()
    return (row['Timestamp'] - pd.DateOffset(days=row['Timestamp'].dayofweek + 1)).date()
hourly_highres['StartOfWeek'] = hourly_highres.apply(figure_StartOfWeek, axis=1)

def figure_weekday(row):
    if row['Timestamp'].hour < 5:
        weekday =  row['Timestamp'].weekday()
    else:
        weekday =  row['Timestamp'].weekday() + 1
    if weekday > 6:
        return 0
    else:
        return weekday
hourly_highres['weekday'] = hourly_highres.apply(figure_weekday, axis=1)
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





''' THanks chatGPT this didn't work for shit

def custom_resampler(array):
    if (array.index.hour < 5).any():
        return array.shift(-1).max()
    else:
        return array.max()


daily_highres = highres2[['Shop','Direct Hours','Earned Hours','weekday','StartOfWeek']].groupby('Shop').resample('D').apply(custom_resampler)
'''

highres2['TimestampShifted'] = highres2.index - datetime.timedelta(hours=4, minutes=59)
highres2 = highres2[~ (highres2['TimestampShifted'].dt.hour == 23) & (highres2['TimestampShifted'].dt.minute > 52)]
highres2['Timestamp'] = highres2.index
highres2 = highres2.set_index(highres2['TimestampShifted'], drop=False)


# x = highres2[(highres2['Shop'] == 'FED') & (highres2['StartOfWeek'] >= datetime.datetime(2023,9,10))]


daily_highres = highres2[['Shop','Direct Hours','Earned Hours','weekday','StartOfWeek']].groupby('Shop').resample('D', closed='right').max()
# this is for troubleshooting so you can see what the max value occurs at
# daily_highres = highres2[['Shop','Direct Hours','Earned Hours','weekday','StartOfWeek','Timestamp']].groupby('Shop').resample('D', closed='right').max()

highres2 = highres2.set_index('Timestamp')
# setting the index back 5 hours works like a charm!!!!!!!
# daily_highres = hourly_highres.set_index(hourly_highres.index + datetime.timedelta(hours=-3))[['Shop','Direct Hours','Earned Hours','weekday','StartOfWeek']].groupby('Shop').resample('D').max()
daily_highres.index = daily_highres.index.droplevel(0)
daily_highres.index.name = 'Timestamp'
daily_highres['Shop'] = daily_highres['Shop'].ffill()
''' NEED TO COUNT THE NUMBER OF NULL DAYS '''


CHECKER_TO_SEE_IF_THETIMESTAMP_TRANSLATION_IS_WORKING = daily_highres[daily_highres['StartOfWeek'] > daily_highres.index]
if CHECKER_TO_SEE_IF_THETIMESTAMP_TRANSLATION_IS_WORKING.shape[0]:
    print('need to look at CHECKER_TO_SEE_IF_THETIMESTAMP_TRANSLATION_IS_WORKING')

#%% Daily - this is where we perfrom checks on missing values


#%% DAILY -- INFILL MISSING VALUES

daily_highres = daily_highres.reset_index(drop=False)
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

def inter_group_interpolate(df, x_col, y_col):
    y_fit_col = y_col + ' Interpolated'
    df_isna = df[df[y_col].isna()].copy()
    df_noNa = df[~df[y_col].isna()].copy()
    x = df_noNa[x_col].to_numpy().reshape(-1,1)
    y = df_noNa[y_col].to_numpy()
    reg = LinearRegression(fit_intercept=True).fit(x,y)
    m = reg.coef_
    b = reg.intercept_
    df_isna[y_fit_col] = df_isna[x_col] * m + b
    if (df_isna[y_fit_col] < 0).any():
        reg = LinearRegression(fit_intercept=False).fit(x,y)
        m = reg.coef_
        b = reg.intercept_
        df_isna[y_fit_col] = df_isna[x_col] * m + b
        
        # sub_zero = df_isna[y_fit_col] < 0 
    # df_isna[sub_zero].copy()[y_fit_col] = 0
    df[y_fit_col] = df[y_col]
    df.loc[df_isna.index, y_fit_col] = df_isna[y_fit_col]
    return df

daily_highres['Direct Hours Interpolated'] = np.nan
daily_highres['Earned Hours Interpolated'] = np.nan
for shop in pd.unique(daily_highres['Shop']):
    chunk_shop = daily_highres[daily_highres['Shop'] == shop]
    for week in pd.unique(chunk_shop['StartOfWeek']):
        chunk = chunk_shop[chunk_shop['StartOfWeek'] == week].copy()
        chunk_interpolated = inter_group_interpolate(chunk, 'weekday', 'Direct Hours')
        chunk_interpolated = inter_group_interpolate(chunk_interpolated, 'weekday', 'Earned Hours')
        daily_highres.loc[chunk.index, ['Direct Hours Interpolated','Earned Hours Interpolated']] = chunk[['Direct Hours Interpolated','Earned Hours Interpolated']]


hourly_highres['Direct Hours Interpolated'] = np.nan
hourly_highres['Earned Hours Interpolated'] = np.nan
hourly_highres = hourly_highres.reset_index(drop=False)
for shop in pd.unique(hourly_highres['Shop']):
    chunk_shop = hourly_highres[hourly_highres['Shop'] == shop]
    for week in pd.unique(chunk_shop['StartOfWeek']):
        chunk_week = chunk_shop[chunk_shop['StartOfWeek'] == week].copy()
        chunk_week_index = chunk_week.index.copy()
        chunk_week = chunk_week.reset_index(drop=True)
        chunk_week['x'] = chunk_week.index
        chunk_interpolated = inter_group_interpolate(chunk_week, 'x','Direct Hours')
        chunk_interpolated = inter_group_interpolate(chunk_interpolated, 'x','Earned Hours')
        hourly_highres.loc[chunk_week.index, ['Direct Hours Interpolated','Earned Hours Interpolated']] = chunk_week[['Direct Hours Interpolated','Earned Hours Interpolated']]
        
        for weekday in pd.unique(chunk_week['weekday']):
            chunk =  chunk_week[chunk_week['weekday'] == weekday]
            chunk_index = chunk.index.copy()
            chunk = chunk.reset_index(drop=True)
            chunk['x'] = chunk.index
            chunk_interpolated = inter_group_interpolate(chunk, 'x', 'Direct Hours')

daily_highres = daily_highres.set_index('Timestamp')

# daily_highres['Direct Hours Interpolate'] = daily_highres['Direct Hours'].interpolate(method='time')

# daily_highres['Earned Hours Interpolated'] = daily_highres['Earned Hours'].interpolate(method='time')
#%% daily resample plots
colors = {'CSM':'tab:blue','CSF':'tab:orange','FED':'tab:green'}




'''
This is exact replica of fig9, except it cuts CSM to have the same start point as CSF/FED

I SHOULD ADD GRIDS ON X TO SHOW UP EVERY 7 DAYS
'''
fig11, (ax110, ax111) = plt.subplots(nrows=2, ncols=1)
for shop in colors.keys():
    chunk = daily_highres[daily_highres['Shop'] == shop]
    chunk = chunk[chunk['StartOfWeek'] >= datetime.datetime(2023,5,21)]
    x = chunk.index
    y2 = chunk['Direct Hours'].cumsum()
    ax111.scatter(x,y2, s=0.6, color=colors[shop])
    for week in pd.unique(chunk['StartOfWeek']):
        y = chunk[chunk['StartOfWeek'] == week] #/ chunk['Direct Hours']
        ax110.plot(y.index,y['Direct Hours Interpolated'], color=colors[shop])
    
ax111.legend(colors.keys(), markerscale=4)
ax110.set_title('Direct Hours by Shop - Daily Resampling')
ax110.set_xticks([])
ax110.set_ylabel('Weekly Hours')
ax111.set_ylabel('Cumulative')
fig11.autofmt_xdate(rotation=45)




# '''
# This is exact replica of fig9, except it cuts CSM to have the same start point as CSF/FED

# I SHOULD ADD GRIDS ON X TO SHOW UP EVERY 7 DAYS
# '''
# fig12, (ax120, ax121) = plt.subplots(nrows=2, ncols=1)
# for shop in colors.keys():
#     chunk = daily_highres[daily_highres['Shop'] == shop]
#     chunk = chunk[chunk['StartOfWeek'] >= datetime.datetime(2023,5,21)]
#     x = chunk.index
#     y2 = chunk['Earned Hours'].cumsum()
#     ax121.scatter(x,y2, s=0.6, color=colors[shop])
#     for week in pd.unique(chunk['StartOfWeek']):
#         y = chunk[chunk['StartOfWeek'] == week] #/ chunk['Direct Hours']
#         ax120.plot(y.index,y['Earned Hours Interpolated'], color=colors[shop])
    
# ax121.legend(colors.keys(), markerscale=4)
# ax120.set_title('Earned Hours by Shop - Daily Resampling')
# ax120.set_xticks([])
# ax120.set_ylabel('Weekly Hours')
# ax121.set_ylabel('Cumulative')
# fig12.autofmt_xdate(rotation=45)


#%% FInding out the ratio distributions of each weekday Contributing to Week

weekly_highres = highres2[['Shop','StartOfWeek','Earned Hours','Direct Hours']].groupby(['Shop','StartOfWeek']).max()
weekly_highres = weekly_highres.reset_index(drop=False)


daily_weekly = pd.merge(daily_highres.reset_index(), weekly_highres, 
                        left_on=['Shop','StartOfWeek'], 
                        right_on = ['Shop','StartOfWeek'],
                        how = 'left',
                        suffixes=('',' WeekMax'))
daily_weekly = daily_weekly.set_index('Timestamp')

daily_weekly['DirectHoursRatio'] = daily_weekly['Direct Hours'] / daily_weekly['Direct Hours WeekMax']
daily_weekly['DirectHoursInterpolatedRatio'] = daily_weekly['Direct Hours Interpolated'] / daily_weekly['Direct Hours WeekMax']


agg_stats = {'DirectHoursRatio':['mean','std','median','count'],'DirectHoursInterpolatedRatio':['mean','std','median','count']}

daily_weekly_shop_stats = daily_weekly.groupby(['Shop','weekday']).agg(agg_stats)
daily_weekly_stats = daily_weekly.groupby('weekday').agg(agg_stats)






#%% Finding out the ratio distributions of each Hour contributing to Day




hourly_highres['Timestamp'] = hourly_highres.index
hourly_highres['Year'] = hourly_highres['Timestamp'].dt.year
hourly_highres['Month'] = hourly_highres['Timestamp'].dt.month
hourly_highres['Day'] = hourly_highres['Timestamp'].dt.day
hourly_highres['Hour'] = hourly_highres['Timestamp'].dt.hour

daily_highres['Timestamp'] = daily_highres.index
daily_highres['Year'] = daily_highres['Timestamp'].dt.year
daily_highres['Month'] = daily_highres['Timestamp'].dt.month
daily_highres['Day'] = daily_highres['Timestamp'].dt.day

hourly_highres = pd.merge(hourly_highres, 
                          daily_highres[['Shop','Year','Month','Day','Direct Hours','Earned Hours']], 
                          how='left',
                          left_on=['Shop','Year','Month','Day'], 
                          right_on=['Shop','Year','Month','Day'],
                          suffixes=('',' DayMax'))

hourly_highres['DirectHoursRatio'] = hourly_highres['Direct Hours'] / hourly_highres['Direct Hours DayMax']
hourly_highres['DirectHoursInterpolatedRatio'] = hourly_highres['Earned Hours'] / hourly_highres['Earned Hours DayMax']
hourly_highres = hourly_highres.set_index('Timestamp')



# custom_order = list(np.arange(6,24,1)) + list(np.arange(0,6))
# fig10_data = []
# for hour in custom_order:
#     chunk = hourly_highres[hourly_highres.index.to_series().dt.hour == hour]
#     fig10_data.append(chunk['Direct Hours'])
# fig10, ax10 = plt.subplots()
# ax10.boxplot(fig10_data)
# ax10.set_xticks(ticks = np.arange(0, len(fig10_data))+1, labels=custom_order)
# ax10.set_title('Distribution of Direct Hours by Hour of Day')












