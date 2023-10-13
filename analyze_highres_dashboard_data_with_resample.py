# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 14:03:21 2023

@author: cwilson
"""

colors = {'CSM':'tab:blue','CSF':'tab:orange','FED':'tab:green'}

import pandas as pd
import numpy as np
import datetime
from retrieve_dfs_from_csvs import load_and_combine_archive_csv_to_df, cleanup_archive_df
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind

weekdays = {0:'Sunday',
            1:'Monday',
            2:'Tuesday',
            3:'Wednesday',
            4:'Thursday',
            5:'Friday',
            6:'Saturday'}

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



#%% checking missing records on 15 minute interval
highres3 = highres.copy()
expected_timestamps = pd.date_range(datetime.datetime(2023,5,21,0,0,0), highres3['Timestamp'].max(), freq='15T')
highres3 = highres3.set_index('Timestamp')
highres3 = highres3[['Shop','Direct Hours']].groupby('Shop').resample('15T').count()

highres3 = highres3.drop(columns='Shop')
highres3 = highres3.reset_index()
new_highres3 = pd.DataFrame(columns=highres3.columns)
for shop in pd.unique(highres3['Shop']):
    chunk = highres3[highres3['Shop'] == shop]
    chunk = chunk.reset_index(drop=True)
    chunk = chunk.set_index('Timestamp')
    chunk = chunk.reindex(expected_timestamps)
    chunk = chunk.reset_index(drop=False)
    chunk = chunk.rename(columns={'index':'Timestamp'})
    chunk['Shop'] = shop
    chunk = chunk.fillna(0)
    new_highres3 = pd.concat([new_highres3, chunk])
    
highres3 = new_highres3.copy()   
highres3['Direct Hours'] = pd.to_numeric(highres3['Direct Hours'])
    
highres3['1'] = highres3['Direct Hours'] > 0
fig11, ax111 = plt.subplots(nrows=1, ncols=1)
for shop in colors.keys():
    chunk = highres3[highres3['Shop'] == shop]
    chunk = chunk[chunk['Timestamp'] >= datetime.datetime(2023,5,21,0,0,0)]
    x = chunk['Timestamp']
    y2 = chunk['1'].cumsum()
    ax111.plot(x,y2, color=colors[shop])
    # ax111.scatter(x,y2, s=0.6, color=colors[shop])

chunk['2'] = 1
ax111.plot(x, chunk['2'].cumsum(), color='k')
  
ax111.legend(list(colors.keys()) + ['No Misses'], markerscale=4)
ax111.set_ylabel('Number of Records')
fig11.suptitle('15 Minute Interval Records Check')
fig11.autofmt_xdate(rotation=45)


expected_number_records = expected_timestamps.shape[0]
number_records_seen = highres3.groupby('Shop').sum()['Direct Hours'].to_frame()
number_records_seen['Ratio Missed'] = (expected_number_records - number_records_seen) / expected_number_records




#%% checking how many hours are missing when we go to hourly


highres4 = highres.copy()
expected_timestamps = pd.date_range(datetime.datetime(2023,5,21,0,0,0), highres4['Timestamp'].max(), freq='H')
highres4 = highres4.set_index('Timestamp')
highres4 = highres4[['Shop','Direct Hours']].groupby('Shop').resample('H').count()

highres4 = highres4.drop(columns='Shop')
highres4 = highres4.reset_index()
new_highres4 = pd.DataFrame(columns=highres4.columns)
for shop in pd.unique(highres4['Shop']):
    chunk = highres4[highres4['Shop'] == shop]
    chunk = chunk.reset_index(drop=True)
    chunk = chunk.set_index('Timestamp')
    chunk = chunk.reindex(expected_timestamps)
    chunk = chunk.reset_index(drop=False)
    chunk = chunk.rename(columns={'index':'Timestamp'})
    chunk['Shop'] = shop
    chunk = chunk.fillna(0)
    new_highres4 = pd.concat([new_highres4, chunk])
    
highres4 = new_highres4.copy()   
highres4['Direct Hours'] = pd.to_numeric(highres4['Direct Hours'])


expected_number_records = expected_timestamps.shape[0]
highres4['HasRecord'] = highres4['Direct Hours'] > 0
number_records_seen = highres4.groupby('Shop').sum()['HasRecord'].to_frame()
number_records_seen['Ratio Missed'] = (expected_number_records - number_records_seen) / expected_number_records






#%%










# remove_anyythin_in_this_hour = 4
# highres = highres[highres['Timestamp'].dt.hour != remove_anyythin_in_this_hour]


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

hourly_highres = highres2[['Shop','StartOfWeek','Direct Hours','Earned Hours']].groupby('Shop').resample('H', label='right').max()
hourly_highres = hourly_highres.drop(columns=['Shop'])
hourly_highres = hourly_highres.reset_index(drop=False)



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

def figure_StartOfWeek(row):
    # sunday and after 5:00
    if row['Timestamp'].dayofweek == 6 and row['Timestamp'].hour >= 5:
        return row['Timestamp'].date()
    # sunday and before 5:00 go to last week
    if row['Timestamp'].dayofweek == 6 and row['Timestamp'].hour < 5:
        return (row['Timestamp'] - pd.DateOffset(days=7)).date()
    # else go to nearest sunday
    return (row['Timestamp'] - pd.DateOffset(days=row['Timestamp'].dayofweek + 1)).date()
hourly_highres['StartOfWeek'] = hourly_highres.apply(figure_StartOfWeek, axis=1)


hourly_highres = hourly_highres.set_index('Timestamp', drop=True)
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
ax80.set_xticks([])



# shit the timestamp back 4 hours 59 minutes
highres2['TimestampShifted'] = highres2.index - datetime.timedelta(hours=4, minutes=59)
# 
# highres2 = highres2[~ ((highres2['TimestampShifted'].dt.hour == 23) & (highres2['TimestampShifted'].dt.minute > 52))]
highres2['Timestamp'] = highres2.index
# we set the index to be shifted b/c once we resample, it will maintain the corrrect day
highres2 = highres2.set_index(highres2['TimestampShifted'], drop=False)
# get the start of week 
highres2['StartOfWeek'] = highres2.apply(figure_StartOfWeek, axis=1)


daily_highres = highres2[['Shop','Direct Hours','Earned Hours','weekday','StartOfWeek','Timestamp']].groupby('Shop').resample('D', closed='right').max()
# this is for troubleshooting so you can see what the max value occurs at
# daily_highres = highres2[['Shop','Direct Hours','Earned Hours','weekday','StartOfWeek','Timestamp']].groupby('Shop').resample('D', closed='right').max()

# set the index back to timestamp
highres2 = highres2.set_index('Timestamp')

# get rid of the shop column
daily_highres = daily_highres.drop(columns=['Shop'])
# reset the index in total so that timestampshifted and shop go into the df
daily_highres = daily_highres.reset_index()
# set the timetsamp shifted back into the index - at this point it is no longer shifted, but rather it is the date that the records belong to 
daily_highres = daily_highres.set_index('TimestampShifted')
# set the name back to timestamp
daily_highres.index.name = 'Timestamp'


''' NEED TO COUNT THE NUMBER OF NULL DAYS '''


CHECKER_TO_SEE_IF_THETIMESTAMP_TRANSLATION_IS_WORKING = daily_highres[daily_highres['StartOfWeek'] > daily_highres.index]
if CHECKER_TO_SEE_IF_THETIMESTAMP_TRANSLATION_IS_WORKING.shape[0]:
    CHECKER_TO_SEE_IF_THETIMESTAMP_TRANSLATION_IS_WORKING
    print('need to look at CHECKER_TO_SEE_IF_THETIMESTAMP_TRANSLATION_IS_WORKING')

daily_highres = daily_highres.drop(columns=['Timestamp'])
#%% Daily - this is where we perfrom checks on missing values


#%% DAILY -- INFILL MISSING VALUES

daily_highres = daily_highres.reset_index(drop=False)
# fix the weekday
daily_highres['weekday'] = daily_highres['Timestamp'].dt.weekday + 1
daily_highres['weekday'] = daily_highres['weekday'].replace(7,0)
# fix the startofweek

def figure_StartOfWeek_daily(row):
      # sunday and after 5:00
      if row['Timestamp'].dayofweek == 6:
          return row['Timestamp'].date()
      else:
          return (row['Timestamp'] - pd.DateOffset(days=row['Timestamp'].dayofweek + 1)).date()
daily_highres['StartOfWeek'] = daily_highres.apply(figure_StartOfWeek_daily, axis=1)
    

def inter_group_interpolate(df, x_col, y_col):
    y_fit_col = y_col + ' Interpolated'
    df_isna = df[df[y_col].isna()].copy()
    df_noNa = df[~df[y_col].isna()].copy()
    if df_isna.shape[0] == df.shape[0]:
        print('bonk')
        df[y_fit_col] = df[y_col]
        return df
    x = df_noNa[x_col].to_numpy().reshape(-1,1)
    y = df_noNa[y_col].to_numpy()
    reg = LinearRegression(fit_intercept=True).fit(x,y)
    m = reg.coef_[0]
    b = reg.intercept_
    df_isna[y_fit_col] = df_isna[x_col] * m + b
    if (df_isna[y_fit_col] < 0).any():
        reg = LinearRegression(fit_intercept=False).fit(x,y)
        m = reg.coef_
        b = reg.intercept_
        df_isna[y_fit_col] = df_isna[x_col] * m + b
        
    
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
        daily_highres.loc[chunk_interpolated.index, ['Direct Hours Interpolated','Earned Hours Interpolated']] = chunk_interpolated[['Direct Hours Interpolated','Earned Hours Interpolated']]


hourly_highres['Direct Hours Interpolated'] = np.nan
hourly_highres['Earned Hours Interpolated'] = np.nan
hourly_highres = hourly_highres.reset_index(drop=False)
for shop in pd.unique(hourly_highres['Shop']):
    chunk_shop = hourly_highres[hourly_highres['Shop'] == shop].copy()
    for week in pd.unique(chunk_shop['StartOfWeek']):
        chunk = chunk_shop[chunk_shop['StartOfWeek'] == week].copy()
        chunk_index = chunk.index.copy()
        chunk = chunk.reset_index(drop=True)
        chunk['x'] = chunk.index
        chunk_interpolated = inter_group_interpolate(chunk, 'x','Direct Hours')
        chunk_interpolated = inter_group_interpolate(chunk_interpolated, 'x','Earned Hours')
        chunk_interpolated.index = chunk_index
        hourly_highres.loc[chunk_interpolated.index, ['Direct Hours Interpolated','Earned Hours Interpolated']] = chunk_interpolated[['Direct Hours Interpolated','Earned Hours Interpolated']]
        

daily_highres = daily_highres.set_index('Timestamp')
hourly_highres = hourly_highres.set_index('Timestamp')
# daily_highres['Direct Hours Interpolate'] = daily_highres['Direct Hours'].interpolate(method='time')

# daily_highres['Earned Hours Interpolated'] = daily_highres['Earned Hours'].interpolate(method='time')
#%% daily resample plots



fig11, (ax110, ax111) = plt.subplots(nrows=2, ncols=1)
for shop in colors.keys():
    chunk = daily_highres[daily_highres['Shop'] == shop]
    chunk = chunk[chunk['StartOfWeek'] >= datetime.date(2023,5,21)]
    x = chunk.index
    y2 = chunk['Direct Hours'].cumsum()
    # ax111.plot(x,y2, color=colors[shop])
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
















hourly_highres['1'] = ~ hourly_highres['Direct Hours'].isna()
fig11, ax111 = plt.subplots(nrows=1, ncols=1)
for shop in colors.keys():
    chunk = hourly_highres[hourly_highres['Shop'] == shop]
    chunk = chunk[chunk['StartOfWeek'] >= datetime.date(2023,5,21)]
    x = chunk.index
    y2 = chunk['1'].cumsum()
    ax111.plot(x,y2, color=colors[shop])
    # ax111.scatter(x,y2, s=0.6, color=colors[shop])

chunk['2'] = 1
ax111.plot(x, chunk['2'].cumsum(), color='k')
  
ax111.legend(list(colors.keys()) + ['No Misses'], markerscale=4)
ax111.set_ylabel('Number of Records')
fig11.suptitle('Hourly Resampled Records Check')
fig11.autofmt_xdate(rotation=45)



'''
This is exact replica of gif11, except it is Earned Hours

I SHOULD ADD GRIDS ON X TO SHOW UP EVERY 7 DAYS
'''
fig12, (ax120, ax121) = plt.subplots(nrows=2, ncols=1)
for shop in colors.keys():
    chunk = daily_highres[daily_highres['Shop'] == shop]
    chunk = chunk[chunk['StartOfWeek'] >= datetime.date(2023,5,21)]
    x = chunk.index
    y2 = chunk['Earned Hours'].cumsum()
    ax121.scatter(x,y2, s=0.6, color=colors[shop])
    for week in pd.unique(chunk['StartOfWeek']):
        y = chunk[chunk['StartOfWeek'] == week] #/ chunk['Direct Hours']
        ax120.plot(y.index,y['Earned Hours Interpolated'], color=colors[shop])
    
ax121.legend(colors.keys(), markerscale=4)
ax120.set_title('Earned Hours by Shop - Daily Resampling')
ax120.set_xticks([])
ax120.set_ylabel('Weekly Hours')
ax121.set_ylabel('Cumulative')
fig12.autofmt_xdate(rotation=45)




# '''
# This is exact replica of gif11, except it is efficiency

# I SHOULD ADD GRIDS ON X TO SHOW UP EVERY 7 DAYS
# '''
# fig13, (ax130, ax131, ax132) = plt.subplots(nrows=3, ncols=1)
# for shop in colors.keys():
#     chunk = daily_highres[daily_highres['Shop'] == shop]
#     chunk = chunk[chunk['StartOfWeek'] >= datetime.date(2023,5,21)]
    
    
#     for week in pd.unique(chunk['StartOfWeek']):
#         y = chunk[chunk['StartOfWeek'] == week] #/ chunk['Direct Hours']
#         if shop == 'CSM':
#             ax130.plot(y.index, y['Earned Hours Interpolated'] / y['Direct Hours Interpolated'], color=colors[shop])
#         elif shop == 'CSF':
#             ax131.plot(y.index, y['Earned Hours Interpolated'] / y['Direct Hours Interpolated'], color=colors[shop])
#         else:
#             ax132.plot(y.index, y['Earned Hours Interpolated'] / y['Direct Hours Interpolated'], color=colors[shop])
    
# # ax130.legend(colors.keys(), markerscale=4)
# ax130.set_title('"Efficiency" = Earned Hours / Direct Hours (Daily Resampling)')
# ax130.set_xticks([])
# ax131.set_xticks([])
# ax130.set_ylim([0,1.5])
# ax131.set_ylim([0,1.5])
# ax132.set_ylim([0,1.5])
# fig13.autofmt_xdate(rotation=45)







#%% FInding out the ratio distributions of each weekday Contributing to Week


daily_highres[['Direct Hours Day',
               'Earned Hours Day',
               'Direct Hours Interpolated Day',
               'Earned Hours Interpolated Day']] = daily_highres.groupby(['Shop','StartOfWeek']).diff()[['Direct Hours','Earned Hours','Direct Hours Interpolated','Earned Hours Interpolated']]

daily_highres['Direct Hours Day'] = daily_highres['Direct Hours Day'].fillna(daily_highres['Direct Hours'])
daily_highres['Earned Hours Day'] = daily_highres['Earned Hours Day'].fillna(daily_highres['Earned Hours'])
daily_highres['Direct Hours Interpolated Day'] = daily_highres['Direct Hours Interpolated Day'].fillna(daily_highres['Direct Hours Interpolated'])
daily_highres['Earned Hours Interpolated Day'] = daily_highres['Earned Hours Interpolated Day'].fillna(daily_highres['Earned Hours Interpolated'])




weekly_highres = highres2[['Shop','StartOfWeek','Earned Hours','Direct Hours']].groupby(['Shop','StartOfWeek']).max()
weekly_highres = weekly_highres.reset_index(drop=False)


daily_highres = pd.merge(daily_highres.reset_index(), weekly_highres, 
                        left_on=['Shop','StartOfWeek'], 
                        right_on = ['Shop','StartOfWeek'],
                        how = 'left',
                        suffixes=('',' WeekMax'))
daily_highres = daily_highres.set_index('Timestamp')

daily_highres['DirectHoursRatio'] = daily_highres['Direct Hours Day'] / daily_highres['Direct Hours WeekMax']
daily_highres['DirectHoursInterpolatedRatio'] = daily_highres['Direct Hours Interpolated Day'] / daily_highres['Direct Hours WeekMax']


agg_stats = {'DirectHoursRatio':['mean','std','median','count'],'DirectHoursInterpolatedRatio':['mean','std','median','count']}

daily_highres_shop_stats = daily_highres.groupby(['Shop','weekday']).agg(agg_stats)
daily_highres_stats = daily_highres.groupby('weekday').agg(agg_stats)


fig,axes = plt.subplots(ncols=2, nrows=len(weekdays), sharey=True, figsize=(5.5,6))
for k in weekdays.keys():
    
    ax = axes[k,0]
    ax1 = axes[k,1]
    
    chunk = daily_highres[daily_highres['weekday'] == k]
    ax.hist(chunk['DirectHoursRatio'], bins=np.arange(0,1,0.05))
    ax1.hist(chunk['DirectHoursInterpolatedRatio'], bins=np.arange(0,1,0.05))
    ax.set_ylabel(weekdays[k][:3])
    if k == list(weekdays.keys())[-1]:
        ax.set_xlim((0,0.5))
        ax1.set_xlim((0,0.5))
        ax.set_xlabel('Ratio of Weekly Hours')
        ax1.set_xlabel('Ratio of Weekly Hours')
    else:
        ax.set_xticks([])
        ax1.set_xticks([])
        
    if k == 0:
        ax.set_title('(No Infill)')
        ax1.set_title('(Linear Interpolation)')
    # ax.set_ylim((0,25))
fig.suptitle('Daily Histograms of Ratio of Weekly Hours Worked')



''' Daily Hours '''
fig14, ax14 = plt.subplots()
fig14_dataset = []
for day in np.arange(0,7):
    chunk = daily_highres[daily_highres['weekday'] == day]
    chunk = chunk[chunk['Direct Hours Interpolated Day'] > 0]
    fig14_dataset.append(chunk['Direct Hours Interpolated Day'])

ax14.boxplot(fig14_dataset)
ax14.set_xticks(ticks = [i+1 for i in weekdays.keys()], labels=[i[:3] for i in weekdays.values()])
ax14.set_title('Hours Worked by Day of Week')
ax14.set_ylabel('Number of Hours')


fig15, ax15 = plt.subplots()
fig15_dataset = []
ax15.violinplot(fig14_dataset)
ax15.set_xticks(ticks = [i+1 for i in weekdays.keys()], labels=[i[:3] for i in weekdays.values()])
ax15.set_title('Hours Worked by Day of Week')
ax15.set_ylabel('Number of Hours')





''' RATIO Daily/Weekly NO INFILL'''
fig16, ax16 = plt.subplots()
fig16_dataset = []
for day in np.arange(0,7):
    chunk = daily_highres[daily_highres['weekday'] == day]
    chunk = chunk[chunk['DirectHoursRatio'] > 0]
    
    fig16_dataset.append(chunk['DirectHoursRatio'])

ax16.boxplot(fig16_dataset)
ax16.set_xticks(ticks = [i+1 for i in weekdays.keys()], labels=[i[:3] for i in weekdays.values()])
ax16.set_title('Ratio Of Weekly Hours Worked By Day')
ax16.set_ylabel('Ratio of Day Hours / Week Hours')
ax16.set_ylim((0,0.5))


fig17, ax17 = plt.subplots()
fig17_dataset = []
ax17.violinplot(fig16_dataset)
ax17.set_xticks(ticks = [i+1 for i in weekdays.keys()], labels=[i[:3] for i in weekdays.values()])
ax17.set_title('Ratio Of Weekly Hours Worked By Day')
ax17.set_ylabel('Ratio of Day Hours / Week Hours')



''' RATIO Daily/Weekly w/ linear interpolation infill'''
fig16, ax16 = plt.subplots()
fig16_dataset = []
for day in np.arange(0,7):
    chunk = daily_highres[daily_highres['weekday'] == day]
    chunk = chunk[chunk['DirectHoursInterpolatedRatio'] > 0]
    # chunk = chunk[chunk['DirectHoursInterpolatedRatio'] < 0.5]
    fig16_dataset.append(chunk['DirectHoursInterpolatedRatio'])

ax16.boxplot(fig16_dataset)
ax16.set_xticks(ticks = [i+1 for i in weekdays.keys()], labels=[i[:3] for i in weekdays.values()])
ax16.set_title('Ratio Of Weekly Hours Worked By Day\n(Infilled Missing Values via Linear Interpolation)')
ax16.set_ylabel('Ratio of Day Hours / Week Hours')
ax16.set_ylim((0,0.5))


fig17, ax17 = plt.subplots()
fig17_dataset = []
ax17.violinplot(fig16_dataset)
ax17.set_xticks(ticks = [i+1 for i in weekdays.keys()], labels=[i[:3] for i in weekdays.values()])
ax17.set_title('Ratio Of Weekly Hours Worked By Day\n(Infilled Missing Values via Linear Interpolation)')
ax17.set_ylabel('Ratio of Day Hours / Week Hours')



#%% Finding out the ratio distributions of each Hour contributing to Day


hourly_highres[['Direct Hours Hour',
               'Earned Hours Hour',
               'Direct Hours Interpolated Hour',
               'Earned Hours Interpolated Hour']] = hourly_highres.groupby(['Shop','StartOfWeek']).diff()[['Direct Hours','Earned Hours','Direct Hours Interpolated','Earned Hours Interpolated']]
hourly_highres['Direct Hours Interpolated Hour'] = hourly_highres['Direct Hours Interpolated Hour'].fillna(hourly_highres['Direct Hours Interpolated'])
hourly_highres['Earned Hours Interpolated Hour'] = hourly_highres['Earned Hours Interpolated Hour'].fillna(hourly_highres['Earned Hours Interpolated'])


# hourly_highres['Direct Hours Interpolated Hour PCTCHANGE'] = hourly_highres[['Shop','StartOfWeek','Direct Hours Interpolated Hour']].groupby(['Shop','StartOfWeek']).pct_change()['Direct Hours Interpolated Hour'].replace([np.inf, -np.inf], np.nan)
# # hourly_highres['Direct Hours Hour'] = hourly_highres['Direct Hours Hour'].fillna(hourly_highres['Direct Hours'])
# # hourly_highres['Earned Hours Hour'] = hourly_highres['Earned Hours Hour'].fillna(hourly_highres['Earned Hours'])
# # hourly_highres['Direct Hours Interpolated Hour'] = hourly_highres['Direct Hours Interpolated Hour'].fillna(hourly_highres['Direct Hours Interpolated'])
# # hourly_highres['Earned Hours Interpolated Hour'] = hourly_highres['Earned Hours Interpolated Hour'].fillna(hourly_highres['Earned Hours Interpolated'])

# fig999, ax999 = plt.subplots()
# ax999.hist(hourly_highres['Direct Hours Interpolated Hour PCTCHANGE'].abs(), bins=[0.01,0.1,1,10])
# ax999.set_xscale('log')
# ax999.set_title('Distribution of PCT Change in Hour over Hour Direct Hours')



hourly_highres = hourly_highres.reset_index(drop=False)
hourly_highres = pd.merge(hourly_highres, 
                          daily_highres[['Shop','StartOfWeek','weekday','Direct Hours Day','Earned Hours Day','Direct Hours Interpolated Day','Earned Hours Interpolated Day']], 
                          how='left',
                          left_on=['Shop','StartOfWeek','weekday'], 
                          right_on=['Shop','StartOfWeek','weekday'],
                          suffixes=('',' DayMax'))


hourly_highres['DirectHoursRatio'] = hourly_highres['Direct Hours Hour'] / hourly_highres['Direct Hours Day']
hourly_highres['DirectHoursInterpolatedRatio'] = hourly_highres['Direct Hours Interpolated Hour'] / hourly_highres['Direct Hours Interpolated Day']
hourly_highres['DirectHoursRatio'] = hourly_highres['DirectHoursRatio'].replace([np.inf, -np.inf], np.nan)
hourly_highres['DirectHoursInterpolatedRatio'] = hourly_highres['DirectHoursInterpolatedRatio'].replace([np.inf, -np.inf], 0)
hourly_highres['Hour'] = hourly_highres['Timestamp'].dt.hour
hourly_highres = hourly_highres.set_index('Timestamp')











fig, (ax0, ax1) = plt.subplots(ncols=2, nrows=1, sharey=True, sharex=True)
ax0.hist(hourly_highres['DirectHoursRatio'], bins=np.arange(-1,1,0.05))
ax0.set_title('(Standard)')
ax0.set_yscale('log')

ax1.hist(hourly_highres['DirectHoursInterpolatedRatio'], bins=np.arange(-1,1,0.05))
ax1.set_title('(Interpolated)')
ax1.set_yscale('log')
fig.suptitle('Distribution of Daily Hours Worked by Hour')



quants = np.arange(0,0.999, 0.001)
fig, (ax0, ax1) = plt.subplots(ncols=2, nrows=1, sharey=True, sharex=True)
ax0.plot(quants, np.quantile(hourly_highres[~hourly_highres['DirectHoursRatio'].isna()]['DirectHoursRatio'], quants))
ax0.set_yscale('log')
ax0.set_title('(Standard)')
ax0.set_ylim((1e-4,1))
ax0.set_ylabel('Ratio of Daily Hours')
ax1.plot(quants, np.quantile(hourly_highres[~hourly_highres['DirectHoursInterpolatedRatio'].isna()]['DirectHoursInterpolatedRatio'], quants))
ax1.set_yscale('log')
ax1.set_title('(Interpolated)')
fig.suptitle('Quantiles of Daily Hours Worked by Hour')


crosses_zero = np.where(np.quantile(hourly_highres[~hourly_highres['DirectHoursRatio'].isna()]['DirectHoursRatio'], quants) >= 1e-4)[0][0]
crosses_zero_DirectHoursRatio_quantile = quants[crosses_zero]
crosses_zero = np.where(np.quantile(hourly_highres[~hourly_highres['DirectHoursInterpolatedRatio'].isna()]['DirectHoursInterpolatedRatio'], quants) >= 1e-4)[0][0]
crosses_zero_DirectHoursInterpolatedRatio_quantile = quants[crosses_zero]



directHoursRatio_quantiles = np.quantile(hourly_highres[~hourly_highres['DirectHoursRatio'].isna()]['DirectHoursRatio'], [0.05,0.1,0.25,0.5,0.75,0.9])

# plt.hist(hourly_highres[~hourly_highres['DirectHoursInterpolatedRatio'].isna()]['DirectHoursInterpolatedRatio'], bins=[-1000,-10,0,10,1000])
directHoursInterpolateRatio_quantiles = np.quantile(hourly_highres[~hourly_highres['DirectHoursInterpolatedRatio'].isna()]['DirectHoursInterpolatedRatio'], [0.01,0.1,0.25,0.5,0.75,0.9,0.99])
directHoursInterpolateRatio_quantiles_filter = np.quantile(hourly_highres[~hourly_highres['DirectHoursInterpolatedRatio'].isna()]['DirectHoursInterpolatedRatio'], [crosses_zero_DirectHoursInterpolatedRatio_quantile,0.8])





agg_stats = {'DirectHoursRatio':['mean','std','median','count'],'DirectHoursInterpolatedRatio':['mean','std','median','count']}
hourly_highres_no_outliers = hourly_highres[(hourly_highres['DirectHoursInterpolatedRatio'] > directHoursInterpolateRatio_quantiles_filter[0]) & (hourly_highres['DirectHoursInterpolatedRatio'] < directHoursInterpolateRatio_quantiles_filter[1])]

hourly_highres_shop_stats = hourly_highres_no_outliers.groupby(['Shop','Hour']).agg(agg_stats)
hourly_highres_stats = hourly_highres_no_outliers.groupby('Hour').agg(agg_stats)




custom_order = list(np.arange(6,24,1)) + list(np.arange(0,6))

''' Hours by Hour '''
fig18, ax18 = plt.subplots()
fig18_dataset = []
for hour in custom_order:
    chunk = hourly_highres[hourly_highres.index.hour == hour]
    chunk = chunk[chunk['Direct Hours Interpolated Hour'] > 0]
    chunk = chunk[chunk['Direct Hours Interpolated Hour'] < 75]
    # chunk = chunk[chunk['Direct Hours Interpolated Hour PCTCHANGE'].abs() < 0.2]
    fig18_dataset.append(chunk['Direct Hours Interpolated Hour'])

ax18.boxplot(fig18_dataset)
ax18.set_xticks(ticks = np.arange(0, len(fig18_dataset))+1, labels=custom_order)
ax18.set_title('Hours Worked of Day by Hour')
ax18.set_ylabel('Number of Hours')
ax18.set_xlabel('Hour of Day')

fig19, ax19 = plt.subplots()
ax19.violinplot(fig18_dataset)
ax19.set_xticks(ticks = np.arange(0, len(fig18_dataset))+1, labels=custom_order)
ax19.set_title('Hours Worked of Day by Hour')
ax19.set_ylabel('Number of Hours')
ax19.set_xlabel('Hour of Day')



''' Ratio of Hours by Hour - NOT INFILLED'''

fig20, ax20 = plt.subplots()
fig20_dataset = []
for hour in custom_order:
    chunk = hourly_highres[hourly_highres.index.hour == hour]
    chunk = chunk[~chunk['DirectHoursRatio'].isna()]
    # chunk = chunk[chunk['DirectHoursRatio'] > directHoursRatio_quantiles[0]]
    # chunk = chunk[chunk['DirectHoursRatio'] < directHoursRatio_quantiles[-1]]
    chunk = chunk[chunk['DirectHoursInterpolatedRatio'] > 0]
    chunk = chunk[chunk['DirectHoursInterpolatedRatio'] < 1]
    # chunk = chunk[chunk['Direct Hours Interpolated Hour PCTCHANGE'].abs() < 0.2]
    fig20_dataset.append(chunk['DirectHoursRatio'])

ax20.boxplot(fig20_dataset)
ax20.set_xticks(ticks = np.arange(0, len(fig20_dataset))+1, labels=custom_order)
ax20.set_title('Ratio Of Daily Hours Worked By Hour Of Day')
ax20.set_ylabel('Ratio of Day\'s Hour Worked By Hour')
ax20.set_xlabel('Hour of Day')

fig21, ax21 = plt.subplots()
ax21.violinplot(fig20_dataset, showextrema=False, showmeans=True, widths=0.7)
# ax21.violinplot(fig20_dataset, showextrema=False, showmeans=True, widths=0.7, quantiles=[[0.1,0.9]]*24)
ax21.set_xticks(ticks = np.arange(0, len(fig20_dataset))+1, labels=custom_order)
ax21.set_title('Ratio Of Daily Hours Worked By Hour Of Day')
ax21.set_ylabel('Ratio of Day\'s Hour Worked By Hour')
ax21.set_xlabel('Hour of Day')


''' Ratio of Hours by Hour - using interpolated'''

fig20, ax20 = plt.subplots()
fig20_dataset = []
for hour in custom_order:
    chunk = hourly_highres[hourly_highres.index.hour == hour]
    chunk = chunk[~chunk['DirectHoursInterpolatedRatio'].isna()]
    chunk = chunk[chunk['DirectHoursInterpolatedRatio'] > directHoursInterpolateRatio_quantiles_filter[0]]
    chunk = chunk[chunk['DirectHoursInterpolatedRatio'] < directHoursInterpolateRatio_quantiles_filter[1]]
    # chunk = chunk[chunk['DirectHoursInterpolatedRatio'] > 0]
    # chunk = chunk[chunk['Direct Hours Interpolated Hour PCTCHANGE'].abs() < 0.2]
    fig20_dataset.append(chunk['DirectHoursInterpolatedRatio'])

ax20.boxplot(fig20_dataset)
ax20.set_xticks(ticks = np.arange(0, len(fig20_dataset))+1, labels=custom_order)
ax20.set_title('Ratio Of Daily Hours Worked By Hour Of Day\n(Infilled Missing Values via Linear Interpolation)')
ax20.set_ylabel('Ratio of Day\'s Hour Worked By Hour')
ax20.set_xlabel('Hour of Day')

fig21, ax21 = plt.subplots()
ax21.violinplot(fig20_dataset, showextrema=False, showmeans=True, widths=0.7)
# ax21.violinplot(fig20_dataset, showextrema=False, showmeans=True, widths=0.7, quantiles=[[0.1,0.9]]*24)
ax21.set_xticks(ticks = np.arange(0, len(fig20_dataset))+1, labels=custom_order)
ax21.set_title('Ratio Of Daily Hours Worked By Hour Of Day\n(Infilled Missing Values via Linear Interpolation)')
ax21.set_ylabel('Ratio of Day\'s Hour Worked By Hour')
ax21.set_xlabel('Hour of Day')


#%% Can we use company averages for day & hour or do we need to use shop level?


def test_means(df, test_col, shop):
    df = df[~df[test_col].isna()]
    shop_df = df[df['Shop'] == shop].copy()
    test = ttest_ind(df[test_col], shop_df[test_col])
    return test[1]


test_col = 'DirectHoursInterpolatedRatio'

test_results = {}
for shop in colors.keys():
    test_results[shop] = {'Daily':{},'Hourly':{},'Overall':{}}
    for day in np.arange(0,7):
        # check to see if the shop's days hours are different from the population's days hours
        thisday = daily_highres[daily_highres['weekday'] == day]
        test_results[shop]['Daily'][day] = test_means(thisday.copy(), test_col, shop)
    for hour in custom_order:
        # check to see if the shop's hour's hours are different from the populations hour's hours
        thishour = hourly_highres[hourly_highres['Hour'] == hour]
        test_results[shop]['Hourly'][hour] = test_means(thishour.copy(), test_col, shop)
    # check to see if the shop's average daily hours & average hourly hours are different from the population
    test_results[shop]['Overall'] = {'Daily':test_means(daily_highres.copy(), test_col, shop),
                                     'Hourly':test_means(hourly_highres.copy(), test_col, shop)}
    
test_results



test_results_daily = pd.DataFrame.from_dict({key: value['Daily'] for key,value in test_results.items()})
test_results_daily.index.name = 'Weekday'
test_results_hourly = pd.DataFrame.from_dict({key: value['Hourly'] for key,value in test_results.items()})
test_results_hourly.index.name = 'Hour'
test_results_overall = pd.DataFrame.from_dict({key: value['Overall'] for key,value in test_results.items()})


stats_columns = (test_col,['mean','std','count'])

test_results_daily_detail = {}
if (test_results_daily < 0.05).any().any():
    print('On a hour basis, one of the shops has a different mean {} than the population'.format(test_col))
    different_means = test_results_daily < 0.05
    different_means_out = test_results_daily[different_means.any(axis=1)]
    
    for shop in different_means_out.columns:
        hours = list(test_results_daily[test_results_daily[shop] < 0.05].index)
        shop_means = daily_highres_shop_stats.loc[(shop,hours), stats_columns]
        pop_means = daily_highres_stats.loc[hours, stats_columns]
        shop_means.index = shop_means.index.droplevel(0)
        shop_means.columns = ['LocalMEAN','LocalSTD','LocalCOUNT']
        pop_means.columns = ['GloablMEAN','GlobalSTD','GloablCOUNT']
        comparison = pd.merge(shop_means, pop_means, left_index=True, right_index=True)
        print(comparison)
        test_results_daily_detail[shop] = comparison  
    
    
test_results_hourly_detail = {}
if (test_results_hourly < 0.05).any().any():
    print('On a hour basis, one of the shops has a different mean {} than the population'.format(test_col))
    different_means = test_results_hourly < 0.05
    different_means_out = test_results_hourly[different_means.any(axis=1)]
    
    for shop in different_means_out.columns:
        hours = list(test_results_hourly[test_results_hourly[shop] < 0.05].index)
        shop_means = hourly_highres_shop_stats.loc[(shop,hours), stats_columns]
        pop_means = hourly_highres_stats.loc[hours, stats_columns]
        shop_means.index = shop_means.index.droplevel(0)
        shop_means.columns = ['LocalMEAN','LocalSTD','LocalCOUNT']
        pop_means.columns = ['GloablMEAN','GlobalSTD','GloablCOUNT']
        comparison = pd.merge(shop_means, pop_means, left_index=True, right_index=True)
        print(comparison)
        test_results_hourly_detail[shop] = comparison
    
    





#%% send the stats to csvs so I can pick em up later

def send_stats_to_csvs():
    zzz = hourly_highres_stats.copy()
    zzz = zzz.loc[:,('DirectHoursInterpolatedRatio',['mean','std','count'])]
    zzz.columns = zzz.columns.get_level_values(1)
    
    zzz.to_csv('ratios_hourly.csv')
    
    zzz = daily_highres_stats.copy()
    zzz = zzz.loc[:, ('DirectHoursInterpolatedRatio',['mean','std','count'])]
    zzz.columns = zzz.columns.get_level_values(1)
    
    zzz.to_csv('ratios_daily.csv')
    
