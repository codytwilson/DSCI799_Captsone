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

hourly_highres = highres2[['Shop','StartOfWeek','Direct Hours','Earned Hours']].groupby('Shop').resample('H').max()
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
highres2 = highres2.set_index(highres2['TimestampShifted'], drop=False)
highres2['StartOfWeek'] = highres2.apply(figure_StartOfWeek, axis=1)


# x = highres2[(highres2['Shop'] == 'FED') & (highres2['StartOfWeek'] >= datetime.datetime(2023,9,10))]

daily_highres = highres2[['Shop','Direct Hours','Earned Hours','weekday','StartOfWeek','Timestamp']].groupby('Shop').resample('D', closed='right').max()
# this is for troubleshooting so you can see what the max value occurs at
# daily_highres = highres2[['Shop','Direct Hours','Earned Hours','weekday','StartOfWeek','Timestamp']].groupby('Shop').resample('D', closed='right').max()

highres2 = highres2.set_index('Timestamp')


daily_highres = daily_highres.drop(columns=['Shop'])
daily_highres = daily_highres.reset_index()
daily_highres = daily_highres.set_index('TimestampShifted')
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
colors = {'CSM':'tab:blue','CSF':'tab:orange','FED':'tab:green'}




'''
This is exact replica of fig9, except it cuts CSM to have the same start point as CSF/FED

I SHOULD ADD GRIDS ON X TO SHOW UP EVERY 7 DAYS
'''
fig11, (ax110, ax111) = plt.subplots(nrows=2, ncols=1)
for shop in colors.keys():
    chunk = daily_highres[daily_highres['Shop'] == shop]
    chunk = chunk[chunk['StartOfWeek'] >= datetime.date(2023,5,21)]
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


daily_weekly = pd.merge(daily_highres.reset_index(), weekly_highres, 
                        left_on=['Shop','StartOfWeek'], 
                        right_on = ['Shop','StartOfWeek'],
                        how = 'left',
                        suffixes=('',' WeekMax'))
daily_weekly = daily_weekly.set_index('Timestamp')

daily_weekly['DirectHoursRatio'] = daily_weekly['Direct Hours Day'] / daily_weekly['Direct Hours WeekMax']
daily_weekly['DirectHoursInterpolatedRatio'] = daily_weekly['Direct Hours Interpolated Day'] / daily_weekly['Direct Hours WeekMax']


agg_stats = {'DirectHoursRatio':['mean','std','median','count'],'DirectHoursInterpolatedRatio':['mean','std','median','count']}

daily_weekly_shop_stats = daily_weekly.groupby(['Shop','weekday']).agg(agg_stats)
daily_weekly_stats = daily_weekly.groupby('weekday').agg(agg_stats)







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

#%% Finding out the ratio distributions of each Hour contributing to Day


hourly_highres[['Direct Hours Hour',
               'Earned Hours Hour',
               'Direct Hours Interpolated Hour',
               'Earned Hours Interpolated Hour']] = hourly_highres.groupby(['Shop','StartOfWeek']).diff()[['Direct Hours','Earned Hours','Direct Hours Interpolated','Earned Hours Interpolated']]


hourly_highres['Direct Hours Interpolated Hour PCTCHANGE'] = hourly_highres.groupby(['Shop','StartOfWeek']).pct_change()['Direct Hours Interpolated Hour'].replace([np.inf, -np.inf], np.nan)
# hourly_highres['Direct Hours Hour'] = hourly_highres['Direct Hours Hour'].fillna(hourly_highres['Direct Hours'])
# hourly_highres['Earned Hours Hour'] = hourly_highres['Earned Hours Hour'].fillna(hourly_highres['Earned Hours'])
# hourly_highres['Direct Hours Interpolated Hour'] = hourly_highres['Direct Hours Interpolated Hour'].fillna(hourly_highres['Direct Hours Interpolated'])
# hourly_highres['Earned Hours Interpolated Hour'] = hourly_highres['Earned Hours Interpolated Hour'].fillna(hourly_highres['Earned Hours Interpolated'])

plt.hist(hourly_highres['Direct Hours Interpolated Hour PCTCHANGE'], bins=[-10,-5,-1,-0.1,,0.1,1,5,10])




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




custom_order = list(np.arange(6,24,1)) + list(np.arange(0,6))


fig16, ax16 = plt.subplots()
fig16_dataset = []
for hour in custom_order:
    chunk = hourly_highres[hourly_highres['Hour'] == hour]
    chunk = chunk[chunk['Direct Hours Interpolated Hour'] > 0]
    chunk = chunk[chunk['Direct Hours Interpolated Hour'] < 75]
    # chunk = chunk[chunk['Direct Hours Interpolated Hour PCTCHANGE'].abs() < 0.2]
    fig16_dataset.append(chunk['Direct Hours Interpolated Hour'])

ax16.boxplot(fig16_dataset)
ax16.set_xticks(ticks = np.arange(0, len(fig16_dataset))+1, labels=custom_order)
ax16.set_title('Hours Worked of Day by Hour')
ax16.set_ylabel('Number of Hours')
ax16.set_xlabel('Hour of Day')

fig17, ax17 = plt.subplots()
fig17_dataset = []
ax17.violinplot(fig16_dataset)
ax17.set_xticks(ticks = np.arange(0, len(fig16_dataset))+1, labels=custom_order)
ax17.set_title('Hours Worked of Day by Hour')
ax17.set_ylabel('Number of Hours')
ax16.set_xlabel('Hour of Day')






#%% Now we need to look at the ratios instead of the number of hours 
















