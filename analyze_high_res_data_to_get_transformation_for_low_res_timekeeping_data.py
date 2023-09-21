# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 10:41:12 2023

@author: codyt
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




weekdays = {0:'Sunday',
            1:'Monday',
            2:'Tuesday',
            3:'Wednesday',
            4:'Thursday',
            5:'Friday',
            6:'Saturday'}

# # befoer 6 am is nightshift
highres['NighShiftHoursMorning'] = highres["Timestamp"].dt.hour <= 5
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






# find the maximum direct hours for each week
maxOfWeek = highres.groupby(['StartOfWeek','Shop']).max()["Direct Hours"]
maxOfWeek = maxOfWeek.rename('MaxOfWeek')
highres = pd.merge(highres, maxOfWeek, left_on=['StartOfWeek','Shop'], right_index=True)
# this is a ratio of the direct horus timestamp to the total number of hours worked for that week
highres['DirectHoursPercentageOfMax'] = highres['Direct Hours'] / highres['MaxOfWeek']

# take the max, mean, and min of the DirectHoursPercentageOfMaxField
hr_weekday = highres[['StartOfWeek','weekday','Shop','DirectHoursPercentageOfMax']].groupby(['StartOfWeek','weekday','Shop']).agg(['max','mean','min'])
hr_weekday.columns = hr_weekday.columns.droplevel(0)
# calcualte the difference in percentage of week worked
hr_weekday['PercentageOfMaxWorked'] = hr_weekday['max'] - hr_weekday['min']

hr_weekday = hr_weekday.reset_index()
percentage_week_worked_dataset = []
for weekday in weekdays.keys():
    # print(weekday, weekdays[weekday])
    percentage_week_worked_dataset.append(hr_weekday[hr_weekday['weekday'] == weekday]['PercentageOfMaxWorked'])

fig1, (ax10, ax11) = plt.subplots(nrows=1, ncols=2, sharey=True)
ax10.boxplot(percentage_week_worked_dataset)
ax11.violinplot(percentage_week_worked_dataset, showextrema=False)
fig1.suptitle('Percentage Of Week Worked by Day of Week')
ax10.set_xticks(ticks = [i+1 for i in weekdays.keys()], labels=[i[:3] for i in weekdays.values()])
ax11.set_xticks(ticks = [i+1 for i in weekdays.keys()], labels=[i[:3] for i in weekdays.values()])



# get the max, mean, min of direct horus for each week/weekday
hr_weekday = highres[['StartOfWeek','weekday','Shop','Direct Hours']].groupby(['StartOfWeek','weekday','Shop']).agg(['max','mean','min'])
hr_weekday.columns = hr_weekday.columns.droplevel(0)
hr_weekday['Hours Worked'] = hr_weekday['max'] - hr_weekday['min']

hr_weekday = hr_weekday.reset_index()
direct_hours_by_day_dataset = []
for weekday in weekdays:
    # print(weekday)
    direct_hours_by_day_dataset.append(hr_weekday[hr_weekday['weekday'] == weekday]['Hours Worked'])


fig2, (ax20, ax21) = plt.subplots(nrows=1, ncols=2, sharey=True)
ax20.boxplot(direct_hours_by_day_dataset)
ax21.violinplot(direct_hours_by_day_dataset, showextrema=False)
fig2.suptitle('Hours Worked by Day of Week')
ax20.set_xticks(ticks = [i+1 for i in weekdays.keys()], labels=[i[:3] for i in weekdays.values()])
ax21.set_xticks(ticks = [i+1 for i in weekdays.keys()], labels=[i[:3] for i in weekdays.values()])




#%% we want to ge tthe distribution of percentage of week worked for each day

x = highres.groupby(['StartOfWeek','weekday','Shop'])[['Timestamp','DirectHoursPercentageOfMax']].max()
x = x.reset_index(drop=False)

fig5, ax5 = plt.subplots()
for weekday in weekdays.keys():
    y = x[x['weekday'] == weekday]
    ax5.hist(y['DirectHoursPercentageOfMax'], alpha=0.75, bins=np.arange(-0.03,1.05,0.05))
ax5.legend(weekdays.values())
ax5.set_title('Distrubtion of Week Completedness in\nHours Worked by Weekday')


#%% this is to see the distrubtion of how many horus each weekday contributes to the total number of hours worked in a week


maxDirect_byWeek = highres.groupby(['StartOfWeek','Shop']).max()['Direct Hours']
maxDirect_byWeekWeekday = highres.groupby(['StartOfWeek','Shop','weekday']).max()['Direct Hours']
# somehow this divides the StartOfWeek-Shop-Weekday values by the StartOfWeek-Shop values
weekday_breakdown = maxDirect_byWeekWeekday / maxDirect_byWeek
weekday_breakdown = weekday_breakdown.sort_index(level=[0,2,1])
# get the rolling difference within the group
ratioDirect_byWeekday = weekday_breakdown.groupby(level=[0,1]).diff()


ratioDirect_byWeekday = ratioDirect_byWeekday.rename('PercentageOfHoursByDay')
weekday_breakdown = weekday_breakdown.reset_index(drop=False)
weekday_breakdown = pd.merge(weekday_breakdown, ratioDirect_byWeekday, left_on=['StartOfWeek','Shop','weekday'], right_index=True)

fix_diff_value = weekday_breakdown[(weekday_breakdown['weekday'] != 0) & (weekday_breakdown['PercentageOfHoursByDay'].isna())]
weekday_breakdown = weekday_breakdown.drop(index=fix_diff_value.index)
fix_diff_value2 = weekday_breakdown[weekday_breakdown['PercentageOfHoursByDay'] < 0]
weekday_breakdown = weekday_breakdown.drop(index=fix_diff_value2.index)
weekday_breakdown['PercentageOfHoursByDay'] = weekday_breakdown['PercentageOfHoursByDay'].fillna(0)

hist_dataset = []
for weekday in weekdays.keys():
    to_plot = weekday_breakdown[weekday_breakdown['weekday'] == weekday]['PercentageOfHoursByDay']
    hist_dataset.append(to_plot)
    
bins_stacked = np.arange(-0.0499,1,0.05)
fig3, ax3 = plt.subplots()
ax3.hist(hist_dataset, alpha=0.7, bins=bins_stacked, stacked=True)
ax3.legend(weekdays.values())
ax3.set_title('Distributions of Ratio of Total Weekly Hours by Weekday')


percentageOfHoursByDay_quantiles = np.quantile(weekday_breakdown['PercentageOfHoursByDay'], [0.1,0.9])


hist_dataset2 = []
for weekday in weekdays.keys():
    to_plot = hist_dataset[weekday]
    to_plot = to_plot[to_plot > percentageOfHoursByDay_quantiles[0]]
    to_plot = to_plot[to_plot < percentageOfHoursByDay_quantiles[1]]
    hist_dataset2.append(to_plot)
    
bins_depth = np.arange(0,0.35,0.012)
fig4, ax4 = plt.subplots()
ax4.hist(hist_dataset2, alpha=0.7, bins=bins_depth, stacked=True)
ax4.set_title('Distributions of Amount of Hours Worked by Day')
ax4.legend(weekdays.values())



WEEKDAY_BREAKDOWN_STATS = pd.DataFrame(index=weekdays.keys(), data=weekdays.values(), columns=['Day'])
weekday_stats = weekday_breakdown[['weekday','PercentageOfHoursByDay']].groupby(['weekday']).agg(['mean','median','std'])
weekday_stats.columns = weekday_stats.columns.droplevel(0)
WEEKDAY_BREAKDOWN_STATS = pd.merge(WEEKDAY_BREAKDOWN_STATS, weekday_stats, left_index=True, right_index=True, how='left')


WEEKDAY_BREAKDOWN_STATS.loc[WEEKDAY_BREAKDOWN_STATS.index.max()+1] = ['All Days', weekday_breakdown['PercentageOfHoursByDay'].mean(), weekday_breakdown['PercentageOfHoursByDay'].median(), weekday_breakdown['PercentageOfHoursByDay'].std()]
weekday_breakdown_weekdays = weekday_breakdown[weekday_breakdown['weekday'].isin([1,2,3,4,5])]
WEEKDAY_BREAKDOWN_STATS.loc[WEEKDAY_BREAKDOWN_STATS.index.max()+1] = ['Weekdays', weekday_breakdown_weekdays['PercentageOfHoursByDay'].mean(), weekday_breakdown_weekdays['PercentageOfHoursByDay'].median(), weekday_breakdown_weekdays['PercentageOfHoursByDay'].std()]
weekday_breakdown_weekends = weekday_breakdown[~weekday_breakdown['weekday'].isin([1,2,3,4,5])]
WEEKDAY_BREAKDOWN_STATS.loc[WEEKDAY_BREAKDOWN_STATS.index.max()+1] = ['Weekends', weekday_breakdown_weekends['PercentageOfHoursByDay'].mean(), weekday_breakdown_weekends['PercentageOfHoursByDay'].median(), weekday_breakdown_weekends['PercentageOfHoursByDay'].std()]




#%% now look at the breakdown by hour of the day

# round the timestamp hour 
highres['TimestampHour'] = highres['Timestamp'].dt.round('H').dt.hour
# highres['TimestampHour'] = highres['Timestamp'].dt.hour

# now we need to get the max direct hours for that weekday 
maxDirect_byWeekWeekdayHour = highres.groupby(['StartOfWeek','Shop','weekday','TimestampHour']).max()['Direct Hours']
# then transform the max at that hour to a ratio
hour_breakdown = maxDirect_byWeekWeekdayHour / maxDirect_byWeekWeekday
hour_breakdown = hour_breakdown.rename('PercentageHoursOfDayCompleted')
# hour_breakdown = hour_breakdown.rename('PercentageOfHoursByHour')
# this gets it in chronological order so it goes by StartOfWeek, Weekday #, Hour #, then Shop
# hour_breakdown = hour_breakdown.sort_index(level=[0,2,3,1])
# hour_breakdown = hour_breakdown.sort_index(level=[1,0,2,3])

hour_breakdown = hour_breakdown.reset_index(drop=False)
# this puts night shift hours after day shift
custom_order = list(np.arange(6,24,1)) + list(np.arange(0,6))
hour_breakdown['TimestampHour'] = hour_breakdown['TimestampHour'].astype('category')
hour_breakdown['TimestampHour'] = hour_breakdown['TimestampHour'].cat.set_categories(custom_order, ordered=True)
hour_breakdown = hour_breakdown.sort_values(['Shop','StartOfWeek','weekday','TimestampHour'])


# get the rolling difference within group of StartOfWeek,Shop,Weekday
ratioDirect_byHour = hour_breakdown[['StartOfWeek','Shop','weekday','PercentageHoursOfDayCompleted']].groupby(['StartOfWeek','Shop','weekday']).diff()
ratioDirect_byHour = ratioDirect_byHour.rename(columns={'PercentageHoursOfDayCompleted':'PercentageOfHoursWorkedByHour'})

hour_breakdown = pd.merge(hour_breakdown, ratioDirect_byHour, left_index=True, right_index=True)


# remove any records that don't have a record before them when they are not the start of the day
fix_diff_value = hour_breakdown[(hour_breakdown['TimestampHour'] != 6) & (hour_breakdown['PercentageOfHoursWorkedByHour'].isna())]
hour_breakdown = hour_breakdown.drop(index=fix_diff_value.index)
# in case there are really small values, just set them to zero to prevent further loss
very_small_change = hour_breakdown[hour_breakdown['PercentageOfHoursWorkedByHour'].abs() < 0.0001]
hour_breakdown.loc[very_small_change.index, 'PercentageOfHoursWorkedByHour'] = 0
fix_diff_value2 = hour_breakdown[hour_breakdown['PercentageOfHoursWorkedByHour'] < 0]
hour_breakdown = hour_breakdown.drop(index=fix_diff_value2.index)
hour_breakdown['PercentageOfHoursWorkedByHour'] = hour_breakdown['PercentageOfHoursWorkedByHour'].fillna(0)


hours_dataset = []
for hour in custom_order:
    # print(hour)
    chunk = hour_breakdown[hour_breakdown['TimestampHour'] == hour]
    hours_dataset.append(chunk['PercentageOfHoursWorkedByHour'])

fig6, ax6 = plt.subplots()    
ax6.boxplot(hours_dataset)
ax6.set_xticks(ticks = np.arange(0,24)+1, labels=custom_order)
ax6.set_xlabel('Hour of Day\n(Hours 0-5 count towards nightshift of previous day)')
ax6.set_ylabel('Percentage of Days Hours Worked')
ax6.set_title('Distributions of Day\'s Hours Worked, by Hour\n(At what hour of the day are hours worked?)')





hour_quantiles_global = np.quantile(hour_breakdown['PercentageOfHoursWorkedByHour'], [0.1,0.9])

hours_dataset_cleaned = []
for hour in custom_order:
    # print(hour)
    chunk = hour_breakdown[hour_breakdown['TimestampHour'] == hour]
    chunk = chunk[chunk['PercentageOfHoursWorkedByHour'] > hour_quantiles_global[0]]
    chunk = chunk[chunk['PercentageOfHoursWorkedByHour'] < hour_quantiles_global[1]]
    hours_dataset_cleaned.append(chunk['PercentageOfHoursWorkedByHour'])

fig7, ax7 = plt.subplots()    
ax7.boxplot(hours_dataset_cleaned)
ax7.set_xticks(ticks = np.arange(0,24)+1, labels=custom_order)
ax7.set_xlabel('Hour of Day\n(Hours 0-5 count towards nightshift of previous day)')
ax7.set_ylabel('Percentage of Days Hours Worked')
ax7.set_title('Distributions of Day\'s Hours Worked, by Hour\n(Cleaned with gloabl 80% median)')



hours_dataset_cleaned = []
for hour in custom_order:
    # print(hour)
    chunk = hour_breakdown[hour_breakdown['TimestampHour'] == hour]
    local_quantiles = np.quantile(chunk['PercentageOfHoursWorkedByHour'], [0.1,0.9])
    if hour == 5:
        local_quantiles = np.quantile(chunk['PercentageOfHoursWorkedByHour'], [0.1,0.8])
    chunk = chunk[chunk['PercentageOfHoursWorkedByHour'] > local_quantiles[0]]
    chunk = chunk[chunk['PercentageOfHoursWorkedByHour'] < local_quantiles[1]]
    hours_dataset_cleaned.append(chunk['PercentageOfHoursWorkedByHour'])

fig7, ax7 = plt.subplots()    
ax7.boxplot(hours_dataset_cleaned)
ax7.set_xticks(ticks = np.arange(0,len(hours_dataset_cleaned))+1, labels=custom_order)
ax7.set_xlabel('Hour of Day\n(Hours 0-5 count towards nightshift of previous day)')
ax7.set_ylabel('Percentage of Days Hours Worked')
ax7.set_title('Distributions of Day\'s Hours Worked, by Hour\n(Cleaned with Local 80% median)')






HOURLY_BREAKDOWN_STATS = pd.DataFrame(data=custom_order, columns=['Hour'])
hour_stats = hour_breakdown[['TimestampHour','PercentageOfHoursWorkedByHour']].groupby(['TimestampHour']).agg(['mean','median','std'])
hour_stats.columns = hour_stats.columns.droplevel(0)
HOURLY_BREAKDOWN_STATS = pd.merge(HOURLY_BREAKDOWN_STATS, hour_stats, left_on=['Hour'], right_index=True, how='left')


HOURLY_BREAKDOWN_STATS.loc[HOURLY_BREAKDOWN_STATS.index.max()+1] = ['All Days', hour_breakdown['PercentageOfHoursWorkedByHour'].mean(), hour_breakdown['PercentageOfHoursWorkedByHour'].median(), hour_breakdown['PercentageOfHoursWorkedByHour'].std()]
hour_breakdown_hours = hour_breakdown[hour_breakdown['weekday'].isin([1,2,3,4,5])]
HOURLY_BREAKDOWN_STATS.loc[HOURLY_BREAKDOWN_STATS.index.max()+1] = ['Weekdays', hour_breakdown_hours['PercentageOfHoursWorkedByHour'].mean(), hour_breakdown_hours['PercentageOfHoursWorkedByHour'].median(), hour_breakdown_hours['PercentageOfHoursWorkedByHour'].std()]
hour_breakdown_weekends = hour_breakdown[~hour_breakdown['weekday'].isin([1,2,3,4,5])]
HOURLY_BREAKDOWN_STATS.loc[HOURLY_BREAKDOWN_STATS.index.max()+1] = ['Weekends', hour_breakdown_weekends['PercentageOfHoursWorkedByHour'].mean(), hour_breakdown_weekends['PercentageOfHoursWorkedByHour'].median(), hour_breakdown_weekends['PercentageOfHoursWorkedByHour'].std()]


