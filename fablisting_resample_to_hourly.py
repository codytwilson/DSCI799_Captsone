# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 11:29:09 2023

@author: cwilson
"""

import pandas as pd
import numpy as np
import datetime
from retrieve_dfs_from_csvs import load_and_combine_fablisting_csv_to_df
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import ttest_ind, ttest_1samp
from scipy.stats import percentileofscore

colors = {'CSM':'tab:blue','CSF':'tab:orange','FED':'tab:green'}

fablisting = load_and_combine_fablisting_csv_to_df()
fablisting = fablisting[['Timestamp','Job #','Lot #','Lot Name','Quantity','Piece Mark - REV','Weight','Earned Hours','Has Model','shop']]

earliest_dates = fablisting[['Timestamp','shop']].groupby('shop').min()
fablisting = fablisting[fablisting['Timestamp'] >= earliest_dates['Timestamp'].max()]
# we gonna do this twice because there is a huge gap in feds data from late 2020 to jan 2021
earliest_dates = fablisting[['Timestamp','shop']].groupby('shop').min()
fablisting = fablisting[fablisting['Timestamp'] >= earliest_dates['Timestamp'].max()]


jobEncoder = LabelEncoder().fit(pd.unique(fablisting['Job #']))


missing_timestamp = fablisting[fablisting['Timestamp'].isna()]
missing_earned_hours = fablisting[(fablisting['Earned Hours'].isna()) | 
                                  (fablisting['Earned Hours'] == 0) | 
                                  (fablisting['Earned Hours'] == np.inf) | 
                                  (fablisting['Earned Hours'] == -np.inf)].copy()
ratio_missing_earned_hours = missing_earned_hours.shape[0] / fablisting.shape[0]

missing_earned_hours['year'] = missing_earned_hours['Timestamp'].dt.year
missing_earned_hours_where = missing_earned_hours.groupby(['year','shop']).count().iloc[:,0]
missing_earned_hours_where = missing_earned_hours_where.reset_index()
missing_earned_hours_where = missing_earned_hours_where.pivot(index='year',columns='shop', values='Timestamp')
missing_has_model = fablisting[~ fablisting['Has Model']]
missing_has_model_and_earned_hours = missing_has_model[missing_has_model['Earned Hours'].isna()]

fablisting['year'] = fablisting['Timestamp'].dt.year
fablisting_where = fablisting.groupby(['year','shop']).count().iloc[:,0]
fablisting_where = fablisting_where.reset_index()
fablisting_where = fablisting_where.pivot(index='year',columns='shop', values='Timestamp')

ratio_missing_where = missing_earned_hours_where / fablisting_where



fablisting_description = fablisting[['Quantity','Weight','Earned Hours']].describe()
fablisting_without_missing_earned_hours = fablisting[~fablisting.index.isin(missing_earned_hours.index)]
fablisting_description.loc['mean', 'Earned Hours'] = fablisting_without_missing_earned_hours['Earned Hours'].mean()
fablisting_description.loc['std', 'Earned Hours'] = fablisting_without_missing_earned_hours['Earned Hours'].std()
fablisting_description.loc['min', 'Earned Hours'] = fablisting_without_missing_earned_hours['Earned Hours'].min()
fablisting_description.loc['25%', 'Earned Hours'] = np.quantile(fablisting_without_missing_earned_hours['Earned Hours'], 0.25)
fablisting_description.loc['50%', 'Earned Hours'] = fablisting_without_missing_earned_hours['Earned Hours'].median()
fablisting_description.loc['75%', 'Earned Hours'] = np.quantile(fablisting_without_missing_earned_hours['Earned Hours'], 0.75)
fablisting_description.loc['max', 'Earned Hours'] = fablisting_without_missing_earned_hours['Earned Hours'].max()
fablisting_description.loc['countNA'] = [np.nan,np.nan,missing_earned_hours.shape[0]]





# looking at distribution of pieces by job number
job_counts = fablisting[['Job #','Timestamp']].groupby('Job #').count()
job_counts_quantiles = np.quantile(job_counts, [0,0.1,0.25,0.5,0.75,0.9,1])
jobs_to_remove = job_counts[job_counts['Timestamp'] < job_counts_quantiles[2]]
jobs_to_remove_list = list(jobs_to_remove.index)


fig0, ax0 = plt.subplots()
ax0.hist(job_counts, bins=job_counts_quantiles[1:])
ax0.set_title('Distribution of records by job')
ax0.set_ylabel('Number of Jobs')
ax0.set_xlabel('Number of Records')








fablisting_without_missing = pd.merge(fablisting.copy(), missing_earned_hours, indicator=True, how='left')

# this gets us a timestamp form the datetime apparently
fablisting_without_missing['kmeansTimestamp'] = fablisting_without_missing['Timestamp'].astype(np.int64) // 10 ** 9
missing_earned_hours['kmeansTimestamp'] = missing_earned_hours['Timestamp'].astype(np.int64) // 10 ** 9

# get rid of the missing ones
fablisting_without_missing = fablisting_without_missing[fablisting_without_missing['_merge'] == 'left_only']



print(f"The mean of Earned Hours {fablisting_without_missing['Earned Hours'].mean()} is the {percentileofscore(fablisting_without_missing['Earned Hours'], fablisting_without_missing['Earned Hours'].mean())} quantile")










''' how are we going to replace the missing model pieces ??? '''
#%% kmeans grouping that we can now 




'''
Kmeans grouping columns:
   *** Weight - continuous, has a correlation to earned hours (known b/c subject matter expert)
    Shop - categorical makes it an issue, but different shops = different jobs = different amount of earned hours
    Job # - categorical - issue. Encoding job # may make jobs appear similar or dissimilar when they have nothing in common simply b/c of way encoding works
    Quantity - basically continuous. This is a maybe
   *** Has Model - different earned hours b/c of earned hour method (model or jobs' hours per ton rating)?
    Earned Hours - our target. if this is part of the model, can we predict with less pieces?
   *** Timestamp - would require a transformation like year.(month/12)
'''

kmeans_vars = ['Weight','Has Model','kmeansTimestamp']

# fablisting_without_missing = fablisting_without_missing[~fablisting_without_missing['Job #'].isin(jobs_to_remove)]

# neighbors = np.arange(100,1000,25)
# kmeans = GridSearchCV(KMeans(random_state=0, n_init="auto"), {'n_clusters':neighbors})
# kmeans.fit(fablisting_without_missing[kmeans_vars])
# kmeans_best = kmeans.best_estimator_



# fig1, ax1 = plt.subplots()
# ax1.set_title('Kmeans on ' + ', '.join(kmeans_vars))
# ax1.plot(neighbors, kmeans.cv_results_['std_test_score'])
# ax1.set_xlable('Neighbors')

# fablisting_without_missing['group'] = kmeans_best.labels_

# fablisting_kmeans_group = fablisting_without_missing.groupby('group').agg({'Earned Hours':['mean','count','std'],'Weight':['mean','std'],'Has Model':['mean','std'],'kmeansTimestamp':['mean','min','max']})
# # fablisting_kmeans_group.columns = fablisting_kmeans_group.columns.get_level_values(1)
# fablisting_kmeans_group.columns = [' '.join(col).strip() for col in fablisting_kmeans_group.columns.values]
# fablisting_kmeans_group['kmeansTimestamp mean'] = pd.to_datetime(fablisting_kmeans_group['kmeansTimestamp mean'], unit='s').dt.date
# fablisting_kmeans_group['kmeansTimestamp max'] = pd.to_datetime(fablisting_kmeans_group['kmeansTimestamp max'], unit='s').dt.date
# fablisting_kmeans_group['kmeansTimestamp min'] = pd.to_datetime(fablisting_kmeans_group['kmeansTimestamp min'], unit='s').dt.date
# fablisting_kmeans_group['datewindow'] = (fablisting_kmeans_group['kmeansTimestamp max'] - fablisting_kmeans_group['kmeansTimestamp min']).dt.days
# fablisting_kmeans_group = fablisting_kmeans_group.sort_values('kmeansTimestamp min')


# ''' Using timestamp makes it looks like it is grouping off of that more than anything '''




# group = fablisting_without_missing[fablisting_without_missing['group'] == 2]


# # missing_earned_hours['shop'] = shopEncoder.transform(missing_earned_hours['shop'])
# missing_earned_hours['kmeansTimestamp'] = missing_earned_hours['Timestamp'].astype(np.int64) // 10 ** 9
# missing_earned_hours['Earned Hours Kmeans'] = kmeans_best.predict(missing_earned_hours[kmeans_vars])
# # missing_earned_hours['shop'] = shopEncoder.inverse_transform(missing_earned_hours['shop'])






#%% now do it with a tree



rf_vars = ['Weight','Job #','kmeansTimestamp','shop','Quantity']


shopEncoder = LabelEncoder().fit(pd.unique(fablisting_without_missing['shop']))
fablisting_without_missing['shop'] = shopEncoder.transform(fablisting_without_missing['shop'])
fablisting_without_missing['Job #'] = jobEncoder.transform(fablisting_without_missing['Job #'])




rf = RandomForestRegressor(random_state=0, max_features='sqrt', n_estimators=150)
# rf.fit(fablisting_without_missing[rf_vars], fablisting_without_missing['Earned Hours'])
X_train, X_test, y_train, y_test = train_test_split(fablisting_without_missing[rf_vars], fablisting_without_missing['Earned Hours'], train_size=0.8)
rf.fit(X_train, y_train)


rf_worth_to_dict = {}
for i in range(0,len(rf_vars)):
    print(rf_vars[i], rf.feature_importances_[i])
    rf_worth_to_dict[rf_vars[i]] =  rf.feature_importances_[i]
rf_worth_to_df = pd.DataFrame.from_dict(rf_worth_to_dict, orient='index')
rf_worth_to_df = rf_worth_to_df.sort_values(by=0, ascending=False)

fablisting_without_missing['Earned Hours rf'] = rf.predict(fablisting_without_missing[rf_vars])
y_pred = rf.predict(X_test)

rf_model_numbers = {
    'Training R2':rf.score(X_train, y_train),
    'Testing R2':rf.score(X_test, y_test),
    'MSE Test vs Predicted':mean_squared_error(y_test, y_pred),
    'T-Test Test vs Predicted':ttest_ind(y_test, y_pred)[1],
    'Full Dataset R2':rf.score(fablisting_without_missing[rf_vars], fablisting_without_missing['Earned Hours']),
    'MSE Full Dataset':mean_squared_error(fablisting_without_missing['Earned Hours'], fablisting_without_missing['Earned Hours rf']),
    'T-Test Full Dataset':ttest_ind(fablisting_without_missing['Earned Hours'], fablisting_without_missing['Earned Hours rf'])[1]
    }

for k in rf_model_numbers.keys():
    print(k, rf_model_numbers[k])

rf_model_numbers_df = pd.DataFrame.from_dict(rf_model_numbers, orient='index')

# print(f"Training Score {rf_model_numbers[]}")
# print(f"Testing Score {rf_model_numbers[]}")
# print(f"MSE of test vs pred {rf_model_numbers[]}")
# print(f"T-Test p-value of test and pred: {rf_model_numbers[]}")


# print(f"Score on full dataset {rf_model_numbers[]}")
# print(f"MSE on full dataset {rf_model_numbers[]}")
# print(f"T-Test p-value of Earned Hours == Earned Hours rf: {rf_model_numbers[]}")

fablisting_without_missing['shop'] = shopEncoder.inverse_transform(fablisting_without_missing['shop'])
fablisting_without_missing['Job #'] = jobEncoder.inverse_transform(fablisting_without_missing['Job #'])


missing_earned_hours['shop'] = shopEncoder.transform(missing_earned_hours['shop'])
missing_earned_hours['Job #'] = jobEncoder.transform(missing_earned_hours['Job #'])
missing_earned_hours['Earned Hours rf'] = rf.predict(missing_earned_hours[rf_vars])
missing_earned_hours['shop'] = shopEncoder.inverse_transform(missing_earned_hours['shop'])
missing_earned_hours['Job #'] = jobEncoder.inverse_transform(missing_earned_hours['Job #'])


ttest_1samp(missing_earned_hours['Earned Hours rf'], fablisting_without_missing['Earned Hours'].median())

missing_earned_hours_description = missing_earned_hours[['Quantity','Weight','Earned Hours rf']].describe()



#%%
fablisting2 = fablisting.copy()
fablisting2.loc[missing_earned_hours.index,'Earned Hours'] = missing_earned_hours['Earned Hours rf']

fablisting_highres = fablisting2.set_index('Timestamp')
fablisting_highres = fablisting_highres[['shop','Quantity','Weight','Earned Hours','Has Model']]
fablisting_highres['Has Model'] = fablisting_highres['Has Model'].astype(int)
fablisting_highres['Earned Hours'] = fablisting_highres['Earned Hours'].fillna(0)
fablisting_highres = fablisting_highres.groupby('shop').resample('H').sum()
fablisting_highres = fablisting_highres.reset_index(drop=False)

for shop in pd.unique(fablisting_highres['shop']):
    chunk = fablisting_highres[fablisting_highres['shop'] == shop].copy()
    chunk['Earned Hours Cumulative'] = chunk['Earned Hours'].cumsum()
    fablisting_highres.loc[chunk.index, 'Earned Hours Cumulative'] = chunk['Earned Hours Cumulative']




plot_col = 'Earned Hours'
years = pd.unique(fablisting_highres['Timestamp'].dt.year)
fig, ax = plt.subplots(nrows=len(years), sharey=True)

shops = pd.unique(fablisting_highres['shop'])

# TODO: Plots showing the daily resolution of earned hours
fablisting_highres_daily = fablisting_highres.set_index('Timestamp').groupby('shop').resample('d').sum()
fablisting_highres_daily = fablisting_highres_daily.reset_index()
fig, axes = plt.subplots(len(shops), 1, sharey=True, figsize=(7,8))
for k, shop in enumerate(shops):
    ax = axes[k]
    chunk = fablisting_highres_daily[fablisting_highres_daily['shop'] == shop].copy()
    chunk = chunk[chunk[plot_col] > 0]
    ax.plot(chunk['Timestamp'], chunk['Earned Hours'], color=colors[shop])
    months = pd.date_range(f'2021-01-01',fablisting_highres_daily['Timestamp'].max(), freq='3MS')
    ax.set_xticks(months)
    ax.set_xticklabels([dt.strftime('%Y-%m') for dt in months])
    ax.set_ylim(0, 1200)  # Replace 'your_max_y_limit' with your desired y-limit
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_ylabel(shop +'\nDaily Earned Hours')
fig.suptitle('Daily Earned Hours')
plt.xticks(rotation=45)
# TODO: Plots showing the daily resolutoin of weekly hours 
fablisting_highres_weekly = fablisting_highres.set_index('Timestamp').groupby('shop').resample('W', label='left').sum()
fablisting_highres_weekly = fablisting_highres_weekly.reset_index()
fig, axes = plt.subplots(len(shops), 1, sharey=True, sharex=True, figsize=(7,8))
for k, shop in enumerate(shops):
    ax = axes[k]
    chunk = fablisting_highres_weekly[fablisting_highres_weekly['shop'] == shop].copy()
    chunk = chunk[chunk[plot_col] > 0]
    ax.plot(chunk['Timestamp'], chunk['Earned Hours'], color=colors[shop])
    months = pd.date_range(f'2021-01-01',fablisting_highres_weekly['Timestamp'].max(), freq='3MS')
    ax.set_xticks(months)
    ax.set_xticklabels([dt.strftime('%Y-%m') for dt in months])
    ax.set_ylim(0, 3000)  # Replace 'your_max_y_limit' with your desired y-limit
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_ylabel(shop +'\nWeekly Earned Hours')
fig.suptitle('Weekly Earned Hours')
plt.xticks(rotation=45)


# TODO: each year showing the weekly hours completed
# Create a figure with shared y-axes
fig, axes = plt.subplots(len(years), 1, sharey=True, figsize=(7, 8))
fig.subplots_adjust(right=0.8)
# Loop through the years and create subplots
for k, year in enumerate(years):
    ax = axes[k]
    for shop in shops:
        # chunk = fablisting_highres[fablisting_highres['shop'] == shop].copy()
        # chunk = chunk[chunk['Timestamp'].dt.year == year]
        # chunk = chunk[chunk[plot_col] > 0]
        # chunk = chunk[chunk[plot_col] < np.quantile(chunk[plot_col], 0.9)]
        # chunk[plot_col] = chunk[plot_col].cumsum()
        chunk = fablisting_highres_weekly[fablisting_highres_weekly['shop'] == shop].copy()
        # chunk = fablisting_highres_daily[fablisting_highres_daily['shop'] == shop].copy()
        chunk = chunk[chunk['Timestamp'].dt.year == year]
        chunk = chunk[chunk[plot_col] > 0]
        # TODO: moving average
        chunk[plot_col] = chunk[plot_col].rolling(window=3).mean()
        # chunk = chunk[chunk[plot_col] < np.quantile(chunk[plot_col], 0.9)]
        # chunk[plot_col] = chunk[plot_col].cumsum()        
        ax.plot(chunk['Timestamp'], chunk[plot_col], color=colors[shop])

    # Set y-limits
    ax.set_ylim(0,3000)  # Replace 'your_max_y_limit' with your desired y-limit
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    # Customize x-axis tick labels to show only months at the beginning of the month
    months = pd.date_range(f'{year}-01-01', f'{year}-12-01', freq='MS')
    ax.set_xticks(months)
    ax.set_xticklabels([dt.strftime('%b') for dt in months])
    
    # Set labels and title
    ax.set_xlabel('Month')
    ax.set_ylabel(str(year )+ '\nWeekly' + plot_col)

legend = fig.legend(shops, loc='center right', title='Shops')
legend.set_bbox_to_anchor((1.15, 0.5))
fig.suptitle('Earned Hours Progression by Year\n(3 Week Moving Average)')
plt.tight_layout()  # Adjust subplots for better spacing
plt.show()

#%%


