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

fablisting = load_and_combine_fablisting_csv_to_df()
fablisting = fablisting[['Timestamp','Job #','Lot #','Lot Name','Quantity','Piece Mark - REV','Weight','Earned Hours','Has Model','shop']]


jobEncoder = LabelEncoder().fit(pd.unique(fablisting['Job #']))


missing_timestamp = fablisting[fablisting['Timestamp'].isna()]
missing_earned_hours = fablisting[(fablisting['Earned Hours'].isna()) | 
                                  (fablisting['Earned Hours'] == 0) | 
                                  (fablisting['Earned Hours'] == np.inf) | 
                                  (fablisting['Earned Hours'] == -np.inf)].copy()
missing_has_model = fablisting[~ fablisting['Has Model']]
missing_has_model_and_earned_hours = missing_has_model[missing_has_model['Earned Hours'].isna()]
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



fablisting_without_missing = pd.merge(fablisting.copy(), missing_earned_hours, indicator=True, how='left')

# this gets us a timestamp form the datetime apparently
fablisting_without_missing['kmeansTimestamp'] = fablisting_without_missing['Timestamp'].astype(np.int64) // 10 ** 9
# get rid of the missing ones
fablisting_without_missing = fablisting_without_missing[fablisting_without_missing['_merge'] == 'left_only']
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



rf_vars = ['Weight','Has Model','shop','kmeansTimestamp','Job #']


shopEncoder = LabelEncoder().fit(pd.unique(fablisting_without_missing['shop']))
fablisting_without_missing['shop'] = shopEncoder.transform(fablisting_without_missing['shop'])
fablisting_without_missing['Job #'] = jobEncoder.transform(fablisting_without_missing['Job #'])




rf = RandomForestRegressor(random_state=0)
rf.fit(fablisting_without_missing[rf_vars], fablisting_without_missing['Earned Hours'])


for i in range(0,len(rf_vars)):
    print(rf_vars[i], rf.feature_importances_[i])


fablisting_without_missing['Earned Hours rf'] = rf.predict(fablisting_without_missing[rf_vars])
fablisting_without_missing['shop'] = shopEncoder.inverse_transform(fablisting_without_missing['shop'])
fablisting_without_missing['Job #'] = jobEncoder.inverse_transform(fablisting_without_missing['Job #'])






missing_earned_hours['shop'] = shopEncoder.transform(missing_earned_hours['shop'])
missing_earned_hours['Job #'] = jobEncoder.transform(missing_earned_hours['Job #'])
missing_earned_hours['Earned Hours rf'] = rf.predict(missing_earned_hours[rf_vars])
missing_earned_hours['shop'] = shopEncoder.inverse_transform(missing_earned_hours['shop'])
missing_earned_hours['Job #'] = jobEncoder.inverse_transform(missing_earned_hours['Job #'])








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









