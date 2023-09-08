# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 16:40:45 2023

@author: codyt
"""



import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

def smape(y_test, y_pred):
    numerator = np.abs(y_test-y_pred)
    denominator = (y_test + np.abs(y_pred)) /200
    return np.mean(np.divide(numerator,denominator))

smape_scorer = make_scorer(smape, greater_is_better=False)


for csv in ['CSM', 'CSF', 'FED']:
    filepath = 'C:\\Users\\codyt\\Documents\\DSCI 799 - Capstone\\Data\\Archive_' + csv + '.csv'
    temp_df = pd.read_csv(filepath)
    temp_df['Shop'] = csv
    if csv == 'CSM':
        df = temp_df
    else:
        df = pd.concat([df, temp_df.reset_index(drop=True)])
# training_csv = 'C:\\Users\\cwilson\\Documents\\Python\\Speedo_Dashboard\\Archive_CSM.csv'
# df = pd.read_csv(training_csv)
df = df.drop(columns=['IsReal','Total Hours'])
df = df.rename(columns={'Date': 'StartOfWeek'})
# df = df[df['StartOfWeek'] == '08/20/2023']
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
df = df[~df['Timestamp'].isna()]
df['StartOfWeek'] = pd.to_datetime(df['StartOfWeek'])
df['EndOfWeek'] = df['StartOfWeek'] + datetime.timedelta(days=6,hours=23,minutes=59)
df['PercentageOfWeek'] = ((7 * 24 * 60 * 60) - (df['EndOfWeek'] - df['Timestamp']).dt.total_seconds()) / (7 * 24 * 60 * 60)
df = df[~((df['Earned Hours'] == 0) & (df['Direct Hours'] == 0))]
df = df.sort_values(by=['Shop','Timestamp'])

X = df[['PercentageOfWeek', 'Direct Hours','Indirect Hours', 'Number Employees', 'Tons','Quantity Pieces']].to_numpy()
y = df['Earned Hours'].to_numpy()

#%%
shop_colors = {'CSM':'blue', 'CSF':'red', 'FED':'green'}
for shop in shop_colors.keys():
    
    color = shop_colors[shop]
    
    df_to_plot = df[df['Shop'] == shop]
    plt.plot(df_to_plot['PercentageOfWeek'], df_to_plot['Earned Hours'], color, linewidth=0.5)
    
plt.xlabel('Percentage of Week Completed')
plt.ylabel('Earned Hours')
plt.title('Earned Hours Across each Week')
plt.legend(list(shop_colors.keys()))
    

#%%
for shop in shop_colors.keys():
    
    color = shop_colors[shop]
    
    df_to_plot = df[df['Shop'] == shop]
    plt.plot(df_to_plot['Timestamp'], df_to_plot['Earned Hours'], color, linewidth=0.5)
    # plt.plot(df_to_plot['Timestamp'], df_to_plot['Efficiency'], color, linewidth=0.5)
    
# plt.xlabel('Percentage of Week Completed')
plt.ylabel('Efficiency')
plt.title('Weekly Earned Hours')
# plt.ylim((0,2.5))`
plt.legend(list(shop_colors.keys()))



#%%

for shop in shop_colors.keys():
    
    color = shop_colors[shop]
    
    df_to_plot = df[df['Shop'] == shop]
    df_to_plot['Earned Hours Cumulative'] = df_to_plot['Earned Hours'].cumsum()
    plt.plot(df_to_plot['Timestamp'], df_to_plot['Earned Hours Cumulative'], color, linewidth=0.5)
    
# plt.xlabel('Percentage of Week Completed')
plt.ylabel('Earned Hours')
plt.title('Cumulative Earned Hours')
plt.legend(list(shop_colors.keys()))

#%% univariate
tester_date = datetime.datetime(year=2023, month=8, day=20)
df_max = df[df['StartOfWeek'] == tester_date]

df_even_interval = pd.DataFrame(data=np.arange(0,7*24*60+15,15), columns=['Offset'])
df_even_interval['StartOfWeek'] = tester_date
df_even_interval['Offset'] = pd.to_timedelta(df_even_interval['Offset'], unit='m')
df_even_interval['Timestamp'] = df_even_interval['Offset'] + df_even_interval['StartOfWeek']
df_even_interval['Type'] = 'NiceAndEven'
df_max['Type'] = 'real'
df_mixed = pd.concat([df_even_interval[['StartOfWeek','Timestamp']], df_max])



#%%

x = df_max['PercentageOfWeek'].to_numpy().reshape(-1,1)
y = df_max['Earned Hours'].to_numpy().reshape(-1,1)
reg = LinearRegression(fit_intercept=False)
reg.fit(x, y)
reg_pred = reg.predict(x)

reg20 = LinearRegression(fit_intercept=False)
index_20 = int(len(x) * 0.2)
reg20.fit(x[:index_20], y[:index_20])
reg20_pred = reg20.predict(x)

reg40 = LinearRegression(fit_intercept=False)
index_40 = int(len(x) * 0.4)
reg40.fit(x[:index_40], y[:index_40])
reg40_pred = reg40.predict(x)

reg60 = LinearRegression(fit_intercept=False)
index_60 = int(len(x) * 0.6)
reg60.fit(x[:index_60], y[:index_60])
reg60_pred = reg60.predict(x)

plt.plot(df_max['PercentageOfWeek'], df_max['Earned Hours'])
plt.plot(df_max['PercentageOfWeek'], reg_pred)
plt.plot(df_max['PercentageOfWeek'], reg20_pred)
plt.plot(df_max['PercentageOfWeek'], reg40_pred)
plt.plot(df_max['PercentageOfWeek'], reg60_pred)
plt.xlabel('% of Week Complete')
plt.ylabel('Earned Hours')
plt.legend(['Actual',
            'Prediction (all)',
            'Prediction (1st 20%)',
            'Prediction (1st 40%)',
            'Prediction (1st 60%)'])


#%%

model = Sequential()
model.add(LSTM(32, input_shape=(1, 20)))
model.add(Dense(1))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model and track loss history
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))
