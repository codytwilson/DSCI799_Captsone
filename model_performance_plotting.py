# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 18:36:57 2023

@author: codyt
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

deepleaerning_csvs = ['iTransformer_202311121024.csv','PatchTST_202311112005.csv',
                      'TimesNet_202311171325.csv','LSTM_202311271835.csv']

csv_dict = {}
for csv in deepleaerning_csvs:
    model_name = csv.split('_')[0]
    csv_dict[model_name] = pd.read_csv('.\\' + csv)


for j in ['Train','Test','Val']:
    plt.title(j + ' Epoch Performance')
    
    for i in csv_dict.keys():
        df = csv_dict[i].iloc[:10]
        plt.plot(df[j], label=i)
    
    plt.legend()
    plt.ylabel('MSE')
    plt.xlabel('Training Epoch')
    plt.show()
    
#%%

arima_csv = 'arima_202311291119.csv'
arimax_csv = 'arimaX_202311291159.csv'

def plot_arima(file, title):
    arima_df = pd.read_csv(file)
    
    def extract_values(row):
        tuple_values = eval(row)  # Using eval to convert string to tuple
        return pd.Series(tuple_values, index=['p', 'q', 'd'])
    arima_df = pd.concat([arima_df, arima_df['orders'].apply(extract_values)], axis=1)
    
    # Set a threshold for outlier detection (adjust as needed)
    val_threshold = 100
    
    # Create subplots with a common color scale
    fig, axes = plt.subplots(ncols=1, nrows=3, sharex=True, figsize=(5, 6))
    fig.suptitle(title + ' ARIMA Validation MSE')
    
    # Create a single Axes instance for the heatmap
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    
    
    arima_df = arima_df[arima_df['Val'] < val_threshold]
    # Calculate common vmin and vmax values
    vmin = arima_df['Val'].min()
    vmax = arima_df['Val'].max()
    
    # Iterate over each subplot
    for i, ax in enumerate(axes):
        ax.set_title(f'p-value {i}')
        ax.set_ylabel('q-value')
    
        # Filter DataFrame for the specific 'p' value and remove outliers
        df = arima_df[arima_df['p'] == i]
        # df = df[df['Val'] < val_threshold]
    
        # Create a 2D grid of 'q' and 'd' values
        q_values = sorted(df['q'].unique())
        d_values = sorted(df['d'].unique())
        heatmap_data = df.pivot(index='d', columns='q', values='Val').reindex(index=d_values, columns=q_values)
    
        # Create a heatmap using imshow with common vmin and vmax values
        heatmap = ax.imshow(heatmap_data, cmap='viridis', aspect='auto', interpolation='nearest', origin='lower', vmin=vmin, vmax=vmax)
    
    # Set labels
    axes[-1].set_xlabel('d')
    
    # Add colorbar with shared color scale
    cbar = fig.colorbar(heatmap, cax=cax, orientation='vertical')
    cbar.set_label('Val')
    
    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    fig.subplots_adjust(top=0.9)
    
    # Show the plot
    plt.show()

plot_arima(arima_csv, 'Univariate')
plot_arima(arimax_csv, 'Multivariate')

#%%

ma_file = 'ma_202311291319.csv'
ma_df = pd.read_csv(ma_file)


plt.plot(ma_df['window_size'], ma_df['Val'])
plt.xlabel('Window Size - Hours')
plt.ylabel('MSE')
plt.title('Moving Average Validation MSE')
plt.show()


#%%

'''
Plot bar chart drawn in notebook for section 'FINAL Selection'

'''