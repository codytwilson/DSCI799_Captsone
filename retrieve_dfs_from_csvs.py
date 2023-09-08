# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 10:30:17 2023

@author: codyt
"""


import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt


def load_and_combine_archive_csv_to_df():
    for csv in ['CSM', 'CSF', 'FED']:
        filepath = '.\\Data\\Archive_' + csv + '.csv'
        temp_df = pd.read_csv(filepath)
        temp_df['Shop'] = csv
        if csv == 'CSM':
            df = temp_df
        else:
            df = pd.concat([df, temp_df.reset_index(drop=True)])
            
    return df


def load_production_worksheet_csv_to_df():
    filepath = '.\\Data\\production_worksheet.csv'
    df = pd.read_csv(filepath)
    
    cutoff_idx = df.index[df['2'] == 'Add new lines above here'][0]
    
    df = df.loc[:cutoff_idx - 1,:]
    return df
   
 
def get_production_worksheet_boundaries(df):
    # get the first column
    col0 = df.iloc[:,0]
    # find the index where we have 'Job #'
    header_index = col0.index[col0 == "Job #"][0]
    # the headers row 
    headers = df.loc[header_index, :]
    # reset the index to ints count
    headers = headers.reset_index(drop=True)
    # get the lsat column we want - which is a space as of 9/7/2023
    last_col = headers.index[headers.str.strip() == ''][0]

    return (header_index, last_col)    
 
    
def get_job_data_from_production_worksheet(df):
    bounds = get_production_worksheet_boundaries(df)
    # the headers row 
    headers = df.loc[bounds[0], :].copy()
    
    # cut the df on the header_idnex and last_column
    df_out = df.iloc[bounds[0]+1:, :bounds[1]+1]
    # rename that last column to shop
    headers[bounds[1]] = 'Shop'
    # set the headers to be headers series
    df_out.columns = headers[:bounds[1]+1]
    # rename some stuff to nicer names
    df_out = df_out.rename(columns={'Hrs./Ton':'HPT', 
                                    'Seq. #':'Sequence',
                                    'Req. Hrs.':'ReqHours'})    
    # drop the shit we dont have a use for
    df_out = df_out.drop(columns=['Hours Worked',
                                  'Tons Completed',
                                  'Remaining Hrs.',
                                  'Complete',
                                  'Headcount'])
    # cleanup the shop column
    df_out['Shop'] = df_out['Shop'].str.strip()
    # convert HPT to numbers
    df_out['HPT'] = df_out['HPT'].replace('\,','', regex=True)
    df_out['HPT'] = pd.to_numeric(df_out['HPT'])
    # convert tonnage to numbers
    df_out['Tonnage'] = df_out['Tonnage'].replace('\,','', regex=True)
    df_out['Tonnage'] = pd.to_numeric(df_out['Tonnage'])
    # convert Required Horus to number
    df_out['ReqHours'] = df_out['ReqHours'].replace('\,','', regex=True)
    df_out['ReqHours'] = pd.to_numeric(df_out['ReqHours'])
    
    
    # spit it out
    return df_out


def get_timeline_data_from_production_worksheet(df):
    bounds = get_production_worksheet_boundaries(df)
    # get just the time portion of the df
    df_narrower = df.iloc[:, bounds[1]:]
    # add the job number & sequence back onto the left
    df_narrower = pd.concat([df.iloc[:,[0,2]], df_narrower], axis=1)
    # add in the name for the shop column
    df_narrower.iloc[bounds[0], 2] = 'Shop'
    
    # is_past = df_narrower.iloc[1, 2:]
    # get the year and week start rows
    dates = df_narrower.iloc[bounds[0]-1:bounds[0]+1, 3:]
    # transpose it
    dates = dates.transpose()
    # forward fill missing values of the year column
    dates[2] = dates[2].fillna(method='ffill')
    # get only the first 4 characters in the year column
    dates[2] = dates[2].str[:4]
    # combine year and date
    dates['date'] = dates[2] + '/' + dates[3]
    # convert to datetime
    dates['date'] = pd.to_datetime(dates['date'], format='%Y/%m/%d', errors='coerce')
    
    # extract the headers
    headers = df_narrower.iloc[bounds[0], :].copy()
    # replace the headers with the dates
    headers.iloc[3:] = dates['date']
    # get rid of the first couple of rows in the dataframe now
    df_narrower = df_narrower.iloc[bounds[0]+1:, :]
    # replace the headers
    df_narrower.columns = headers
    # cleanup the shop column
    df_narrower['Shop'] = df_narrower['Shop'].str.strip()
    # before replacing the strings to numbers, lets change NAN to zero
    df_narrower.iloc[:, 3:] = df_narrower.iloc[:, 3:].fillna(0)
    # replace any commas with blanks
    df_narrower.iloc[:, 3:] = df_narrower.iloc[:, 3:].replace('[^0-9]','', regex=True)
    # convert the hours to numbers
    df_narrower.iloc[:, 3:] = df_narrower.iloc[:, 3:].apply(pd.to_numeric, errors='coerce')
    # and send the remaining shit to zero
    df_narrower.iloc[:, 3:] = df_narrower.iloc[:, 3:].fillna(0)    
    # get rid of NaT columns
    df_narrower = df_narrower.loc[:, ~df_narrower.columns.isna()]
    
    return df_narrower


def get_timeline_data_as_cumsum(timeline_df):
    # first 3 columns are identification purposes
    date_columns = timeline_df.columns[3:]
    # make a copy
    cumulative_df = timeline_df.copy()
    # Calculate cumulative hours worked for each date column
    cumulative_df[date_columns] = timeline_df[date_columns].cumsum(axis=1)
    
    return cumulative_df


def cleanup_archive_df(df):
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
    return df
#%%
df = load_and_combine_archive_csv_to_df()
df = cleanup_archive_df(df)

pw = load_production_worksheet_csv_to_df()
# job_data.shape[0] == time_data.shape[0]
job_data = get_job_data_from_production_worksheet(pw)
time_data = get_timeline_data_from_production_worksheet(pw)

# this represents the amount of work completed on the job
time_data_cumulative = get_timeline_data_as_cumsum(time_data)
# this represents the amount of work remaining on the job

# they should have the same index
time_data_remaining = time_data_cumulative.copy()
# Subtract the Required Hours from the cumulative horus worked
# negative numbers means amount of future work
# positive number means overworked the job
time_data_remaining.iloc[:, 3:] = time_data_cumulative.iloc[:,3:].subtract(job_data['ReqHours'], axis=0)
# convert any 'overwork' to zero ?
# time_data_remaining.iloc[:, 3:] = time_data_remaining.iloc[:, 3:].applymap(lambda x: 0 if x > 0 else x)

