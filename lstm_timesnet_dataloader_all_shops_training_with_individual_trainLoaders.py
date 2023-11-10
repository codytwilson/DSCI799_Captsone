# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 14:08:56 2023

@author: cwilson
"""

'''
https://www.crosstab.io/articles/time-series-pytorch-lstm/
'''
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

df = pd.read_csv('.\\data\\data_for_ml.csv', index_col=0)
df = df.set_index('Timestamp')
df.index = pd.to_datetime(df.index)
# df = df[df['Shop'] == 'CSM']

weekdays = {0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}
df['Weekday'] = df['Weekday'].apply(lambda x: weekdays[x])
weekdays_oh = pd.get_dummies(df['Weekday'])
df = df.drop(columns=['Weekday'])
df = pd.concat([df, weekdays_oh], axis=1)

df['Manhours'] = df['Manhours'].rolling(window=3, min_periods=0).mean()
# df['Worth'] = df['Worth'].rolling(window=24, min_periods=0).mean()
# i = 27
# this is basically the amount of training units which is 4 weeks to predict next hours value
sequence_length = 24*7*6
# forecast length
forecast_units = 24*7*4
# extra for testing so we can calculate bigger period of test mse
extra_testing_units = 24*7*12
# forecast_units = 5
batch_size = 50


df_shop_oh = pd.get_dummies(df['Shop'])
'''
# df_shop_oh_index = df_shop_oh.index
# df_shop_oh = df_shop_oh.reset_index()
# 
df_shop = df['Shop']
df = df.drop(columns=['Shop'])
df = pd.concat([df, df_shop_oh], axis=1)
# ???
# I can zero pad the number for sequence_length between each shop's ?

df = df.reset_index()
empty_data = [np.nan] + [0] * (len(df.columns)-1)
empty_df_like = pd.DataFrame(columns=df.columns, data=[empty_data], index=range(0,sequence_length))
# basically zero padding the seuqnece length between shops so no interaction between shop data
testing_dfs = {}
for shop in df_shop_oh.columns:
    testing_dfs[shop] = {}
    # start index of the shop
    min_idx = df[df[shop] == True].index.min()
    # end index of the shop
    max_idx = df[df[shop] == True].index.max()
    # get df before start of the shop
    before = df.iloc[:min_idx, :]
    # get df after the end of the shop
    after = df.iloc[max_idx:, :]
    # subject is inbetween min and max indexes
    subject = df.iloc[min_idx:max_idx, :]
    # get the test df - go back sequence length + forecast units + some extra
    test = subject.iloc[-sequence_length-forecast_units-extra_testing_units:, :]
    # stick it in the dict
    testing_dfs[shop]['df'] = test.set_index('Timestamp')
    # get rid of the testing bits
    remove_from_subject = ~subject.index.isin( test.index )
    subject = subject.loc[remove_from_subject, :] 
    
    
    infill_index = pd.date_range(subject['Timestamp'].max() + pd.DateOffset(hours=1), periods=len(empty_df_like), freq='H')
    empty_df_like['Timestamp'] = infill_index

    # recreate the dataframe with empty padding ebtween shops
    df = pd.concat([before, subject, empty_df_like, after], axis=0, ignore_index=True)

df = df.set_index('Timestamp')

'''






target = 'Worth'
features = ['Quantity', 'Weight', 'OustandingWeight', 'Manhours', 'DayShift'] + list(weekdays.values())
# features = ['Quantity', 'Weight', 'OustandingWeight', 'Manhours', 'DayShift'] + list(weekdays.values())
# features = ['Weight', 'OustandingWeight', 'Manhours'] + list(df_shop_oh.columns)

shop = df['Shop']
df_to_tensor = df.copy()[features + [target]]
scaler_x = MinMaxScaler()
scaler_x.fit(df_to_tensor[features])
scaler_y = MinMaxScaler()
scaler_y.fit(df_to_tensor[[target]])
df_to_tensor[features] = scaler_x.transform(df_to_tensor[features])
df_to_tensor[target] = scaler_y.transform(df_to_tensor[[target]])
df_to_tensor['Shop'] = shop
print('Dataset sizes\n--------------------')
print(f"Train:  {df_to_tensor.shape}")
testing_dfs = {k:{} for k in df_shop_oh.columns}

for i,k in enumerate(testing_dfs.keys()):
    testing_dfs[k]['df_ready'] = df_to_tensor[df_to_tensor['Shop'] == k].copy()[features + [target]]
    train,test = train_test_split(testing_dfs[k]['df_ready'], test_size = 0.3, shuffle=False)
    val,test = train_test_split(test, test_size=0.4, shuffle=False)
    testing_dfs[k]['df_train'] = train.copy()
    testing_dfs[k]['df_test'] = test.copy()
    testing_dfs[k]['df_val'] = val.copy()
    print(f'Train  {i}:  {train.shape}')
    print(f'Val    {i}:  {val.shape}')
    print(f'Test   {i}:  {test.shape}')


class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length, pred_len, label_len=0):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.label_len = label_len
        self.pred_len = pred_len
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()
  
    
    def __getitem__(self, index):
        #given an index, calculate the positions after this index to truncate the dataset
        s_begin = index
        s_end = s_begin + self.sequence_length
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
    
        #input and output sequence
        seq_x = self.X[s_begin:s_end]
        seq_y = self.y[r_begin:r_end]
    
        #time mark
        #seq_x_mark = self.data_stamp[s_begin:s_end]
        #seq_y_mark = self.data_stamp[r_begin:r_end]
    
        return seq_x, seq_y#, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.X) - self.sequence_length - self.pred_len + 1
     
    


# train_dataset = SequenceDataset(
#     df_train,
#     target = target,
#     features = features,
#     sequence_length = sequence_length,
#     pred_len = forecast_units)

# train_dataset = SequenceDataset(
#     df_to_tensor,
#     target = target,
#     features = features,
#     sequence_length = sequence_length,
#     pred_len = forecast_units)


for k in testing_dfs.keys():
    
    testing_dfs[k]['train_dataset'] = SequenceDataset(
                                                testing_dfs[k]['df_train'],
                                                target = target,
                                                features = features,
                                                sequence_length = sequence_length,
                                                pred_len = forecast_units)
    testing_dfs[k]['train_loader'] = DataLoader(testing_dfs[k]['train_dataset'], batch_size=batch_size, shuffle=True)
    
    testing_dfs[k]['test_dataset'] = SequenceDataset(
                                                testing_dfs[k]['df_test'],
                                                target = target,
                                                features = features,
                                                sequence_length = sequence_length,
                                                pred_len = forecast_units)
    testing_dfs[k]['test_loader'] = DataLoader(testing_dfs[k]['test_dataset'], batch_size=batch_size, shuffle=False)
    
    testing_dfs[k]['val_dataset'] = SequenceDataset(
                                                testing_dfs[k]['df_val'],
                                                target = target,
                                                features = features,
                                                sequence_length = sequence_length,
                                                pred_len = forecast_units)
    testing_dfs[k]['val_loader'] = DataLoader(testing_dfs[k]['val_dataset'], batch_size=batch_size, shuffle=False)    
    
    testing_dfs[k]['train_eval_loader'] = DataLoader(testing_dfs[k]['train_dataset'], batch_size=1, shuffle=False)

    X,y = next(iter(testing_dfs[k]['train_loader']))
    print(f'Features size: {X.shape}')
    print(f'Target size: {y.shape}')
    


# train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
# test1_loader = DataLoader(test1_dataset, batch_size = batch_size, shuffle=False)
# test2_loader = DataLoader(test2_dataset, batch_size = batch_size, shuffle=False)
# test3_loader = DataLoader(test3_dataset, batch_size = batch_size, shuffle=False)

# val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False)
# get an unshuffled version for later
# train_eval_loader = DataLoader(train_dataset, batch_size = 1, shuffle=False)






class myLSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units, pred_len):
        super().__init__()
        
        # number of features
        self.num_sensors = num_sensors
        self.hidden_units = hidden_units
        self.num_layers = 1
        
        # self.linear_init = nn.Linear(num_sensors, num_sensors)
        
        self.lstm = nn.LSTM(
            input_size = num_sensors,
            hidden_size = hidden_units,
            batch_first = True,
            num_layers = self.num_layers,
            bidirectional = True
        )
        # self.linear = nn.Sequential(
        #     nn.Linear(in_features = self.hidden_units, out_features = 16),
        #     nn.LayerNorm(16),
        #     nn.ReLU(16),
        #     nn.Linear(16, 8),
        #     nn.LayerNorm(8),
        #     nn.ReLU(8),
        #     nn.Linear(8, pred_len),
        #     nn.ReLU(pred_len)
        #     )
        self.linear = nn.Sequential(
            nn.LayerNorm(self.hidden_units),
            nn.Linear(in_features = self.hidden_units, out_features = 16),
            nn.LayerNorm(16),
            nn.ReLU(16),
            nn.Linear(16, pred_len),
            nn.ReLU(pred_len)
            )
        # self.linear = nn.Linear(in_features = self.hidden_units, out_features = pred_len)
        
    def forward(self, x):
        
        # init = self.linear_init(x)
        
        batch_size = x.shape[0]
        # print(batch_size)
        h0 = torch.zeros(self.num_layers * 2,
                         batch_size,
                         self.hidden_units
                         ).requires_grad_()
        c0 = torch.zeros(self.num_layers * 2,
                         batch_size,
                         self.hidden_units
                         ).requires_grad_()
        
        _, (hn, _) = self.lstm(x, (h0, c0))
        
        out = self.linear(hn[0]).flatten()
        
        # return out
        # I hope this is the right way to reshape it # ???
        # it is return 'out' as pred_len * batch_size 
        return out.reshape(x.shape[0], out.shape[0]//x.shape[0])


#%%
learning_rate = 1e-5
# weight_decay = 0.001
num_hidden_units = 8

model = myLSTM(num_sensors = len(features), hidden_units = num_hidden_units, pred_len=forecast_units)
loss_function = nn.MSELoss()
# this is mse when y-yhat > 1
# loss_function = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print(model)

def train_model(dataloader, model, loss_function, optimier):
    num_batches = len(dataloader)
    total_loss = 0
    # flag training mode
    model.train()
    
    for batch, (X,y) in enumerate(dataloader):
        # print(batch, X.shape, y.shape)
        # break
        output = model(X)
        # calculate loss as the difference in cumulative worth at the end of the period
        loss = loss_function(output.sum(axis=1), y.sum(axis=1))
        # loss = loss_function(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"[{current:>5d}/{num_batches*len(X):>5d}]    loss: {loss:>7f}")
    
    avg_loss = total_loss / num_batches
    print(f'Train Loss: {avg_loss}')
    return avg_loss
    
def test_model(dataloader, model, loss_function):
    num_batches = len(dataloader)
    total_loss = 0
    model.eval()
    
    with torch.no_grad():
        for X,y in dataloader:
            # print(X.shape, y.shape)
            # break
            output = model(X)
            # calculate loss as the difference in cumulative worth at the end of the period
            total_loss += loss_function(output.sum(axis=1), y.sum(axis=1)).item()
            # total_loss += loss_function(output, y).item()

    
    total_loss = total_loss
    avg_loss = total_loss / num_batches
    print(f'Test Loss:  {avg_loss}\n')
    return avg_loss

def predict(dataloader, model):
    
    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X,_ in dataloader:
            # print(X.shape)
            y_hat = model(X)
            output = torch.cat((output, y_hat), 0)
    
    return output

def predict_singular(X, model):
    X = X.reshape(1, sequence_length, len(features))
    model.eval()
    with torch.no_grad():
        y_hat = model(X)
    return y_hat

losses_train, losses_val = [], []

#%%

print('untrained test\n---------------------')
for i,k in enumerate(testing_dfs.keys()):
    print(f'Test {i}:', end='\t')
    test_model(testing_dfs[k]['test_loader'], model, loss_function)

print('\n')



epochs = 20
for epoch in range(epochs):
    print(f'Epoch #{epoch}----------------------')
    epoch_train_loss, epoch_val_loss = [], []
    for k in testing_dfs.keys():
        loss_train = train_model(testing_dfs[k]['train_loader'], model, loss_function, optimizer)
        epoch_train_loss.append(loss_train)
    # do the val checks after training all 3 
    for k in testing_dfs.keys():
        loss_val = test_model(testing_dfs[k]['val_loader'], model, loss_function)
        epoch_val_loss.append(loss_val)
    
        
    # loss_val = test_model(val_loader, model, loss_function)
    
    losses_train.append(epoch_train_loss)
    losses_val.append(epoch_val_loss)
    
losses_train_total = [sum(i) for i in losses_train]
losses_val_total = [sum(i) for i in losses_val]

fig,(ax0,ax1) = plt.subplots(ncols=1, nrows=2)
ax0.plot(losses_train_total, label='Train Loss')
ax0.plot(losses_val_total, label='Val Loss')
ax0.set_title('Total')
ax0.set_yscale('log')
ax0.legend(loc='best')
ax1.plot(losses_train, label=['Train1','Train2','Train3'])
ax1.plot(losses_val, label=['Val1','Val2','Val3'])
ax1.set_yscale('log')
ax1.legend(loc='best')
ax1.set_xlabel('Epoch #')
# plt.plot(losses_train_total)
# plt.plot(losses_val_total)
# plt.plot(losses_train, label=['Loss 1','Loss 2','Loss 3'])
# plt.plot(losses_val, label=['Val 1','Val 2','Val 3'])
# plt.legend(['Train Loss','Val Loss', 'Loss 1','Loss 2','Loss 3','Val 1','Val 2','Val 3'])
# plt.yscale('log')
# plt.show()


print('trained test\n---------------------')
for i,k in enumerate(testing_dfs.keys()):
    print(f'Test {i}:', end='\t')
    test_model(testing_dfs[k]['test_loader'], model, loss_function)
print('\n')





#%%

yhat_col = 'Model Forecast'

def determine_if_inverse_transform_needed(df):
    if df[target].max() < 3:
        df[target] = scaler_y.inverse_transform(df[[target]])
    if df[features].max().max() < 3:
        df[features] = scaler_x.inverse_transform(df[features])
    if df[yhat_col].max() < 3:
        df[yhat_col] = scaler_y.inverse_transform(df[[yhat_col]])
    return df


def average_predictions(preds, df):
    from scipy.stats import t
    df_rows = df.shape[0]
    preds_placeholder = np.empty((df_rows - sequence_length, preds.shape[0]))
    preds_placeholder[:] = np.nan
    for i in range(preds.shape[0]):
        if i+forecast_units <= preds_placeholder.shape[0]:
            preds_placeholder[i:i+forecast_units, i] = preds[i,:]
        else:
            preds_placeholder[i:i+forecast_units-1, i] = preds[i,:-1]
    preds_placeholder_mean = np.nanmean(preds_placeholder, axis=1)
    preds_placeholder_std = np.nanstd(preds_placeholder, axis=1)
    preds_placeholder_count = np.count_nonzero(~np.isnan(preds_placeholder), axis=1)
    critical_value = t.ppf(1 - (1-0.97)/2, preds_placeholder_count-1)
    preds_placeholder_CI_upper = preds_placeholder_mean + critical_value * preds_placeholder_std / np.sqrt(preds_placeholder_count)
    preds_placeholder_CI_lower = preds_placeholder_mean - critical_value * preds_placeholder_std / np.sqrt(preds_placeholder_count)
    return preds_placeholder_mean, preds_placeholder_CI_upper, preds_placeholder_CI_lower


# test_preds = predict(test_loader, model)
# test_pred_mean, test_pred_upper, test_pred_lower = average_predictions(test_preds, df_test)
# start_idx = df_test.index[-len(test_pred_mean)]
# df_test.loc[start_idx:,f'{yhat_col}_mean'] = scaler_y.inverse_transform(test_pred_mean.reshape(-1,1))
# df_test.loc[start_idx:,f'{yhat_col}_upper'] = scaler_y.inverse_transform(test_pred_upper.reshape(-1,1))
# df_test.loc[start_idx:,f'{yhat_col}_lower'] = scaler_y.inverse_transform(test_pred_lower.reshape(-1,1))

# plt.plot(df_test.index, df_test[[target, f'{yhat_col}_mean', f'{yhat_col}_upper', f'{yhat_col}_lower']])
# plt.ylim(min(df_test[target]), max(df_test[target]))



def compare_predictions(dataset, df_to_check, resampler, random_index, neg_2_zero=True, title=''):   
    preds = predict_singular(dataset[random_index][0], model).numpy()
    df_to_check[yhat_col] = np.nan
    idx_start = random_index + sequence_length - 1
    idx_end = idx_start + forecast_units
    df_to_check.iloc[idx_start : idx_end, -1] = preds.reshape(-1,1)
    df_to_check = determine_if_inverse_transform_needed(df_to_check)
    if neg_2_zero:
        df_pred_subzero = df_to_check[yhat_col] < 0
        df_to_check.loc[df_pred_subzero, yhat_col] = 0
    
    # df_plot = df_to_check[~df_to_check[yhat_col].isna()]
    df_plot = df_to_check.copy()
    df_plot.index = pd.to_datetime(df_plot.index)
    df_plot = df_plot.resample(resampler).sum()
    
    
    # fig,ax = plt.subplots()
    # ax.plot(df_plot[['Worth','Model Forecast']])
    # ax.legend(['Actual','Pred'], loc='best')
    # ax.set_title('Actual vs Predicted Worth')
    # fig.suptitle(title)
    # plt.xticks(rotation=45)
    # plt.show()
    
    df_plot = df_to_check[~df_to_check[yhat_col].isna()]
    # df_plot = df_to_check.copy()
    df_plot.index = pd.to_datetime(df_plot.index)
    df_plot = df_plot.resample(resampler).sum()    
    
    for col in [target, yhat_col]:
        df_plot[f'{col}_cumulative'] = df_plot[col].cumsum()
    
    df_plot['Difference_cumulative'] = abs(df_plot['Worth_cumulative'] - df_plot['Model Forecast_cumulative'])
    df_plot['Difference_percentage'] = df_plot['Difference_cumulative'] / df_plot['Worth_cumulative']
    df_plot['Difference_percentage'] = df_plot['Difference_percentage'].apply(lambda x: x if abs(x) < 2 else np.nan)
    
    fig, ax = plt.subplots()
    plt.xticks(rotation=45)
    ax.plot(df_plot[['Worth_cumulative']], label='Actual')
    ax.plot(df_plot[['Model Forecast_cumulative']], label='Pred')
    ax.set_ylabel('Worth')
    ax.set_title('Cumulative Worth')
    ax0 = ax.twinx()
    ax0.plot(df_plot['Difference_percentage']*100, color='red', linestyle='--', label='Diff')
    # this is hokie but it gets the value in the legend
    ax.plot(df_plot['Worth_cumulative'].iloc[0:1], color='red', linestyle='--', label='Diff')
    ax0.set_ylabel('Cumulative Difference (Percentage)')
    ax.legend(loc='best')
    fig.suptitle(title)


for i,k in enumerate(testing_dfs.keys()):
    compare_predictions(testing_dfs[k]['test_dataset'], 
                        testing_dfs[k]['df_test'], 
                        '24H', 
                        len(testing_dfs[k]['test_dataset']), 
                        neg_2_zero=True, 
                        title=f'Test {i}: Final Test Pred')
    
    # compare_predictions(testing_dfs[k]['train_dataset'], 
    #                     testing_dfs[k]['df_train'], 
    #                     '24H', 
    #                     len(testing_dfs[k]['train_dataset']) - 1000, 
    #                     neg_2_zero=True, 
    #                     title=f'Test {i}: Final Test Pred')   
    
    
 

#%%

def save_model():
    torch.save(obj=model.state_dict(), f='C:\\Users\\cwilson\\downloads\\capstone_lstm_individual_trainers_per_shop1.pth')

def load_model():
    model = myLSTM()
    model.load_state_dict(torch.load('C:\\Users\\cwilson\\downloads\\capstone_lstm_0.pth'))
