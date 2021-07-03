import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def extract_and_combine_data(order_book, date):
    dates = set()
    for root, _, files in os.walk(order_book, topdown=False):
        for name in files:
            dates.add(pd.to_datetime(name.split()[1].split('.csv')[0]).strftime('%Y-%m-%d'))
    print(list(dates)) # ['2020-08-01', '2020-08-15', '2020-08-02']
    order = []
    for root, _, files in os.walk(order_book, topdown=False):
        for name in files:
            if date in name:
                order.append(pd.read_csv(os.path.join(root, name)).set_index('receiveTs').sort_index())
    order = pd.concat(order)
    order.index = pd.to_datetime(order.index)
    return order

    # order_1 = []
    # order_2 = []
    # order_3 = []
    # for root, _, files in os.walk(order_book, topdown=False):
    #     for name in files:
    #         if '2020-08-01' in name:
    #             order_1.append(pd.read_csv(os.path.join(root, name)).set_index('receiveTs').sort_index())
    #         elif '2020-08-02' in name:
    #             order_2.append(pd.read_csv(os.path.join(root, name)).set_index('receiveTs').sort_index())
    #         elif '2020-08-15' in name:
    #             order_3.append(pd.read_csv(os.path.join(root, name)).set_index('receiveTs').sort_index())
    #         else:
    #             print('Unknow Date')

    # order_1 = pd.concat(order_1).apply(pd.to_numeric)
    # order_2 = pd.concat(order_2).apply(pd.to_numeric)
    # order_3 = pd.concat(order_3).apply(pd.to_numeric)
    # order_1.index = pd.to_datetime(order_1.index)
    # order_2.index = pd.to_datetime(order_2.index)
    # order_3.index = pd.to_datetime(order_3.index)
    # print(order_1.info())
    # print(order_2.info())
    # print(order_3.info())

    # return order_1, order_2, order_3

# turn the unevenly spaced time series into evenly freq spaced time series
def normalize_data(df, freq='s'):
    return df.resample(freq).last().ffill()

def generate_features(df):
    '''
    generate features:
        1. raw data, 11-level LOB data
        2. Time-insensitive data: spread & mid-price, price differences, price and volume means, accumulated differences
    :param df:
        evenly sampled time series LOB data
    :return:
        dataframe with features
    '''
    # Spread & Mid-price
    df['Mid'] = (df['Pa_1'] + df['Pb_1']) / 2
    df['Spread'] = df['Pa_1'] - df['Pb_1']
    # Price differences
    askcol = ['Pa_{}'.format(i) for i in range(1, 12)]
    bidcol = ['Pb_{}'.format(i) for i in range(1, 12)]
    askvol = ['Va_{}'.format(i) for i in range(1, 12)]
    bidvol = ['Vb_{}'.format(i) for i in range(1, 12)]
    for i in range(1, 11):
        df['askdiff_1_{}'.format(i + 1)] = df[askcol[i]].sub(df['Pa_1'])
        df['biddiff_1_{}'.format(i + 1)] = df[bidcol[i]].sub(df['Pb_1'])
    # Price & Volume means
    df['Mean_ask'] = df[askcol].mean(axis=1)
    df['Mean_bid'] = df[bidcol].mean(axis=1)
    df['Mean_ask_vol'] = df[askvol].mean(axis=1)
    df['Mean_bid_vol'] = df[bidvol].mean(axis=1)
    # Accumulated differences
    df['Accum_price'] = 0
    df['Accum_vol'] = 0
    for ask, bid in zip(askcol, bidcol):
        df['Accum_price'] += df[ask].sub(df[bid])
    df['Accum_price'].div(len(askcol))
    for askvol, bidvol in zip(askvol, bidvol):
        df['Accum_vol'] += df[askvol].sub(df[bidvol])
    df['Accum_vol'].div(len(askvol))
    return df

def generate_x_y(df, forecast_horizon, regression=True, threshold=0.002):
    '''
        split the dataframe into regressors X and response y according to forecast horizon, regression or classification
    :param df:
        dataframe with all features and response variable
        forecast_horizon: forecast horizon
        regression: regression or classification
        threshold: only used in classification

    :return:
        features X and response y
    '''

    if regression:
        # response variable is the mid price
        y = df.shift(-forecast_horizon)['Mid'][:-forecast_horizon]
        return df[:-forecast_horizon], y
    else:
        # calculates percentage change of mid-price as (midprice(t + forecast_horizon) - current mid price) / current mid price
        # assign label 1 if pct change > 0.2%, 2 if pct change < -0.2%, 3 otherwise
        df['pct_change'] = (df.shift(-forecast_horizon)['Mid'] - df['Mid']) / df['Mid']
        df['label'] = np.where(df['pct_change'] > threshold, 1, np.nan)
        df['label'] = np.where(df['pct_change'] < -threshold, 2, df['label'])
        df['label'] = df['label'].fillna(3)
        y = df['label'][:-forecast_horizon]
        return df.drop(['pct_change', 'label'], axis=1)[:-forecast_horizon], y

def train_test_split(data_x, data_y, train_size):
    assert data_x.shape[0] == len(data_y)
    n = len(data_y)
    n_train = int(n * train_size)
    x_train = data_x[:n_train]
    y_train = data_y[:n_train]
    x_test = data_x[n_train:]
    y_test = data_y[n_train:]
    return x_train, y_train, x_test, y_test
