
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler


def process_data(path):
    """
    Retrieves and preprocesses the base dataset
    """
    data = {}
    for root, _, files in os.walk(path, topdown=False):
        for file in files:
            if file.endswith('.csv'):
                data[file.split('.csv')[0]] = pd.read_csv(os.path.join(root, file)).set_index('time').sort_index()

    data = pd.concat([data['ask'], data['bid'], data['askVol'], data['bidVol'], data['rtn']],
                     keys=('ask', 'bid', 'askVol', 'bidVol', 'returns'), axis=1)
    data.columns.names = ['features', 'tickers']
    data.dropna(inplace=True)

    return data

def generate_features(data, freq, lags):
    data = data.iloc[::freq, :]
    data = data.stack('tickers').swaplevel().sort_index(level='tickers')
    data['mid'] = (data['ask'] + data['bid']) / 2
    data['spread'] = (data['ask'] - data['bid']) / 2
    data['imbalance'] = data['askVol'] - data['bidVol']
    data['avgVol'] = (data['askVol'] + data['bidVol']) / 2
    data['imbal_ratio'] = data['imbalance'].div(data['avgVol'])

    for lag in range(1, lags + 1):
        data[f'midChange_lag{lag}'] = np.log(data['mid'] / data['mid'].shift(lag))
        data[f'mom_lag{lag}'] = data[f'midChange_lag{lag}'].sub(data.midChange_lag1)

    return data.fillna(0)

def train_test_split(data, train_size):
    train_idx = int(data.shape[0] * train_size)
    train_data = data[:train_idx]
    test_data = data[train_idx:]
    return train_data, test_data

def generate_signal(data, train_size, model, fitPCA=True, n_components=5):
    train_data, test_data = train_test_split(data, train_size)
    train_x = train_data.drop('returns', axis=1)
    train_y = np.sign(train_data['returns'])
    test_x = test_data.drop('returns', axis=1)
    test_y = np.sign(test_data['returns'])

    if fitPCA:
        pipe = make_pipeline(MinMaxScaler(), PCA(n_components=n_components))
        train_x = pipe.fit_transform(train_x)
        test_x = pipe.transform(test_x)
    if model == 'LinearRegression':
        model = LinearRegression()
    elif model == 'LogisticRegression':
        model = LogisticRegression(C=1e6, solver='lbfgs', multi_class='ovr', max_iter=1000)
    elif model == 'SVC':
        model = SVC(decision_function_shape='ovo')
    elif model == 'XGBoost':
        model = XGBClassifier()
    else:
        raise NotImplementedError
    model.fit(train_x, train_y)
    test_data['signal'] = model.predict(test_x)
    return test_data


def generate_weights(data, train_size, model, fitPCA=True, n_components=5):
    unique_tickers = data.unstack('tickers')['ask'].columns.unique()
    weights = {}
    for i in range(unique_tickers.nunique()):
        data_i = data.groupby('tickers').get_group(unique_tickers[i]).droplevel('tickers')
        test_i = generate_signal(data_i, train_size, model, fitPCA=PCA, n_components=n_components)
        weights[unique_tickers[i]] = test_i['signal']

    weights = pd.DataFrame.from_dict(weights)
    weights = weights.div(np.abs(weights).sum(axis=1), axis=0)
    return weights

def compute_pnl(weights, data, cost=0.002):
    wgt_rtn = pd.concat([weights, data['returns'].loc[weights.index, :]], keys=('weights', 'returns'), axis=1)
    pnl = wgt_rtn.weights.mul(wgt_rtn.returns, axis=0).sum(axis=1)
    pnl = pnl.to_frame().rename(columns={0: 'pnl_1'})
    trades = wgt_rtn.weights.diff().fillna(0)
    pnl['pnl_2'] = pnl['pnl_1'] - cost * trades.sum(axis=1)
    pnl['Benchmark'] = data['returns'].loc[pnl.index].mean(axis=1)
    SR_ratio = pd.DataFrame(index=['Sharpe Ratio'], columns=['Gross','ex-Cost'])
    SR_ratio.loc['Sharpe Ratio', 'Gross'] = pnl['pnl_1'].mean() / pnl['pnl_1'].std()
    SR_ratio.loc['Sharpe Ratio', 'ex-Cost'] = pnl['pnl_2'].mean() / pnl['pnl_2'].std()
    SR_ratio.loc['Sharpe Ratio', 'Benchmark'] = pnl['Benchmark'].mean() / pnl['Benchmark'].std()
    return pnl, SR_ratio


if __name__ == '__main__':
    path = '../Data/Trading Alpha'
    resample_freq = 60
    mom_lags = 12
    train_size = 0.7
    model = 'LogisticRegression'
    fitPCA = True
    n_components = 5
    transaction_cost = 0.002

    data = process_data(path)
    features = generate_features(data, resample_freq, mom_lags)
    weights = generate_weights(features, train_size=train_size, model=model, fitPCA=fitPCA, n_components=n_components)
    print('Target position:')
    print('==============')
    print(weights.head())
    print('==============')
    pnl, SR = compute_pnl(weights, data, transaction_cost)
    print(pnl.head())
    print('==============')
    print(SR)











