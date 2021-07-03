## main file for midprice prediction

import os
import numpy as np
import pandas as pd
import data_preprocess
import regression_models
import classification_models

if __name__ == '__main__':
    data_path = '../Data/'
    order_book = os.path.join(data_path, 'Price Prediction')
    date = '2020-08-01' # Take order_1 '2020-08-01' as an example
    forecast_horizon = 5  # target as midPrice in 5 seconds later
    train_size = 0.8
    regression = False # True for regression, False for classification
    if regression:
        n_components = 10 # used in PCA
        model = 'XGBoost' # choose from 'LinearRegression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet', 'Random Forest', 'LightGBM', 'XGBoost'
    else:
        threshold = 0.002
        model = 'SVC' # choose from 'LogisticRegression', 'Ridge', 'SVC', 'KNC', 'Random Forest', 'LightGBM', 'XGBoost'

    # 1. load the data
    # order_1, order_2, order_3 = data_preprocess.extract_and_combine_data(order_book)
    order = data_preprocess.extract_and_combine_data(order_book, date)
    # 2. normalize the data
    data = data_preprocess.normalize_data(order)
    # 3. feature engineering
    df = data_preprocess.generate_features(data) # 360+ features
    if regression:
        # 4. generate regressors and response(target)
        data_x, data_y = data_preprocess.generate_x_y(df, forecast_horizon, regression=True)
        # 5. split the dataset into train and test sets
        x_train, y_train, x_test, y_test = data_preprocess.train_test_split(data_x, data_y, train_size)
        # 6. apply PCA to reduce dimensionality
        _, x_train_pca, x_test_pca, _ = regression_models.apply_pca(x_train, x_test, n_components, standardize=True)
        # 7. train the model and get result
        table = regression_models.regress_model(x_train_pca, y_train, x_test_pca, y_test, model, optimized=False,
                                                nfolds=10, verbose=False)
    else:
        data_x, data_y = data_preprocess.generate_x_y(df, forecast_horizon, regression=False, threshold=threshold)
        x_train, y_train, x_test, y_test = data_preprocess.train_test_split(data_x, data_y, train_size)
        table = classification_models.classification_model(x_train, y_train, x_test, y_test, model, optimized=False,
                                                           nfolds=10, verbose=False, random_state=123)

    print(table)
