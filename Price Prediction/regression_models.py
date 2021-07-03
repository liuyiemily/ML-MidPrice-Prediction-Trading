## Implement various regression models

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgbm
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

def apply_pca(x_train, x_test, num_pca=10, standardize=True):
    # Standardize
    if standardize:
        x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0)
        x_test = (x_test - x_train.mean(axis=0)) / x_train.std(axis=0) # use training set mean and std

    # Create principal components
    pca = PCA(n_components=num_pca).fit(x_train)
    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)

    # Convert to dataframe
    component_names = [f"PC{i + 1}" for i in range(x_train_pca.shape[1])]
    x_train_pca = pd.DataFrame(x_train_pca, columns=component_names)
    x_test_pca = pd.DataFrame(x_test_pca, columns=component_names)

    # Create loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=component_names,
        index=x_train.columns
    )

    return pca, x_train_pca, x_test_pca, loadings

def regress_model(x_train, y_train, x_test, y_test, model, optimized=False, nfolds=10, verbose=False):
    """
    Parameters
    ----------
    x_train, y_train, x_test, y_test: DataFrame
        train and test sets
    model: str
        str, can be 'LinearRegression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet', 'Random Forest', 'LightGBM', 'XGBoost'
    nfolds: int
        number of folds for GridSearch, default to be 10
    optimized: bool
        True if run GridSearchCV, False otherwise

    Returns
    -------
    r2_score and mean_squared_error for both train and test set
    """

    table = pd.DataFrame(index=['R2', 'RMSE'], columns=['Train', 'Test'])
    if model == 'LinearRegression':
        print('Running Linear Regression Model')
        mdl = LinearRegression()

    elif model == 'Ridge':
        print('Running Ridge Regression Model')
        if optimized:
            params_ridge = {
                'alpha': [.01, .1, .5, .7, .9, .95, .99, 1, 5, 10, 20],
                'fit_intercept': [True, False],
                'normalize': [True, False],
                'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
            }
            ridge_mdl = Ridge()
            mdl = GridSearchCV(ridge_mdl, params_ridge, scoring='neg_root_mean_squared_error', cv=nfolds, n_jobs=-1)
        else:
            mdl = Ridge()

    elif model == 'Lasso':
        print('Running Lasso Regression Model')
        if optimized:
            params_lasso = {
                'alpha': [.01, .1, .5, .7, .9, .95, .99, 1, 5, 10, 20],
                'fit_intercept': [True, False],
                'normalize': [True, False]
            }
            lasso_mdl = Lasso()
            mdl = GridSearchCV(lasso_mdl, params_lasso, scoring='neg_root_mean_squared_error', cv=nfolds, n_jobs=-1)
        else:
            mdl = Lasso()

    elif model == 'ElasticNet':
        print('Running ElasticNet Model')
        if optimized:
            mdl = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], eps=5e-2, cv=nfolds, n_jobs=-1)
        else:
            mdl = ElasticNet()

    elif model == 'RandomForest':
        print('Running Random Forest Model')
        if optimized:
            params_rf = {
                'max_depth': [10, 30, 35, 50, 65, 75, 100],
                'max_features': [.3, .4, .5, .6],
                'min_samples_leaf': [3, 4, 5],
                'min_samples_split': [8, 10, 12],
                'n_estimators': [30, 50, 100, 200]
            }
            rf = RandomForestRegressor()
            mdl = GridSearchCV(rf, params_rf, scoring='neg_root_mean_squared_error', cv=nfolds, n_jobs=-1)
        else:
            mdl = RandomForestRegressor()

    elif model == 'LightGBM':
        print('Running Light GBM Model')
        if optimized:
            params_lgbm = {
                'learning_rate': [.01, .1, .5, .7, .9, .95, .99, 1],
                'boosting': ['gbdt'],
                'metric': ['l1'],
                'feature_fraction': [.3, .4, .5, 1],
                'num_leaves': [20],
                'min_data': [10],
                'max_depth': [10],
                'n_estimators': [10, 30, 50, 100]
            }

            lgb = lgbm.LGBMRegressor()
            mdl = GridSearchCV(lgb, params_lgbm, scoring='neg_root_mean_squared_error', cv=nfolds, n_jobs=-1)
        else:
            mdl = lgbm.LGBMRegressor()

    elif model == 'XGBoost':
        if optimized:
            params_xgb = {
                'learning_rate': [.1, .5, .7, .9, .95, .99, 1],
                'colsample_bytree': [.3, .4, .5, .6],
                'max_depth': [4],
                'alpha': [3],
                'subsample': [.5],
                'n_estimators': [30, 70, 100, 200]
            }

            xgb_model = XGBRegressor()
            mdl = GridSearchCV(xgb_model, params_xgb, scoring='neg_root_mean_squared_error', cv=nfolds, n_jobs=-1)
        else:
            mdl = XGBRegressor()

    else:
        raise NotImplementedError

    mdl.fit(x_train, y_train)
    y_train_pred = mdl.predict(x_train)
    y_test_pred = mdl.predict(x_test)

    train_r2 = r2_score(y_train_pred, y_train)
    test_r2 = r2_score(y_test_pred, y_test)
    train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))
    test_rmse = np.sqrt(mean_squared_error(y_test_pred, y_test))
    table.loc['R2', 'Train'] = train_r2
    table.loc['R2', 'Test'] = test_r2
    table.loc['RMSE', 'Train'] = train_rmse
    table.loc['RMSE', 'Test'] = test_rmse

    if verbose:
        print("Train r2 score: {:.2f}".format(train_r2))
        print("Test r2 score: {:.2f}".format(test_r2))
        print("Train RMSE: {:.4f}".format(train_rmse))
        print("Validation RMSE: {:.4f}".format(test_rmse))

    return table
