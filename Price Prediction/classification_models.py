import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgbm
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tensorflow import keras

def classification_model(x_train, y_train, x_test, y_test, model, optimized=False, nfolds=10, verbose=False, random_state=123):
    '''
       :param
           x_train, y_train, x_test, y_test: train and test sets
           model: str, can be 'LogisticRegression', 'Ridge', 'SVC', 'KNC', 'Random Forest', 'LightGBM', 'XGBoost'
           nfolds: number of folds for GridSearch, default to be 10
           optimized: True if run GridSearchCV, False otherwise

       :return:
           r2_score and mean_squared_error for both train and test set
       '''

    table = pd.DataFrame(index=['F1', 'Accuracy', 'Precision', 'Recall'], columns=['Train', 'Test'])
    if model == 'LogisticRegression':
        mdl = make_pipeline(StandardScaler(), LogisticRegression(random_state=random_state))
    elif model == 'Ridge':
        mdl = make_pipeline(StandardScaler(), RidgeClassifier(random_state=random_state))
    elif model == 'SVC':
        # Note SVC is quite sensitive to scaling
        mdl = make_pipeline(StandardScaler(), SVC(decision_function_shape='ovo', random_state=random_state))
    elif model == 'KNC':
        mdl = make_pipeline(StandardScaler(), KNeighborsClassifier(random_state=random_state))
    elif model == 'Random Forest': # don't need to pre-scale for tree methods
        mdl = RandomForestClassifier(random_state=random_state)
    elif model == 'LightGBM':
        mdl = lgbm.LGBMRegressor()
    elif model == 'XGBoost':
        mdl = XGBClassifier()
    else:
        raise NotImplementedError

    mdl.fit(x_train, y_train)
    y_train_pred = mdl.predict(x_train)
    y_test_pred = mdl.predict(x_test)

    train_acc, test_acc = accuracy_score(y_train_pred, y_train), accuracy_score(y_test_pred, y_test)
    train_precision, test_precision = precision_score(y_train_pred, y_train, average='macro'), precision_score(y_test_pred, y_test, average='macro')
    train_recall, test_recall = recall_score(y_train_pred, y_train, average='macro'), recall_score(y_test_pred, y_test, average='macro')
    train_f1, test_f1 = f1_score(y_train_pred, y_train, average='macro'), f1_score(y_test_pred, y_test, average='macro')
    table.loc['F1', :] = [train_f1, test_f1]
    table.loc['Accuracy', :] = [train_acc, test_acc]
    table.loc['Precision', :] = [train_precision, test_precision]
    table.loc['Recall', :] = [train_recall, test_recall]
    return table


def cnn1d_model(x_train, y_train, x_test, y_test, model, epochs=1, batch_size=100, dropout=0.2, learning_rate=0.001):
    print('Running CNN1D with Maxpooling')

    input_features = keras.Input(shape=(x_train.shape))

    x = keras.layers.Conv1D(15, 5, activation='relu')(input_features)
    x = keras.layers.MaxPooling1D(5)(x)
    x = keras.layers.Dropout(dropout)(x)

    x = keras.layers.Conv1D(15, 5, activation='relu')(x)
    x = keras.layers.MaxPooling1D(5)(x)
    x = keras.layers.Dropout(dropout)(x)

    x = keras.layers.Conv1D(15, 5, activation='relu')(x)
    x = keras.layers.MaxPooling1D(15)(x)  # global max pooling

    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(3, activation='softmax')(x)

    mdl = keras.Model(inputs=input_features, outputs=outputs)
    adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)

    mdl.compile(optimizer=adam, loss='categorical_crossentropy', metrics=[keras.metrics.AUC(), 'accuracy'])
    mdl.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verose=0)
