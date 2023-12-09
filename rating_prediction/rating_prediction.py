import pandas as pd
import numpy as np
from datetime import datetime
import time
import random
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance


def XGBoost_model(X_train, X_test, y_train, y_test, random_state):
    xgb = XGBRegressor(random_state=random_state)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    MAE = mean_absolute_error(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(y_test, y_pred)
    perm_importance = permutation_importance(xgb, X_test, y_test)
    perm_importance_mean = np.round(perm_importance.importances_mean, 3)
    return MAE, MSE, RMSE, R2, perm_importance_mean


def RandomForest_model(X_train, X_test, y_train, y_test, random_state):
    rf = RandomForestRegressor(random_state=random_state)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    MAE = mean_absolute_error(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(y_test, y_pred)
    perm_importance = permutation_importance(rf, X_test, y_test)
    perm_importance_mean = np.round(perm_importance.importances_mean, 3)
    return MAE, MSE, RMSE, R2, perm_importance_mean


def SVR_model(X_train, X_test, y_train, y_test, random_state):
    svr = SVR()
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)
    MAE = mean_absolute_error(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(y_test, y_pred)
    perm_importance = permutation_importance(svr, X_test, y_test)
    perm_importance_mean = np.round(perm_importance.importances_mean, 3)
    return MAE, MSE, RMSE, R2, perm_importance_mean


def DecisionTree_model(X_train, X_test, y_train, y_test, random_state):
    dt = DecisionTreeRegressor(random_state=random_state)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    MAE = mean_absolute_error(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(y_test, y_pred)
    perm_importance = permutation_importance(dt, X_test, y_test)
    perm_importance_mean = np.round(perm_importance.importances_mean, 3)
    return MAE, MSE, RMSE, R2, perm_importance_mean


def LinearRegression_model(X_train, X_test, y_train, y_test, random_state):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    MAE = mean_absolute_error(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(y_test, y_pred)
    perm_importance = permutation_importance(lr, X_test, y_test)
    perm_importance_mean = np.round(perm_importance.importances_mean, 3)
    return MAE, MSE, RMSE, R2, perm_importance_mean


if __name__ == '__main__':
    processed_dataset = pd.read_csv("universal_studio_processed_updated.csv")
    processed_dataset['Date'] = pd.to_datetime(processed_dataset['Date'])
    processed_dataset['is_peak_month'] = processed_dataset['Date'].dt.month.between(6, 8)
    processed_dataset['is_weekend'] = processed_dataset['Date'].dt.dayofweek.isin(
        [5, 6])
    # remove the date column
    processed_dataset = processed_dataset.drop(columns=['Date'])
    # apply one hot encoding to crowd_status
    processed_dataset = pd.get_dummies(processed_dataset, columns=['crowd_status'])
    # move user_satisfaction to the last column
    processed_dataset = processed_dataset[[c for c in processed_dataset if c not in ['User_satisfaction']] + ['User_satisfaction']]
    print(processed_dataset.head())
    X = processed_dataset.iloc[:, :-1].values
    y = processed_dataset.iloc[:, -1].values

    XGboost_mae_list = []
    XGboost_mse_list = []
    XGboost_rmse_list = []
    XGboost_r2_list = []
    XGboost_permutation_importance_list = []
    RandomForest_mae_list = []
    RandomForest_mse_list = []
    RandomForest_rmse_list = []
    RandomForest_r2_list = []
    RandomForest_permutation_importance_list = []
    SVR_mae_list = []
    SVR_mse_list = []
    SVR_rmse_list = []
    SVR_r2_list = []
    SVR_permutation_importance_list = []
    DecisionTree_mae_list = []
    DecisionTree_mse_list = []
    DecisionTree_rmse_list = []
    DecisionTree_r2_list = []
    DecisionTree_permutation_importance_list = []
    LinearRegression_mae_list = []
    LinearRegression_mse_list = []
    LinearRegression_rmse_list = []
    LinearRegression_r2_list = []
    LinearRegression_permutation_importance_list = []

    # iterate 5 times
    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i+666)

        # XGBoost model
        XGboost_mae, XGboost_mse, XGboost_rmse, XGboost_r2, XGboost_permutation_importance = \
            XGBoost_model(X_train, X_test, y_train, y_test, i+666)
        XGboost_mae_list.append(XGboost_mae)
        XGboost_mse_list.append(XGboost_mse)
        XGboost_rmse_list.append(XGboost_rmse)
        XGboost_r2_list.append(XGboost_r2)
        XGboost_permutation_importance_list.append(XGboost_permutation_importance)

        # Random Forest model
        RandomForest_mae, RandomForest_mse, RandomForest_rmse, RandomForest_r2, RandomForest_permutation_importance = \
            RandomForest_model(X_train, X_test, y_train, y_test, i+666)
        RandomForest_mae_list.append(RandomForest_mae)
        RandomForest_mse_list.append(RandomForest_mse)
        RandomForest_rmse_list.append(RandomForest_rmse)
        RandomForest_r2_list.append(RandomForest_r2)
        RandomForest_permutation_importance_list.append(RandomForest_permutation_importance)

        # SVR model
        SVR_mae, SVR_mse, SVR_rmse, SVR_r2, SVR_permutation_importance = \
            SVR_model(X_train, X_test, y_train, y_test, i+666)
        SVR_mae_list.append(SVR_mae)
        SVR_mse_list.append(SVR_mse)
        SVR_rmse_list.append(SVR_rmse)
        SVR_r2_list.append(SVR_r2)
        SVR_permutation_importance_list.append(SVR_permutation_importance)

        # Decision Tree model
        DecisionTree_mae, DecisionTree_mse, DecisionTree_rmse, DecisionTree_r2, DecisionTree_permutation_importance = \
            DecisionTree_model(X_train, X_test, y_train, y_test, i+666)
        DecisionTree_mae_list.append(DecisionTree_mae)
        DecisionTree_mse_list.append(DecisionTree_mse)
        DecisionTree_rmse_list.append(DecisionTree_rmse)
        DecisionTree_r2_list.append(DecisionTree_r2)
        DecisionTree_permutation_importance_list.append(DecisionTree_permutation_importance)


        # Linear Regression model
        LinearRegression_mae, LinearRegression_mse, LinearRegression_rmse, LinearRegression_r2, LinearRegression_permutation_importance = \
            LinearRegression_model(X_train, X_test, y_train, y_test, i+666)
        LinearRegression_mae_list.append(LinearRegression_mae)
        LinearRegression_mse_list.append(LinearRegression_mse)
        LinearRegression_rmse_list.append(LinearRegression_rmse)
        LinearRegression_r2_list.append(LinearRegression_r2)
        LinearRegression_permutation_importance_list.append(LinearRegression_permutation_importance)


    print("XGBoost MAE: ", np.mean(XGboost_mae_list))
    print("XGBoost MSE: ", np.mean(XGboost_mse_list))
    print("XGBoost RMSE: ", np.mean(XGboost_rmse_list))
    print("XGBoost R2: ", np.mean(XGboost_r2_list))
    # standard deviation
    print("XGBoost MAE std: ", np.std(XGboost_mae_list))
    print("XGBoost MSE std: ", np.std(XGboost_mse_list))
    print("XGBoost RMSE std: ", np.std(XGboost_rmse_list))
    print("XGBoost R2 std: ", np.std(XGboost_r2_list))
    print("--------------------------------------")
    print("Random Forest MAE: ", np.mean(RandomForest_mae_list))
    print("Random Forest MSE: ", np.mean(RandomForest_mse_list))
    print("Random Forest RMSE: ", np.mean(RandomForest_rmse_list))
    print("Random Forest R2: ", np.mean(RandomForest_r2_list))
    # standard deviation
    print("Random Forest MAE std: ", np.std(RandomForest_mae_list))
    print("Random Forest MSE std: ", np.std(RandomForest_mse_list))
    print("Random Forest RMSE std: ", np.std(RandomForest_rmse_list))
    print("Random Forest R2 std: ", np.std(RandomForest_r2_list))
    print("--------------------------------------")
    print("SVR MAE: ", np.mean(SVR_mae_list))
    print("SVR MSE: ", np.mean(SVR_mse_list))
    print("SVR RMSE: ", np.mean(SVR_rmse_list))
    print("SVR R2: ", np.mean(SVR_r2_list))
    # standard deviation
    print("SVR MAE std: ", np.std(SVR_mae_list))
    print("SVR MSE std: ", np.std(SVR_mse_list))
    print("SVR RMSE std: ", np.std(SVR_rmse_list))
    print("SVR R2 std: ", np.std(SVR_r2_list))
    print("--------------------------------------")
    print("Decision Tree MAE: ", np.mean(DecisionTree_mae_list))
    print("Decision Tree MSE: ", np.mean(DecisionTree_mse_list))
    print("Decision Tree RMSE: ", np.mean(DecisionTree_rmse_list))
    print("Decision Tree R2: ", np.mean(DecisionTree_r2_list))
    # standard deviation
    print("Decision Tree MAE std: ", np.std(DecisionTree_mae_list))
    print("Decision Tree MSE std: ", np.std(DecisionTree_mse_list))
    print("Decision Tree RMSE std: ", np.std(DecisionTree_rmse_list))
    print("Decision Tree R2 std: ", np.std(DecisionTree_r2_list))
    print("--------------------------------------")
    print("Linear Regression MAE: ", np.mean(LinearRegression_mae_list))
    print("Linear Regression MSE: ", np.mean(LinearRegression_mse_list))
    print("Linear Regression RMSE: ", np.mean(LinearRegression_rmse_list))
    print("Linear Regression R2: ", np.mean(LinearRegression_r2_list))
    # standard deviation
    print("Linear Regression MAE std: ", np.std(LinearRegression_mae_list))
    print("Linear Regression MSE std: ", np.std(LinearRegression_mse_list))
    print("Linear Regression RMSE std: ", np.std(LinearRegression_rmse_list))
    print("Linear Regression R2 std: ", np.std(LinearRegression_r2_list))
    print("--------------------------------------")










