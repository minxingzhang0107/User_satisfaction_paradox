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
import warnings
warnings.filterwarnings("ignore")


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
    # coefficent
    coefficient_list = lr.coef_
    return MAE, MSE, RMSE, R2, perm_importance_mean, coefficient_list


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
    # remove crowd_status prefix
    processed_dataset.columns = processed_dataset.columns.str.replace(
        'crowd_status_', '')
    # move user_satisfaction to the last column
    processed_dataset = processed_dataset[[c for c in processed_dataset if c not in ['User_satisfaction']] + ['User_satisfaction']]
    print(processed_dataset.head())
    X = processed_dataset.iloc[:, :-1].values
    y = processed_dataset.iloc[:, -1].values

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # feature importance for SVR
    MAE, MSE, RMSE, R2, perm_importance_mean_SVR = SVR_model(
        X_train, X_test, y_train, y_test, 42)
    # add rainbow palette for the bar chart
    plt.bar([x for x in range(len(perm_importance_mean_SVR))], perm_importance_mean_SVR, color=
    plt.cm.rainbow(np.linspace(0, 1, len(perm_importance_mean_SVR))))
    plt.xticks([x for x in range(len(perm_importance_mean_SVR))], processed_dataset.columns[:-1], rotation='horizontal')
    plt.xticks(fontsize=6)
    plt.title('Feature importance for SVR')
    plt.savefig('SVR_feature_importance.png')
    plt.show()

    # feature importance for XGBoost
    MAE, MSE, RMSE, R2, perm_importance_mean_XGBoost = XGBoost_model(
        X_train, X_test, y_train, y_test, 42)
    plt.bar([x for x in range(len(perm_importance_mean_XGBoost))], perm_importance_mean_XGBoost, color=
            plt.cm.rainbow(np.linspace(0, 1, len(perm_importance_mean_XGBoost))))
    plt.xticks([x for x in range(len(perm_importance_mean_XGBoost))], processed_dataset.columns[:-1], rotation='horizontal')
    plt.xticks(fontsize=6)
    plt.title('Feature importance for XGBoost')
    plt.savefig('XGBoost_feature_importance.png')
    plt.show()

    # feature importance for RandomForest
    MAE, MSE, RMSE, R2, perm_importance_mean_RF = RandomForest_model(
        X_train, X_test, y_train, y_test, 42)
    plt.bar([x for x in range(len(perm_importance_mean_RF))], perm_importance_mean_RF, color=
            plt.cm.rainbow(np.linspace(0, 1, len(perm_importance_mean_RF))))
    plt.xticks([x for x in range(len(perm_importance_mean_RF))], processed_dataset.columns[:-1], rotation='horizontal')
    plt.xticks(fontsize=6)
    plt.title('Feature importance for RandomForest')
    plt.savefig('RandomForest_feature_importance.png')
    plt.show()

    # feature importance for DecisionTree
    MAE, MSE, RMSE, R2, perm_importance_mean_DT = DecisionTree_model(
        X_train, X_test, y_train, y_test, 42)
    plt.bar([x for x in range(len(perm_importance_mean_DT))], perm_importance_mean_DT, color=
            plt.cm.rainbow(np.linspace(0, 1, len(perm_importance_mean_DT))))
    plt.xticks([x for x in range(len(perm_importance_mean_DT))], processed_dataset.columns[:-1], rotation='horizontal')
    plt.xticks(fontsize=6)
    plt.title('Feature importance for DecisionTree')
    plt.savefig('DecisionTree_feature_importance.png')
    plt.show()

    # feature importance for LinearRegression
    MAE, MSE, RMSE, R2, perm_importance_mean_LR, coefficient_list = LinearRegression_model(
        X_train, X_test, y_train, y_test, 42)
    plt.bar([x for x in range(len(perm_importance_mean_LR))], perm_importance_mean_LR, color=
            plt.cm.rainbow(np.linspace(0, 1, len(perm_importance_mean_LR))))
    plt.xticks([x for x in range(len(perm_importance_mean_LR))], processed_dataset.columns[:-1], rotation='horizontal')
    plt.xticks(fontsize=6)
    plt.title('Feature importance for LinearRegression')
    plt.savefig('LinearRegression_feature_importance.png')
    plt.show()
    # also plot the coefficient for LinearRegression
    plt.bar([x for x in range(len(coefficient_list))], coefficient_list, color=
            plt.cm.rainbow(np.linspace(0, 1, len(coefficient_list))))
    plt.xticks([x for x in range(len(coefficient_list))], processed_dataset.columns[:-1], rotation='horizontal')
    plt.xticks(fontsize=6)
    plt.title('Coefficient for LinearRegression')
    plt.savefig('LinearRegression_coefficient.png')
    plt.show()




