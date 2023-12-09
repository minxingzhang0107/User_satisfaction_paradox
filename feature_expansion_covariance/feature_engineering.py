import pandas as pd
import numpy as np
from datetime import datetime
import time
import random
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


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

    # Correlation between different variables
    corr = processed_dataset.corr()
    # Set up the matplotlib plot configuration
    f, ax = plt.subplots(figsize=(18, 15))
    # Configure a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    # Draw the heatmap
    sns.heatmap(corr, annot=False, cmap=cmap)
    for i in range(len(corr)):
        for j in range(len(corr)):
            plt.text(i + 0.3, j + 0.5, str(round(corr.iloc[i, j], 2)), fontsize=10)
    plt.title('Correlation between different variables', fontsize=25)
    plt.savefig('correlation.png')
    plt.show()

    # save the processed dataset
    processed_dataset.to_csv('universal_studio_feature_expansion.csv', index=False)





