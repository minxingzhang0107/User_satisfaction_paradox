import pandas as pd
import numpy as np
from datetime import datetime
import time
import random
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    processed_dataset = pd.read_csv("universal_studio_processed_updated.csv")
    processed_dataset['Date'] = pd.to_datetime(processed_dataset['Date'])
    processed_dataset['is_peak_month'] = processed_dataset['Date'].dt.month.between(6, 8)
    processed_dataset['is_weekend'] = processed_dataset['Date'].dt.dayofweek.isin(
        [5, 6])
    # add a new column to indicate day of week, i.e. from Monday to Sunday
    processed_dataset['day_of_week'] = processed_dataset['Date'].dt.day_name()
    print(processed_dataset.head())


    # apply one hot encoding to crowd_status
    processed_dataset = pd.get_dummies(processed_dataset, columns=['crowd_status'])
    # remove crowd_status prefix
    processed_dataset.columns = processed_dataset.columns.str.replace(
        'crowd_status_', '')
    # move user_satisfaction to the last column
    processed_dataset = processed_dataset[[c for c in processed_dataset if c not in ['User_satisfaction']] + ['User_satisfaction']]
    print(processed_dataset.head())

    # plot the crowd_level for weekdays and weekends
    plt.figure(figsize=(10, 6))
    plt.title('Crowd Level for Weekdays and Weekends')
    plt.xlabel('Crowd Level')
    plt.ylabel('Frequency')
    sns.distplot(processed_dataset[processed_dataset['is_weekend'] == False]['crowd_level'], label='Weekday', color='blue')
    sns.distplot(processed_dataset[processed_dataset['is_weekend'] == True]['crowd_level'], label='Weekend', color='red')
    plt.legend()
    plt.savefig('crowd_level_weekday_weekend.png')
    plt.show()

    # plot the average crowd level for each day of week
    average_crowd_level_for_each_day_of_week_list = []
    day_of_week_name_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for day_of_week in day_of_week_name_list:
        average_crowd_level_for_each_day_of_week_list.append(
            processed_dataset[processed_dataset['day_of_week'] == day_of_week]['crowd_level'].mean())
    plt.figure(figsize=(10, 6))
    plt.title('Average Crowd Level for Each Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Average Crowd Level')
    # plot with annotation
    sns.barplot(x=day_of_week_name_list, y=average_crowd_level_for_each_day_of_week_list)
    for i in range(len(day_of_week_name_list)):
        plt.annotate(round(average_crowd_level_for_each_day_of_week_list[i], 2),
                     xy=(i, average_crowd_level_for_each_day_of_week_list[i]), ha='center', va='bottom')
    plt.savefig('crowd_level_day_of_week.png')
    plt.show()


