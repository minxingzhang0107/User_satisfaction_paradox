import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime
import time
import random
import os


if __name__ == '__main__':
    review_dataset = pd.read_csv("universal_studio_branches.csv")
    review_dataset["written_date"] = review_dataset["written_date"].apply(lambda x: datetime.strptime(x, '%B %d, %Y'))
    # remove all rows with date before 2018-01-01
    review_dataset = review_dataset[review_dataset["written_date"] >= datetime.strptime("2018-01-01", '%Y-%m-%d')]
    # save the updated dataset
    review_dataset.to_csv("universal_studio_branches_original_updated.csv", index=False)
    new_dataset_first_5000 = pd.read_csv("universal_studio_updated_first_5000.csv")
    new_dataset_last_5000 = pd.read_csv("universal_studio_updated_5000_to_end.csv")
    new_dataset = pd.concat([new_dataset_first_5000, new_dataset_last_5000], ignore_index=True)
    # remove the Original_row_index column
    new_dataset = new_dataset.drop(columns=["Original_row_index"])
    # # remove duplicate rows
    # new_dataset = new_dataset.drop_duplicates()
    new_dataset.to_csv("universal_studio_processed_updated.csv", index=False)


