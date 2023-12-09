import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime
import time
import random
import os
import warnings
warnings.filterwarnings("ignore")


def scrape_specific_date(input_url):
    response = requests.get(input_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        all_span_elements = soup.find_all("span", class_="has-text-weight-normal")
        crowd_level_list = [span for span in all_span_elements if span.get_text(strip=True) == "Crowd level"]
        # Extract the crowd level percentage
        crowd_level_percentage_span = crowd_level_list[0]
        crowd_level_percentage = crowd_level_percentage_span.find_next_sibling("span").text.strip()
        # Extract the crowd status
        crowd_status_element = crowd_level_list[1]
        crowd_status = crowd_status_element.find_next_sibling("span").text.strip()
        # return crowd_level_percentage, crowd_status
        return crowd_level_percentage, crowd_status
    else:
        print("Failed to retrieve data. Status Code:", response.status_code)
        return 0, 0


if __name__ == '__main__':
    review_dataset = pd.read_csv("universal_studio_branches.csv")
    review_dataset["written_date"] = review_dataset["written_date"].apply(lambda x: datetime.strptime(x, '%B %d, %Y'))
    # remove all rows with date before 2018-01-01
    review_dataset = review_dataset[review_dataset["written_date"] >= datetime.strptime("2018-01-01", '%Y-%m-%d')]
    # new_df = pd.DataFrame(columns=["Date", "crowd_level", "prediction_error", "crowd_status", "User_satisfaction",
    #                                "Original_row_index"])
    new_df = pd.DataFrame(columns=["Date", "crowd_level", "crowd_status", "User_satisfaction",
                                   "Original_row_index"])
    # for i in range(len(review_dataset)):
    # process rows from 5000 to the end
    for i in range(5000, len(review_dataset)):
        # # process the first 5000 rows
        # if i == 5000:
        #     break
        print(f"Processing {i}th row")
        current_date = review_dataset.iloc[i]["written_date"]
        current_user_satisfaction = review_dataset.iloc[i]["rating"]
        original_row_index = i
        if current_date in new_df["Date"].values:
            same_row = new_df[new_df["Date"] == current_date]
            same_row_to_list = same_row.values.tolist()
            # replace the 4th element with the real user satisfaction
            same_row_to_list[0][3] = current_user_satisfaction
            same_row_to_list = same_row_to_list[0]
            new_df.loc[i] = same_row_to_list
            continue
        current_year = current_date.year
        current_month = current_date.month
        # for months with single digit, add a 0 in front of it
        if current_month < 10:
            current_month = f"0{current_month}"
        current_day = current_date.day
        # for days with single digit, add a 0 in front of it
        if current_day < 10:
            current_day = f"0{current_day}"
        input_url = f"https://queue-times.com/en-US/parks/65/calendar/{current_year}/{current_month}/{current_day}"
        print(input_url)
        # crowd_level, prediction_error, crowd_status = scrape_specific_date(input_url)
        crowd_level, crowd_status = scrape_specific_date(input_url)
        if crowd_level == 0 and crowd_status == 0:
            print("Failed to retrieve data. Skip this row")
            continue
        crowd_level = float(crowd_level.strip("%"))
        crowd_level = crowd_level / 100
        # prediction_error = float(prediction_error.strip("%"))
        # prediction_error = prediction_error / 100
        # current_row = [current_date, crowd_level, prediction_error, crowd_status, current_user_satisfaction,
        #                original_row_index]
        current_row = [current_date, crowd_level, crowd_status, current_user_satisfaction,
                       original_row_index]
        new_df.loc[i] = current_row
        time.sleep(random.randint(1, 5))
    # new_df.to_csv("universal_studio_updated_first_5000.csv", index=False)
    new_df.to_csv("universal_studio_updated_5000_to_end.csv", index=False)






