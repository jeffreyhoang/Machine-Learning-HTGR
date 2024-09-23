# This is Data Cleaning with Pandas. We're selecting 12 columns out of 61:
# "AccMeanX", "AccMeanY", "AccMeanZ", "AccMedianX", "AccMedianY", "AccMedianZ", 
# "GyroMeanX", "GyroMeanY", "GyroMeanZ", "GyroMedianX", "GyroMedianY", and "GyroMedianZ" 

import pandas as pd

input_filename = "features_14.csv"

# Read the csv file
total_data = pd.read_csv(input_filename)
# Displaying file information
# total_data.info()

# Select the 12 column names
columns_of_interest = ["Target", "AccMeanX", "AccMeanY", "AccMeanZ",
                       "AccMedianX", "AccMedianY", "AccMedianZ",
                       "GyroMeanX", "GyroMeanY", "GyroMeanZ",
                       "GyroMedianX", "GyroMedianY", "GyroMedianZ"]
filtered_data = total_data[columns_of_interest]

# Save the filtered data to a new CSV file
# Exclude the index column
# Format: up to 5 decimal places
filtered_data.to_csv("filtered_data.csv", index=False, float_format="%.5f")
