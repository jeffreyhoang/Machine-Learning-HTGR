#This file splits data and then scales it 
import pandas as pd
import numpy as np

input_filename = "filtered_data.csv"

# Read the csv file
total_data = pd.read_csv(input_filename)

#train,valid,test = np.split(total_data.sample(frac=1), [int(0.7*len(data))], [int(0.8*len(data))])


# Replace 'target_values' with a list of the target values you want to check
target_values = [1, 2, 3,4]

# Initialize empty dictionaries to store the splits
train_splits = {target: None for target in target_values}
valid_splits = {target: None for target in target_values}
test_splits = {target: None for target in target_values}

# Split the data for each target value
for target in target_values:
    target_data = total_data[total_data['Target'] == target]
    train_splits[target] = target_data.sample(frac=0.7)
    valid_splits[target] = target_data.sample(frac=0.1)
    test_splits[target] = target_data.sample(frac=0.2)

# Extract the resulting data splits for each target value
train = pd.concat([train_splits[target] for target in target_values])
valid = pd.concat([valid_splits[target] for target in target_values])
test = pd.concat([test_splits[target] for target in target_values])


train.to_csv("train.csv", index=False, float_format="%.5f")
valid.to_csv("valid.csv", index=False, float_format="%.5f")
test.to_csv("test.csv", index=False, float_format="%.5f")