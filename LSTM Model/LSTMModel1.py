import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

# Read the csv file
df = pd.read_csv('train.csv', skiprows=0)

cols = list(df)[:14] # define the columns we want to use from data for training

df_for_training = df[cols].astype(float) # grab data from specified columns

# Normalize the dataset
scaler = StandardScaler() 
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)

# Reshape input data into n_samples x timesteps x n_features
# Empty lists to be populated using formatted training data
train_X = [] # training data input
train_Y = [] # prediction

n_future = 1   # number of measurments forward we want to predict
n_past = 5  # number of past measurments we want to use to predict next target

# Reformat input data as (n_samples x timesteps x n_features)
for i in range(n_past, len(df_for_training) - n_future + 1):
    train_X.append(df_for_training_scaled[i - n_past:i, 0:df.shape[1]]) 
    train_Y.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

# Convert train_X and train_Y from lists to arrays
train_X, train_Y = np.array(train_X), np.array(train_Y)

# print(train_X)
# print(train_Y)

# Print shape of trainX and trainY
print('train_X shape == {}.'.format(train_X.shape))
print('train_Y shape == {}.'.format(train_Y.shape))

# Define the Autoencoder model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(train_Y.shape[1]))

model.compile(optimizer='adam', loss='mse') # complile model
model.summary() # print model summary

# Train the model
history = model.fit(train_X, train_Y, epochs=10, batch_size=16, validation_split=0.1, verbose=1)

# Plot training for visualization
# plt.plot(history.history['loss'], label='Training loss')
# plt.plot(history.history['val_loss'], label='Validation loss')
# plt.show()

# Predict
prediction = model.predict(train_X) # use model to predict target
prediction_copies = np.repeat(prediction, df_for_training.shape[1], axis = -1) # convert prediction vector to orignial shape
final_prediction = scaler.inverse_transform(prediction_copies)[:,0] # convert prediction values to original scale

# Convert training data to original scale
train_Y_copies = np.repeat(train_Y, df_for_training.shape[1], axis = -1) 
orignial_train_Y = scaler.inverse_transform(train_Y_copies)[:,0]

# Display results
plt.plot(final_prediction, label='Prediction')
plt.plot(orignial_train_Y, label='Actuals')
plt.show()
