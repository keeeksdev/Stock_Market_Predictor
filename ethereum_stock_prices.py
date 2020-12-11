import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# get the stock quote
df = web.DataReader('ETH', data_source='yahoo', start='2012-01-01', end='2020-12-09')

# print the data
print(df)

# Get the number of rows and column in the data set
print("Number of rows and columns...")
print(df.shape)

# Visualise the closing price history
plt.figure(figsize=(16, 8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD', fontsize=18)
plt.show()

# Create a new dataframe with only the 'Close Column'
data = df.filter(['Close'])

# Convert the dataframe to a numpy array
dataset = data.values

# Get the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * 0.8)

# print the training_data_set....
print("\nGrabbing the length of the trained dataset ... @80%")
print(training_data_len)

# Scale the dat
# In the real world it is always advantageous to apply
# pre processed transformation scaling or normalisation to the input
# data before it is fed to a neural network
# Good practise.
print("\nAttempting to scale the data between 0, 1 inclusive ... ")
scaler = MinMaxScaler(feature_range=(0, 1))

# fit_transform computes the min and max values for scaling and it transforms the data based on these 2 values.
# 0 to 1 inclusive.
scaled_data = scaler.fit_transform(dataset)
print(scaled_data)

# Create the training data set
# create the scaled training data set
train_data = scaled_data[0:training_data_len, :]

# split the data into x_train and y_train data sets
x_train = []
y_train = []

# append the last 60 values to the x_train
for i in range(60, len(train_data)):
    x_train.append(train_data[i - 60:i, 0])
    y_train.append(train_data[i, 0])

    if i <= 60:
        print("\nx_train dataset ... ")
        print(x_train)
        print()
        print("\ny_train dataset ... ")
        print(y_train)

# Covert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
print("\nreshaping the data...")
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)

# Build the LSTM model
print("\nBuilding the LSTM model...")
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

for layer in model.layers:
    print(layer.output_shape)

# Compile the model
# an optimizer is used to improve upon the loss function
# the loss function is used to measure how well the model did on traning
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
print("\ntraining the model ...")
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Creating the testing dataset
print("\n creating the testing dataset...")

test_data = scaled_data[training_data_len - 60:, :]
print(test_data)

# Create the data sets x_tests and y_tests
print("\ncreate the data sets x_tests and y_tests...")
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60:i, 0])

print(y_test, x_test)

# Convert data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
# Undo scaling

# Get the root mean squared error (RMSE)
# a good measure of how accurate does the model predict the responses.
# the lower the value the better
print("\nget the root mean squared error...")
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
print(rmse)

# Plot/Create the data for the graph
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

# Visualize the data
plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

# Get the quote
apple_quote = web.DataReader('ETH', data_source='yahoo', start='2012-01-01', end='2020-12-09')

# Create a new dataframe
new_df = apple_quote.filter(['Close'])

# Get the last 60 day closing price
last_60_days = new_df[-60:].values

# Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)

# Create an empty list
X_test = []

# Append teh past 60 days
X_test.append(last_60_days_scaled)

# Convert the X_test data set to a numpy array
X_test = np.array(X_test)

# Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Get the predicted scaled price
pred_price = model.predict(X_test)

# undo the scaling
print("\ncalculating the predicted price...")
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)

print("\ncalculating the actual price...")

# Get the quote
apple_quote_2 = web.DataReader('ETH', data_source='yahoo', start='2020-12-10', end='2020-12-10')
print(apple_quote_2['Close'])

