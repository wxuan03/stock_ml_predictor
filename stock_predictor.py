import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import yfinance as yf

# Download stock data
ticker = 'GOOG'
data = yf.download(ticker, start='2010-01-01', end='2023-01-01')
data.Close.plot(title=f"{ticker} Stock Price")

# Use only the 'Close' column
close_prices = data[['Close']]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(close_prices)

# Define a function to create datasets
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# Create the training and testing datasets
time_step = 100
X, y = create_dataset(scaled_data, time_step)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM network model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)),
    tf.keras.layers.LSTM(units=50, return_sequences=False),
    tf.keras.layers.Dense(units=25),
    tf.keras.layers.Dense(units=1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=1, epochs=10)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Transform back to original form
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Plotting
look_back = time_step
trainPredictPlot = np.empty_like(scaled_data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

# Shift test predictions for plotting with dynamic slicing
testPredictPlot = np.empty_like(scaled_data)
testPredictPlot[:, :] = np.nan

# Calculate the indices for slicing
train_size = len(y_train)
total_size = len(scaled_data)
test_size = len(test_predict)

start_index = train_size + look_back
end_index = start_index + test_size

# Ensure the indices do not exceed the total size
start_index = max(start_index, 0)  # Ensure it's not negative
end_index = min(end_index, total_size - 1)  # Ensure it doesn't go past the end

# Check for sufficient space and assign the predictions if space is sufficient
if end_index - start_index == test_size:
    testPredictPlot[start_index:end_index, :] = test_predict
else:
    print("Adjusted mismatch in space for test predictions!")

# Plot baseline and predictions
plt.plot(scaler.inverse_transform(scaled_data))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

from sklearn.metrics import mean_squared_error

# Calculate RMSE
trainScore = np.sqrt(mean_squared_error(y_train, train_predict[:,0]))
print(f'Train Score: {trainScore:.2f} RMSE')
testScore = np.sqrt(mean_squared_error(y_test, test_predict[:,0]))
print(f'Test Score: {testScore:.2f} RMSE')
