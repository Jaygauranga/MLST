import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Load the data
data = pd.read_csv('yahoo_stock.csv', date_parser=True)
data = data.sort_values('Date')
data = data.set_index('Date')
training_data = data.iloc[:len(data)-30, :].values
testing_data = data.iloc[len(data)-30:, :].values

# Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_training_data = scaler.fit_transform(training_data)

# Create the training data
X_train = []
y_train = []

for i in range(60, len(training_data)):
    X_train.append(scaled_training_data[i-60:i, 0])
    y_train.append(scaled_training_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Prepare the test data
scaled_testing_data = scaler.fit_transform(testing_data)
X_test = []
for i in range(60, 90):
    X_test.append(scaled_testing_data[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Predict the test data
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)

# Visualize the results
plt.plot(testing_data[:, 0], color='blue', label='Actual Yahoo Stock Price')
plt.plot(y_pred, color='red', label='Predicted Yahoo Stock Price')
plt.title('Yahoo Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
