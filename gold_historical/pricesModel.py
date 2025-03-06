from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
#import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

#predict price then calc return
"""
data = pd.read_csv('./gold_historical/Prices.csv', usecols=lambda col: col != 'index')

#data = data.iloc[::-1].reset_index(drop=True)
data.dropna(inplace=True)
print(data.shape)
data.to_csv('Downloads/Prices_cleaned.csv', index=False)
data1 = pd.read_csv('Downloads/Prices_cleaned.csv', usecols=lambda col: col != 'index')
print(data1.head())
"""
data = pd.read_csv('./gold_historical/Prices_cleaned.csv', usecols=lambda col: col != 'index')
# Parse the date and price columns
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data.set_index('Date', inplace=True)

# Convert the 'Price' column to numeric
data['Price'] = data['Price'].str.replace(',', '').astype(float)

# Calculate returns
#data['Return'] = data['Price'].pct_change()
# log returns
data['Return'] = np.log(data['Price'] / data['Price'].shift(1))

data.dropna(subset=['Return'], inplace=True)

print(data.head())
print(data.size)


scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['Price']])


train_size = int(len(scaled_data) * 0.95)
train_data = scaled_data[0:int(train_size), :]

X_train = []
Y_train = []

look_back = 14
# 60 batch size


for i in range(look_back, len(train_data)):
    X_train.append(train_data[i - look_back: i, 0])
    Y_train.append(train_data[i, 0])

X_train, Y_train = np.array(X_train), np.array(Y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

testing = scaled_data[train_size - look_back:, :]
X_test = []

Y_test = scaled_data[train_size:, :]
for i in range(look_back, len(testing)):
    X_test.append(testing[i - look_back: i, 0])

X_test, Y_test = np.array(X_test), np.array(Y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#implanted code
from keras import regularizers

model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(look_back, 1)))
#model.add(keras.layers.LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=64))
model.add(Dense(128, kernel_regularizer=regularizers.L2(0.002)))
model.add(keras.layers.Dropout(0.5))
model.add(Dense(1))

from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

learning_rate = 0.0008
optimizer = Adam(learning_rate=learning_rate)

# Implement early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

model.compile(optimizer=optimizer, loss='mae', metrics=[RootMeanSquaredError()])

history = model.fit(X_train, Y_train, epochs=40, callbacks=[early_stopping, reduce_lr], validation_split=0.2)


pred = model.predict(X_test)

train = data[:train_size]
test = data[train_size:].copy(deep=True)
#test = test.iloc[-len(pred):]
test['pred'] = scaler.inverse_transform(pred)

#test['pred'] = pred
Y_test = Y_test.reshape(-1)
test_loss, test_rmse = model.evaluate(X_test, Y_test)

print(f"Test Loss (MAE): {test_loss}")
print(f"Test RMSE: {test_rmse}")

from sklearn.metrics import mean_absolute_error

# Ensure Y_test is a 2D array before inverse transforming
y_test_inv = scaler.inverse_transform(Y_test.reshape(-1, 1))

# Calculate mean absolute error
#mae = mean_absolute_error(y_test_inv.flatten(), test['pred'])
mae = mean_absolute_error(y_test_inv, test['pred'])

print(f"Mean Absolute Error (MAE): {mae:.2f}")

#end implanted 

# Evaluate the model
#train_score = model.evaluate(X_train, Y_train, verbose=0)[0]
#test_score = model.evaluate(X_test, Y_test, verbose=0)[0]

#print(f'Train Score: {train_score:.2f} MSE')
#print(f'Test Score: {test_score:.2f} MSE')

# Make predictions
#predictions = model.predict(X_test)

# Inverse transform the predictions
#predictions = scaler.inverse_transform(predictions)


plt.figure(figsize=(10, 8))
plt.plot(train["Price"], c="b")
plt.plot(test[["Price", "pred"]])
plt.ylabel("return")
plt.legend(['train', 'test', 'predic'])
plt.show()

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# Calculate log returns for the predicted prices
test['pred_log_return'] = np.log(test['pred'] / test['pred'].shift(1))

# Calculate log returns for the actual prices
test['actual_log_return'] = np.log(test['Price'] / test['Price'].shift(1))

# Drop the first row with NaN values in log returns
test = test.dropna()

# Calculate mean absolute error and mean percentage error in predicted returns compared to actual returns
mae_log_return = mean_absolute_error(test['actual_log_return'], test['pred_log_return'])
mape_log_return = mean_absolute_percentage_error(test['actual_log_return'], test['pred_log_return'])

print(f"Mean Absolute Error (MAE) in log returns: {mae_log_return:.6f}")
print(f"Mean Absolute Percentage Error (MAPE) in log returns: {mape_log_return:.6f}")

# Plot the actual and predicted log returns
plt.figure(figsize=(10, 8))
plt.plot(test.index, test['actual_log_return'], label='Actual Log Return', color='blue')
plt.plot(test.index, test['pred_log_return'], label='Predicted Log Return', color='red')
plt.xlabel('Date')
plt.ylabel('Log Return')
plt.legend()
plt.show()

# Predict tomorrow's price and return
last_sequence = scaled_data[-look_back:]
last_sequence = np.reshape(last_sequence, (1, look_back, 1))

# Predict the next price
predicted_price_scaled = model.predict(last_sequence)
predicted_price = scaler.inverse_transform(predicted_price_scaled)

# Calculate the predicted return for tomorrow
last_price = data['Price'].iloc[-1]
#predicted_return = (predicted_price[0][0] - last_price) / last_price
predicted_return = np.log(predicted_price[0][0] / last_price)

print(f"Predicted Price for Tomorrow: {predicted_price[0][0]:.2f}")
print(f"Predicted Return for Tomorrow: {predicted_return:.6f}")
