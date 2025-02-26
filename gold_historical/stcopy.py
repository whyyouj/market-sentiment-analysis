from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
#import seaborn as sns

mu = pd.read_csv('./gold_futures.csv', usecols=lambda col: col != 'index')

print(mu.head())
mu = mu.iloc[::-1].reset_index(drop=True)

mu['Date'] = pd.to_datetime(mu['Date'], errors='coerce')
mu.set_index('Date', inplace=True)

# convert 'Price' to numeric
mu['Price'] = mu['Price'].str.replace(',', '').astype(float)
#mu['Change'] = mu['Change'].str.replace('%', '').astype(float)


"""
print(mu.head())
print(mu.shape)
print(mu.info())
print(mu.describe())
"""


#mu["Date"] = pd.to_datetime(mu["Date"])

"""
sns.heatmap(mu.select_dtypes(include=np.number).corr(), annot=True, cbar=False)
plt.show()
"""


mu_close = mu.filter(["Price"])
#mu_close = mu.filter(["Open", "High", "Low", "Close/Last"])
dataset = mu_close.values
training = int(np.ceil(len(dataset) * 0.95))

ss1 = StandardScaler()
ss = ss1.fit_transform(dataset)
train_data = ss[0:int(training), :]

x_train = []
y_train = []

batch_size = 14
# 60 batch size


"""close only
"""
for i in range(batch_size, len(train_data)):
    x_train.append(train_data[i - batch_size: i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


"""all factors
for i in range(batch_size, len(train_data)):
    x_train.append(train_data[i - batch_size: i])
    y_train.append(train_data[i, 3])

x_train, y_train = np.array(x_train), np.array(y_train)
X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
"""

from keras import regularizers

model = keras.models.Sequential()
model.add(keras.layers.LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
#model.add(keras.layers.LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(keras.layers.LSTM(units=64))
model.add(keras.layers.Dense(128, kernel_regularizer=regularizers.L2(0.002)))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))

# print(model.summary())

from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

learning_rate = 0.0008
optimizer = Adam(learning_rate=learning_rate)

# early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

model.compile(optimizer=optimizer, loss='mae', metrics=[RootMeanSquaredError()])

history = model.fit(X_train, y_train, epochs=40, callbacks=[early_stopping, reduce_lr], validation_split=0.2)


testing = ss[training - batch_size:, :]
x_test = []

"""all factors
y_test = dataset[training:, 3]
for i in range(batch_size, len(testing)):
    x_test.append(testing[i - batch_size: i])

x_test, y_test = np.array(x_test), np.array(y_test)
X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))
"""

"""close only
"""
y_test = ss[training:, :]
for i in range(batch_size, len(testing)):
    x_test.append(testing[i - batch_size: i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

"""
"""
#y_test_inv = ss1.inverse_transform(np.concatenate((np.zeros((len(y_test), 3)), y_test), axis=1))[:, 3]
y_test_inv = ss1.inverse_transform(y_test)
y_test = y_test.flatten()
pred = model.predict(X_test)

train = mu[:training]
test = mu[training:].copy(deep=True)
test['pred'] = ss1.inverse_transform(np.concatenate((np.zeros((len(pred), 3)), pred), axis=1))[:, 3]
#test['pred'] = pred
test_loss, test_rmse = model.evaluate(X_test, y_test)

print(f"Test Loss (MAE): {test_loss}")
print(f"Test RMSE: {test_rmse}")

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


mape = mean_absolute_percentage_error(y_test_inv, test['pred'])
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")



#print(train.iloc[::-1])
#print(test)

plt.figure(figsize=(10, 8))
plt.plot(train["Price"], c="b")
plt.plot(test[["Price", "pred"]])
plt.ylabel("price")
plt.legend(['train', 'test', 'predic'])
plt.show()


"""
residuals = y_test - pred.flatten()

plt.figure(figsize=(10, 6))
plt.scatter(range(len(residuals)), residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals Plot')
plt.xlabel('Index')
plt.ylabel('Residuals')
plt.show()


plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=50, alpha=0.75)
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()
"""

# input sequence
input_seq = ss[-batch_size:]
input_seq = np.reshape(input_seq, (1, batch_size, 1))

# prediction
latest_pred = model.predict(input_seq)

# inverse transform the prediction to get the actual value
latest_prediction_actual = ss1.inverse_transform(latest_pred)

print(f"Predicted value for the latest data: {latest_prediction_actual[0][0]}")
returns = ((latest_prediction_actual[0][0] - dataset[len(dataset) - 1]) /  dataset[len(dataset) - 1]) * 100
print(returns)
