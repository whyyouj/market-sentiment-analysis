import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras import regularizers
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ==================== LSTM Modelling Functions ====================
def prepare_data(data):
    data = data.copy().dropna()

    # Scale the data
    scaler = MinMaxScaler()
    train_data = data.values
    scaled_train_data = scaler.fit_transform(train_data)
    # scaled_test_data = np.vstack([scaled_train_data[-seq_length:], scaled_test_data])
    
    return scaled_train_data, scaler

def create_sequences(data, seq_length, target_col=0):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, target_col].reshape(1))
    
    return np.array(X), np.array(y)

def build_lstm_model(input_shape, best_hyperparameters):    
    # Extract hyperparameters
    num_lstm_layers = best_hyperparameters['num_lstm_layers']
    lstm_units = [
        best_hyperparameters['lstm_units_0'],
        best_hyperparameters['lstm_units_1'],
        best_hyperparameters['lstm_units_2']
    ]
    dense_units = best_hyperparameters['dense_units']
    l2_reg = best_hyperparameters['l2_reg']
    dropout_rate = best_hyperparameters['dropout_rate']
    learning_rate = best_hyperparameters['learning_rate']
    
    model = Sequential()
    model.add(Input(shape=input_shape))
    
    # Add LSTM layers
    for i in range(num_lstm_layers):
        return_sequences = True if i < num_lstm_layers - 1 else False
        model.add(LSTM(
            units=lstm_units[i],
            return_sequences=return_sequences,
            kernel_regularizer=regularizers.l2(l2_reg) # added to each LSTM layer
        ))
    
    # Add dense layer
    model.add(Dense(
        units=dense_units,
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg)
    ))
    model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(units=1))
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mean_squared_error',
        metrics=[RootMeanSquaredError()]
    )
    
    return model

def train_lstm(data, best_hyperparameters, features, seq_length=30):
    data = data[features]
    
    # Prepare training data
    train_data, scaler = prepare_data(data)
    
    # Create sequences for train set
    X_train, y_train = create_sequences(train_data, seq_length)

    print(f"[train_lstm] Training data shape: {X_train.shape}")
    
    # Get input shape
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Build the model
    model = build_lstm_model(input_shape, best_hyperparameters)
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    )
    
    # Train the model
    model.fit(
        X_train, y_train,
        epochs=40,
        verbose=1,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr]
    )

    # Create models directory if it does not exist
    models_path = os.path.join(os.getcwd(), 'trained_models')
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    # Save the trained model in models directory
    if len(features) == 1 and 'Price' in features:
        # Model trained with price only
        model.save('trained_models/lstm_price')
    else:
        # Model trained with full feature set
        model.save('trained_models/lstm_full')

    # Output scaler for prediction function
    return scaler

def lstm_predict(model, data, seq_length, features, scaler, target_col=0):
    # Prepare the input data
    input_data = data[features].copy()
    scaled_input = scaler.transform(input_data)
    
    # Create a sequence for the latest seq_length days of data
    last_sequence = scaled_input[-seq_length:].reshape(1, seq_length, len(features))
    
    # Predict the next value
    y_pred = model.predict(last_sequence, verbose=0)[0][0]

    # Create a template for inverse transformation with the correct shape
    y_pred_template = np.zeros((1, len(features)))
    
    # Place the prediction in the right column
    y_pred_template[0, target_col] = y_pred
    
    # Inverse transform to get actual value
    y_pred_denorm = scaler.inverse_transform(y_pred_template)[0, target_col]
    
    return y_pred_denorm