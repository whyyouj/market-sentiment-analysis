import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import os
from .utils import prepare_data, create_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras import regularizers
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ==================== LSTM Modelling Functions ====================

# Model building
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

# Model training
def train_lstm(df, best_hyperparameters, features, model_name, seq_length=60):
    
    # Prepare training data
    train_data, scaler_dict, features = prepare_data(df, features)
    
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
        epochs=30, # changed to 30 to be consistent with transformer
        verbose=1,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr]
    )

    # Create models directory if it does not exist
    models_path = os.path.join(os.getcwd(), 'dags', 'trained_models')
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    # Save the model trained with full feature set
    model.save(f'dags/trained_models/{model_name}.keras')
    abs_model_path = os.path.abspath(os.path.join(models_path, f"{model_name}.keras"))
    print(f'{model_name} saved to {abs_model_path}')

    # Output scaler for prediction function
    return scaler_dict