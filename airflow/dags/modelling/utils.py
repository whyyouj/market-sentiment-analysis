import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error
import pickle

# Data Preparation
def prepare_data(df, features=["Price"]):
    # Select the relevant columns
    data = df[features].copy().dropna()

    # Normalize the data
    scaler_dict = {}
    for column in data.columns:
        scaler = MinMaxScaler()
        data[column] = scaler.fit_transform(data[column].values.reshape(-1, 1))
        scaler_dict[column] = scaler
    
    return data, scaler_dict, features

# Create sequences
def create_sequences(data, seq_length, target_column='Price'):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i + seq_length)].values
        y = data.iloc[i + seq_length][target_column]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Load saved scaler
def load_scaler():
    scaler_path = os.path.join(os.getcwd(), 'trained_models', 'scalers', 'scaler.pkl')
    if not os.path.exists(scaler_path):
        print(f"Error: File '{scaler_path}' does not exist.")
        return None
            
    # Open the file in binary read mode and load its contents
    with open(scaler_path, 'rb') as f:
        scaler_dict = pickle.load(f)

    return scaler_dict

# Price prediction using model
def predict(model, scaler_dict, data, features):
    # Apply the same scalers to test data
    data_scaled = data[features].copy()
    for column in data_scaled.columns:
        data_scaled[column] = scaler_dict[column].transform(
            data_scaled[column].values.reshape(-1, 1)
        )

    # Create a sequence for the latest seq_length days of data
    sequence = list(data_scaled.values)
    sequence = np.array(sequence)

    # Make +1 day prediction
    y_pred = model.predict(sequence)
    
    # Denormalize prediction
    price_scaler = scaler_dict['Price']
    y_pred_denorm = price_scaler.inverse_transform(y_pred).flatten()
    
    return y_pred_denorm

# Prepare data for feature importance computation
def prepare_data_for_analysis(scaler_dict, data, features, seq_length=60):
    # Apply the same scalers to test data
    data_scaled = data[features].copy()
    for column in data_scaled.columns:
        data_scaled[column] = scaler_dict[column].transform(
            data_scaled[column].values.reshape(-1, 1)
        )

    # Create sequences for the data
    X_test, y_test = create_sequences(data_scaled, seq_length, target_column='Actual')

    return X_test, y_test

# Analyse feature importances
def analyse_feature_importance(model, X_test, y_test, feature_names, 
                              batch_size=None, 
                              n_repeats=5,
                              normalize=True):
    """
    Analyse feature importance using timestep-aware permutation for sequential models.
    """
    
    # Get baseline predictions and baseline MSE
    baseline_preds = model.predict(X_test, batch_size=batch_size, verbose=0)
    baseline_mse = mean_squared_error(y_test, baseline_preds)
    
    # Initialize arrays to store results
    n_features = X_test.shape[2]
    importances = np.zeros((n_repeats, n_features))
    
    # For each feature
    for i in range(n_features):
        # For each repetition
        for r in range(n_repeats):
            # Create a copy of the test data
            X_permuted = X_test.copy()
            
            # Permute feature values per timestep to maintain sequential structure
            # (Works for both LSTM and Transformer)
            for t in range(X_permuted.shape[1]):  # Loop over timesteps
                X_permuted[:, t, i] = np.random.permutation(X_permuted[:, t, i])
            
            # Get predictions with permuted feature
            permuted_preds = model.predict(X_permuted, batch_size=batch_size, verbose=0)
            permuted_mse = mean_squared_error(y_test, permuted_preds)
            
            # Calculate importance: how much the performance drops due to permutation
            # Higher value = more important feature
            importances[r, i] = permuted_mse - baseline_mse
    
    # Calculate mean of importance scores across repetitions
    mean_importances = np.mean(importances, axis=0)
    
    # Force non-negative importance values (some might be negative due to randomness)
    mean_importances = np.maximum(0, mean_importances)
    
    # Normalize if requested and possible
    if normalize and np.sum(mean_importances) > 0:
        mean_importances = mean_importances / np.sum(mean_importances)
    
    # Return results as a dictionary
    results = dict(zip(feature_names, mean_importances))
    
    return results