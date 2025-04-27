import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import os
from .utils import prepare_data, create_sequences
from keras.saving import register_keras_serializable

# ==================== Transformer Model Architecture ====================

@register_keras_serializable(package="Custom")
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_seq_length=1000):
        super().__init__()
        
        # Create positional encodings
        pos = np.arange(max_seq_length)[:, np.newaxis]  # Shape: [max_seq_length, 1]
        i = np.arange(d_model)[np.newaxis, :]  # Shape: [1, d_model]

        # Compute the angular frequencies
        div_term = np.exp(-np.log(10000.0) * (2 * (i // 2) / d_model))

        # Apply sine to even indices and cosine to odd indices
        pos_encoding = np.zeros((max_seq_length, d_model))
        pos_encoding[:, 0::2] = np.sin(pos * div_term[:, 0::2])  # Even indices
        pos_encoding[:, 1::2] = np.cos(pos * div_term[:, 1::2])  # Odd indices

        self.pos_encoding = tf.cast(pos_encoding[np.newaxis, :, :], tf.float32)  # Add batch dimension: [1, max_seq_len, d_model]

        self.d_model = d_model
        self.max_seq_length = max_seq_length

    def call(self, inputs):
        seq_length = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_length, :]
    
    def get_config(self):
        config = {
            "d_model": self.d_model,
            "max_seq_length": self.max_seq_length
        }
        return config

@register_keras_serializable(package="Custom")
class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout_rate=0.1, attention_decay_factor=0.01, ff_dim=None):
        super().__init__()
        if ff_dim is None:
            ff_dim = 4 * d_model

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model//num_heads)
        
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2=0.01)),
            tf.keras.layers.Dense(d_model, kernel_regularizer=tf.keras.regularizers.l2(l2=0.01))
        ])
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        
        # Store attention weights
        self.attention_weights = None

        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.attention_decay_factor = attention_decay_factor
        self.ff_dim = ff_dim
        
    def call(self, inputs, training=None, mask=None):
        # Multi-head attention with residual connection and layer normalization
        attn_output, attention_weights = self.mha(
            inputs, inputs, 
            return_attention_scores=True,
            attention_mask=mask
        )
        
        # Store attention weights for later analysis
        self.attention_weights = attention_weights

        # Add attention decay regularization calculation
        if training:
            seq_length = tf.shape(attention_weights)[-1]
        
            # Create position indices
            positions = tf.range(seq_length, dtype=tf.float32)
            
            # Calculate distance matrix (how far each position is from other positions)
            pos_i = tf.expand_dims(positions, 0)  # [1, seq_len]
            pos_j = tf.expand_dims(positions, 1)  # [seq_len, 1]
            distance_matrix = tf.abs(pos_i - pos_j)  # [seq_len, seq_len]
            
            # Create decay mask - HIGHER weights for DISTANT positions
            decay_mask = 1.0 - tf.exp(-self.attention_decay_factor * distance_matrix)
            
            # Average attention across batch and heads
            avg_attention = tf.reduce_mean(attention_weights, axis=[0, 1])
            
            # Calculate penalty - higher when distant positions get low attention
            recency_penalty = tf.reduce_sum(avg_attention * decay_mask) 
            
            self.add_loss(self.attention_decay_factor * recency_penalty)
        
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward network with residual connection and layer normalization
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = {
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "attention_decay_factor": self.attention_decay_factor,
            "ff_dim": self.ff_dim
        }
        return config

@register_keras_serializable(package="Custom")
class GoldPriceTransformer(tf.keras.Model):
    def __init__(self, input_dim, output_dim, d_model=128, num_heads=8, 
                 num_layers=3, dropout_rate=0.1, attention_decay_factor=0.01):
        super().__init__()
        
        self.input_projection = tf.keras.layers.Dense(d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        self.encoder_layers = [
            TransformerEncoderLayer(
                d_model, 
                num_heads, 
                dropout_rate, 
                attention_decay_factor=attention_decay_factor
            )
            for _ in range(num_layers)
        ]
        
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.output_projection = tf.keras.layers.Dense(output_dim)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.attention_decay_factor = attention_decay_factor
        
    def call(self, inputs, training=None, mask=None):
        # inputs shape: [batch_size, seq_len, input_dim]
        x = self.input_projection(inputs)
        x = self.positional_encoding(x)
        x = self.dropout(x, training=training)
        
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, training=training, mask=mask)
            
        # Extract the last time step for forecasting
        final_output = x[:, -1, :]
        return self.output_projection(final_output)
    
    def get_config(self):
        config = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "dropout_rate": self.dropout_rate,
            "attention_decay_factor": self.attention_decay_factor
        }
        return config

# ==================== Transformer Modelling Functions ====================

# Model training
def train_transformer(df, best_hyperparameters, features, model_name, seq_length=60):
    # Prepare training data
    if "Price" not in features:
        features.append("Price")
    train_data, scaler_dict, features = prepare_data(df, features)

    # Create sequences for train set
    X_train, y_train = create_sequences(train_data, seq_length)
    
    print(f"[train_transformer] Training data shape: {X_train.shape}")
    
    # Initialize the model
    input_dim = X_train.shape[2]
    output_dim = 1
    
    model = GoldPriceTransformer(
        input_dim=input_dim,
        output_dim=output_dim,
        d_model=best_hyperparameters['d_model'],
        num_heads=best_hyperparameters['num_heads'],
        num_layers=best_hyperparameters['num_layers'],
        dropout_rate=best_hyperparameters['dropout_rate'],
        attention_decay_factor=best_hyperparameters['attention_decay_factor']
    )
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mean_squared_error",
    )
    
    # Define callbacks
    callbacks = []
    
    # Add early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    callbacks.append(early_stopping)

    # Train the model
    model.fit(
        X_train, y_train,
        validation_split=0.2,
        batch_size=best_hyperparameters['batch_size'],
        epochs=best_hyperparameters['epochs'],
        callbacks=callbacks,
        verbose=1
    )

    # Create models directory if it does not exist
    models_path = os.path.join(os.getcwd(), 'dags', 'trained_models')
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    # Save the model trained with full feature set
    model.save(f'dags/trained_models/{model_name}.keras')
    abs_model_path = os.path.abspath(os.path.join(models_path, f"{model_name}.keras"))
    print(f'{model_name} saved to {abs_model_path}')

    return scaler_dict, abs_model_path