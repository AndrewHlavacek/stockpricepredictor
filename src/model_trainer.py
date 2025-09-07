"""
Model Training Module for Stock Price Prediction
Handles LSTM model creation, training, and evaluation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """Handles LSTM model training and preparation"""
    
    def __init__(self, sequence_length=60, features=None):
        """
        Initialize the model trainer
        
        Args:
            sequence_length (int): Number of time steps to look back
            features (list): List of features to use for training
        """
        self.sequence_length = sequence_length
        self.features = features or [
            'close', 'volume', 'sma_5', 'sma_20', 'sma_50',
            'ema_12', 'ema_26', 'macd', 'macd_signal', 'rsi',
            'bb_upper', 'bb_lower', 'price_change', 'volume_ratio'
        ]
        self.scaler = MinMaxScaler()
        
    def prepare_data(self, df):
        """
        Prepare data for LSTM training
        
        Args:
            df (pd.DataFrame): Stock data with technical indicators
            
        Returns:
            tuple: (X, y) arrays for training
        """
        # Select features
        feature_data = df[self.features].values
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(feature_data)
        
        # Create sequences
        X, y = self._create_sequences(scaled_data)
        
        return X, y
    
    def _create_sequences(self, data):
        """
        Create sequences for LSTM training
        
        Args:
            data (np.array): Scaled feature data
            
        Returns:
            tuple: (X, y) arrays
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            # Use all features for input
            X.append(data[i-self.sequence_length:i])
            # Predict only the close price
            y.append(data[i, 0])  # Assuming 'close' is the first feature
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """
        Build LSTM model architecture
        
        Args:
            input_shape (tuple): Input shape for the model
            
        Returns:
            tf.keras.Model: Compiled LSTM model
        """
        model = Sequential([
            # First LSTM layer
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.2),
            
            # Second LSTM layer
            LSTM(units=50, return_sequences=True),
            BatchNormalization(),
            Dropout(0.2),
            
            # Third LSTM layer
            LSTM(units=50, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            
            # Dense layers
            Dense(units=25, activation='relu'),
            Dropout(0.2),
            Dense(units=1, activation='linear')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_model(self, df, test_size=0.2, validation_split=0.2):
        """
        Train the LSTM model
        
        Args:
            df (pd.DataFrame): Stock data
            test_size (float): Proportion of data for testing
            validation_split (float): Proportion of training data for validation
            
        Returns:
            tuple: (model, scaler, train_data, test_data)
        """
        print("🔄 Preparing data for training...")
        
        # Prepare data
        X, y = self.prepare_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        print(f"📊 Training samples: {len(X_train)}")
        print(f"📊 Test samples: {len(X_test)}")
        
        # Build model
        print("🏗️ Building LSTM model...")
        model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        # Print model summary
        print("\n📋 Model Architecture:")
        model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=0.0001,
                verbose=1
            )
        ]
        
        # Train model
        print("\n🚀 Starting model training...")
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=100,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Prepare return data
        train_data = (X_train, y_train)
        test_data = (X_test, y_test)
        
        print("✅ Model training completed!")
        
        return model, self.scaler, train_data, test_data
    
    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            model: Trained LSTM model
            X_test: Test input data
            y_test: Test target data
            
        Returns:
            dict: Evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = np.mean((y_test - y_pred.flatten()) ** 2)
        mae = np.mean(np.abs(y_test - y_pred.flatten()))
        rmse = np.sqrt(mse)
        
        # Calculate directional accuracy
        y_test_diff = np.diff(y_test)
        y_pred_diff = np.diff(y_pred.flatten())
        directional_accuracy = np.mean((y_test_diff * y_pred_diff) > 0) * 100
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'directional_accuracy': directional_accuracy
        }
        
        print("\n📈 Model Performance Metrics:")
        print(f"   MSE: {mse:.6f}")
        print(f"   MAE: {mae:.6f}")
        print(f"   RMSE: {rmse:.6f}")
        print(f"   Directional Accuracy: {directional_accuracy:.2f}%")
        
        return metrics
