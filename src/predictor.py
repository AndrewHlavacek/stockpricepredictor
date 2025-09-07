"""
Prediction Module for Stock Price Prediction
Handles making predictions and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    """Handles stock price predictions and visualization"""
    
    def __init__(self, model, scaler):
        """
        Initialize the predictor
        
        Args:
            model: Trained LSTM model
            scaler: Fitted MinMaxScaler
        """
        self.model = model
        self.scaler = scaler
        
    def predict_next_days(self, test_data, days=30):
        """
        Predict stock prices for the next N days
        
        Args:
            test_data (tuple): (X_test, y_test) from training
            days (int): Number of days to predict ahead
            
        Returns:
            tuple: (predictions, actual_values)
        """
        X_test, y_test = test_data
        
        # Get the last sequence for prediction
        last_sequence = X_test[-1].reshape(1, X_test.shape[1], X_test.shape[2])
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        print(f"🔮 Predicting next {days} days...")
        
        for i in range(days):
            # Make prediction
            next_pred = self.model.predict(current_sequence, verbose=0)
            predictions.append(next_pred[0, 0])
            
            # Update sequence for next prediction
            # Shift the sequence and add the new prediction
            new_row = current_sequence[0, -1].copy()
            new_row[0] = next_pred[0, 0]  # Update close price
            
            # Shift and append
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1] = new_row
        
        predictions = np.array(predictions)
        
        # Get actual values for comparison (if available)
        actual = y_test[-days:] if len(y_test) >= days else y_test
        
        return predictions, actual
    
    def evaluate_model(self, actual, predictions):
        """
        Evaluate prediction performance
        
        Args:
            actual (np.array): Actual values
            predictions (np.array): Predicted values
        """
        if len(actual) == 0:
            print("⚠️ No actual values available for comparison")
            return
        
        # Ensure arrays have the same length for comparison
        min_length = min(len(actual), len(predictions))
        actual = actual[:min_length]
        predictions = predictions[:min_length]
        
        # Calculate metrics
        mse = mean_squared_error(actual, predictions)
        mae = mean_absolute_error(actual, predictions)
        rmse = np.sqrt(mse)
        
        # Calculate percentage errors
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100
        
        # Directional accuracy
        if len(actual) > 1:
            actual_direction = np.diff(actual)
            pred_direction = np.diff(predictions)
            directional_accuracy = np.mean((actual_direction * pred_direction) > 0) * 100
        else:
            directional_accuracy = 0
        
        print("\n📊 Prediction Performance:")
        print(f"   MSE: {mse:.6f}")
        print(f"   MAE: {mae:.6f}")
        print(f"   RMSE: {rmse:.6f}")
        print(f"   MAPE: {mape:.2f}%")
        print(f"   Directional Accuracy: {directional_accuracy:.2f}%")
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'directional_accuracy': directional_accuracy
        }
    
    def plot_predictions(self, actual, predictions, symbol, days_ahead=30):
        """
        Create visualization of predictions vs actual values
        
        Args:
            actual (np.array): Actual values
            predictions (np.array): Predicted values
            symbol (str): Stock symbol
            days_ahead (int): Number of days predicted
        """
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{symbol} Stock Price Prediction Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Predictions vs Actual
        ax1 = axes[0, 0]
        days_range = range(len(predictions))
        
        if len(actual) > 0:
            ax1.plot(days_range[:len(actual)], actual, 'b-', label='Actual', linewidth=2)
        ax1.plot(days_range, predictions, 'r--', label='Predicted', linewidth=2)
        
        ax1.set_title('Predictions vs Actual Values')
        ax1.set_xlabel('Days Ahead')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Prediction Error
        ax2 = axes[0, 1]
        if len(actual) > 0:
            # Ensure arrays have the same length
            min_length = min(len(actual), len(predictions))
            actual_trimmed = actual[:min_length]
            predictions_trimmed = predictions[:min_length]
            errors = actual_trimmed - predictions_trimmed
            ax2.plot(days_range[:len(errors)], errors, 'g-', linewidth=2)
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax2.set_title('Prediction Errors')
            ax2.set_xlabel('Days Ahead')
            ax2.set_ylabel('Error (Actual - Predicted)')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No actual values\nfor comparison', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Prediction Errors')
        
        # Plot 3: Price Distribution
        ax3 = axes[1, 0]
        ax3.hist(predictions, bins=20, alpha=0.7, color='red', label='Predicted')
        if len(actual) > 0:
            min_length = min(len(actual), len(predictions))
            ax3.hist(actual[:min_length], bins=20, alpha=0.7, color='blue', label='Actual')
        ax3.set_title('Price Distribution')
        ax3.set_xlabel('Price')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Prediction Confidence (using prediction variance)
        ax4 = axes[1, 1]
        if len(actual) > 0:
            # Calculate rolling prediction accuracy
            window = min(5, len(actual))
            accuracy_scores = []
            for i in range(window, len(actual)):
                recent_actual = actual[i-window:i]
                recent_pred = predictions[i-window:i]
                accuracy = 1 - np.mean(np.abs(recent_actual - recent_pred) / recent_actual)
                accuracy_scores.append(accuracy)
            
            ax4.plot(range(window, len(actual)), accuracy_scores, 'purple', linewidth=2)
            ax4.set_title('Rolling Prediction Accuracy')
            ax4.set_xlabel('Days')
            ax4.set_ylabel('Accuracy Score')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No actual values\nfor accuracy calculation', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Prediction Accuracy')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print(f"\n📈 Prediction Summary for {symbol}:")
        print(f"   Predicted {len(predictions)} days ahead")
        print(f"   Price range: ${predictions.min():.2f} - ${predictions.max():.2f}")
        print(f"   Average predicted price: ${predictions.mean():.2f}")
        
        if len(actual) > 0:
            print(f"   Average actual price: ${actual.mean():.2f}")
            print(f"   Price trend: {'📈 Upward' if predictions[-1] > predictions[0] else '📉 Downward'}")
    
    def get_feature_importance(self):
        """
        Get feature importance from the model (simplified approach)
        
        Returns:
            dict: Feature importance scores
        """
        # This is a simplified approach - in practice, you might want to use
        # techniques like SHAP or permutation importance for LSTM models
        
        feature_names = [
            'close', 'volume', 'sma_5', 'sma_20', 'sma_50',
            'ema_12', 'ema_26', 'macd', 'macd_signal', 'rsi',
            'bb_upper', 'bb_lower', 'price_change', 'volume_ratio'
        ]
        
        # Get model weights (simplified approach)
        try:
            # Extract weights from the first LSTM layer
            lstm_weights = self.model.layers[0].get_weights()[0]
            feature_importance = np.mean(np.abs(lstm_weights), axis=1)
            
            # Normalize importance scores
            feature_importance = feature_importance / np.sum(feature_importance)
            
            importance_dict = dict(zip(feature_names, feature_importance))
            
            print("\n🔍 Feature Importance (Top 5):")
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            for i, (feature, importance) in enumerate(sorted_features[:5]):
                print(f"   {i+1}. {feature}: {importance:.4f}")
            
            return importance_dict
            
        except Exception as e:
            print(f"⚠️ Could not extract feature importance: {str(e)}")
            return {}
