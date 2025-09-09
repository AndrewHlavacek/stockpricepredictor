"""
Stock Price Prediction using LSTM and Alpaca API
A lightweight and explainable stock price prediction project
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from src.data_collector import DataCollector
from src.model_trainer import ModelTrainer
from src.predictor import StockPredictor

def main():
    """Main function to run the stock price prediction pipeline"""
    
    # Configuration
    API_KEY = "PKU21LNR9BAPGDEJGRB3"
    API_SECRET = "axPiIoCPh2JLwN5v0lk902b7L1PODtwJ6rRCSaWk"
    SYMBOL = "TSLA"
    
    print("🚀 Starting Stock Price Prediction Pipeline")
    print("=" * 50)
    
    # Step 1: Collect data
    print("📊 Collecting stock data...")
    collector = DataCollector(API_KEY, API_SECRET)
    df = collector.get_stock_data(SYMBOL, days_back=365)
    print(f"✅ Collected {len(df)} days of data")
    
    # Step 2: Train model
    print("\n🧠 Training LSTM model...")
    trainer = ModelTrainer()
    model, scaler, train_data, test_data = trainer.train_model(df)
    print("✅ Model training completed")
    
    # Step 3: Make predictions
    print("\n🔮 Making predictions...")
    predictor = StockPredictor(model, scaler)
    predictions, actual = predictor.predict_next_days(test_data, days=30)
    
    # Step 4: Evaluate and visualize
    print("\n📈 Evaluating model performance...")
    predictor.evaluate_model(actual, predictions)
    predictor.plot_predictions(actual, predictions, SYMBOL)
    
    print("\n🎉 Pipeline completed successfully!")

if __name__ == "__main__":
    main()