"""
Data Collection Module for Stock Price Prediction
Handles fetching stock data from Alpaca API
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import warnings
warnings.filterwarnings('ignore')

class DataCollector:
    """Handles data collection from Alpaca API"""
    
    def __init__(self, api_key, api_secret):
        """
        Initialize the data collector with API credentials
        
        Args:
            api_key (str): Alpaca API key
            api_secret (str): Alpaca API secret
        """
        self.client = StockHistoricalDataClient(api_key, api_secret)
        
    def get_stock_data(self, symbol, days_back=365, timeframe="1Day"):
        """
        Fetch historical stock data for a given symbol
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            days_back (int): Number of days to look back
            timeframe (str): Data timeframe ('1Day', '1Hour', etc.)
            
        Returns:
            pd.DataFrame: Historical stock data with OHLCV columns
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Create request
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date
            )
            
            # Fetch data
            bars = self.client.get_stock_bars(request_params)
            
            # Convert to DataFrame
            df = bars.df
            
            if df.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Reset index to get symbol and timestamp as columns
            df = df.reset_index()
            
            # Rename columns for clarity
            df.columns = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']
            
            # Select relevant columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Add technical indicators
            df = self._add_technical_indicators(df)
            
            print(f"✅ Successfully collected {len(df)} days of data for {symbol}")
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {str(e)}")
            raise
    
    def _add_technical_indicators(self, df):
        """
        Add technical indicators to the dataframe
        
        Args:
            df (pd.DataFrame): Stock data dataframe
            
        Returns:
            pd.DataFrame: Dataframe with technical indicators
        """
        # Simple Moving Averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Price change features
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Remove rows with NaN values (due to rolling calculations)
        df = df.dropna().reset_index(drop=True)
        
        return df
    
    def _calculate_rsi(self, prices, window=14):
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            prices (pd.Series): Price series
            window (int): RSI window period
            
        Returns:
            pd.Series: RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def get_multiple_symbols(self, symbols, days_back=365):
        """
        Fetch data for multiple symbols
        
        Args:
            symbols (list): List of stock symbols
            days_back (int): Number of days to look back
            
        Returns:
            dict: Dictionary with symbol as key and DataFrame as value
        """
        data_dict = {}
        
        for symbol in symbols:
            try:
                data_dict[symbol] = self.get_stock_data(symbol, days_back)
            except Exception as e:
                print(f"⚠️ Skipping {symbol}: {str(e)}")
                continue
        
        return data_dict
