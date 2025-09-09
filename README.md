# Stock Price Predictor 🚀📈

A lightweight and explainable stock price prediction project using LSTM neural networks and the Alpaca API.

## Features ✨

- **Real-time Data Collection**: Fetches historical stock data from Alpaca API
- **Advanced Technical Indicators**: Includes SMA, EMA, MACD, RSI, Bollinger Bands
- **LSTM Neural Network**: Deep learning model for time series prediction
- **Comprehensive Evaluation**: Multiple metrics including directional accuracy
- **Beautiful Visualizations**: Interactive charts and analysis plots
- **Explainable AI**: Feature importance analysis and model interpretability

## Project Structure 📁

```
stockpricepredictor/
├── main.py                 # Main pipeline script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── src/                   # Source code modules
│   ├── __init__.py
│   ├── data_collector.py  # Alpaca API data collection
│   ├── model_trainer.py   # LSTM model training
│   └── predictor.py       # Prediction and visualization
├── data/                  # Data storage directory
├── notebooks/             # Jupyter notebooks
└── tests/                 # Unit tests
```

## Installation 🛠️

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd stockpricepredictor
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Alpaca API credentials**:
   - Get your API key and secret from [Alpaca](https://alpaca.markets/)
   - Update the credentials in `main.py`

## Usage 🚀

   - source venv/bin/activate
   - pip install -r requirements.txt
   - python main.py

### Quick Start

Run the complete pipeline:

```bash
python main.py
```

This will:
1. 📊 Collect 365 days of stock data for AAPL
2. 🧠 Train an LSTM model with technical indicators
3. 🔮 Make predictions for the next 30 days
4. 📈 Evaluate performance and create visualizations

### Custom Configuration

You can modify the following parameters in `main.py`:

```python
# Configuration
API_KEY = "your_api_key"
API_SECRET = "your_api_secret"
SYMBOL = "AAPL"  # Change to any stock symbol
```

### Advanced Usage

#### Data Collection Only
```python
from src.data_collector import DataCollector

collector = DataCollector(API_KEY, API_SECRET)
df = collector.get_stock_data("AAPL", days_back=365)
```

#### Model Training Only
```python
from src.model_trainer import ModelTrainer

trainer = ModelTrainer(sequence_length=60)
model, scaler, train_data, test_data = trainer.train_model(df)
```

#### Making Predictions
```python
from src.predictor import StockPredictor

predictor = StockPredictor(model, scaler)
predictions, actual = predictor.predict_next_days(test_data, days=30)
```

## Technical Details 🔬

### Model Architecture

- **Input**: 60-day sequences of 14 technical indicators
- **Architecture**: 3-layer LSTM with Batch Normalization and Dropout
- **Output**: Single price prediction
- **Optimizer**: Adam with learning rate scheduling
- **Regularization**: Early stopping and dropout

### Technical Indicators

1. **Moving Averages**: SMA (5, 20, 50), EMA (12, 26)
2. **MACD**: MACD line, signal line, histogram
3. **RSI**: Relative Strength Index
4. **Bollinger Bands**: Upper, middle, lower bands
5. **Volume Analysis**: Volume ratios and trends
6. **Price Features**: Price changes and ratios

### Evaluation Metrics

- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error
- **Directional Accuracy**: Percentage of correct trend predictions

## Model Performance 📊

The model typically achieves:
- **RMSE**: 2-5% of stock price
- **Directional Accuracy**: 60-70%
- **Training Time**: 5-15 minutes (depending on hardware)

## Explainability Features 🔍

1. **Feature Importance**: Shows which indicators are most influential
2. **Prediction Confidence**: Rolling accuracy scores
3. **Error Analysis**: Detailed error distribution and patterns
4. **Visual Interpretability**: Multiple chart types for analysis

## API Requirements 🔑

### Alpaca API Setup

1. Sign up at [Alpaca](https://alpaca.markets/)
2. Get your API key and secret
3. Update credentials in `main.py`

**Note**: The project uses Alpaca's free tier which provides:
- Real-time market data
- Historical data access
- No minimum balance requirements

## Dependencies 📦

- **alpaca-py**: Alpaca API client
- **tensorflow**: Deep learning framework
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning utilities
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical visualization

## Contributing 🤝

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License 📄

This project is open source and available under the MIT License.

## Disclaimer ⚠️

This project is for educational and research purposes only. Stock market predictions are inherently uncertain, and this tool should not be used as the sole basis for investment decisions. Always do your own research and consider consulting with financial professionals.

## Support 💬

If you encounter any issues or have questions:
1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed information

---

**Happy Predicting!** 🎯📈