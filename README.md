# Plus500 AutoTrader

An automated trading system for EUR/USD on Plus500 using LSTM neural networks and Playwright browser automation.

## 🚀 Overview

This project implements a complete automated trading solution that:

1. **Predicts EUR/USD price movements** using an LSTM neural network trained on 15-minute historical data
2. **Automates trading execution** on Plus500 using Playwright browser automation with coordinate-based clicking
3. **Manages risk** with configurable stop-loss, take-profit, and position sizing
4. **Operates continuously** with real-time decision making

## 🏗️ Architecture

```
plus500-autotrader/
├── config.py          # Configuration and settings
├── data_handler.py     # Data loading, processing, and LSTM preparation
├── engine.py           # LSTM model implementation and training
├── trader.py           # Playwright automation for Plus500 interface
├── main.py            # Main orchestration script
├── requirements.txt   # Python dependencies
├── setup_env.sh      # Environment setup script
├── data/             # Historical data storage
├── models/           # Trained model storage
├── logs/             # Application logs
└── tests/            # Test scripts
```

## 🎯 Key Features

### LSTM Engine
- **Time Series Prediction**: LSTM model trained on EUR/USD 15-minute intervals
- **Feature Engineering**: Incorporates lagged values, rolling statistics, and time-based features
- **Model Persistence**: Automatic saving and loading of trained models
- **Confidence Scoring**: Provides confidence metrics for trading decisions

### Playwright Automation
- **Browser Automation**: Automated interaction with Plus500 web platform
- **Coordinate-Based Clicking**: Reliable UI interaction using X/Y coordinates
- **Trade Execution**: Automated buy/sell order placement with stop-loss and take-profit
- **Position Management**: Opening and closing positions based on model signals

### Risk Management
- **Position Sizing**: Configurable risk percentage per trade (default: 2%)
- **Stop Loss**: Automatic stop-loss placement (default: 20 pips)
- **Take Profit**: Automatic take-profit placement (default: 40 pips)
- **Max Concurrent Trades**: Limits simultaneous open positions

## 📋 Prerequisites

- Python 3.8 or higher
- Plus500 account (demo account recommended for testing)
- Chrome/Chromium browser (for Playwright)

## 🛠️ Installation

### 1. Clone and Setup Environment

```bash
cd /home/maksich/Documents/plus500-autotrader
chmod +x setup_env.sh
./setup_env.sh
```

### 2. Activate Virtual Environment

```bash
source venv/bin/activate
```

### 3. Configure Credentials (Optional)

```bash
export PLUS500_USERNAME="your_username"
export PLUS500_PASSWORD="your_password"
```

## 🎮 Usage

### Training Mode
Train a new LSTM model on historical data:

```bash
python main.py --mode train
```

### Prediction Mode (Default)
Get a single trading prediction without executing trades:

```bash
python main.py --mode predict
```

### Trading Mode
Run automated trading (requires credentials):

```bash
python main.py --mode trade --username your_username --password your_password --duration 8
```

### Test Individual Components

Test data processing:
```bash
python data_handler.py
```

Test LSTM model:
```bash
python engine.py
```

Test Playwright automation:
```bash
python trader.py
```

## ⚙️ Configuration

### Trading Parameters (`config.py`)

```python
@dataclass
class TradingConfig:
    risk_percentage: float = 0.02      # 2% of account per trade
    stop_loss_pips: int = 20           # Stop loss in pips
    take_profit_pips: int = 40         # Take profit in pips
    max_concurrent_trades: int = 3     # Max simultaneous positions
    trading_start_hour: int = 8        # Trading start time (8 AM)
    trading_end_hour: int = 18         # Trading end time (6 PM)
    min_confidence_threshold: float = 0.7  # Minimum prediction confidence
```

### Model Parameters

```python
@dataclass
class ModelConfig:
    lstm_units: int = 50               # LSTM layer neurons
    dropout_rate: float = 0.2          # Dropout rate for regularization
    sequence_length: int = 60          # Input sequence length (15 hours)
    epochs: int = 100                  # Training epochs
    batch_size: int = 32               # Training batch size
```

### Coordinate Calibration

**Important**: The Plus500 coordinate mappings in `config.py` need to be calibrated for your specific screen resolution and browser setup:

```python
@dataclass
class Plus500Coordinates:
    eur_usd_pair: Tuple[int, int] = (300, 400)     # EUR/USD selection
    buy_button: Tuple[int, int] = (600, 500)       # Buy button
    sell_button: Tuple[int, int] = (400, 500)      # Sell button
    # ... other coordinates
```

## 📊 Data Format

The system expects EUR/USD data in the following format:

```csv
timestamp,change_percent,hour,day_of_week,minute
2023-10-01 08:00:00,0.22,8,6,0
2023-10-01 08:15:00,-0.13,8,6,15
2023-10-01 08:30:00,0.07,8,6,30
```

If no historical data is provided, the system will generate realistic sample data for training.

## 🔒 Security Considerations

### Credential Management
- Use environment variables for credentials
- Never commit passwords to version control
- Consider using demo accounts for testing

### Browser Security
- Playwright runs in a controlled browser environment
- Uses custom user agents and disables automation detection
- Screenshots are saved to `logs/` for debugging

### Risk Management
- Start with small position sizes
- Use stop-losses on all trades
- Monitor system performance closely
- Test thoroughly on demo accounts first

## 🧪 Testing

### Unit Tests
Run the test suite:
```bash
pytest tests/
```

### Manual Testing
Test individual components:
```bash
# Test data handler
python data_handler.py

# Test LSTM engine
python engine.py

# Test trader (browser automation)
python trader.py
```

## 📝 Logs and Monitoring

### Log Files
- Application logs: `logs/autotrader_YYYYMMDD.log`
- Screenshots: `logs/screenshot_*.png`
- Model files: `models/lstm_model.h5`
- Scalers: `models/scaler.pkl`

### Monitoring Metrics
- Trading signals and confidence scores
- Model prediction accuracy
- Trade execution success rates
- P&L tracking (to be implemented)

## ⚠️ Important Disclaimers

### Trading Risks
- **Automated trading involves significant financial risk**
- Past performance does not guarantee future results
- Use only funds you can afford to lose
- Test extensively on demo accounts before live trading

### Legal Compliance
- Ensure compliance with your local financial regulations
- Plus500 terms of service must be followed
- Some jurisdictions may restrict automated trading

### Technical Limitations
- Coordinate-based clicking may break if Plus500 updates their interface
- Internet connectivity issues can disrupt trading
- The LSTM model requires sufficient historical data for accurate predictions

## 🔧 Troubleshooting

### Common Issues

**Model Training Fails**
- Ensure sufficient historical data (at least 3 months)
- Check TensorFlow/GPU compatibility
- Verify data format and column names

**Browser Automation Breaks**
- Recalibrate coordinates in `config.py`
- Check if Plus500 interface has changed
- Verify Playwright browser installation

**Trading Signals Not Generated**
- Check if model is properly trained and loaded
- Ensure sufficient recent data for predictions
- Verify confidence thresholds

### Debug Mode
Enable debug logging:
```bash
export DEBUG=true
python main.py --mode predict
```

## 🚀 Future Enhancements

- [ ] Multi-currency pair support
- [ ] Advanced backtesting framework
- [ ] Real-time P&L tracking
- [ ] Web dashboard for monitoring
- [ ] Integration with multiple brokers
- [ ] Advanced feature engineering (technical indicators)
- [ ] Ensemble model predictions
- [ ] Position sizing optimization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is provided as-is for educational purposes. Use at your own risk.

## 🙋‍♂️ Support

For questions or issues:
1. Check the troubleshooting section
2. Review the logs in `logs/`
3. Test individual components
4. Create an issue with detailed information

---

**Remember**: This system is designed for educational and research purposes. Always test thoroughly and trade responsibly!