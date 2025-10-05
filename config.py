"""
Configuration module for Plus500 AutoTrader
Contains all settings and parameters for the trading system
"""

import os
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class TradingConfig:
    """Trading-specific configuration parameters"""
    # Trading parameters
    risk_percentage: float = 0.02  # 2% of account per trade
    stop_loss_pips: int = 20
    take_profit_pips: int = 40
    max_concurrent_trades: int = 3
    
    # Time windows
    trading_start_hour: int = 8  # 8 AM
    trading_end_hour: int = 18   # 6 PM
    
    # Model confidence threshold
    min_confidence_threshold: float = 0.7


@dataclass
class Plus500Coordinates:
    """Plus500 UI coordinate mappings for Playwright automation"""
    # Login coordinates (example - need to be calibrated)
    login_button: Tuple[int, int] = (500, 300)
    username_field: Tuple[int, int] = (400, 250)
    password_field: Tuple[int, int] = (400, 290)
    login_submit: Tuple[int, int] = (400, 330)
    
    # Trading coordinates
    eur_usd_pair: Tuple[int, int] = (300, 400)
    buy_button: Tuple[int, int] = (600, 500)
    sell_button: Tuple[int, int] = (400, 500)
    close_position: Tuple[int, int] = (700, 600)
    
    # Amount input
    amount_field: Tuple[int, int] = (500, 450)
    
    # Stop loss and take profit
    stop_loss_field: Tuple[int, int] = (450, 480)
    take_profit_field: Tuple[int, int] = (550, 480)


@dataclass
class ModelConfig:
    """LSTM model configuration"""
    # Model architecture
    lstm_units: int = 50
    dropout_rate: float = 0.2
    dense_units: int = 25
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    validation_split: float = 0.2
    
    # Data parameters
    sequence_length: int = 60  # 15 hours of 15-min intervals
    prediction_horizon: int = 4  # Predict 1 hour ahead (4x15min)
    
    # Model persistence
    model_save_path: str = "models/lstm_model.h5"
    scaler_save_path: str = "models/scaler.pkl"


@dataclass
class DataConfig:
    """Data handling configuration"""
    # Data source
    data_file: str = "data/eur_usd_15min.csv"
    
    # Data processing
    train_test_split: float = 0.8
    
    # Features
    feature_columns: list = None
    target_column: str = "change_percent"
    
    def __post_init__(self):
        if self.feature_columns is None:
            self.feature_columns = ["change_percent", "hour", "day_of_week"]


class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.trading = TradingConfig()
        self.coordinates = Plus500Coordinates()
        self.model = ModelConfig()
        self.data = DataConfig()
        
        # Environment-specific settings
        self.debug_mode = os.getenv('DEBUG', 'False').lower() == 'true'
        self.headless_browser = os.getenv('HEADLESS', 'True').lower() == 'true'
        
        # Paths
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.project_root, "data")
        self.models_dir = os.path.join(self.project_root, "models")
        self.logs_dir = os.path.join(self.project_root, "logs")
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)


# Global config instance
config = Config()