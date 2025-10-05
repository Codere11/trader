"""
Data Handler for Plus500 AutoTrader
Handles EUR/USD data loading, processing, and preparation for LSTM model
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Optional
import logging

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataHandler:
    """Handles all data operations for the AutoTrader system"""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.data = None
        self.processed_data = None
        
    def create_sample_data(self, months: int = 3) -> pd.DataFrame:
        """
        Create sample EUR/USD 15-minute data for the last N months
        Format: timestamp, change_percent
        Example: 2023-10-01 10:00:00, +0.22
        """
        logger.info(f"Creating sample data for {months} months")
        
        # Generate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months * 30)
        
        # Create 15-minute intervals
        date_range = pd.date_range(
            start=start_date, 
            end=end_date, 
            freq='15T'  # 15 minutes
        )
        
        # Remove weekends (forex market closed)
        date_range = date_range[date_range.weekday < 5]
        
        # Generate realistic EUR/USD change percentages
        # Using random walk with some trend and volatility
        np.random.seed(42)  # For reproducibility
        n_points = len(date_range)
        
        # Base random changes
        changes = np.random.normal(0, 0.15, n_points)  # Mean 0, std 0.15%
        
        # Add some autocorrelation (trending behavior)
        for i in range(1, len(changes)):
            changes[i] += 0.3 * changes[i-1]
        
        # Add time-of-day effects (higher volatility during market open/close)
        for i, dt in enumerate(date_range):
            hour = dt.hour
            if hour in [8, 9, 15, 16, 17]:  # Market open/close hours
                changes[i] *= 1.5
            elif hour in [12, 13, 14]:  # Lunch time - lower volatility
                changes[i] *= 0.7
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': date_range,
            'change_percent': np.round(changes, 2)
        })
        
        # Add additional features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['minute'] = df['timestamp'].dt.minute
        
        logger.info(f"Created {len(df)} data points from {start_date.date()} to {end_date.date()}")
        return df
    
    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load EUR/USD data from file or create sample data"""
        if file_path is None:
            file_path = config.data.data_file
        
        try:
            # Try to load existing data
            self.data = pd.read_csv(file_path)
            logger.info(f"Loaded data from {file_path}")
        except FileNotFoundError:
            # Create sample data if file doesn't exist
            logger.warning(f"Data file {file_path} not found. Creating sample data.")
            self.data = self.create_sample_data()
            
            # Save the sample data
            self.data.to_csv(file_path, index=False)
            logger.info(f"Sample data saved to {file_path}")
        
        # Ensure timestamp column is datetime
        if 'timestamp' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        
        return self.data
    
    def preprocess_data(self) -> pd.DataFrame:
        """Preprocess data for LSTM training"""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info("Preprocessing data for LSTM training")
        
        # Sort by timestamp
        df = self.data.sort_values('timestamp').copy()
        
        # Remove any NaN values
        df = df.dropna()
        
        # Create additional features
        if 'hour' not in df.columns:
            df['hour'] = df['timestamp'].dt.hour
        if 'day_of_week' not in df.columns:
            df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Normalize hour and day_of_week to 0-1 range
        df['hour_norm'] = df['hour'] / 23.0
        df['day_of_week_norm'] = df['day_of_week'] / 6.0
        
        # Create lagged features
        for lag in [1, 2, 3, 5, 10]:
            df[f'change_lag_{lag}'] = df['change_percent'].shift(lag)
        
        # Create rolling statistics
        for window in [5, 10, 20]:
            df[f'change_rolling_mean_{window}'] = df['change_percent'].rolling(window=window).mean()
            df[f'change_rolling_std_{window}'] = df['change_percent'].rolling(window=window).std()
        
        # Remove rows with NaN values created by lagging/rolling
        df = df.dropna()
        
        self.processed_data = df
        logger.info(f"Preprocessed data shape: {df.shape}")
        
        return df
    
    def create_sequences(self, data: pd.DataFrame, 
                        sequence_length: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training
        Returns X (features) and y (targets) arrays
        """
        if sequence_length is None:
            sequence_length = config.model.sequence_length
        
        logger.info(f"Creating sequences with length {sequence_length}")
        
        # Select feature columns
        feature_cols = [
            'change_percent', 'hour_norm', 'day_of_week_norm'
        ] + [col for col in data.columns if 'lag_' in col or 'rolling_' in col]
        
        # Ensure all feature columns exist
        available_cols = [col for col in feature_cols if col in data.columns]
        logger.info(f"Using features: {available_cols}")
        
        features = data[available_cols].values
        target = data['change_percent'].values
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        X, y = [], []
        
        for i in range(sequence_length, len(features_scaled)):
            X.append(features_scaled[i-sequence_length:i])
            y.append(target[i])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Created sequences - X shape: {X.shape}, y shape: {y.shape}")
        
        return X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   train_ratio: float = None) -> Tuple[np.ndarray, ...]:
        """Split data into train and test sets"""
        if train_ratio is None:
            train_ratio = config.data.train_test_split
        
        split_idx = int(len(X) * train_ratio)
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"Data split - Train: {len(X_train)}, Test: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def save_scaler(self, path: str = None):
        """Save the fitted scaler for later use"""
        if path is None:
            path = config.model.scaler_save_path
        
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Scaler saved to {path}")
    
    def load_scaler(self, path: str = None):
        """Load a previously fitted scaler"""
        if path is None:
            path = config.model.scaler_save_path
        
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)
        logger.info(f"Scaler loaded from {path}")
    
    def prepare_real_time_data(self, recent_data: List[float]) -> np.ndarray:
        """
        Prepare real-time data for prediction
        recent_data should be a list of recent change_percent values
        """
        if len(recent_data) < config.model.sequence_length:
            raise ValueError(f"Need at least {config.model.sequence_length} data points")
        
        # Take the last sequence_length points
        sequence = recent_data[-config.model.sequence_length:]
        
        # For now, just use change_percent (would need full feature engineering in production)
        # This is a simplified version for demo purposes
        features = np.array(sequence).reshape(-1, 1)
        features_scaled = self.scaler.transform(features)
        
        return features_scaled.reshape(1, config.model.sequence_length, -1)


if __name__ == "__main__":
    # Test the data handler
    handler = DataHandler()
    
    # Load/create data
    data = handler.load_data()
    print(f"Loaded data shape: {data.shape}")
    print(data.head())
    
    # Preprocess data
    processed = handler.preprocess_data()
    print(f"Processed data shape: {processed.shape}")
    
    # Create sequences
    X, y = handler.create_sequences(processed)
    print(f"Sequences - X: {X.shape}, y: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = handler.split_data(X, y)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")