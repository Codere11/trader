"""
LSTM Engine for Plus500 AutoTrader
Implements LSTM model for EUR/USD price change prediction
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import logging
from typing import Tuple, Optional
import os

from config import config
from data_handler import DataHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMEngine:
    """LSTM model for EUR/USD price change prediction"""
    
    def __init__(self):
        self.model = None
        self.data_handler = DataHandler()
        self.is_trained = False
        
    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Build LSTM model architecture
        
        Args:
            input_shape: (sequence_length, n_features)
        
        Returns:
            Compiled Sequential model
        """
        logger.info(f"Building LSTM model with input shape: {input_shape}")
        
        model = Sequential([
            # First LSTM layer with return sequences
            LSTM(config.model.lstm_units, 
                 return_sequences=True, 
                 input_shape=input_shape),
            Dropout(config.model.dropout_rate),
            
            # Second LSTM layer
            LSTM(config.model.lstm_units // 2, 
                 return_sequences=False),
            Dropout(config.model.dropout_rate),
            
            # Dense layers
            Dense(config.model.dense_units, activation='relu'),
            Dropout(config.model.dropout_rate),
            
            # Output layer (regression for price change prediction)
            Dense(1, activation='linear')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info("Model architecture:")
        model.summary(print_fn=logger.info)
        
        self.model = model
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              save_model: bool = True) -> dict:
        """
        Train the LSTM model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            save_model: Whether to save the trained model
        
        Returns:
            Training history
        """
        if self.model is None:
            # Build model based on input shape
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.build_model(input_shape)
        
        logger.info(f"Training model with {len(X_train)} samples")
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            logger.info(f"Using {len(X_val)} validation samples")
        
        # Callbacks
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss' if validation_data else 'loss',
            patience=10,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
        
        # Model checkpoint
        if save_model:
            checkpoint_path = config.model.model_save_path
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            
            checkpoint = ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss' if validation_data else 'loss',
                save_best_only=True,
                verbose=1
            )
            callbacks.append(checkpoint)
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=config.model.epochs,
            batch_size=config.model.batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        logger.info("Model training completed")
        
        # Save scaler
        self.data_handler.save_scaler()
        
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            X: Input features
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() or load_model() first.")
        
        if not self.is_trained:
            logger.warning("Model may not be trained yet.")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
    
    def predict_single(self, sequence: np.ndarray) -> Tuple[float, float]:
        """
        Make a single prediction and return confidence
        
        Args:
            sequence: Single sequence of shape (1, sequence_length, n_features)
            
        Returns:
            Tuple of (prediction, confidence_score)
        """
        prediction = self.predict(sequence)[0]
        
        # Simple confidence calculation (can be improved)
        # For now, use the absolute value as inverse confidence
        # (smaller absolute predictions are more confident)
        confidence = 1.0 / (1.0 + abs(prediction))
        
        return prediction, confidence
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate the model on test data
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() or load_model() first.")
        
        logger.info(f"Evaluating model on {len(X_test)} test samples")
        
        # Get predictions
        predictions = self.predict(X_test)
        
        # Calculate metrics
        mse = np.mean((y_test - predictions) ** 2)
        mae = np.mean(np.abs(y_test - predictions))
        rmse = np.sqrt(mse)
        
        # Direction accuracy (for trading, direction matters more than exact value)
        y_direction = np.sign(y_test)
        pred_direction = np.sign(predictions)
        direction_accuracy = np.mean(y_direction == pred_direction)
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'direction_accuracy': direction_accuracy
        }
        
        logger.info("Evaluation metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def save_model(self, path: str = None):
        """Save the trained model"""
        if path is None:
            path = config.model.model_save_path
        
        if self.model is None:
            raise ValueError("No model to save")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str = None):
        """Load a previously trained model"""
        if path is None:
            path = config.model.model_save_path
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        self.model = load_model(path)
        self.is_trained = True
        logger.info(f"Model loaded from {path}")
        
        # Also load the scaler
        try:
            self.data_handler.load_scaler()
        except FileNotFoundError:
            logger.warning("Scaler file not found. You may need to retrain or provide scaler.")
    
    def get_trading_signal(self, recent_data: list) -> dict:
        """
        Get trading signal based on recent price data
        
        Args:
            recent_data: List of recent price change percentages
            
        Returns:
            Dictionary with signal information
        """
        if len(recent_data) < config.model.sequence_length:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'prediction': 0.0,
                'reason': f'Insufficient data (need {config.model.sequence_length} points)'
            }
        
        # Prepare data for prediction
        try:
            X = self.data_handler.prepare_real_time_data(recent_data)
            prediction, confidence = self.predict_single(X)
            
            # Determine signal based on prediction and confidence
            if confidence < config.trading.min_confidence_threshold:
                signal = 'HOLD'
                reason = f'Low confidence ({confidence:.2f})'
            elif prediction > 0.1:  # Expect upward movement > 0.1%
                signal = 'BUY'
                reason = f'Predicted upward movement: {prediction:.2f}%'
            elif prediction < -0.1:  # Expect downward movement > 0.1%
                signal = 'SELL'
                reason = f'Predicted downward movement: {prediction:.2f}%'
            else:
                signal = 'HOLD'
                reason = f'Prediction too small: {prediction:.2f}%'
            
            return {
                'signal': signal,
                'confidence': confidence,
                'prediction': prediction,
                'reason': reason
            }
            
        except Exception as e:
            logger.error(f"Error generating trading signal: {e}")
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'prediction': 0.0,
                'reason': f'Error: {str(e)}'
            }


def train_model():
    """Standalone function to train the LSTM model"""
    logger.info("Starting model training process")
    
    # Initialize components
    engine = LSTMEngine()
    data_handler = DataHandler()
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    data = data_handler.load_data()
    processed_data = data_handler.preprocess_data()
    
    # Create sequences
    X, y = data_handler.create_sequences(processed_data)
    
    # Split data
    X_train, X_test, y_train, y_test = data_handler.split_data(X, y)
    
    # Train model
    history = engine.train(X_train, y_train, X_test, y_test)
    
    # Evaluate model
    metrics = engine.evaluate(X_test, y_test)
    
    logger.info("Training completed successfully!")
    logger.info(f"Final metrics: {metrics}")
    
    return engine, history, metrics


if __name__ == "__main__":
    # Train the model when run directly
    train_model()