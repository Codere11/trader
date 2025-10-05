#!/usr/bin/env python3
"""
Main execution script for Plus500 AutoTrader
Orchestrates the entire system: data collection, LSTM prediction, and automated trading
"""

import asyncio
import os
import sys
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import argparse

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from data_handler import DataHandler
from engine import LSTMEngine, train_model
from trader import TradingSession, Plus500Trader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.logs_dir, f'autotrader_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AutoTrader:
    """Main AutoTrader orchestration class"""
    
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        
        # Initialize components
        self.data_handler = DataHandler()
        self.engine = LSTMEngine()
        self.trading_session = None
        
        # State tracking
        self.is_model_loaded = False
        self.recent_data = []
        self.last_signal_time = None
        
    async def initialize(self):
        """Initialize the AutoTrader system"""
        logger.info("Initializing AutoTrader system...")
        
        # Load or train the LSTM model
        await self.load_or_train_model()
        
        # Load recent historical data for context
        self.load_recent_data()
        
        logger.info("AutoTrader initialization completed")
    
    async def load_or_train_model(self):
        """Load existing model or train a new one"""
        model_path = config.model.model_save_path
        
        if os.path.exists(model_path):
            try:
                logger.info(f"Loading existing model from {model_path}")
                self.engine.load_model(model_path)
                self.is_model_loaded = True
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                logger.info("Training new model...")
                await self.train_new_model()
        else:
            logger.info("No existing model found. Training new model...")
            await self.train_new_model()
    
    async def train_new_model(self):
        """Train a new LSTM model"""
        try:
            logger.info("Starting model training...")
            
            # Use the standalone train_model function
            engine, history, metrics = train_model()
            self.engine = engine
            self.is_model_loaded = True
            
            logger.info(f"Model training completed. Final metrics: {metrics}")
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise
    
    def load_recent_data(self):
        """Load recent historical data for making predictions"""
        try:
            data = self.data_handler.load_data()
            
            # Get the last N data points for prediction context
            sequence_length = config.model.sequence_length
            self.recent_data = data['change_percent'].tail(sequence_length * 2).tolist()
            
            logger.info(f"Loaded {len(self.recent_data)} recent data points")
            
        except Exception as e:
            logger.error(f"Failed to load recent data: {e}")
            self.recent_data = []
    
    def update_recent_data(self, new_change: float):
        """Update recent data with new price change"""
        self.recent_data.append(new_change)
        
        # Keep only the last N*2 points (for buffer)
        max_length = config.model.sequence_length * 2
        if len(self.recent_data) > max_length:
            self.recent_data = self.recent_data[-max_length:]
    
    async def get_trading_signal(self) -> Dict:
        """Get trading signal from LSTM engine"""
        if not self.is_model_loaded:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'prediction': 0.0,
                'reason': 'Model not loaded'
            }
        
        if len(self.recent_data) < config.model.sequence_length:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'prediction': 0.0,
                'reason': 'Insufficient historical data'
            }
        
        try:
            signal = self.engine.get_trading_signal(self.recent_data)
            self.last_signal_time = datetime.now()
            return signal
        
        except Exception as e:
            logger.error(f"Error generating trading signal: {e}")
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'prediction': 0.0,
                'reason': f'Signal generation error: {str(e)}'
            }
    
    async def execute_trading_cycle(self) -> Dict:
        """Execute one complete trading cycle"""
        logger.info("Executing trading cycle...")
        
        cycle_result = {
            'timestamp': datetime.now(),
            'signal': None,
            'execution': None,
            'error': None
        }
        
        try:
            # Get trading signal
            signal = await self.get_trading_signal()
            cycle_result['signal'] = signal
            
            logger.info(f"Generated signal: {signal['signal']} "
                       f"(confidence: {signal['confidence']:.2f}, "
                       f"prediction: {signal['prediction']:.2f}%)")
            
            # Execute signal if we have an active trading session
            if self.trading_session and signal['signal'] != 'HOLD':
                execution_result = await self.trading_session.execute_trading_signal(signal)
                cycle_result['execution'] = execution_result
                
                if execution_result['success']:
                    logger.info(f"Successfully executed {execution_result['action']} trade")
                else:
                    logger.error(f"Failed to execute trade: {execution_result.get('error')}")
            else:
                cycle_result['execution'] = {'action': 'HOLD', 'reason': signal['reason']}
        
        except Exception as e:
            error_msg = f"Error in trading cycle: {e}"
            logger.error(error_msg)
            cycle_result['error'] = error_msg
        
        return cycle_result
    
    async def run_continuous_trading(self, duration_hours: int = 8):
        """Run continuous automated trading"""
        logger.info(f"Starting continuous trading for {duration_hours} hours...")
        
        end_time = datetime.now() + timedelta(hours=duration_hours)
        cycle_count = 0
        
        try:
            # Start trading session
            async with TradingSession(self.username, self.password) as session:
                self.trading_session = session
                
                logger.info("Trading session established")
                
                while datetime.now() < end_time:
                    cycle_count += 1
                    logger.info(f"Starting trading cycle #{cycle_count}")
                    
                    # Execute trading cycle
                    cycle_result = await self.execute_trading_cycle()
                    
                    # Log cycle summary
                    signal = cycle_result['signal']
                    if signal:
                        logger.info(f"Cycle #{cycle_count} - Signal: {signal['signal']}, "
                                   f"Reason: {signal['reason']}")
                    
                    # Wait before next cycle (15 minutes to match data frequency)
                    await asyncio.sleep(15 * 60)  # 15 minutes
                
                logger.info(f"Continuous trading completed after {cycle_count} cycles")
        
        except KeyboardInterrupt:
            logger.info("Trading interrupted by user")
        except Exception as e:
            logger.error(f"Error in continuous trading: {e}")
        finally:
            self.trading_session = None
    
    async def run_single_prediction(self):
        """Run a single prediction without trading"""
        logger.info("Running single prediction...")
        
        signal = await self.get_trading_signal()
        
        print("\n" + "="*50)
        print("TRADING SIGNAL")
        print("="*50)
        print(f"Signal: {signal['signal']}")
        print(f"Confidence: {signal['confidence']:.2f}")
        print(f"Prediction: {signal['prediction']:.2f}%")
        print(f"Reason: {signal['reason']}")
        print("="*50)
        
        return signal
    
    async def run_backtest(self, start_date: str, end_date: str):
        """Run a backtest on historical data"""
        logger.info(f"Running backtest from {start_date} to {end_date}")
        
        # This would need to be implemented with historical data
        # For now, just a placeholder
        logger.warning("Backtesting not yet implemented")
        
        return {'status': 'not_implemented'}


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Plus500 AutoTrader')
    parser.add_argument('--mode', choices=['train', 'predict', 'trade', 'backtest'], 
                       default='predict', help='Operation mode')
    parser.add_argument('--username', help='Plus500 username')
    parser.add_argument('--password', help='Plus500 password')
    parser.add_argument('--duration', type=int, default=8, 
                       help='Trading duration in hours (for trade mode)')
    parser.add_argument('--start-date', help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date for backtest (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Initialize AutoTrader (credentials only needed for trading mode)
    username = args.username or os.getenv('PLUS500_USERNAME', 'demo_user')
    password = args.password or os.getenv('PLUS500_PASSWORD', 'demo_pass')
    
    autotrader = AutoTrader(username, password)
    
    try:
        # Initialize system
        await autotrader.initialize()
        
        # Execute based on mode
        if args.mode == 'train':
            logger.info("Training mode selected")
            await autotrader.train_new_model()
            print("Model training completed!")
            
        elif args.mode == 'predict':
            logger.info("Prediction mode selected")
            await autotrader.run_single_prediction()
            
        elif args.mode == 'trade':
            logger.info("Trading mode selected")
            if not args.username or not args.password:
                logger.error("Username and password required for trading mode")
                print("Please provide --username and --password for trading mode")
                return
            
            await autotrader.run_continuous_trading(args.duration)
            
        elif args.mode == 'backtest':
            logger.info("Backtest mode selected")
            start_date = args.start_date or '2023-10-01'
            end_date = args.end_date or '2023-11-01'
            await autotrader.run_backtest(start_date, end_date)
    
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())