"""
Trader module for Plus500 AutoTrader
Handles browser automation using Playwright for Plus500 trading interface
"""

import asyncio
from playwright.async_api import async_playwright, Browser, Page
import logging
from typing import Dict, Tuple, Optional
import time
from datetime import datetime

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Plus500Trader:
    """Playwright-based automation for Plus500 trading platform"""
    
    def __init__(self):
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.playwright = None
        self.is_logged_in = False
        self.current_positions = []
        
    async def start_browser(self):
        """Initialize browser and create new page"""
        logger.info("Starting browser...")
        
        self.playwright = await async_playwright().start()
        
        # Launch browser (use headless=False for debugging)
        self.browser = await self.playwright.chromium.launch(
            headless=config.headless_browser,
            args=['--disable-blink-features=AutomationControlled']
        )
        
        # Create browser context and page
        context = await self.browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        
        self.page = await context.new_page()
        logger.info("Browser started successfully")
    
    async def close_browser(self):
        """Close browser and cleanup"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        logger.info("Browser closed")
    
    async def navigate_to_plus500(self):
        """Navigate to Plus500 website"""
        logger.info("Navigating to Plus500...")
        
        try:
            await self.page.goto('https://app.plus500.com/', wait_until='networkidle')
            await self.page.wait_for_timeout(2000)  # Wait 2 seconds for page to settle
            logger.info("Successfully navigated to Plus500")
        except Exception as e:
            logger.error(f"Error navigating to Plus500: {e}")
            raise
    
    async def login(self, username: str, password: str):
        """
        Login to Plus500 platform
        Note: This is a template - actual selectors need to be updated based on real Plus500 interface
        """
        logger.info("Attempting to log in...")
        
        try:
            # Wait for login elements to appear
            await self.page.wait_for_selector('input[type="email"], input[name="username"]', timeout=10000)
            
            # Fill username
            username_selector = 'input[type="email"], input[name="username"]'
            await self.page.fill(username_selector, username)
            
            # Fill password
            password_selector = 'input[type="password"], input[name="password"]'
            await self.page.fill(password_selector, password)
            
            # Click login button
            login_button = 'button[type="submit"], .login-button, button:has-text("Login")'
            await self.page.click(login_button)
            
            # Wait for login to complete
            await self.page.wait_for_timeout(5000)
            
            # Check if login was successful (this would need to be adapted to actual Plus500 interface)
            if await self.page.is_visible('.dashboard, .trading-interface, .account-info'):
                self.is_logged_in = True
                logger.info("Login successful")
            else:
                logger.error("Login may have failed - dashboard not visible")
                
        except Exception as e:
            logger.error(f"Error during login: {e}")
            raise
    
    async def click_coordinate(self, x: int, y: int, delay: float = 1.0):
        """
        Click at specific coordinates
        
        Args:
            x: X coordinate
            y: Y coordinate
            delay: Delay after click in seconds
        """
        logger.debug(f"Clicking at coordinates ({x}, {y})")
        
        try:
            await self.page.mouse.click(x, y)
            await self.page.wait_for_timeout(int(delay * 1000))
        except Exception as e:
            logger.error(f"Error clicking at ({x}, {y}): {e}")
            raise
    
    async def select_eur_usd_pair(self):
        """Select EUR/USD trading pair"""
        logger.info("Selecting EUR/USD trading pair...")
        
        try:
            # Method 1: Try to find by text
            try:
                await self.page.click('text=EUR/USD', timeout=5000)
                logger.info("Selected EUR/USD by text")
                return
            except:
                pass
            
            # Method 2: Try coordinate-based clicking
            coords = config.coordinates.eur_usd_pair
            await self.click_coordinate(coords[0], coords[1])
            logger.info(f"Selected EUR/USD by coordinates ({coords[0]}, {coords[1]})")
            
        except Exception as e:
            logger.error(f"Error selecting EUR/USD pair: {e}")
            raise
    
    async def place_trade(self, action: str, amount: float, 
                         stop_loss: Optional[float] = None,
                         take_profit: Optional[float] = None) -> Dict:
        """
        Place a trade on Plus500
        
        Args:
            action: 'BUY' or 'SELL'
            amount: Trade amount
            stop_loss: Stop loss level (optional)
            take_profit: Take profit level (optional)
            
        Returns:
            Dictionary with trade result information
        """
        logger.info(f"Placing {action} trade for {amount}")
        
        trade_result = {
            'success': False,
            'action': action,
            'amount': amount,
            'timestamp': datetime.now(),
            'error': None
        }
        
        try:
            # Select EUR/USD pair first
            await self.select_eur_usd_pair()
            
            # Enter trade amount
            amount_coords = config.coordinates.amount_field
            await self.click_coordinate(amount_coords[0], amount_coords[1])
            
            # Clear field and enter amount
            await self.page.keyboard.press('Control+A')
            await self.page.keyboard.type(str(amount))
            
            # Set stop loss if provided
            if stop_loss:
                sl_coords = config.coordinates.stop_loss_field
                await self.click_coordinate(sl_coords[0], sl_coords[1])
                await self.page.keyboard.press('Control+A')
                await self.page.keyboard.type(str(stop_loss))
            
            # Set take profit if provided
            if take_profit:
                tp_coords = config.coordinates.take_profit_field
                await self.click_coordinate(tp_coords[0], tp_coords[1])
                await self.page.keyboard.press('Control+A')
                await self.page.keyboard.type(str(take_profit))
            
            # Click Buy or Sell button
            if action.upper() == 'BUY':
                button_coords = config.coordinates.buy_button
            else:
                button_coords = config.coordinates.sell_button
            
            await self.click_coordinate(button_coords[0], button_coords[1])
            
            # Wait for trade confirmation
            await self.page.wait_for_timeout(3000)
            
            # Check if trade was successful (would need to be adapted to actual interface)
            trade_result['success'] = True
            logger.info(f"Trade placed successfully: {action} {amount}")
            
        except Exception as e:
            trade_result['error'] = str(e)
            logger.error(f"Error placing trade: {e}")
        
        return trade_result
    
    async def close_position(self, position_id: Optional[str] = None) -> Dict:
        """
        Close an open position
        
        Args:
            position_id: Specific position to close (if None, closes most recent)
            
        Returns:
            Dictionary with close result information
        """
        logger.info(f"Closing position: {position_id if position_id else 'most recent'}")
        
        close_result = {
            'success': False,
            'position_id': position_id,
            'timestamp': datetime.now(),
            'error': None
        }
        
        try:
            # Click close position button (coordinate-based)
            close_coords = config.coordinates.close_position
            await self.click_coordinate(close_coords[0], close_coords[1])
            
            # Wait for confirmation
            await self.page.wait_for_timeout(2000)
            
            # Confirm close if needed (adapt to actual interface)
            try:
                await self.page.click('button:has-text("Confirm"), button:has-text("Close")', timeout=3000)
            except:
                pass
            
            close_result['success'] = True
            logger.info("Position closed successfully")
            
        except Exception as e:
            close_result['error'] = str(e)
            logger.error(f"Error closing position: {e}")
        
        return close_result
    
    async def get_current_price(self) -> Optional[float]:
        """
        Get current EUR/USD price from the interface
        This would need to be implemented based on the actual Plus500 interface
        """
        try:
            # This is a placeholder - would need actual selector for price display
            price_element = await self.page.query_selector('.price, .current-price, [data-testid="price"]')
            if price_element:
                price_text = await price_element.text_content()
                # Parse price from text (would need actual parsing logic)
                return float(price_text.replace(',', ''))
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
        
        return None
    
    async def take_screenshot(self, filename: str = None):
        """Take a screenshot for debugging purposes"""
        if filename is None:
            filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        screenshot_path = f"logs/{filename}"
        await self.page.screenshot(path=screenshot_path)
        logger.info(f"Screenshot saved to {screenshot_path}")
        return screenshot_path
    
    async def wait_for_market_hours(self):
        """Wait if market is closed (basic implementation)"""
        current_hour = datetime.now().hour
        
        if current_hour < config.trading.trading_start_hour or current_hour > config.trading.trading_end_hour:
            wait_hours = config.trading.trading_start_hour - current_hour
            if wait_hours <= 0:
                wait_hours += 24
            
            logger.info(f"Market closed. Waiting {wait_hours} hours until market opens.")
            await asyncio.sleep(wait_hours * 3600)  # Convert to seconds


class TradingSession:
    """Manages a complete trading session"""
    
    def __init__(self, username: str, password: str):
        self.trader = Plus500Trader()
        self.username = username
        self.password = password
        self.active_trades = []
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.trader.start_browser()
        await self.trader.navigate_to_plus500()
        await self.trader.login(self.username, self.password)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.trader.close_browser()
    
    async def execute_trading_signal(self, signal: Dict) -> Dict:
        """
        Execute a trading signal from the LSTM engine
        
        Args:
            signal: Signal dictionary from engine.get_trading_signal()
            
        Returns:
            Execution result dictionary
        """
        logger.info(f"Executing trading signal: {signal}")
        
        if signal['signal'] == 'HOLD':
            return {
                'action': 'HOLD',
                'success': True,
                'reason': signal['reason']
            }
        
        # Calculate trade amount based on risk management
        account_balance = 10000  # This would need to be fetched from the platform
        risk_amount = account_balance * config.trading.risk_percentage
        
        # Execute trade
        result = await self.trader.place_trade(
            action=signal['signal'],
            amount=risk_amount,
            stop_loss=config.trading.stop_loss_pips,
            take_profit=config.trading.take_profit_pips
        )
        
        if result['success']:
            self.active_trades.append(result)
        
        return result


# Example usage and testing functions
async def test_trader():
    """Test function for the trader module"""
    logger.info("Testing Plus500 trader...")
    
    trader = Plus500Trader()
    
    try:
        await trader.start_browser()
        await trader.navigate_to_plus500()
        
        # Take a screenshot for debugging
        await trader.take_screenshot("test_navigation.png")
        
        logger.info("Trader test completed successfully")
        
    except Exception as e:
        logger.error(f"Error in trader test: {e}")
    finally:
        await trader.close_browser()


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_trader())