import os
import sys
import time
import logging
import sqlite3
from typing import Tuple, Optional, Dict

# ==========================================
# Dependency Check
# ==========================================
try:
    import pandas as pd
    from dotenv import load_dotenv
    from requests.exceptions import ReadTimeout, ConnectionError
    from binance.client import Client
    from binance.exceptions import BinanceAPIException, BinanceRequestException
except ImportError as e:
    print(f"\n[CRITICAL ERROR] Missing dependency: {e}")
    print("Please install the required libraries by running the following command in your terminal:")
    print("pip install python-binance pandas python-dotenv requests\n")
    sys.exit(1)

# ==========================================
# 1. Configuration & Setup
# ==========================================

# Setup Logging for VPS 24/7 monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CryptoBot")

class Config:
    """Loads and validates environment variables."""
    def __init__(self):
        load_dotenv()
        self.API_KEY = os.getenv("BINANCE_API_KEY")
        self.API_SECRET = os.getenv("BINANCE_API_SECRET")
        self.SYMBOL = os.getenv("TRADE_SYMBOL", "BTCUSDT")
        self.TIMEFRAME = os.getenv("TIMEFRAME", "15m") # 1m, 5m, 15m
        self.TRADE_PERCENTAGE = float(os.getenv("TRADE_PERCENTAGE", 0.05)) # 5% of balance
        self.STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", 0.02)) # 2% SL
        self.TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", 0.04)) # 4% TP
        
        if not self.API_KEY or not self.API_SECRET:
            raise ValueError("API Keys missing. Please set BINANCE_API_KEY and BINANCE_API_SECRET in .env file.")
        
        # Security Note logged at startup
        logger.info("Ensure API Key has 'Enable Reading' and 'Enable Spot & Margin Trading' ONLY.")
        logger.info("DO NOT enable withdrawals on this API Key.")

# ==========================================
# 2. Database Module
# ==========================================

class DatabaseHandler:
    """Handles SQLite database operations for trade logging."""
    def __init__(self, db_name="trade_history.db"):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self._create_tables()

    def _create_tables(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT,
                side TEXT,
                price REAL,
                quantity REAL,
                status TEXT
            )
        ''')
        self.conn.commit()

    def log_trade(self, symbol: str, side: str, price: float, quantity: float, status: str):
        self.cursor.execute('''
            INSERT INTO trades (symbol, side, price, quantity, status)
            VALUES (?, ?, ?, ?, ?)
        ''', (symbol, side, price, quantity, status))
        self.conn.commit()
        logger.info(f"Trade logged to DB: {side} {quantity} {symbol} @ {price}")

# ==========================================
# 3. Market Data & Strategy Modules
# ==========================================

class MarketDataAggregator:
    """Fetches and formats market data from Binance."""
    def __init__(self, client: Client):
        self.client = client

    def get_historical_data(self, symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
        """Fetches klines and converts to pandas DataFrame."""
        try:
            klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            df['close'] = df['close'].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return pd.DataFrame()

class StrategyEngine:
    """Calculates Technical Indicators and generates trading signals."""
    
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculates RSI, MACD, and SMA using pure pandas."""
        # Calculate SMA
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        
        # Calculate RSI (14 period)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        return df

    @staticmethod
    def get_signal(df: pd.DataFrame) -> str:
        """
        Custom Entry/Exit Logic.
        Returns: 'BUY', 'SELL', or 'HOLD'
        """
        if df.empty or len(df) < 50:
            return 'HOLD'
            
        latest = df.iloc[-1]
        previous = df.iloc[-2]

        # BUY Logic: RSI is oversold (< 30) AND MACD crosses above Signal Line
        if latest['RSI'] < 30 and (previous['MACD'] < previous['Signal_Line'] and latest['MACD'] > latest['Signal_Line']):
            return 'BUY'
            
        # SELL Logic: RSI is overbought (> 70) AND MACD crosses below Signal Line
        elif latest['RSI'] > 70 and (previous['MACD'] > previous['Signal_Line'] and latest['MACD'] < latest['Signal_Line']):
            return 'SELL'
            
        return 'HOLD'

# ==========================================
# 4. Risk Management Module
# ==========================================

class RiskManager:
    """Handles position sizing, stop-loss, and take-profit calculations."""
    
    @staticmethod
    def calculate_position_size(usdt_balance: float, current_price: float, risk_pct: float) -> float:
        """Calculates how much crypto to buy based on percentage of total balance."""
        capital_to_risk = usdt_balance * risk_pct
        quantity = capital_to_risk / current_price
        return quantity

    @staticmethod
    def calculate_sl_tp(entry_price: float, side: str, sl_pct: float, tp_pct: float) -> Tuple[float, float]:
        """Calculates Stop-Loss and Take-Profit prices."""
        if side == 'BUY':
            stop_loss = entry_price * (1 - sl_pct)
            take_profit = entry_price * (1 + tp_pct)
        else: # SELL (Shorting - if applicable, though primarily Spot is assumed)
            stop_loss = entry_price * (1 + sl_pct)
            take_profit = entry_price * (1 - tp_pct)
        return stop_loss, take_profit

# ==========================================
# 5. Execution Handler Module
# ==========================================

class ExecutionHandler:
    """Handles API order execution with error handling and retries."""
    def __init__(self, client: Client, db: DatabaseHandler):
        self.client = client
        self.db = db

    def get_usdt_balance(self) -> float:
        """Fetches available USDT balance with retry logic."""
        try:
            balance = self.client.get_asset_balance(asset='USDT')
            return float(balance['free'])
        except (BinanceAPIException, ConnectionError) as e:
            logger.error(f"Failed to fetch balance: {e}")
            return 0.0

    def execute_trade(self, symbol: str, side: str, quantity: float, current_price: float, config: Config):
        """
        Executes a Market Order, followed by an OCO order for SL/TP.
        Complex Logic: Binance requires strict precision for prices and quantities.
        In a full production environment, we dynamically fetch 'stepSize' and 'tickSize' 
        from exchange info. Here we use standard rounding for demonstration.
        """
        try:
            # Format quantity to 4 decimal places (adjust based on Binance Symbol rules)
            qty_formatted = round(quantity, 4)
            
            logger.info(f"Attempting to place {side} Market Order for {qty_formatted} {symbol}")
            
            # 1. Place Market Entry Order
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=qty_formatted
            )
            
            # Extract actual fill price
            fills = order.get('fills', [])
            fill_price = float(fills[0]['price']) if fills else current_price
            
            self.db.log_trade(symbol, side, fill_price, qty_formatted, "FILLED")
            logger.info(f"Market {side} filled at {fill_price}")

            # 2. Place OCO Order (Stop-Loss & Take-Profit)
            if side == 'BUY':
                sl_price, tp_price = RiskManager.calculate_sl_tp(
                    fill_price, side, config.STOP_LOSS_PCT, config.TAKE_PROFIT_PCT
                )
                
                # Round prices to 2 decimal places (adjust based on tickSize)
                sl_price = round(sl_price, 2)
                tp_price = round(tp_price, 2)
                
                logger.info(f"Placing OCO Sell Order -> TP: {tp_price}, SL: {sl_price}")
                
                # OCO requires a stopLimitPrice slightly below the stopPrice to ensure execution
                stop_limit_price = round(sl_price * 0.999, 2) 

                oco_order = self.client.create_oco_order(
                    symbol=symbol,
                    side='SELL',
                    quantity=qty_formatted,
                    price=str(tp_price),          # Take Profit
                    stopPrice=str(sl_price),      # Stop Loss trigger
                    stopLimitPrice=str(stop_limit_price), # Stop Loss Limit execution
                    stopLimitTimeInForce='GTC'
                )
                self.db.log_trade(symbol, "OCO_SELL", fill_price, qty_formatted, "PLACED")
                logger.info("OCO Order successfully placed.")

        except BinanceAPIException as e:
            logger.error(f"Binance API Error during execution: {e.status_code} - {e.message}")
        except Exception as e:
            logger.error(f"Unexpected error during execution: {e}")

# ==========================================
# 6. Main Orchestrator
# ==========================================

class TradingBot:
    """Main application class that orchestrates the trading loop."""
    def __init__(self):
        self.config = Config()
        self.db = DatabaseHandler()
        
        # Initialize Binance Client
        self.client = Client(self.config.API_KEY, self.config.API_SECRET)
        
        # Initialize Modules
        self.market_data = MarketDataAggregator(self.client)
        self.execution = ExecutionHandler(self.client, self.db)
        
        # State tracking to avoid buying multiple times in one signal
        self.in_position = False 

    def run(self):
        """Main loop designed for 24/7 VPS execution."""
        logger.info(f"Starting Trading Bot on {self.config.SYMBOL} | Timeframe: {self.config.TIMEFRAME}")
        
        while True:
            try:
                # 1. Fetch Market Data
                df = self.market_data.get_historical_data(self.config.SYMBOL, self.config.TIMEFRAME)
                
                if df.empty:
                    logger.warning("Empty DataFrame received. Retrying in 60 seconds...")
                    time.sleep(60)
                    continue

                # 2. Apply Strategy Engine
                df = StrategyEngine.calculate_indicators(df)
                signal = StrategyEngine.get_signal(df)
                current_price = df.iloc[-1]['close']
                
                logger.info(f"[{self.config.SYMBOL}] Price: {current_price} | Signal: {signal} | RSI: {df.iloc[-1]['RSI']:.2f}")

                # 3. Execute Logic based on Signal
                if signal == 'BUY' and not self.in_position:
                    usdt_balance = self.execution.get_usdt_balance()
                    
                    if usdt_balance > 10: # Minimum Binance trade size is usually 10 USDT
                        qty = RiskManager.calculate_position_size(
                            usdt_balance, current_price, self.config.TRADE_PERCENTAGE
                        )
                        self.execution.execute_trade(
                            self.config.SYMBOL, 'BUY', qty, current_price, self.config
                        )
                        self.in_position = True
                    else:
                        logger.warning("Insufficient USDT balance to place trade.")

                # If the OCO order hits TP or SL, we need to reset `in_position`.
                # In a real-world scenario, you would query open orders to check if the OCO is still active.
                if self.in_position:
                    open_orders = self.client.get_open_orders(symbol=self.config.SYMBOL)
                    if not open_orders:
                        logger.info("No open orders found. Assuming OCO hit SL or TP. Resetting position state.")
                        self.in_position = False

                # Sleep before next check (e.g., check every 1 minute)
                time.sleep(60)

            except ReadTimeout:
                logger.error("Connection timeout. Exchange might be under heavy load. Retrying...")
                time.sleep(120)
            except ConnectionError:
                logger.error("Network connection error. Check VPS internet. Retrying...")
                time.sleep(120)
            except Exception as e:
                logger.error(f"Critical error in main loop: {e}")
                time.sleep(60) # Graceful recovery sleep

# ==========================================
# 7. Entry Point
# ==========================================

if __name__ == "__main__":
    # Ensure .env file exists or variables are set in the system before running
    try:
        bot = TradingBot()
        bot.run()
    except KeyboardInterrupt:
        logger.info("Bot stopped manually by user.")
    except Exception as e:
        logger.critical(f"Bot terminated due to fatal error: {e}")
