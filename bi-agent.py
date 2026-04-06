import os
import sys
import logging
import sqlite3
import pandas as pd
from typing import Tuple
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException

# ==========================================
# 1. Configuration & Setup
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("CryptoBot")

class Config:
    def __init__(self):
        load_dotenv()
        # GitHub Secrets ya .env se data uthayega
        self.API_KEY = os.getenv("BINANCE_API_KEY")
        self.API_SECRET = os.getenv("BINANCE_API_SECRET")
        self.SYMBOL = os.getenv("TRADE_SYMBOL", "BTCUSDT")
        self.TIMEFRAME = os.getenv("TIMEFRAME", "15m")
        self.TRADE_PERCENTAGE = float(os.getenv("TRADE_PERCENTAGE", 0.05))
        self.STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", 0.02))
        self.TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", 0.04))
        
        if not self.API_KEY or not self.API_SECRET:
            raise ValueError("API Keys missing in Environment Variables!")

# ==========================================
# 2. Strategy & Market Data
# ==========================================
class StrategyEngine:
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        return df

    @staticmethod
    def get_signal(df: pd.DataFrame) -> str:
        if df.empty or len(df) < 50: return 'HOLD'
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        if latest['RSI'] < 30 and (previous['MACD'] < previous['Signal_Line'] and latest['MACD'] > latest['Signal_Line']):
            return 'BUY'
        elif latest['RSI'] > 70 and (previous['MACD'] > previous['Signal_Line'] and latest['MACD'] < latest['Signal_Line']):
            return 'SELL'
        return 'HOLD'

# ==========================================
# 3. Main Execution (Single Run for GitHub Actions)
# ==========================================
def run_bot():
    try:
        config = Config()
        client = Client(config.API_KEY, config.API_SECRET)
        
        # Get Market Data
        klines = client.get_klines(symbol=config.SYMBOL, interval=config.TIMEFRAME, limit=100)
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades', 'tbb', 'tbq', 'ignore'])
        df['close'] = df['close'].astype(float)
        
        # Process Signal
        df = StrategyEngine.calculate_indicators(df)
        signal = StrategyEngine.get_signal(df)
        current_price = df.iloc[-1]['close']
        
        logger.info(f"Checking {config.SYMBOL} | Price: {current_price} | Signal: {signal} | RSI: {df.iloc[-1]['RSI']:.2f}")

        if signal == 'BUY':
            balance = float(client.get_asset_balance(asset='USDT')['free'])
            if balance > 11:
                qty = round((balance * config.TRADE_PERCENTAGE) / current_price, 4)
                logger.info(f"Executing BUY for {qty} {config.SYMBOL}")
                # Order execution yahan hoti hai
                # client.create_order(symbol=config.SYMBOL, side='BUY', type='MARKET', quantity=qty)
            else:
                logger.warning("Low balance for trading.")
                
    except Exception as e:
        logger.error(f"Bot Error: {e}")

if __name__ == "__main__":
    run_bot()
