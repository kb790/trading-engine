import pandas as pd
from datetime import datetime, timedelta

# Alpaca SDK Imports
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

API_KEY = "PK6WKT3DXMOF7HZFTBHK3A5XAM"
SECRET_KEY = "AEd4t5GycPo3GKV1tWRhEwv4eykKh13XEuGVRz6tzgNS"
PAPER = True

# Global Clients exported for main_bot.py
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)


def get_latest_price(symbol: str) -> float:
    """Fetches the most recent close price for a symbol."""
    start_time = datetime.now() - timedelta(days=5)
    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start_time,
        feed=DataFeed.IEX
    )
    bars = data_client.get_stock_bars(request_params)
    df = bars.df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")
    return float(df.iloc[-1]['close'])


def get_existing_position(symbol: str) -> float:
    """Checks open position quantity for a symbol."""
    try:
        position = trading_client.get_open_position(symbol)
        return float(position.qty)
    except Exception:
        return 0.0


def display_portfolio():
    """Prints current account equity and active positions."""
    try:
        account = trading_client.get_account()
        positions = trading_client.get_all_positions()
        
        print("\n=== PORTFOLIO SUMMARY ===")
        print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"Cash Balance:    ${float(account.cash):,.2f}")
        print("Active Positions:")
        if not positions:
            print("  None")
        else:
            for p in positions:
                print(f"  - {p.symbol}: {p.qty} shares @ avg price ${float(p.avg_entry_price):.2f}")
        print("=========================\n")
    except Exception as e:
        print(f"[ERROR] Could not fetch portfolio summary: {e}")


def execute_bracket_buy(symbol: str, current_price: float, cash_allocation_pct: float = 0.20, profit_pct: float = 0.05, stop_pct: float = 0.02):
    """Submits a Bracket BUY order (+5% Profit Target / -2% Stop Loss)."""
    account = trading_client.get_account()
    available_cash = float(account.cash)
    
    trade_capital = available_cash * cash_allocation_pct
    qty = int(trade_capital // current_price)

    if qty <= 0:
        print(f"[GUARDRAIL] Insufficient cash to buy 1 share of {symbol}.")
        return None

    take_profit_price = round(current_price * (1 + profit_pct), 2)
    stop_loss_price = round(current_price * (1 - stop_pct), 2)

    order_data = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.GTC,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=take_profit_price),
        stop_loss=StopLossRequest(stop_price=stop_loss_price)
    )

    try:
        order = trading_client.submit_order(order_data)
        print(f"[SUCCESS] Bracket BUY order submitted for {qty} shares of {symbol}")
        print(f" -> Take Profit Target: ${take_profit_price}")
        print(f" -> Stop Loss Guardrail: ${stop_loss_price}")
        return order
    except Exception as e:
        print(f"[ERROR] Failed to execute Bracket Buy for {symbol}: {e}")
        return None


def execute_exit_sell(symbol: str, qty: float):
    """Submits a Market SELL order to close an open position."""
    order_data = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.GTC
    )
    try:
        order = trading_client.submit_order(order_data)
        print(f"[SUCCESS] Market SELL executed for {qty} shares of {symbol}.")
        return order
    except Exception as e:
        print(f"[ERROR] Sell order failed for {symbol}: {e}")
        return None


def process_trade_signal(symbol: str, signal: str, current_price: float):
    """
    Main signal router invoked by main_bot.py.
    Checks open positions and dispatches to Bracket Buy or Sell handlers.
    """
    open_qty = get_existing_position(symbol)

    if signal == "BUY" and open_qty == 0:
        print(f"[SIGNAL] Golden Cross detected for {symbol}. Executing Bracket Buy...")
        return execute_bracket_buy(symbol, current_price)

    elif signal == "SELL" and open_qty > 0:
        print(f"[SIGNAL] Death Cross detected for {symbol}. Closing position...")
        return execute_exit_sell(symbol, open_qty)

    elif signal == "BUY" and open_qty > 0:
        print(f"[GUARDRAIL] Buy signal triggered, but position already exists for {symbol}.")

    else:
        print(f"[INFO] No action needed for {symbol}. (Signal: {signal} | Open Shares: {open_qty})")