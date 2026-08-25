from dotenv import load_dotenv
load_dotenv()

import os
import time
import logging
import datetime
import json
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, TakeProfitRequest, StopLossRequest, GetCalendarRequest
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 1. Configuration & Safety Parameters
# ------------------------------------------------------------------
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
if not API_KEY or not SECRET_KEY:
    raise RuntimeError(
        "ALPACA_API_KEY / ALPACA_SECRET_KEY not found in environment. "
        "Set them via a .env file (UTF-8 encoded, no quotes) in the same "
        "directory you run this script from -- never hardcode keys in "
        "the script."
    )

PAPER_TRADING = True

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "ORCL", "CSCO", "IBM", "ADBE",
    "JPM", "BAC", "GS", "MS", "WFC", "C", "AXP", "BLK",
    "XOM", "CVX", "COP", "SLB",
    "JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT",
    "WMT", "PG", "KO", "PEP", "COST", "MCD", "NKE", "HD", "DIS",
    "CAT", "HON", "BA", "GE", "MMM", "UPS",
    "NEE", "DUK", "SO", "T", "VZ",
]
BENCHMARK = "SPY"
TOP_K = 5
HORIZON = 60

TAKE_PROFIT_PCT = 0.15
STOP_LOSS_PCT = 0.08
MAX_POS_ALLOCATION_PCT = 0.20
MAX_PORTFOLIO_DRAWDOWN = 0.15
MIN_PROB_SPREAD = 0.02

STATE_FILE = "rebalance_state.json"
NY_TZ = ZoneInfo("America/New_York")
SLEEP_CHUNK_SECONDS = 6 * 3600

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER_TRADING)


# ------------------------------------------------------------------
# 2. Risk Management & Guardrail Functions
# ------------------------------------------------------------------
def verify_account_health(trading_client: TradingClient) -> bool:
    account = trading_client.get_account()
    current_equity = float(account.equity)
    last_equity = float(account.last_equity)
    drawdown = (last_equity - current_equity) / last_equity if last_equity > 0 else 0.0

    log.info(f"Current Account Equity: ${current_equity:,.2f}")
    log.info(f"Daily Drawdown: {drawdown:.2%}")

    if drawdown >= MAX_PORTFOLIO_DRAWDOWN:
        log.warning(f"ALERT: Portfolio drawdown ({drawdown:.2%}) exceeded "
                     f"safety threshold ({MAX_PORTFOLIO_DRAWDOWN:.2%}).")
        return False
    return True


# ------------------------------------------------------------------
# 3. Model Pipeline & Signal Generation
# ------------------------------------------------------------------
def generate_top_signals() -> list:
    log.info("Fetching market data...")
    all_symbols = TICKERS + [BENCHMARK]
    raw = yf.download(all_symbols, period="3y", progress=False)
    raw_close = raw['Close']
    raw_volume = raw['Volume']
    spy_return = raw_close[BENCHMARK].pct_change()

    base_frames = []
    for ticker in TICKERS:
        if ticker not in raw_close.columns:
            continue
        close = raw_close[ticker]
        volume = raw_volume[ticker]
        if close.dropna().shape[0] < 200:
            continue

        d = pd.DataFrame(index=close.index)
        d['Ticker'] = ticker
        d['Close'] = close
        d['Lag_1'] = close.pct_change(1)
        d['Lag_5'] = close.pct_change(5)
        d['Lag_10'] = close.pct_change(10)

        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        d['Dist_SMA50'] = (close - sma50) / sma50
        d['Dist_SMA200'] = (close - sma200) / sma200

        change = close.diff()
        gain = change.clip(lower=0)
        loss = -change.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        d['RSI'] = 100 - (100 / (1 + rs))

        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        d['MACD'] = exp1 - exp2
        d['MACD_Signal'] = d['MACD'].ewm(span=9, adjust=False).mean()

        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        d['BB_Width'] = (bb_mid + 2 * bb_std - (bb_mid - 2 * bb_std)) / bb_mid

        d['Volume_ZScore'] = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()
        d['SPY_Return'] = spy_return

        base_frames.append(d)

    panel = pd.concat(base_frames).reset_index()

    raw_feature_cols = [
        'Lag_1', 'Lag_5', 'Lag_10', 'Dist_SMA50', 'Dist_SMA200',
        'RSI', 'MACD', 'MACD_Signal', 'BB_Width', 'Volume_ZScore',
    ]
    for col in raw_feature_cols:
        panel[f'{col}_CSRank'] = panel.groupby('Date')[col].rank(pct=True)

    feature_cols = raw_feature_cols + [f'{c}_CSRank' for c in raw_feature_cols] + ['SPY_Return']

    panel['Fwd_Return'] = panel.groupby('Ticker')['Close'].transform(
        lambda s: s.shift(-HORIZON) / s - 1
    )
    panel['CS_Median_Fwd'] = panel.groupby('Date')['Fwd_Return'].transform('median')
    panel['Target'] = (panel['Fwd_Return'] > panel['CS_Median_Fwd']).astype(int)

    train_data = panel.dropna(subset=feature_cols + ['Target']).copy()

    panel_sorted = panel.sort_values('Date')
    latest_data = panel_sorted.groupby('Ticker').tail(1).reset_index(drop=True)
    latest_data = latest_data.dropna(subset=feature_cols).copy()

    log.info("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=30,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    model.fit(train_data[feature_cols], train_data['Target'])

    latest_data['Prob_Up'] = model.predict_proba(latest_data[feature_cols])[:, 1]

    prob_spread = latest_data['Prob_Up'].std()
    log.info(f"Prob_Up today: min={latest_data['Prob_Up'].min():.3f}, "
             f"max={latest_data['Prob_Up'].max():.3f}, std={prob_spread:.4f}")
    if prob_spread < MIN_PROB_SPREAD:
        log.warning("Probabilities are tightly clustered today -- the top-K "
                    "selection may be close to arbitrary tie-breaking rather "
                    "than a meaningful ranking. Proceeding, but treat today's "
                    "picks with extra skepticism.")

    top_picks = latest_data.sort_values(by='Prob_Up', ascending=False).head(TOP_K)
    return top_picks[['Ticker', 'Close']].to_dict(orient='records')


# ------------------------------------------------------------------
# 4. Portfolio Execution & Rebalancing
# ------------------------------------------------------------------
def execute_rebalance():
    if not verify_account_health(trading_client):
        log.warning("Execution halted due to safety guardrails.")
        return

    top_signals = generate_top_signals()
    target_tickers = [item['Ticker'] for item in top_signals]
    log.info(f"Target Tickers ({TOP_K}): {target_tickers}")

    account = trading_client.get_account()
    total_portfolio_value = float(account.portfolio_value)

    trading_client.cancel_orders()

    current_positions = trading_client.get_all_positions()
    current_symbols = {p.symbol: p for p in current_positions}

    for symbol, pos in current_symbols.items():
        if symbol in TICKERS and symbol not in target_tickers:
            log.info(f"Liquidating {symbol} (no longer in top target list)...")
            trading_client.close_position(symbol)

    time.sleep(3)

    raw_allocation = total_portfolio_value / TOP_K
    max_allowed = total_portfolio_value * MAX_POS_ALLOCATION_PCT
    target_allocation = min(raw_allocation, max_allowed)

    for item in top_signals:
        symbol = item['Ticker']
        last_price = item['Close']

        if symbol in current_symbols:
            log.info(f"Position in {symbol} already exists. Skipping entry.")
            continue

        shares = int(target_allocation // last_price)
        if shares <= 0:
            log.info(f"Insufficient allocation to buy 1 share of {symbol}. Skipping.")
            continue

        tp_price = round(last_price * (1 + TAKE_PROFIT_PCT), 2)
        sl_price = round(last_price * (1 - STOP_LOSS_PCT), 2)

        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=shares,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=tp_price),
            stop_loss=StopLossRequest(stop_price=sl_price)
        )

        try:
            log.info(f"Placing Bracket Order: Buy {shares} shares of {symbol} "
                     f"| TP: ${tp_price} | SL: ${sl_price}")
            trading_client.submit_order(order_request)
        except Exception as e:
            log.error(f"Failed to execute order for {symbol}: {e}")

    log.info("Rebalancing complete.")


# ------------------------------------------------------------------
# 5. Calendar-driven continuous scheduling
# ------------------------------------------------------------------
def load_last_rebalance_date():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            data = json.load(f)
            return datetime.date.fromisoformat(data["last_rebalance_date"])
    return None


def save_last_rebalance_date(d: datetime.date):
    with open(STATE_FILE, "w") as f:
        json.dump({"last_rebalance_date": d.isoformat()}, f)


def is_market_open(trading_client: TradingClient) -> bool:
    clock = trading_client.get_clock()
    return clock.is_open


def get_next_rebalance_open(trading_client: TradingClient, last_rebalance_date):
    """Uses Alpaca's own market calendar (via GetCalendarRequest -- the
    modern alpaca-py API takes a filters object, not raw start/end kwargs)
    to find the exact date/time of the next scheduled rebalance."""
    if last_rebalance_date is None:
        start = datetime.date.today()
    else:
        start = last_rebalance_date + datetime.timedelta(days=1)

    end = start + datetime.timedelta(days=200)
    calendar_filter = GetCalendarRequest(start=start, end=end)
    calendar = trading_client.get_calendar(filters=calendar_filter)

    if not calendar:
        raise RuntimeError(f"Alpaca returned no trading days between {start} and {end}.")

    if last_rebalance_date is None:
        target_day = calendar[0]
    else:
        idx = HORIZON - 1
        if idx >= len(calendar):
            raise RuntimeError(
                f"Only {len(calendar)} trading days returned but HORIZON={HORIZON} "
                f"requires at least {HORIZON} -- widen the calendar query window."
            )
        target_day = calendar[idx]

    open_val = target_day.open
    if isinstance(open_val, datetime.datetime):
        # SDK returned a full datetime -- attach/convert timezone as needed.
        target_open_dt = (
            open_val.replace(tzinfo=NY_TZ) if open_val.tzinfo is None
            else open_val.astimezone(NY_TZ)
        )
    else:
        # SDK returned a plain time -- combine with the date ourselves.
        target_open_dt = datetime.datetime.combine(target_day.date, open_val, tzinfo=NY_TZ)

    return target_open_dt


def sleep_until(target_dt: datetime.datetime):
    while True:
        now = datetime.datetime.now(NY_TZ)
        remaining = (target_dt - now).total_seconds()
        if remaining <= 0:
            return
        chunk = min(remaining, SLEEP_CHUNK_SECONDS)
        time.sleep(chunk)
        remaining_after = (target_dt - datetime.datetime.now(NY_TZ)).total_seconds()
        if remaining_after > 0:
            log.info(f"Waiting for next rebalance at {target_dt} "
                     f"(~{remaining_after / 3600:.1f} hours remaining).")


def run_continuously():
    log.info("Starting calendar-driven continuous rebalance scheduler. Press Ctrl+C to stop.")
    last_rebalance = load_last_rebalance_date()
    if last_rebalance:
        log.info(f"Loaded last rebalance date from disk: {last_rebalance}")
    else:
        log.info("No prior rebalance recorded -- scheduling for the next open trading day.")

    while True:
        try:
            target_open_dt = get_next_rebalance_open(trading_client, last_rebalance)
            log.info(f"Next rebalance scheduled for {target_open_dt} (market open, America/New_York).")

            sleep_until(target_open_dt)

            time.sleep(60)
            if not is_market_open(trading_client):
                log.warning("Target time reached but market clock reports closed -- "
                            "recomputing next rebalance date without advancing state.")
                continue

            log.info("Target rebalance time reached -- executing rebalance.")
            execute_rebalance()
            last_rebalance = datetime.date.today()
            save_last_rebalance_date(last_rebalance)

        except Exception:
            log.exception("Error in scheduling loop -- retrying in 1 hour.")
            time.sleep(3600)


# ------------------------------------------------------------------
# 6. Script Entry Point
# ------------------------------------------------------------------
if __name__ == "__main__":
    run_continuously()