import os
import pickle
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "ORCL", "CSCO", "IBM", "ADBE",
    "JPM", "BAC", "GS", "MS", "WFC", "C", "AXP", "BLK",
    "XOM", "CVX", "COP", "SLB",
    "JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT",
    "WMT", "PG", "KO", "PEP", "COST", "MCD", "NKE", "HD", "DIS",
    "CAT", "HON", "BA", "GE", "MMM", "UPS",
    "NEE", "DUK", "SO", "T", "VZ",
]
benchmark = "SPY"
start, end = "2005-01-01", "2026-01-01"
HORIZON = 60
TOP_K = 5

CACHE_DIR = ".cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def cache_path(name):
    return os.path.join(CACHE_DIR, f"{name}.pkl")


def load_cached(name):
    path = cache_path(name)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def save_cache(name, obj):
    with open(cache_path(name), "wb") as f:
        pickle.dump(obj, f)


# Download price/volume data
all_symbols = tickers + [benchmark]
cached = load_cached("price_data")
if cached is not None:
    raw_close, raw_volume = cached
else:
    raw = yf.download(all_symbols, start=start, end=end, progress=False)
    raw_close = raw['Close']
    raw_volume = raw['Volume']
    if isinstance(raw_close.columns, pd.MultiIndex):
        raw_close.columns = raw_close.columns.get_level_values(0)
    if isinstance(raw_volume.columns, pd.MultiIndex):
        raw_volume.columns = raw_volume.columns.get_level_values(0)
    save_cache("price_data", (raw_close, raw_volume))

spy_return = raw_close[benchmark].pct_change()


# Fetch fundamentals sequentially
def fetch_fundamentals(ticker):
    try:
        t_obj = yf.Ticker(ticker)
        return t_obj.quarterly_financials, t_obj.upgrades_downgrades
    except Exception:
        return None, None


fundamentals_cache = load_cached("fundamentals")
if fundamentals_cache is None:
    fundamentals_cache = {}
    for t in tickers:
        q_fin, ud = fetch_fundamentals(t)
        fundamentals_cache[t] = (q_fin, ud)
    save_cache("fundamentals", fundamentals_cache)

# Feature engineering
base_frames = []
for ticker in tickers:
    if ticker not in raw_close.columns:
        continue

    close = raw_close[ticker]
    volume = raw_volume[ticker]

    if close.dropna().shape[0] < 500:
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

    q_fin, ud = fundamentals_cache.get(ticker, (None, None))
    d['Rev_Growth_YoY'] = 0.0
    d['SUE'] = 0.0
    d['Analyst_Momentum'] = 0.0

    try:
        if q_fin is not None and not q_fin.empty and 'Total Revenue' in q_fin.index:
            rev = q_fin.loc['Total Revenue'].sort_index()
            rev_yoy = rev.pct_change(4)
            rev_yoy.index = pd.to_datetime(rev_yoy.index) + pd.Timedelta(days=45)
            d['Rev_Growth_YoY'] = d.index.map(lambda x: rev_yoy.asof(x)).fillna(0.0)

        eps_row = 'Diluted EPS' if (q_fin is not None and 'Diluted EPS' in q_fin.index) \
            else ('Basic EPS' if (q_fin is not None and 'Basic EPS' in q_fin.index) else None)
        if q_fin is not None and not q_fin.empty and eps_row is not None:
            eps = q_fin.loc[eps_row].sort_index().astype(float)
            eps_diff = eps.diff(4)
            eps_std = eps_diff.rolling(8, min_periods=4).std()
            eps_std = eps_std.replace(0, np.nan).fillna(eps_diff.abs().mean())
            sue = eps_diff / eps_std
            sue.index = pd.to_datetime(sue.index) + pd.Timedelta(days=45)
            d['SUE'] = d.index.map(lambda x: sue.asof(x)).fillna(0.0)
    except Exception:
        pass

    try:
        if ud is not None and not ud.empty and 'Action' in ud.columns:
            ud = ud.copy()
            ud.index = pd.to_datetime(ud.index)
            ud = ud.sort_index()
            ud['Score'] = ud['Action'].map({
                'up': 1, 'upgrade': 1, 'init': 0, 'main': 0,
                'down': -1, 'downgrade': -1, 'reit': 0
            }).fillna(0)
            daily_score = ud['Score'].resample('D').sum()
            rolling_90d = daily_score.rolling('90D').sum()
            min_idx = rolling_90d.index.min()
            d['Analyst_Momentum'] = d.index.map(
                lambda x: rolling_90d.asof(x) if x >= min_idx else 0.0
            ).fillna(0.0)
    except Exception:
        pass

    base_frames.append(d)

panel = pd.concat(base_frames)
panel.index.name = 'Date'
panel = panel.reset_index()

raw_feature_cols = [
    'Lag_1', 'Lag_5', 'Lag_10', 'Dist_SMA50', 'Dist_SMA200',
    'RSI', 'MACD', 'MACD_Signal', 'BB_Width', 'Volume_ZScore',
    'Rev_Growth_YoY', 'SUE', 'Analyst_Momentum'
]
for col in raw_feature_cols:
    panel[f'{col}_CSRank'] = panel.groupby('Date')[col].rank(pct=True)

feature_cols = raw_feature_cols + [f'{c}_CSRank' for c in raw_feature_cols] + ['SPY_Return']

panel['Fwd_Return'] = panel.groupby('Ticker')['Close'].transform(
    lambda s: s.shift(-HORIZON) / s - 1
)
panel['CS_Median_Fwd'] = panel.groupby('Date')['Fwd_Return'].transform('median')
panel['Target'] = (panel['Fwd_Return'] > panel['CS_Median_Fwd']).astype(int)

panel = panel.dropna(subset=feature_cols + ['Target', 'Fwd_Return']).copy()
panel = panel.sort_values('Date').reset_index(drop=True)

close_by_ticker = panel.pivot(index='Date', columns='Ticker', values='Close')

# Model Setup
model_params = {
    'n_estimators': 100,
    'max_depth': 6,
    'min_samples_leaf': 30,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1,
}

train_days = 750
val_days = 250
step_days = 60
cost_per_trade = 0.0005

# Walk-forward loop
unique_dates = np.sort(panel['Date'].unique())
n_dates = len(unique_dates)

test_records = []
for start_i in range(0, n_dates - train_days - val_days - step_days, step_days):
    train_end_i = start_i + train_days
    val_end_i = train_end_i + val_days
    test_end_i = val_end_i + step_days

    train_dates = unique_dates[start_i:train_end_i]
    test_dates = unique_dates[val_end_i:test_end_i]

    train = panel[panel['Date'].isin(train_dates)]
    test = panel[panel['Date'].isin(test_dates)]

    if len(train) < 200 or len(test) < 10:
        continue

    model = RandomForestClassifier(**model_params)
    model.fit(train[feature_cols], train['Target'])

    test = test.copy()
    test['Prob_Up'] = model.predict_proba(test[feature_cols])[:, 1]
    test_records.append(test[['Date', 'Ticker', 'Close', 'Fwd_Return', 'Prob_Up']])

df_test = pd.concat(test_records).sort_values(['Ticker', 'Date']).reset_index(drop=True)

# Backtest calculations
wide_prob = df_test.pivot(index='Date', columns='Ticker', values='Prob_Up')
wide_close = close_by_ticker.reindex(wide_prob.index)
wide_ret = wide_close.pct_change()

rebalance_days = wide_prob.index[::HORIZON]
prob_at_rebalance = wide_prob.reindex(rebalance_days)

long_mask_sparse = prob_at_rebalance.rank(axis=1, ascending=False) <= TOP_K
long_mask = long_mask_sparse.reindex(wide_prob.index).ffill().shift(1).fillna(False).astype(bool)

long_ret = wide_ret.where(long_mask).mean(axis=1)
long_turnover = long_mask.astype(int).diff().abs().sum(axis=1) / (2 * TOP_K)
long_only_ret = (long_ret - long_turnover.fillna(0) * cost_per_trade).fillna(0)

equal_weight_bh_ret = wide_ret.mean(axis=1).fillna(0)

cum_long_only = (1 + long_only_ret).cumprod()
cum_bh = (1 + equal_weight_bh_ret).cumprod()

# SINGLE GRAPH DISPLAY (Without Sharpe)
plt.figure(figsize=(12, 7))
plt.plot(cum_bh.index, cum_bh,
         label=f"Buy & Hold (equal-weight, {panel['Ticker'].nunique()} stocks)  |  Final ${cum_bh.iloc[-1]:.2f}",
         color='gray', linestyle='--', linewidth=2)
plt.plot(cum_long_only.index, cum_long_only,
         label=f"AI Strategy (Top-{TOP_K} long only)  |  Final ${cum_long_only.iloc[-1]:.2f}",
         color='navy', linewidth=2)
plt.title(f"AI Strategy vs. Buy & Hold  |  Horizon = {HORIZON} days")
plt.ylabel('Growth of $1')
plt.xlabel('Date')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()