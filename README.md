# Quant Backtester: Systematic Evaluation of ML-Based Equity Signals

A research project testing whether machine learning models can find a real, exploitable
edge in predicting equity returns using public data  -  and a case study in how easily
a backtest can look successful before it survives rigorous scrutiny.

**Short version of the finding:** across multiple targets, feature sets, model families,
and a systematic search for statistical arbitrage pairs, no approach here produced a
signal that reliably survives significance testing. That's not a failed project  -  it's
the actual, well-evidenced answer this repo set out to find, and it's consistent with
decades of academic finance research on market efficiency in liquid, heavily-analyzed
large-cap equities.

**What's in this repo, in one line each:**
- **`main.py`**  -  the backtest. Trains the ranking model on historical data and
  produces a chart comparing the AI strategy against buy & hold. No API keys
  needed; run this to see the results.
- **`main_bot.py`**  -  the live bot. Connects to Alpaca's paper-trading API and
  actually places (simulated) trades based on the same model, on a schedule.
  Requires Alpaca API credentials in a `.env` file.

---

## Table of Contents
- [Project Structure](#project-structure)
- [Setup](#setup)
- [The Research Journey](#the-research-journey)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [The Alpaca Trading Bot](#the-alpaca-trading-bot)
- [Limitations & Honest Caveats](#limitations--honest-caveats)
- [What Would Be Worth Trying Next](#what-would-be-worth-trying-next)
- [Disclaimer](#disclaimer)

---

## Project Structure

```
quant_backtester/
├── python/
│   ├── main.py                    # Backtest: cross-sectional ranking model
│   │                               # (Top-K long / Bottom-K short) vs. buy & hold
│   └── main_bot.py                # Live Alpaca paper-trading bot
├── .env                            # Alpaca API credentials -- only needed for
│                                    # main_bot.py; main.py runs on yfinance alone
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Running `main.py` (the backtest) requires no credentials at all -- it only
pulls public price data via `yfinance`.

`main_bot.py` (live paper trading) needs Alpaca credentials. Create a `.env`
file in the project root (UTF-8 encoded, no quotes, never committed):
```
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
```

Run the backtest:
```bash
python python/main.py
```

Run the live paper-trading bot:
```bash
python python/main_bot.py
```

---

## The Research Journey

This project didn't start with the methodology described below  -  it evolved through
several rounds of finding a flaw, fixing it, and re-testing. That process is the actual
substance of the project, so it's documented here rather than hidden:

1. **Naive single-stock backtest.** Random Forest predicting next-day direction from
   RSI and volatility alone, single train/test split. Looked promising on a chart, but
   had no walk-forward validation and only two weak, collinear features.

2. **Added richer technical features + proper walk-forward validation.** Retrained
   periodically on rolling windows instead of a single static split. Results still
   looked inconsistent across tickers.

3. **Diagnosed the actual bottleneck: AUC ≈ 0.50.** Before trusting any P&L curve, we
   checked whether the model's raw predictions had any classification skill at all,
   using AUC and log-loss. They didn't  -  consistently, across single-stock and
   cross-sectional targets, multiple forward-looking horizons, and two model families
   (Random Forest, LightGBM).

4. **Reframed as a cross-sectional ranking problem.** Instead of predicting absolute
   direction, the model predicts whether a stock will outperform the *median* of a
   20-49 stock universe over a fixed horizon  -  a more tractable question, since it
   cancels out a lot of market-wide noise.

5. **Fixed a real target/trading-rule mismatch.** An earlier version used the
   relative ranking to make an absolute per-stock hold/cash decision, which
   structurally favors buy & hold in an up-trending sample regardless of model skill.
   Fixed by switching to genuine Top-K long / Bottom-K short portfolio construction.

6. **Replaced AUC with Information Coefficient (IC)** as the primary evaluation
   metric  -  the correct tool for a ranking model, since binarizing at the median
   throws away information that a rank-correlation captures.

7. **Caught and corrected a lookahead bias risk** in point-in-time fundamentals
   (reporting-date lag) and discovered a **data coverage wall**: free `yfinance`
   fundamentals endpoints only return ~5-7 quarters of history, making a
   multi-year fundamentals-only backtest unsupportable on this data source.

8. **Corrected an overlap-inflated significance test.** A 60-day-ahead forward
   return computed daily produces autocorrelated IC observations that overstate
   the effective sample size by ~60x. Re-tested using only non-overlapping,
   independent rebalance periods  -  a much smaller but honest sample.

9. **Ran an empirical statistical-arbitrage / pairs-trading search** across ~380
   candidate pairs, with a rolling-window cointegration stability check  -  this
   surfaces a real lesson in multiple-testing bias (testing many pairs at
   p<0.05 will produce ~5% false positives by chance alone).

---

## Methodology

- **Universe:** 20-49 US large-cap equities across multiple sectors (tech,
  financials, energy, healthcare, consumer, industrials, utilities).
- **Features:** technical (lagged returns, SMA distance, RSI, MACD, Bollinger
  Band width, volume z-score) and point-in-time-lagged fundamentals (revenue
  growth YoY, standardized unanticipated earnings, analyst upgrade/downgrade
  momentum).
- **Target:** does a stock's forward N-day return exceed the cross-sectional
  median return of the universe on that date? (Horizons tested: 5, 10, 20, 60
  trading days.)
- **Validation:** rolling walk-forward  -  train on ~3 years, validate threshold
  selection on the following ~1 year, test out-of-sample on the next ~2 months,
  then roll forward and repeat.
- **Evaluation metrics:**
  - **AUC / log-loss**  -  raw classification skill on the binarized target.
  - **Information Coefficient (IC)**  -  daily Spearman correlation between
    predicted probability and actual forward return; the metric that actually
    matches what a ranking model is trained to do.
  - **Overlap-corrected significance test**  -  IC re-computed on only
    non-overlapping rebalance dates, with a t-test against zero, to avoid
    overstating confidence from autocorrelated daily samples.
  - **Long-short (market-neutral) Sharpe**  -  Top-K long minus Bottom-K short,
    which cancels general market beta and isolates real ranking skill from
    "the strategy just happened to be invested during a bull market."

---

## Key Findings

| Test | Result |
|---|---|
| Single-stock direction prediction (RF, LightGBM) | AUC ≈ 0.45-0.52 across all tickers tested |
| Cross-sectional ranking, technical features only | AUC ≈ 0.50, IC ≈ 0.00-0.03 |
| + Fundamentals (revenue growth, SUE, analyst momentum) | No measurable change in AUC or IC |
| Horizon sweep (5/10/20/60 days) | 60-day horizon strongest, but not significant after correcting for multiple-horizon search and overlap bias (p ≈ 0.10-0.13) |
| Empirical pairs-trading search (~380 pairs) | 1 surviving pair after two-stage screening  -  consistent with the false-positive rate expected from testing that many candidates, not a validated edge |
| Long-short market-neutral spread (best-case setup) | Sharpe ~0.2-0.75 depending on configuration; statistical significance borderline to absent |

**Honest conclusion:** no configuration tested here produced a signal that
clearly and reliably survives rigorous out-of-sample, significance-corrected
testing. Apparent wins in earlier iterations were traced to either overlap bias
in the significance test, a target/trading-rule mismatch that favored buy & hold
by construction and was masking the real comparison, or risk-management overlays
(trend filters, volatility scaling) improving risk-adjusted returns independent
of whether the underlying ranking had any real skill.

---

## The Alpaca Trading Bot

`main_bot.py` runs the cross-sectional model live against Alpaca's paper-trading
API. It:
- Retrains on the most recent 3 years of data at each rebalance
- Selects the Top-5 stocks by predicted probability
- Places bracket orders (take-profit +15% / stop-loss -8%) with position-size
  caps and a daily drawdown circuit breaker
- Rebalances on a calendar-aware schedule matched to the model's 60-day horizon,
  using Alpaca's own market calendar (not a naive day-count) to skip weekends
  and holidays correctly
- Logs a warning when today's predicted probabilities are too tightly
  clustered to represent a meaningful ranking, rather than presenting a
  low-confidence pick with false certainty

**This is a paper-trading / engineering exercise, not a validated strategy.**
Given the findings above, the bot should be understood as a demonstration of a
correctly-built live trading pipeline (scheduling, risk controls, order
management), not as evidence the underlying picks have real predictive value.

---

## Limitations & Honest Caveats

- **Free data ceiling.** `yfinance` fundamentals only cover ~1.5-2 years of
  history regardless of ticker, making full-history fundamentals backtesting
  unsupportable without a paid data vendor (e.g. Financial Modeling Prep,
  Sharadar, Polygon).
- **Multiple-comparisons risk.** Several results in this repo (horizon
  selection, pairs search) involved testing many candidates and reporting the
  best  -  the README states this explicitly rather than presenting the winning
  result in isolation, since that framing materially changes how much
  confidence the result deserves.
- **Short interest** was checked but excluded as a feature  -  `yfinance` only
  exposes a current snapshot, not a historical series, so including it would
  have introduced lookahead bias.
- **Transaction cost and borrow-cost assumptions** are simplified (a flat cost
  per trade); real-world short-selling costs, especially for less liquid
  names, are not modeled.

## What Would Be Worth Trying Next

- Point-in-time-correct fundamentals from a paid vendor or SEC EDGAR's XBRL
  API, which provides real historical filing data with actual filing dates
- Options-market data (implied volatility skew, put/call ratios)  -  reflects
  informed positioning rather than historical price shape
- A genuinely out-of-sample pairs search: pre-register a smaller candidate set
  based on business logic, rather than scanning hundreds of combinations
- Extending the universe to less-efficient markets (small/mid-caps, less
  analyst coverage) where a real edge is more plausible

## Disclaimer

This is a research and educational project. Nothing here is financial advice,
and no results in this repository should be interpreted as a validated,
profitable trading strategy. The Alpaca integration is configured for paper
trading; treat any live-trading use as entirely at your own risk.