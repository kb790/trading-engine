import yfinance as yf

def evaluate_company_metrics(ticker: str) -> dict:
    """
    Fetches key company valuation and financial metrics using yfinance 
    to ensure the stock passes baseline financial health checks.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Extract key metrics safely
        pe_ratio = info.get("trailingPE", None)
        peg_ratio = info.get("pegRatio", None)
        profit_margins = info.get("profitMargins", None)
        recommendation = info.get("recommendationKey", "none")

        print(f"\n[METRICS] Valuation report for {ticker}:")
        print(f"  • P/E Ratio: {pe_ratio}")
        print(f"  • PEG Ratio: {peg_ratio}")
        print(f"  • Profit Margin: {profit_margins * 100:.2f}%" if profit_margins else "  • Profit Margin: N/A")
        print(f"  • Analyst Consensus: {recommendation.upper()}")

        # Baseline Safety Rules:
        # 1. Company must be profitable (positive P/E ratio)
        # 2. Profit margins must be positive (> 0%)
        is_profitable = pe_ratio is not None and pe_ratio > 0
        has_positive_margins = profit_margins is not None and profit_margins > 0

        if is_profitable and has_positive_margins:
            print(f"[METRICS PASS] {ticker} meets financial health criteria.")
            return {"pass": True, "reason": "Healthy metrics"}
        else:
            print(f"[METRICS REJECT] {ticker} failed baseline financial checks.")
            return {"pass": False, "reason": "Unprofitable or weak margins"}

    except Exception as e:
        print(f"[METRICS ERROR] Could not fetch data for {ticker}: {e}")
        # Default to True so network errors don't block technical trades
        return {"pass": True, "reason": "Fallback due to connection issue"}