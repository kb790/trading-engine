from company_metrics import evaluate_company_metrics

def generate_current_signal(ticker: str):
    # ... existing technical SMA analysis code ...
    
    # Assume technical_signal is produced by your 20/70/200 SMA calculation
    technical_signal = "ACTION_BUY"  # (Example signal)
    current_price = 225.50           # (Example price)

    if technical_signal == "ACTION_BUY":
        # Run company health check before placing the buy
        metrics_check = evaluate_company_metrics(ticker)
        
        if metrics_check["pass"]:
            print(f"[HYBRID SIGNAL] Technical BUY confirmed by financial health check for {ticker}.")
            return "ACTION_BUY", current_price
        else:
            print(f"[HYBRID SIGNAL] Technical BUY rejected due to weak company metrics.")
            return "ACTION_HOLD", current_price

    return technical_signal, current_price