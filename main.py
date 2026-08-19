import pandas as pd
from bist_data_cache import ensure_cache_dirs, fetch_and_cache_data, DEFAULT_TICKERS, TIMEFRAME_MAP
from ifvg_engine import run_ifvg_backtest

def run_all_tests():
    ensure_cache_dirs()
    
    test_tickers = DEFAULT_TICKERS
    timeframes = ["15m"]
    
    results_list = []
    current_recommendations = []
    
    print(f"Starting 15m Backtest on {len(test_tickers)} BIST100 Stocks (LONG ONLY)...")
    
    for tf in timeframes:
        for ticker in test_tickers:
            df = fetch_and_cache_data(ticker, tf)
            if df is None or len(df) < 50:
                print(f"Skipping {ticker} on {tf} (Not enough data)")
                continue
                
            # Run backtest
            trades_df = run_ifvg_backtest(df, mintick=0.01)
            
            if len(trades_df) > 0:
                # FILTER FOR LONG TRADES ONLY (dir == 1)
                long_trades = trades_df[trades_df['dir'] == 1]
                
                completed = long_trades[long_trades['status'] != 'OPEN']
                wins = len(completed[completed['status'] == 'WIN'])
                losses = len(completed[completed['status'] == 'LOSS'])
                total_long_completed = len(completed)
                
                win_rate = (wins / total_long_completed * 100) if total_long_completed > 0 else 0
                total_rr = completed['pnl'].sum()
                
                if total_long_completed > 0:
                    results_list.append({
                        'Ticker': ticker,
                        'Total Trades': total_long_completed,
                        'Wins': wins,
                        'Losses': losses,
                        'Win Rate (%)': round(win_rate, 2),
                        'Total RR': round(total_rr, 2)
                    })
                
                # Check for CURRENT OPEN Long positions (Recommendation)
                open_longs = long_trades[long_trades['status'] == 'OPEN']
                if not open_longs.empty:
                    last_open = open_longs.iloc[-1]
                    current_recommendations.append({
                        'Ticker': ticker,
                        'Entry Price': round(last_open['entry_price'], 2),
                        'Stop Loss': round(last_open['sl'], 2),
                        'Take Profit': round(last_open['tp'], 2),
                        'Time': last_open['entry_time']
                    })
                    
    results_df = pd.DataFrame(results_list)
    
    if not results_df.empty:
        # Sort by Win Rate (highest first), then by Total RR
        results_df = results_df.sort_values(by=['Win Rate (%)', 'Total RR'], ascending=[False, False])
        
        print("\n" + "="*60)
        print("🏆 TOP 15M LONG POSITION PERFORMERS")
        print("="*60)
        print(results_df.head(20).to_string(index=False)) # Top 20
        
        results_df.to_csv("backtest_results_15m_long.csv", index=False)
        print("\nFull results saved to backtest_results_15m_long.csv")
    else:
        print("No completed long trades found.")

    print("\n" + "="*60)
    print("🎯 CURRENT LIVE RECOMMENDATIONS (OPEN LONG IFVGs)")
    print("="*60)
    if current_recommendations:
        recs_df = pd.DataFrame(current_recommendations)
        print(recs_df.to_string(index=False))
    else:
        print("No open long signals at the moment.")

if __name__ == "__main__":
    run_all_tests()
