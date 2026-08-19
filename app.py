import streamlit as st
import pandas as pd
import time
from bist_data_cache import ensure_cache_dirs, fetch_and_cache_data, get_tickers_from_file
from ifvg_engine import run_ifvg_backtest

# --- Page Config ---
st.set_page_config(page_title="IFVG Sniper BIST Scanner", page_icon="🎯", layout="wide")

# --- CSS ---
st.markdown("""
    <style>
    .stProgress > div > div > div > div {
        background-color: #00ff00;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
st.title("🎯 IFVG Sniper BIST Scanner")
st.markdown("Scan Borsa Istanbul stocks for Inversion Fair Value Gap (IFVG) setups on the 15-minute timeframe.")

# --- Load Tickers ---
@st.cache_data
def load_tickers():
    return get_tickers_from_file("stox.txt")

tickers = load_tickers()
st.sidebar.header("Scanner Settings")
st.sidebar.write(f"Loaded **{len(tickers)}** tickers from stox.txt")

timeframe = st.sidebar.selectbox("Timeframe", ["15m", "1h", "1d"], index=0)
run_button = st.sidebar.button("Run Scanner 🚀", use_container_width=True)
update_button = st.sidebar.button("🔄 Update Data & Append", use_container_width=True)

if run_button or update_button:
    ensure_cache_dirs()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results_list = []
    current_recommendations = []
    
    total_tickers = len(tickers)
    
    for i, ticker in enumerate(tickers):
        # Update UI
        progress = (i + 1) / total_tickers
        progress_bar.progress(progress)
        
        if update_button:
            status_text.text(f"Updating data for {i+1}/{total_tickers}: {ticker}")
            df = fetch_and_cache_data(ticker, timeframe, force_update=True, append=True)
        else:
            status_text.text(f"Scanning {i+1}/{total_tickers}: {ticker}")
            df = fetch_and_cache_data(ticker, timeframe, force_update=False)
            
        if df is None or len(df) < 50:
            continue
            
        # Run Backtest
        trades_df = run_ifvg_backtest(df, mintick=0.01)
        
        if len(trades_df) > 0:
            # Filter for Long Trades Only
            long_trades = trades_df[trades_df['dir'] == 1]
            
            completed = long_trades[long_trades['status'] != 'OPEN']
            wins = len(completed[completed['status'] == 'WIN'])
            losses = len(completed[completed['status'] == 'LOSS'])
            total_long_completed = len(completed)
            
            win_rate = (wins / total_long_completed * 100) if total_long_completed > 0 else 0
            total_rr = completed['pnl'].sum()
            
            # Save Historical Stats
            if total_long_completed > 0:
                results_list.append({
                    'Ticker': ticker,
                    'Total Trades': total_long_completed,
                    'Wins': wins,
                    'Losses': losses,
                    'Win Rate (%)': round(win_rate, 2),
                    'Total RR': round(total_rr, 2)
                })
            
            # Save Current Open Signals (Recommendations)
            open_longs = long_trades[long_trades['status'] == 'OPEN']
            if not open_longs.empty:
                last_open = open_longs.iloc[-1]
                current_recommendations.append({
                    'Ticker': ticker,
                    'Entry Price': round(last_open['entry_price'], 2),
                    'Stop Loss': round(last_open['sl'], 2),
                    'Take Profit': round(last_open['tp'], 2),
                    'Time': last_open['entry_time'],
                    'Win Rate (%)': round(win_rate, 2), # Attach Win Rate for sorting
                    'Total Trades': total_long_completed  # Attach Total Trades for sorting
                })

    # Processing Complete
    progress_bar.empty()
    status_text.success(f"Scan complete! Analyzed {total_tickers} stocks.")
    
    # --- Display Results ---
    results_df = pd.DataFrame(results_list)
    recs_df = pd.DataFrame(current_recommendations)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎯 Live Recommendations")
        if not recs_df.empty:
            # Hierarchical Sorting: 1. Win Rate (Desc), 2. Total Trades (Desc)
            recs_df = recs_df.sort_values(by=['Win Rate (%)', 'Total Trades'], ascending=[False, False])
            
            # Reorder columns for display
            display_recs = recs_df[['Ticker', 'Win Rate (%)', 'Total Trades', 'Entry Price', 'Stop Loss', 'Take Profit', 'Time']]
            st.dataframe(display_recs, use_container_width=True, hide_index=True)
        else:
            st.info("No open long signals at the moment.")
            
    with col2:
        st.subheader("🏆 Historical Top Performers")
        if not results_df.empty:
            results_df = results_df.sort_values(by=['Win Rate (%)', 'Total RR'], ascending=[False, False])
            st.dataframe(results_df, use_container_width=True, hide_index=True)
        else:
            st.info("No completed long trades found.")
            
else:
    st.info("👈 Click **Run Scanner** to begin backtesting.")
