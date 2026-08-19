import streamlit as st
import pandas as pd
import plotly.graph_objects as go
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
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
st.title("🎯 IFVG finder backtest")
st.markdown("Inversion Fair Value Gap engine for Borsa Istanbul — scan, inspect, and review setups.")

# --- Load Tickers ---
@st.cache_data
def load_tickers():
    return get_tickers_from_file("stox.txt")

tickers = load_tickers()

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab_scanner, tab_inspector = st.tabs(["🚀 Scanner", "📈 Chart Inspector"])


# ══════════════════════════════════════════════════════════════
# TAB 1 — SCANNER  (original logic, untouched)
# ══════════════════════════════════════════════════════════════
with tab_scanner:
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

            trades_df = run_ifvg_backtest(df, mintick=0.01)

            if len(trades_df) > 0:
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

                open_longs = long_trades[long_trades['status'] == 'OPEN']
                if not open_longs.empty:
                    last_open = open_longs.iloc[-1]
                    current_recommendations.append({
                        'Ticker': ticker,
                        'Entry Price': round(last_open['entry_price'], 2),
                        'Stop Loss': round(last_open['sl'], 2),
                        'Take Profit': round(last_open['tp'], 2),
                        'Time': last_open['entry_time'],
                        'Win Rate (%)': round(win_rate, 2),
                        'Total Trades': total_long_completed
                    })

        progress_bar.empty()
        status_text.success(f"Scan complete! Analyzed {total_tickers} stocks.")

        results_df = pd.DataFrame(results_list)
        recs_df = pd.DataFrame(current_recommendations)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("🎯 Live Recommendations")
            if not recs_df.empty:
                recs_df = recs_df.sort_values(by=['Win Rate (%)', 'Total Trades'], ascending=[False, False])
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


# ══════════════════════════════════════════════════════════════
# TAB 2 — CHART INSPECTOR
# ══════════════════════════════════════════════════════════════
with tab_inspector:
    st.subheader("📈 Chart Inspector — IFVG Trade Overlay")
    st.markdown(
        "Select a stock and timeframe, then run the IFVG engine and inspect every trade "
        "directly on the candlestick chart. Green boxes = reward zone (Entry → TP), "
        "Red boxes = risk zone (Entry → SL)."
    )

    ci_col1, ci_col2, ci_col3 = st.columns([2, 1, 1])

    with ci_col1:
        # Build a sorted, human-readable ticker list
        ticker_options = sorted(tickers)
        # Try to pre-select THYAO.IS if present
        default_idx = ticker_options.index("THYAO.IS") if "THYAO.IS" in ticker_options else 0
        selected_ticker = st.selectbox(
            "Select Stock",
            options=ticker_options,
            index=default_idx,
            key="ci_ticker"
        )

    with ci_col2:
        selected_tf = st.selectbox(
            "Timeframe",
            options=["15m", "1h", "1d"],
            index=0,
            key="ci_tf"
        )

    with ci_col3:
        candle_limit = st.selectbox(
            "Candles to show",
            options=[200, 500, 1000, 2000, "All"],
            index=1,
            key="ci_candles"
        )

    inspect_btn = st.button("🔍 Inspect Chart", use_container_width=False, key="ci_run")

    if inspect_btn:
        with st.spinner(f"Fetching data and running IFVG engine for {selected_ticker}..."):
            ensure_cache_dirs()
            df = fetch_and_cache_data(selected_ticker, selected_tf, force_update=False)

        if df is None or len(df) < 50:
            st.error(f"Not enough data for **{selected_ticker}** on the {selected_tf} timeframe. "
                     "Try running **🔄 Update Data & Append** from the Scanner tab first.")
        else:
            trades_df = run_ifvg_backtest(df.copy(), mintick=0.01)

            # ── Slice candles ──────────────────────────────────────────
            if candle_limit != "All":
                plot_df = df.iloc[-int(candle_limit):]
            else:
                plot_df = df.copy()

            # ── Build candlestick figure ──────────────────────────────
            fig = go.Figure()

            fig.add_trace(go.Candlestick(
                x=plot_df.index,
                open=plot_df['Open'],
                high=plot_df['High'],
                low=plot_df['Low'],
                close=plot_df['Close'],
                name=selected_ticker,
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
                increasing_fillcolor="#26a69a",
                decreasing_fillcolor="#ef5350",
            ))

            # ── Overlay trade boxes ───────────────────────────────────
            plot_start = plot_df.index[0]
            plot_end   = plot_df.index[-1]

            shapes = []
            annotations = []

            for _, trade in trades_df.iterrows():
                entry_time = trade['entry_time']

                # Only draw trades that fall within our visible candle window
                if entry_time < plot_start:
                    continue

                # Determine the right x-boundary of each box
                if trade['status'] == 'OPEN':
                    # Extend box to the right edge of the chart
                    exit_time = plot_end
                else:
                    exit_time = trade['exit_time']
                    # Clamp to visible range
                    if exit_time > plot_end:
                        exit_time = plot_end
                    if exit_time < plot_start:
                        continue

                entry = trade['entry_price']
                sl    = trade['sl']
                tp    = trade['tp']
                direction = trade['dir']
                status = trade['status']

                # ── Colour logic ──────────────────────────────
                if status == 'WIN':
                    tp_fill  = "rgba(38, 166, 154, 0.18)"   # teal
                    sl_fill  = "rgba(239, 83, 80, 0.10)"    # red faded
                    border   = "rgba(38, 166, 154, 0.70)"
                elif status == 'LOSS':
                    tp_fill  = "rgba(38, 166, 154, 0.07)"   # faded
                    sl_fill  = "rgba(239, 83, 80, 0.22)"    # red stronger
                    border   = "rgba(239, 83, 80, 0.70)"
                else:  # OPEN
                    tp_fill  = "rgba(255, 214, 0, 0.14)"    # gold
                    sl_fill  = "rgba(239, 83, 80, 0.14)"
                    border   = "rgba(255, 214, 0, 0.85)"

                # For long trades: TP box above entry, SL box below entry
                if direction == 1:
                    tp_y0, tp_y1 = entry, tp
                    sl_y0, sl_y1 = sl, entry
                else:
                    tp_y0, tp_y1 = tp, entry
                    sl_y0, sl_y1 = entry, sl

                # ── TP (reward) box ───────────────────────────
                shapes.append(dict(
                    type="rect",
                    xref="x", yref="y",
                    x0=entry_time, x1=exit_time,
                    y0=tp_y0, y1=tp_y1,
                    fillcolor=tp_fill,
                    line=dict(color=border, width=1),
                    layer="below"
                ))

                # ── SL (risk) box ─────────────────────────────
                shapes.append(dict(
                    type="rect",
                    xref="x", yref="y",
                    x0=entry_time, x1=exit_time,
                    y0=sl_y0, y1=sl_y1,
                    fillcolor=sl_fill,
                    line=dict(color="rgba(239, 83, 80, 0.50)", width=1),
                    layer="below"
                ))

                # ── Entry line ────────────────────────────────
                shapes.append(dict(
                    type="line",
                    xref="x", yref="y",
                    x0=entry_time, x1=exit_time,
                    y0=entry, y1=entry,
                    line=dict(color=border, width=1.2, dash="dot"),
                    layer="above"
                ))

                # ── Annotations ───────────────────────────────
                label_suffix = f" ({status})" if status != 'OPEN' else " (OPEN)"
                ann_x = entry_time

                annotations.append(dict(
                    x=ann_x, y=tp,
                    xref="x", yref="y",
                    text=f"TP {tp:.2f}{label_suffix}",
                    showarrow=False,
                    font=dict(size=9, color="#26a69a"),
                    xanchor="left",
                    yanchor="bottom",
                    bgcolor="rgba(0,0,0,0.5)",
                    borderpad=2,
                ))
                annotations.append(dict(
                    x=ann_x, y=sl,
                    xref="x", yref="y",
                    text=f"SL {sl:.2f}",
                    showarrow=False,
                    font=dict(size=9, color="#ef5350"),
                    xanchor="left",
                    yanchor="top",
                    bgcolor="rgba(0,0,0,0.5)",
                    borderpad=2,
                ))
                annotations.append(dict(
                    x=ann_x, y=entry,
                    xref="x", yref="y",
                    text=f"Entry {entry:.2f}",
                    showarrow=False,
                    font=dict(size=9, color=border),
                    xanchor="left",
                    yanchor="bottom",
                    bgcolor="rgba(0,0,0,0.5)",
                    borderpad=2,
                ))

            # ── Layout ────────────────────────────────────────────────
            fig.update_layout(
                title=dict(
                    text=f"<b>{selected_ticker}</b> — {selected_tf} IFVG Trade Map",
                    font=dict(size=18, color="#e0e0e0"),
                ),
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                xaxis=dict(
                    rangeslider=dict(visible=False),
                    gridcolor="#1e2329",
                    tickfont=dict(color="#9e9e9e"),
                    type="date",
                ),
                yaxis=dict(
                    gridcolor="#1e2329",
                    tickfont=dict(color="#9e9e9e"),
                    side="right",
                ),
                shapes=shapes,
                annotations=annotations,
                legend=dict(
                    font=dict(color="#9e9e9e"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                margin=dict(l=10, r=60, t=60, b=30),
                height=680,
            )

            st.plotly_chart(fig, use_container_width=True)

            # ── Trade Summary Table ───────────────────────────────────
            if not trades_df.empty:
                long_trades = trades_df[trades_df['dir'] == 1]
                completed   = long_trades[long_trades['status'] != 'OPEN']
                wins        = len(completed[completed['status'] == 'WIN'])
                losses      = len(completed[completed['status'] == 'LOSS'])
                open_trades = len(long_trades[long_trades['status'] == 'OPEN'])
                total_rr    = completed['pnl'].sum()
                win_rate    = (wins / len(completed) * 100) if len(completed) > 0 else 0

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Total Long Trades", len(long_trades))
                m2.metric("Wins ✅", wins)
                m3.metric("Losses ❌", losses)
                m4.metric("Open 🟡", open_trades)
                m5.metric("Win Rate", f"{win_rate:.1f}%")

                st.markdown("#### All Trades")
                display_df = trades_df.copy()
                display_df['entry_time'] = pd.to_datetime(display_df['entry_time']).dt.strftime('%Y-%m-%d %H:%M')
                display_df['exit_time']  = pd.to_datetime(display_df['exit_time']).dt.strftime('%Y-%m-%d %H:%M').fillna("—")
                display_df['dir']  = display_df['dir'].map({1: "Long 📈", -1: "Short 📉"})
                display_df['pnl']  = display_df['pnl'].round(2)
                display_df.columns = ['Entry Time', 'Direction', 'Entry Price', 'Stop Loss', 'Take Profit',
                                       'Status', 'Exit Time', 'PnL (R)']
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("No trades found for this stock on the selected timeframe.")

    else:
        st.info("👆 Select a stock and click **Inspect Chart** to visualise IFVG setups.")
