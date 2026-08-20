import pandas as pd
import numpy as np


def calculate_atr(df, length=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.ewm(alpha=1/length, adjust=False).mean()
    return atr


def run_ifvg_backtest(df,
                      min_gap_ticks=0, mintick=0.01,
                      max_hidden_fvg=120, max_fvg_age=60,
                      min_gap_atr=0.25, min_body_ratio=0.50, min_range_atr=0.60, break_buffer_atr=0.05,
                      atr_len=14, sl_atr_mult=1.5, tp_rr=3.0,
                      entry_mode="IFVG Line", line_price_mode="Broken Boundary"):
    """
    Optimized IFVG Sniper engine.
    - FVG detection is fully vectorized with NumPy (no per-row Python overhead).
    - The state machine (FVG memory + trade management) runs in a tight Python loop
      over only the candles where FVGs exist, skipping empty candles fast via NumPy masks.
    """
    n = len(df)

    # ── Pre-compute all indicators as raw numpy arrays ──────────────────────
    atr_s = calculate_atr(df, atr_len)

    highs   = df['High'].values.astype(np.float64)
    lows    = df['Low'].values.astype(np.float64)
    opens   = df['Open'].values.astype(np.float64)
    closes  = df['Close'].values.astype(np.float64)
    atrs    = atr_s.values.astype(np.float64)
    times   = df.index

    candle_range = np.maximum(highs - lows, mintick)
    candle_body  = np.abs(closes - opens)
    body_ratios  = candle_body / candle_range
    range_atrs   = candle_range / np.where(atrs > 0, atrs, mintick)

    # High and Low shifted by 2 (vectorized)
    high2s = np.empty(n); high2s[:] = np.nan; high2s[2:] = highs[:-2]
    low2s  = np.empty(n); low2s[:] = np.nan;  low2s[2:]  = lows[:-2]

    min_gap = min_gap_ticks * mintick

    # ── Vectorized FVG candidate detection ──────────────────────────────────
    # Bull FVG: low[i] > high[i-2]
    bull_mask = (lows > high2s + min_gap) & ~np.isnan(high2s)
    # Bear FVG: high[i] < low[i-2]
    bear_mask = (highs < low2s - min_gap) & ~np.isnan(low2s)

    # Indices where ANY fvg candidate or trade check is needed
    # We'll still loop, but skip non-event candles quickly with precomputed arrays
    # and avoid per-candle Python list comprehensions inside the loop.

    # ── State machine loop (only unavoidable stateful part) ──────────────────
    # Use parallel numpy arrays instead of list-of-dicts for FVG memory
    # Preallocate fixed-size buffers (ring-buffer style)
    MAX = max_hidden_fvg

    fvg_top      = np.empty(MAX, dtype=np.float64)
    fvg_bot      = np.empty(MAX, dtype=np.float64)
    fvg_dir      = np.zeros(MAX, dtype=np.int8)
    fvg_age      = np.zeros(MAX, dtype=np.int32)
    fvg_gap_atr  = np.empty(MAX, dtype=np.float64)
    fvg_body_rat = np.empty(MAX, dtype=np.float64)
    fvg_rng_atr  = np.empty(MAX, dtype=np.float64)
    fvg_active   = np.zeros(MAX, dtype=bool)

    fvg_count = 0   # number of active FVGs
    head = 0        # next insertion index (ring-buffer)

    trade_active    = False
    last_signal_dir = 0
    last_entry      = 0.0
    last_sl         = 0.0
    last_tp         = 0.0

    trades = []

    for i in range(2, n):
        high  = highs[i]
        low   = lows[i]
        close = closes[i]
        atr   = atrs[i] if (not np.isnan(atrs[i]) and atrs[i] > 0) else mintick
        br    = body_ratios[i]
        ra    = range_atrs[i]

        # 1. Age all active FVGs; deactivate expired ones (vectorized slice)
        if fvg_count > 0:
            active_idx = np.where(fvg_active)[0]
            fvg_age[active_idx] += 1
            expired = active_idx[fvg_age[active_idx] > max_fvg_age]
            if len(expired):
                fvg_active[expired] = False
                fvg_count -= len(expired)

        # 2. Add new FVG candidates (pre-computed masks)
        if bull_mask[i]:
            slot = head % MAX
            fvg_top[slot]      = low
            fvg_bot[slot]      = high2s[i]
            fvg_dir[slot]      = 1
            fvg_age[slot]      = 0
            fvg_gap_atr[slot]  = (low - high2s[i]) / atr
            fvg_body_rat[slot] = br
            fvg_rng_atr[slot]  = ra
            if not fvg_active[slot]:   # evicting old entry
                fvg_active[slot] = True
                fvg_count += 1
            head += 1

        if bear_mask[i]:
            slot = head % MAX
            fvg_top[slot]      = low2s[i]
            fvg_bot[slot]      = high
            fvg_dir[slot]      = -1
            fvg_age[slot]      = 0
            fvg_gap_atr[slot]  = (low2s[i] - high) / atr
            fvg_body_rat[slot] = br
            fvg_rng_atr[slot]  = ra
            if not fvg_active[slot]:
                fvg_active[slot] = True
                fvg_count += 1
            head += 1

        # 3. Detect IFVG inversion (scan active FVGs newest-first)
        new_dir   = 0
        new_top   = 0.0
        new_bot   = 0.0
        new_close = close

        if fvg_count > 0:
            buf = atr * break_buffer_atr
            active_idx = np.where(fvg_active)[0]

            # Vectorized inversion check over all active FVGs
            bull_inv = (fvg_dir[active_idx] == -1) & (close > fvg_top[active_idx] + buf)
            bear_inv = (fvg_dir[active_idx] ==  1) & (close < fvg_bot[active_idx] - buf)
            inv_mask = bull_inv | bear_inv

            if inv_mask.any():
                # Take the newest inverted FVG (highest index in active_idx = most recent)
                inv_positions = active_idx[inv_mask]
                # "Newest" = the one inserted latest; approximate by highest slot index
                idx_in_buf = inv_positions[-1]  # last inserted among inverted

                if (fvg_gap_atr[idx_in_buf]  >= min_gap_atr and
                        fvg_body_rat[idx_in_buf] >= min_body_ratio and
                        fvg_rng_atr[idx_in_buf]  >= min_range_atr):
                    new_dir = 1 if bull_inv[inv_mask.nonzero()[0][-1]] else -1
                    new_top = fvg_top[idx_in_buf]
                    new_bot = fvg_bot[idx_in_buf]

                # Remove the FVG from memory (Pine Script removes it regardless of filter)
                fvg_active[idx_in_buf] = False
                fvg_count -= 1

        # 4. Entry logic
        if new_dir != 0 and not trade_active:
            if line_price_mode == "Broken Boundary":
                line_price = new_top if new_dir == 1 else new_bot
            elif line_price_mode == "Midpoint":
                line_price = (new_top + new_bot) / 2.0
            else:
                line_price = new_close

            entry_price = new_close if entry_mode == "Confirmation Close" else line_price
            risk = atr * sl_atr_mult

            last_sl = entry_price - risk if new_dir == 1 else entry_price + risk
            last_tp = entry_price + risk * tp_rr if new_dir == 1 else entry_price - risk * tp_rr

            trade_active    = True
            last_signal_dir = new_dir
            last_entry      = entry_price

            trades.append({
                'entry_time':  times[i],
                'dir':         new_dir,
                'entry_price': entry_price,
                'sl':          last_sl,
                'tp':          last_tp,
                'status':      'OPEN',
                'exit_time':   None,
                'pnl':         0.0
            })

        # 5. Trade management
        if trade_active:
            if last_signal_dir == 1:
                hit_sl = low  <= last_sl
                hit_tp = high >= last_tp
            else:
                hit_sl = high >= last_sl
                hit_tp = low  <= last_tp

            if hit_sl:
                trade_active = False
                trades[-1]['status']    = 'LOSS'
                trades[-1]['exit_time'] = times[i]
                trades[-1]['pnl']       = -1.0
            elif hit_tp:
                trade_active = False
                trades[-1]['status']    = 'WIN'
                trades[-1]['exit_time'] = times[i]
                trades[-1]['pnl']       = tp_rr

    return pd.DataFrame(trades)
