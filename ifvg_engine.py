import pandas as pd
import numpy as np

def calculate_atr(df, length=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    # Simple moving average of TR for ATR as Pine's ta.atr usually matches RMA, 
    # but simple is close enough. Actually Pine's ta.atr is RMA.
    # RMA formula: alpha = 1/length
    atr = true_range.ewm(alpha=1/length, adjust=False).mean()
    return atr

def run_ifvg_backtest(df, 
                      min_gap_ticks=0, mintick=0.01,
                      max_hidden_fvg=120, max_fvg_age=60,
                      min_gap_atr=0.25, min_body_ratio=0.50, min_range_atr=0.60, break_buffer_atr=0.05,
                      atr_len=14, sl_atr_mult=1.5, tp_rr=3.0,
                      entry_mode="IFVG Line", line_price_mode="Broken Boundary"):
    """
    Runs the IFVG Sniper engine on a DataFrame containing High, Low, Open, Close.
    Returns trade history and metrics.
    """
    # Pre-calculate indicators
    df['ATR'] = calculate_atr(df, atr_len)
    df['CandleRange'] = np.maximum(df['High'] - df['Low'], mintick)
    df['CandleBody'] = np.abs(df['Close'] - df['Open'])
    df['BodyRatio'] = df['CandleBody'] / df['CandleRange']
    df['RangeAtr'] = df['CandleRange'] / df['ATR']
    
    # Pre-calculate raw FVGs
    df['High_2'] = df['High'].shift(2)
    df['Low_2'] = df['Low'].shift(2)
    
    # We will iterate row by row to match Pine Script's stateful nature.
    # While slow in pure Python, it guarantees 1:1 logic translation.
    
    # State variables
    hidden_fvgs = [] # List of dicts
    
    trade_active = False
    last_signal_dir = 0
    last_entry = np.nan
    last_sl = np.nan
    last_tp = np.nan
    
    trades = []
    
    # Extract arrays for faster iteration
    highs = df['High'].values
    lows = df['Low'].values
    opens = df['Open'].values
    closes = df['Close'].values
    high2s = df['High_2'].values
    low2s = df['Low_2'].values
    atrs = df['ATR'].values
    body_ratios = df['BodyRatio'].values
    range_atrs = df['RangeAtr'].values
    times = df.index
    
    for i in range(2, len(df)):
        current_time = times[i]
        high = highs[i]
        low = lows[i]
        close = closes[i]
        atr = atrs[i] if pd.notna(atrs[i]) and atrs[i] > 0 else mintick
        
        # 1. Update FVG ages and remove expired
        for fvg in hidden_fvgs:
            fvg['age'] += 1
            
        hidden_fvgs = [fvg for fvg in hidden_fvgs if fvg['age'] <= max_fvg_age]
        
        # 2. FVG Detection
        min_gap = min_gap_ticks * mintick
        raw_bull_fvg = not pd.isna(high2s[i]) and low > high2s[i] and (low - high2s[i]) >= min_gap
        raw_bear_fvg = not pd.isna(low2s[i]) and high < low2s[i] and (low2s[i] - high) >= min_gap
        
        if raw_bull_fvg:
            gap_atr = (low - high2s[i]) / atr
            hidden_fvgs.append({
                'top': low, 'bot': high2s[i], 'dir': 1, 'age': 0,
                'gap_atr': gap_atr, 'body_ratio': body_ratios[i], 'range_atr': range_atrs[i]
            })
            
        if raw_bear_fvg:
            gap_atr = (low2s[i] - high) / atr
            hidden_fvgs.append({
                'top': low2s[i], 'bot': high, 'dir': -1, 'age': 0,
                'gap_atr': gap_atr, 'body_ratio': body_ratios[i], 'range_atr': range_atrs[i]
            })
            
        # Limit memory size
        if len(hidden_fvgs) > max_hidden_fvg:
            hidden_fvgs = hidden_fvgs[-max_hidden_fvg:]
            
        # 3. Detect IFVG Inversion
        new_bull_ifvg = False
        new_bear_ifvg = False
        new_dir = 0
        new_top = np.nan
        new_bot = np.nan
        new_confirm_close = np.nan
        
        if len(hidden_fvgs) > 0:
            # Loop backwards
            for idx in range(len(hidden_fvgs)-1, -1, -1):
                fvg = hidden_fvgs[idx]
                clean_break_buffer = atr * break_buffer_atr
                
                bullish_inversion = (fvg['dir'] == -1) and (close > fvg['top'] + clean_break_buffer)
                bearish_inversion = (fvg['dir'] == 1) and (close < fvg['bot'] - clean_break_buffer)
                
                if bullish_inversion or bearish_inversion:
                    # Filter check
                    if (fvg['gap_atr'] >= min_gap_atr and 
                        fvg['body_ratio'] >= min_body_ratio and 
                        fvg['range_atr'] >= min_range_atr):
                        
                        new_bull_ifvg = bullish_inversion
                        new_bear_ifvg = bearish_inversion
                        new_dir = 1 if bullish_inversion else -1
                        new_top = fvg['top']
                        new_bot = fvg['bot']
                        new_confirm_close = close
                        
                    # Remove from memory whether passed filter or not (as per Pine Script)
                    hidden_fvgs.pop(idx)
                    break
        
        # 4. Entry Logic
        if new_bull_ifvg or new_bear_ifvg:
            if not trade_active:
                if line_price_mode == "Broken Boundary":
                    line_price = new_top if new_dir == 1 else new_bot
                elif line_price_mode == "Midpoint":
                    line_price = (new_top + new_bot) / 2.0
                else:
                    line_price = new_confirm_close
                    
                entry_price = new_confirm_close if entry_mode == "Confirmation Close" else line_price
                risk = atr * sl_atr_mult
                
                sl = entry_price - risk if new_dir == 1 else entry_price + risk
                tp = entry_price + risk * tp_rr if new_dir == 1 else entry_price - risk * tp_rr
                
                trade_active = True
                last_signal_dir = new_dir
                last_entry = entry_price
                last_sl = sl
                last_tp = tp
                
                trades.append({
                    'entry_time': current_time,
                    'dir': new_dir,
                    'entry_price': entry_price,
                    'sl': sl,
                    'tp': tp,
                    'status': 'OPEN',
                    'exit_time': None,
                    'pnl': 0
                })
                
        # 5. Trade Management (SL/TP Checks)
        if trade_active:
            hit_sl = (low <= last_sl) if last_signal_dir == 1 else (high >= last_sl)
            hit_tp = (high >= last_tp) if last_signal_dir == 1 else (low <= last_tp)
            
            if hit_sl:
                trade_active = False
                trades[-1]['status'] = 'LOSS'
                trades[-1]['exit_time'] = current_time
                trades[-1]['pnl'] = -1.0 # 1R loss
            elif hit_tp:
                trade_active = False
                trades[-1]['status'] = 'WIN'
                trades[-1]['exit_time'] = current_time
                trades[-1]['pnl'] = tp_rr # tp_rr R win
                
    return pd.DataFrame(trades)
