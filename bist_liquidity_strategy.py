import yfinance as yf
import pandas as pd
import numpy as np
import time
import warnings

# Suppress FutureWarning from yfinance
warnings.filterwarnings('ignore', category=FutureWarning)

def get_bist_symbols():
    try:
        url = "https://raw.githubusercontent.com/ahmeterenodaci/Istanbul-Stock-Exchange--BIST--including-symbols-and-logos/main/bist.json"
        df = pd.read_json(url)
        tickers = [f"{symbol}.IS" for symbol in df['symbol']]
        return tickers
    except Exception as e:
        print(f"BIST listesi webden çekilirken hata: {e}")
        print("Yedek liste kullanılıyor...")
        # Fallback to BIST30 as minimum
        return ["AKBNK.IS", "ARCLK.IS", "ASELS.IS", "BIMAS.IS", "EKGYO.IS", "EREGL.IS", 
                "FROTO.IS", "GARAN.IS", "GUBRF.IS", "HEKTS.IS", "ISCTR.IS", "KCHOL.IS", 
                "KOZAA.IS", "KOZAL.IS", "KRDMD.IS", "ODAS.IS", "PETKM.IS", "PGSUS.IS", 
                "SAHOL.IS", "SASA.IS", "SISE.IS", "TAVHL.IS", "TCELL.IS", "THYAO.IS", 
                "TKFEN.IS", "TOASO.IS", "TUPRS.IS", "VAKBN.IS", "YKBNK.IS"]

def fetch_and_resample(ticker):
    try:
        # yfinance allows max 730 days for 1h interval
        data = yf.download(ticker, period="730d", interval="1h", progress=False)
        if data.empty:
            return None
        
        # Flatten MultiIndex columns if yf returns them (happens in newer yfinance versions)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
            
        # Resample 1h candles to 4h
        df_4h = data.resample('4h', label='right').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        return df_4h
    except Exception:
        return None

def analyze_stock(df):
    trades = []
    
    # BIST is open roughly 10:00 to 18:00 (8 hours) -> 2 4h-candles per day
    # 6 months is roughly 126 trading days = 252 candles
    window = 252
    
    # 1. 6 Months Consolidation (< 20% price change)
    df['RollingMax'] = df['High'].rolling(window=window).max()
    df['RollingMin'] = df['Low'].rolling(window=window).min()
    # To avoid division by zero
    df['RollingMin'] = df['RollingMin'].replace(0, np.nan)
    df['Consolidation'] = ((df['RollingMax'] - df['RollingMin']) / df['RollingMin']) <= 0.20
    
    # 2. 8-10 Candle Downtrend
    # We define it as price dropping over 10 candles and majority of candles being red
    df['IsRed'] = df['Close'] < df['Open']
    df['RedCount10'] = df['IsRed'].rolling(window=10).sum()
    df['PriceDrop10'] = df['Close'] < df['Close'].shift(10)
    df['Downtrend'] = df['PriceDrop10'] & (df['RedCount10'] >= 6)
    
    # 3. Liquidity Grab (Long Stop Hunts)
    df['Body'] = abs(df['Open'] - df['Close'])
    df['LowerWick'] = np.minimum(df['Open'], df['Close']) - df['Low']
    df['UpperWick'] = df['High'] - np.maximum(df['Open'], df['Close'])
    
    # A stop hunt: lower wick is more than twice the body, and greater than upper wick
    df['IsStopHunt'] = (df['LowerWick'] > 2 * df['Body']) & (df['LowerWick'] > df['UpperWick']) & (df['Body'] > 0)
    
    # Multiple stop hunts recently (at least 2 in the last 15 candles)
    df['RecentStopHunts'] = df['IsStopHunt'].rolling(window=15).sum()
    
    # Final Signal: Consolidation AND recent downtrend AND multiple stop hunts
    df['Signal'] = df['Consolidation'] & df['Downtrend'] & (df['RecentStopHunts'] >= 2)
    
    # Backtest Loop - Looking for Explosions (>50% return)
    in_trade = False
    entry_price = 0.0
    entry_time = None
    max_price = 0.0
    
    explosion_target = 0.50
    sl_pct = 0.05
    
    for i in range(window, len(df)):
        if not in_trade:
            if df['Signal'].iloc[i]:
                # Enter at next candle's open
                if i + 1 < len(df):
                    in_trade = True
                    entry_time = df.index[i + 1]
                    entry_price = df['Open'].iloc[i + 1]
                    max_price = entry_price
        else:
            high = df['High'].iloc[i]
            low = df['Low'].iloc[i]
            
            if high > max_price:
                max_price = high
                
            # Check Explosion
            if high >= entry_price * (1 + explosion_target):
                trades.append({
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': df.index[i],
                    'exit_price': entry_price * (1 + explosion_target),
                    'max_price': max_price,
                    'status': 'explosion'
                })
                in_trade = False
            # Check SL (Setup invalidated)
            elif low <= entry_price * (1 - sl_pct):
                trades.append({
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': df.index[i],
                    'exit_price': entry_price * (1 - sl_pct),
                    'max_price': max_price,
                    'status': 'loss'
                })
                in_trade = False
            # Trade still open at the end of data
            elif i == len(df) - 1:
                max_return = ((max_price - entry_price) / entry_price) * 100
                trades.append({
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': df.index[i],
                    'exit_price': df['Close'].iloc[i],
                    'max_price': max_price,
                    'status': 'drift',
                    'max_return_pct': max_return
                })
                in_trade = False
                
    return trades

def main():
    print("BIST hisse listesi çekiliyor...")
    symbols = get_bist_symbols()
    print(f"Toplam {len(symbols)} hisse senedi bulundu.")
    print("Tüm hisseler için 2 yıllık veriler taranıyor ve backtest yapılıyor. Lütfen bekleyin...")
    
    all_trades = []
    processed = 0
    
    for symbol in symbols:
        df = fetch_and_resample(symbol)
        # Ensure we have enough data to calculate the 6-month window and still have days left to trade
        if df is not None and len(df) > 300: 
            trades = analyze_stock(df)
            for t in trades:
                t['symbol'] = symbol
            all_trades.extend(trades)
            
        processed += 1
        if processed % 50 == 0:
            print(f"Taranan Hisse: {processed}/{len(symbols)}")
            time.sleep(0.5) # Slight delay to avoid rate limiting
            
    # Calculate Results
    total_trades = len(all_trades)
    explosions = sum(1 for t in all_trades if t['status'] == 'explosion')
    losses = sum(1 for t in all_trades if t['status'] == 'loss')
    drifts = sum(1 for t in all_trades if t['status'] == 'drift')
    explosion_rate = (explosions / total_trades * 100) if total_trades > 0 else 0
    
    print("\n========================================")
    print("BACKTEST SONUÇLARI (2 Yıl, 4 Saatlik Mum)")
    print("========================================")
    print("Strateji Kuralları:")
    print(" 1) Son 6 ayda fiyat değişimi maksimum %20 (Konsolidasyon / Toplanma)")
    print(" 2) Son 8-10 mumluk süreçte belirgin düşüş trendi")
    print(" 3) Birden fazla uzun alt fitilli mum (Likidite avı / Stop hunt)")
    print(" Çıkış: %50 Kâr (Patlama), %5 Zarar Kes (İptal)")
    print("----------------------------------------")
    print(f"Toplam Üretilen Sinyal / İşlem Sayısı: {total_trades}")
    print(f"Patlama (> %50 Getiri): {explosions}")
    print(f"Stop Oldu (<-%5 Kayıp): {losses}")
    print(f"Süre Doldu (Ne patladı ne stop oldu): {drifts}")
    print(f"Patlama Yakalama Oranı: %{explosion_rate:.2f}")
    
    if total_trades > 0:
        print("\nÖrnek Patlama Gerçekleşen Hisseler:")
        explosion_trades = [t for t in all_trades if t['status'] == 'explosion']
        for t in explosion_trades[:5]: # İlk 5'ini göster
            print(f" - {t['symbol']}: Giriş {t['entry_time']} @ {t['entry_price']:.2f} -> %50 Hedefine Ulaşıldı: {t['exit_time']} @ {t['exit_price']:.2f} (Zirve: {t['max_price']:.2f})")
            
        print("\nSüre Dolup Kalan (Drift) Hisselerin Getirileri:")
        drift_trades = [t for t in all_trades if t['status'] == 'drift']
        for t in drift_trades[:3]:
            print(f" - {t['symbol']}: Gördüğü Zirve Getiri %{t['max_return_pct']:.2f} (Maks Fiyat: {t['max_price']:.2f})")

if __name__ == "__main__":
    main()
