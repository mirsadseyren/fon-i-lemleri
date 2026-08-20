import yfinance as yf
import pandas as pd
import os
import time
from datetime import datetime, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Setup session for yfinance to avoid timeouts and retry on failures
YF_SESSION = requests.Session()
retry = Retry(connect=5, read=5, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
YF_SESSION.mount('http://', adapter)
YF_SESSION.mount('https://', adapter)

CACHE_DIR = "bist_data_cache"

# BIST100 Tickers as fallback
DEFAULT_TICKERS = [
    "AEFES.IS", "AGHOL.IS", "AHGAZ.IS", "AKBNK.IS", "AKCNS.IS", "AKFGY.IS", "AKFYE.IS", "AKSA.IS", "AKSEN.IS", "ALARK.IS", 
    "ALBRK.IS", "ALFAS.IS", "ARCLK.IS", "ASELS.IS", "ASTOR.IS", "ASUZU.IS", "AYDEM.IS", "BAGFS.IS", "BERA.IS", "BIENY.IS", 
    "BIMAS.IS", "BIOEN.IS", "BOBET.IS", "BRSAN.IS", "BRYAT.IS", "BUCIM.IS", "CANTE.IS", "CCOLA.IS", "CEMAS.IS", "CIMSA.IS", 
    "CWENE.IS", "DOHOL.IS", "DOAS.IS", "ECILC.IS", "ECZYT.IS", "EGEEN.IS", "EKGYO.IS", "ENERY.IS", "ENJSA.IS", "ENKAI.IS", 
    "EUPWR.IS", "EUREN.IS", "FROTO.IS", "GARAN.IS", "GENIL.IS", "GESAN.IS", "GLYHO.IS", "GUBRF.IS", "GWIND.IS", "HALKB.IS", 
    "HEKTS.IS", "HKTM.IS", "HLGYO.IS", "IMASM.IS", "INVEO.IS", "INVES.IS", "IPEKE.IS", "ISCTR.IS", "ISDMR.IS", "ISGYO.IS", 
    "ISMEN.IS", "IZENR.IS", "KALES.IS", "KARSN.IS", "KAYSE.IS", "KCAER.IS", "KCHOL.IS", "KMPUR.IS", "KONTR.IS", "KONYA.IS", 
    "KOZAA.IS", "KOZAL.IS", "KRDMD.IS", "KZBGY.IS", "MAVI.IS", "MGROS.IS", "MIATK.IS", "ODAS.IS", "OTKAR.IS", "OYAKC.IS", 
    "PENTA.IS", "PETKM.IS", "PGSUS.IS", "PNLSN.IS", "QUAGR.IS", "SAHOL.IS", "SASA.IS", "SELEC.IS", "SISE.IS", "SMRTG.IS", 
    "SOKM.IS", "TAVHL.IS", "TCELL.IS", "THYAO.IS", "TKFEN.IS", "TOASO.IS", "TTKOM.IS", "TTRAK.IS", "TUKAS.IS", "TUPRS.IS", 
    "ULKER.IS", "VAKBN.IS", "VESBE.IS", "VESTL.IS", "YEOTK.IS", "YKBNK.IS", "YYLGD.IS", "ZOREN.IS"
]

def get_tickers_from_file(filepath="stox.txt"):
    """Reads tickers from a file and appends .IS suffix."""
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Using default BIST100 tickers.")
        return DEFAULT_TICKERS
        
    tickers = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                ticker = line.strip()
                if ticker:
                    if not ticker.endswith(".IS"):
                        ticker = f"{ticker}.IS"
                    tickers.append(ticker)
        print(f"Loaded {len(tickers)} tickers from {filepath}")
        return tickers
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return DEFAULT_TICKERS

TIMEFRAME_MAP = {
    "1m": {"period": "7d", "interval": "1m"},     # max 7 days
    "5m": {"period": "60d", "interval": "5m"},    # max 60 days
    "15m": {"period": "60d", "interval": "15m"},  # max 60 days
    "1h": {"period": "730d", "interval": "1h"},   # max 730 days (often capped at 60d for some stocks)
    "1d": {"period": "10y", "interval": "1d"}     # Max long term
}

def ensure_cache_dirs():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    for tf in TIMEFRAME_MAP.keys():
        tf_dir = os.path.join(CACHE_DIR, tf)
        if not os.path.exists(tf_dir):
            os.makedirs(tf_dir)

def get_cached_data(ticker, timeframe):
    """Load data from cache if exists."""
    file_path = os.path.join(CACHE_DIR, timeframe, f"{ticker}.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        return df
    return None

def fetch_and_cache_data(ticker, timeframe, force_update=False, append=False):
    """Fetch data from yfinance and save to cache. Optionally append to existing."""
    file_path = os.path.join(CACHE_DIR, timeframe, f"{ticker}.csv")
    
    existing_df = None
    # Check if we should use cache
    if os.path.exists(file_path):
        existing_df = get_cached_data(ticker, timeframe)
        if not force_update:
            return existing_df
            
    tf_settings = TIMEFRAME_MAP[timeframe]
    
    try:
        # Rate limit protection
        time.sleep(0.05) 
        
        # If appending and we have old data, just fetch the missing days
        if append and existing_df is not None and not existing_df.empty:
            last_date = existing_df.index[-1]
            start_date = (last_date - timedelta(days=1)).strftime('%Y-%m-%d')
            df = yf.download(ticker, start=start_date, interval=tf_settings["interval"], progress=False, session=YF_SESSION)
        else:
            df = yf.download(ticker, period=tf_settings["period"], interval=tf_settings["interval"], progress=False, session=YF_SESSION)
        
        if df is not None and not df.empty:
            # Drop multi-index columns if they exist (yfinance sometimes does this)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            
            # Ensure index is timezone naive or consistent
            if df.index.tz is not None:
                df.index = df.index.tz_convert("Europe/Istanbul").tz_localize(None)
                
            if append and existing_df is not None and not existing_df.empty:
                # Combine old and new data
                combined_df = pd.concat([existing_df, df])
                # Remove duplicate dates, keeping the most recently downloaded version
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                combined_df.sort_index(inplace=True)
                df = combined_df
                
            df.to_csv(file_path)
            return df
        else:
            return existing_df if (append and existing_df is not None) else None
    except Exception as e:
        print(f"Error fetching {ticker} on {timeframe}: {e}")
        return existing_df if (append and existing_df is not None) else None

def pre_cache_all(tickers=DEFAULT_TICKERS, timeframes=["15m", "1h", "1d"]):
    """Utility to run and cache everything before backtesting."""
    ensure_cache_dirs()
    print(f"Pre-caching data for {len(tickers)} tickers across {timeframes}...")
    for ticker in tickers:
        for tf in timeframes:
            fetch_and_cache_data(ticker, tf, force_update=True)
    print("Pre-caching complete.")

if __name__ == "__main__":
    # Test pre-caching for a couple of stocks
    pre_cache_all(tickers=["THYAO.IS", "GARAN.IS"], timeframes=["15m", "1h"])
