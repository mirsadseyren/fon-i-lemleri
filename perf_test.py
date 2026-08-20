import time
from bist_data_cache import get_cached_data, DEFAULT_TICKERS
from ifvg_engine import run_ifvg_backtest
import concurrent.futures

tickers = DEFAULT_TICKERS[:20]

start = time.time()
for t in tickers:
    df = get_cached_data(t, "15m")
    if df is not None:
        run_ifvg_backtest(df)
print(f"Sequential time: {time.time() - start:.2f}s")

start = time.time()
def process(t):
    df = get_cached_data(t, "15m")
    if df is not None:
        run_ifvg_backtest(df)

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    list(executor.map(process, tickers))
print(f"ThreadPool time: {time.time() - start:.2f}s")

start = time.time()
with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
    list(executor.map(process, tickers))
print(f"ProcessPool time: {time.time() - start:.2f}s")
