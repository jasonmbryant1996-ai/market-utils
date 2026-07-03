"""
features.py
===========
Complete feature engineering pipeline for live BTC regime inference.
Fetches the last N 5-min bars from Binance, computes all 35 features,
and returns a scaled numpy array ready for model input.

All computations are CAUSAL — no future data leaks into the features.
"""

import time
import numpy as np
import pandas as pd
import requests
import yfinance as yf


# ── Constants ──────────────────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    "open_log_return", "high_log_return", "low_log_return",
    "close_log_return", "volume_log_return",
    "spx_log_return", "dxy_log_return", "eth_btc_log_return",
    "SMA_Distance_10", "SMA_Distance_30", "SMA_Distance_60",
    "volatility_10",
    "time_of_day_sin", "time_of_day_cos", "day_of_week_sin", "day_of_week_cos",
    "RSI_5m", "RSI_1h", "RSI_4h", "RSI_1d",
    "RSI_5m_SMA", "RSI_1h_SMA", "RSI_4h_SMA", "RSI_1d_SMA",
    "dist_to_1d_bullish_ifvg", "dist_to_1d_bearish_ifvg",
    "dist_to_4h_bullish_ifvg", "dist_to_4h_bearish_ifvg",
    "dist_to_1h_bullish_ifvg", "dist_to_1h_bearish_ifvg",
    "dist_to_15m_bullish_ifvg", "dist_to_15m_bearish_ifvg",
    "dist_to_5m_bullish_ifvg", "dist_to_5m_bearish_ifvg",
    "taker_buy_ratio",
]
assert len(FEATURE_COLUMNS) == 35


# ── Data fetching ──────────────────────────────────────────────────────────────

def fetch_btc_bars(n: int = 8700, symbol: str = "BTCUSDT") -> pd.DataFrame:
    """
    Fetch the last n closed 5-min bars from Binance.
    Uses spot klines endpoint (no auth required).
    n=8700 ≈ 30 days, which is enough for all feature warmups.
    """
    BASE = "https://api.binance.com/api/v3/klines"
    end_ms   = int(time.time() * 1000)
    start_ms = end_ms - n * 5 * 60 * 1000   # n bars × 5 min × 60 s × 1000 ms

    all_rows = []
    ts = start_ms
    while ts < end_ms:
        last_exc = None
        for attempt in range(4):
            try:
                r = requests.get(
                    BASE,
                    params={"symbol": symbol, "interval": "5m",
                            "startTime": ts, "endTime": end_ms, "limit": 1000},
                    timeout=15,
                )
                r.raise_for_status()
                data = r.json()
                break
            except Exception as exc:
                last_exc = exc
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"Binance klines fetch failed after retries: {last_exc}")

        if not data:
            break
        all_rows.extend(data)
        ts = int(data[-1][0]) + 1
        if len(data) < 1000:
            break
        time.sleep(0.05)

    COLS = ["ts","open","high","low","close","volume",
            "cts","qv","n_trades","tbv","tbqv","x"]
    df = pd.DataFrame(all_rows, columns=COLS)
    df.index = pd.to_datetime(df["ts"].astype(int), unit="ms", utc=True)
    df.index.name = "datetime"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    for c in ["open","high","low","close","volume","tbv"]:
        df[c] = df[c].astype(float)
    df["taker_buy_ratio"] = (df["tbv"] / df["volume"].replace(0, 1e-8)).clip(0, 1)

    # Drop the currently open (incomplete) bar — last row
    df = df.iloc[:-1]
    print(f"  Fetched {len(df):,} closed bars  "
          f"(latest: {df.index[-1].strftime('%Y-%m-%d %H:%M')} UTC)")
    return df[["open","high","low","close","volume","taker_buy_ratio"]]


def fetch_macro(start_dt: str, end_dt: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Download SPX, DXY, and ETH/BTC daily log-returns.
    All are shifted 1 day (causal: yesterday's return is today's feature)
    and forward-filled over weekends/holidays.

    Returns three date-indexed Series (index = date objects).
    """
    def _safe_download(ticker, **kw):
        """Download with yfinance, return None on failure."""
        try:
            df = yf.download(ticker, progress=False, auto_adjust=True, **kw)
            if df.empty:
                return None
            return df
        except Exception:
            return None

    def _to_daily_map(df_raw) -> pd.Series:
        if df_raw is None:
            return pd.Series(dtype=float)
        close = df_raw["Close"].squeeze()
        lr    = np.log(close / close.shift(1))
        # Shift 1 day (causal) then forward-fill gaps
        full_idx = pd.date_range(start=start_dt, end=end_dt, freq="D")
        s = lr.shift(1).reindex(full_idx).ffill()
        s.index = s.index.date
        return s

    spx     = _safe_download("^GSPC",    start=start_dt, end=end_dt)
    dxy     = _safe_download("DX-Y.NYB", start=start_dt, end=end_dt)
    eth_usd = _safe_download("ETH-USD",  start=start_dt, end=end_dt)
    btc_usd = _safe_download("BTC-USD",  start=start_dt, end=end_dt)

    spx_map = _to_daily_map(spx)
    dxy_map = _to_daily_map(dxy)

    # ETH/BTC ratio log-return
    if eth_usd is not None and btc_usd is not None:
        ratio = eth_usd["Close"].squeeze() / btc_usd["Close"].squeeze()
        ratio = ratio.reindex(eth_usd.index).ffill()
        lr    = np.log(ratio / ratio.shift(1))
        full_idx = pd.date_range(start=start_dt, end=end_dt, freq="D")
        eth_btc_map = lr.shift(1).reindex(full_idx).ffill()
        eth_btc_map.index = eth_btc_map.index.date
    else:
        eth_btc_map = pd.Series(dtype=float)

    return spx_map, dxy_map, eth_btc_map


# ── Indicator helpers ──────────────────────────────────────────────────────────

def _rolling_rsi(series: pd.Series, window: int) -> pd.Series:
    delta    = series.diff()
    gain     = delta.where(delta > 0, 0.0)
    loss     = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, adjust=False).mean()
    return 100.0 - 100.0 / (1.0 + avg_gain / (avg_loss + 1e-8))


def _compute_ifvgs(df_tf: pd.DataFrame) -> pd.DataFrame:
    """
    Pure-Python causal IFVG state machine.
    Returns bull_ifvg_dist and bear_ifvg_dist aligned to df_tf.index.
    """
    n = len(df_tf)
    H, L, C = df_tf["high"].values, df_tf["low"].values, df_tf["close"].values
    active_bull_fvgs, active_bear_fvgs   = [], []
    active_bull_ifvgs, active_bear_ifvgs = [], []
    bull_dist = np.zeros(n)
    bear_dist = np.zeros(n)

    for i in range(2, n):
        c = C[i]

        # Identify new FVGs
        if L[i] > H[i - 2]:
            active_bull_fvgs.append((H[i - 2], L[i]))
        if H[i] < L[i - 2]:
            active_bear_fvgs.append((H[i], L[i - 2]))

        # FVG → IFVG inversions
        for fvg in list(active_bull_fvgs):
            if c < fvg[0]:
                active_bull_fvgs.remove(fvg)
                active_bear_ifvgs.append(fvg)
        for fvg in list(active_bear_fvgs):
            if c > fvg[1]:
                active_bear_fvgs.remove(fvg)
                active_bull_ifvgs.append(fvg)

        # IFVG mitigations
        for ifvg in list(active_bull_ifvgs):
            if c < ifvg[0]:
                active_bull_ifvgs.remove(ifvg)
        for ifvg in list(active_bear_ifvgs):
            if c > ifvg[1]:
                active_bear_ifvgs.remove(ifvg)

        if active_bull_ifvgs:
            top = max(f[1] for f in active_bull_ifvgs)
            bull_dist[i] = (c - top) / (top + 1e-10)
        if active_bear_ifvgs:
            bot = min(f[0] for f in active_bear_ifvgs)
            bear_dist[i] = (c - bot) / (bot + 1e-10)

    return pd.DataFrame(
        {"bull_ifvg_dist": bull_dist, "bear_ifvg_dist": bear_dist},
        index=df_tf.index,
    )


# ── Main feature builder ───────────────────────────────────────────────────────

def build_features(
    df_raw:      pd.DataFrame,
    spx_map:     pd.Series,
    dxy_map:     pd.Series,
    eth_btc_map: pd.Series,
) -> pd.DataFrame:
    """
    Takes the raw OHLCV+taker_buy_ratio DataFrame and produces all 35
    FEATURE_COLUMNS. Same logic as Phase 2 in the training notebook.

    Parameters
    ----------
    df_raw      : OHLCV + taker_buy_ratio, 5-min UTC DatetimeIndex
    spx_map     : date → SPX daily log-return (causal, shifted 1 day)
    dxy_map     : date → DXY daily log-return (causal, shifted 1 day)
    eth_btc_map : date → ETH/BTC ratio log-return (causal, shifted 1 day)
    """
    df = df_raw.copy()

    # ── 1. OHLCV log-returns ─────────────────────────────────────────────────
    for col, src in [
        ("open_log_return",   "open"),
        ("high_log_return",   "high"),
        ("low_log_return",    "low"),
        ("close_log_return",  "close"),
    ]:
        df[col] = np.log(df[src] / df[src].shift(1))
    df["volume_log_return"] = np.log(
        df["volume"].replace(0, 1e-8) / df["volume"].shift(1).replace(0, 1e-8)
    )

    # ── 2. SMA distances ─────────────────────────────────────────────────────
    for win, col in [(10, "SMA_Distance_10"), (30, "SMA_Distance_30"), (60, "SMA_Distance_60")]:
        sma      = df["close"].rolling(win).mean()
        df[col]  = (df["close"] - sma) / sma

    # ── 3. Volatility ────────────────────────────────────────────────────────
    df["volatility_10"] = df["close_log_return"].rolling(10).std()

    # ── 4. Cyclic time ───────────────────────────────────────────────────────
    frac_h = df.index.hour + df.index.minute / 60.0
    df["time_of_day_sin"] = np.sin(2 * np.pi * frac_h / 24.0)
    df["time_of_day_cos"] = np.cos(2 * np.pi * frac_h / 24.0)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7.0)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7.0)

    # ── 5. Multi-timeframe RSIs ──────────────────────────────────────────────
    df["RSI_5m"] = _rolling_rsi(df["close"], 14)
    df["RSI_1h"] = _rolling_rsi(df["close"], 168)
    df["RSI_4h"] = _rolling_rsi(df["close"], 672)
    df["RSI_1d"] = _rolling_rsi(df["close"], 4032)
    df["RSI_5m_SMA"] = df["RSI_5m"].rolling(14).mean()
    df["RSI_1h_SMA"] = df["RSI_1h"].rolling(168).mean()
    df["RSI_4h_SMA"] = df["RSI_4h"].rolling(672).mean()
    df["RSI_1d_SMA"] = df["RSI_1d"].rolling(4032).mean()

    # ── 6. Multi-timeframe IFVGs ─────────────────────────────────────────────
    _ohlc = {"open": "first", "high": "max", "low": "min", "close": "last"}

    i1d = _compute_ifvgs(df.resample("1D").agg(_ohlc).dropna()).shift(1)
    i1d.index = i1d.index.date
    i4h = _compute_ifvgs(df.resample("4h").agg(_ohlc).dropna()).shift(1)
    i1h = _compute_ifvgs(df.resample("1h").agg(_ohlc).dropna()).shift(1)
    i15 = _compute_ifvgs(df.resample("15min").agg(_ohlc).dropna()).shift(1)
    i5m = _compute_ifvgs(df).shift(1)

    df["_d"]  = df.index.date
    df["_4h"] = df.index.floor("4h")
    df["_1h"] = df.index.floor("1h")
    df["_15"] = df.index.floor("15min")

    df["dist_to_1d_bullish_ifvg"]  = df["_d"].map(i1d["bull_ifvg_dist"])
    df["dist_to_1d_bearish_ifvg"]  = df["_d"].map(i1d["bear_ifvg_dist"])
    df["dist_to_4h_bullish_ifvg"]  = df["_4h"].map(i4h["bull_ifvg_dist"])
    df["dist_to_4h_bearish_ifvg"]  = df["_4h"].map(i4h["bear_ifvg_dist"])
    df["dist_to_1h_bullish_ifvg"]  = df["_1h"].map(i1h["bull_ifvg_dist"])
    df["dist_to_1h_bearish_ifvg"]  = df["_1h"].map(i1h["bear_ifvg_dist"])
    df["dist_to_15m_bullish_ifvg"] = df["_15"].map(i15["bull_ifvg_dist"])
    df["dist_to_15m_bearish_ifvg"] = df["_15"].map(i15["bear_ifvg_dist"])
    df["dist_to_5m_bullish_ifvg"]  = i5m["bull_ifvg_dist"]
    df["dist_to_5m_bearish_ifvg"]  = i5m["bear_ifvg_dist"]
    df.drop(columns=["_d", "_4h", "_1h", "_15"], inplace=True)

    # ── 7. Macro proxies ─────────────────────────────────────────────────────
    dates = pd.Series(df.index.date, index=df.index)
    df["spx_log_return"]     = dates.map(spx_map).fillna(0.0)
    df["dxy_log_return"]     = dates.map(dxy_map).fillna(0.0)
    df["eth_btc_log_return"] = dates.map(eth_btc_map).fillna(0.0)

    df.dropna(subset=FEATURE_COLUMNS, inplace=True)
    return df