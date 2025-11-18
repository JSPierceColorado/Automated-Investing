import os
import json
import sys
import math
from datetime import datetime, timedelta, timezone

import requests
import gspread
from google.oauth2.service_account import Credentials

from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed  # use IEX instead of SIP


# -----------------------------
# Helpers / config
# -----------------------------

def get_env(name: str, default=None, required: bool = False):
    value = os.getenv(name, default)
    if required and (value is None or value.strip() == ""):
        print(f"Missing required environment variable: {name}", file=sys.stderr)
        sys.exit(1)
    return value


def parse_symbol_list(env_var: str):
    raw = os.getenv(env_var, "")
    return [s.strip() for s in raw.split(",") if s.strip()]


def get_google_client():
    """Authorize gspread using a service account JSON from env."""
    json_str = get_env("GOOGLE_SERVICE_ACCOUNT_JSON", required=True)
    info = json.loads(json_str)
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    return gspread.authorize(creds)


def open_target_worksheet(gc):
    sheet_name = get_env("GOOGLE_SHEETS_SPREADSHEET_NAME", "Active-Investing")
    tab_name = get_env("GOOGLE_SHEETS_WORKSHEET_NAME", "Automation-Screener")

    sh = gc.open(sheet_name)

    try:
        ws = sh.worksheet(tab_name)
    except gspread.exceptions.WorksheetNotFound:
        # create if missing
        ws = sh.add_worksheet(title=tab_name, rows=1000, cols=30)

    return ws


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


# -----------------------------
# Indicator logic (Pine-style)
# -----------------------------

def compute_atr_rma(highs, lows, closes, length=10):
    """
    Pine's ta.atr = RMA(TrueRange, length).
    Returns list of ATR values aligned with input (None where not yet defined).
    """
    n = len(closes)
    if n == 0:
        return []

    tr = [0.0] * n
    tr[0] = highs[0] - lows[0]

    for i in range(1, n):
        tr_hl = highs[i] - lows[i]
        tr_hc = abs(highs[i] - closes[i - 1])
        tr_lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(tr_hl, tr_hc, tr_lc)

    atr = [None] * n
    if n < length:
        return atr

    # seed RMA with SMA of first `length` TR values
    seed = sum(tr[0:length]) / length
    atr[length - 1] = seed

    for i in range(length, n):
        atr[i] = (atr[i - 1] * (length - 1) + tr[i]) / length

    return atr


def compute_supertrend_like_trailing_stop(closes, atr, atr_mult=1.2):
    """
    SuperTrend-style trailing stop:

    entryLoss = atr * atr_mult
    if price > prevStop and pricePrev > prevStop:  # long regime
        stop = max(prevStop, price - entryLoss)
    elif price < prevStop and pricePrev < prevStop:  # short regime
        stop = min(prevStop, price + entryLoss)
    elif price > prevStop:  # flip to long
        stop = price - entryLoss
    else:                   # flip to short
        stop = price + entryLoss
    """
    n = len(closes)
    stop = [None] * n
    if n == 0:
        return stop

    entry_loss = [a * atr_mult if a is not None else None for a in atr]

    # find first index where ATR is defined
    first_idx = None
    for i, el in enumerate(entry_loss):
        if el is not None and not math.isnan(el):
            first_idx = i
            break

    if first_idx is None:
        return stop

    stop[first_idx] = closes[first_idx] - entry_loss[first_idx]

    for i in range(first_idx + 1, n):
        if entry_loss[i] is None:
            continue
        prev_stop = stop[i - 1]
        price = closes[i]
        price_prev = closes[i - 1]

        if prev_stop is None:
            stop[i] = price - entry_loss[i]
        else:
            if price > prev_stop and price_prev > prev_stop:
                # Long regime: trail up
                stop[i] = max(prev_stop, price - entry_loss[i])
            elif price < prev_stop and price_prev < prev_stop:
                # Short regime: trail down
                stop[i] = min(prev_stop, price + entry_loss[i])
            elif price > prev_stop:
                # Flip to long
                stop[i] = price - entry_loss[i]
            else:
                # Flip to short
                stop[i] = price + entry_loss[i]

    return stop


def compute_sma(values, length):
    """
    Pine ta.sma equivalent for a fixed window.
    Returns list with None until enough data is present.
    (Kept for possible future use; MA for latest bar is
     now computed directly in build_metrics_from_ohlcv.)
    """
    n = len(values)
    sma = [None] * n
    if n < length:
        return sma

    window_sum = sum(values[0:length])
    sma[length - 1] = window_sum / length

    for i in range(length, n):
        window_sum += values[i] - values[i - length]
        sma[i] = window_sum / length

    return sma


def compute_buy_signals(closes, trailing_stop):
    """
    UT-style long buy:

    crossover = close crosses above trailing stop
    buy = (close > stop) and crossover
    """
    n = len(closes)
    buys = [False] * n
    for i in range(1, n):
        s = trailing_stop[i]
        s_prev = trailing_stop[i - 1]
        if s is None or s_prev is None:
            continue
        crossover = (closes[i - 1] <= s_prev) and (closes[i] > s)
        buys[i] = (closes[i] > s) and crossover
    return buys


def build_metrics_from_ohlcv(times, opens, highs, lows, closes, volumes, is_crypto: bool):
    """
    Given full OHLCV arrays, compute all metrics needed for your strategy on the latest bar.
    For MA:
      - Crypto: target length = 240
      - Stocks: target length = 825
      - Younger assets: use effective_len = min(target_len, num_bars)
    """
    n = len(closes)
    if n == 0:
        return {"status": "NO_DATA"}

    # 1) ATR(10) and trailing stop
    atr = compute_atr_rma(highs, lows, closes, length=10)
    trailing = compute_supertrend_like_trailing_stop(closes, atr, atr_mult=1.2)
    buys = compute_buy_signals(closes, trailing)

    idx = n - 1
    last_time = times[idx]
    last_open = opens[idx]
    last_high = highs[idx]
    last_low = lows[idx]
    last_close = closes[idx]
    last_volume = volumes[idx]

    atr_last = atr[idx]
    trailing_last = trailing[idx]
    buy_last = buys[idx]

    # 2) Long MA: 240 (crypto) vs 825 (stocks), but allow shorter for young assets
    target_ma_len = 240 if is_crypto else 825
    effective_ma_len = min(target_ma_len, n)

    # Always compute an MA on the latest bar using as many bars as we have (up to target_ma_len)
    window_closes = closes[n - effective_ma_len : n]
    sma_last = sum(window_closes) / effective_ma_len if effective_ma_len > 0 else None

    price_minus_ma = None
    pct_diff = None
    above_zone = None

    if sma_last is not None and sma_last != 0:
        price_minus_ma = last_close - sma_last
        pct_diff = (price_minus_ma / sma_last) * 100.0
        above_zone = price_minus_ma >= 0

    long_regime = None
    if trailing_last is not None:
        long_regime = last_close > trailing_last

    # "Enough history" for status = OK is driven by ATR (10 bars),
    # but MA-related columns are always filled as long as n >= 1.
    had_enough_history = atr_last is not None

    return {
        "status": "OK" if had_enough_history else "INSUFFICIENT_HISTORY",
        "last_time": last_time,
        "last_open": last_open,
        "last_high": last_high,
        "last_low": last_low,
        "last_close": last_close,
        "last_volume": last_volume,
        "atr_10": atr_last,
        "atr_entry_loss": atr_last * 1.2 if atr_last is not None else None,
        "trailing_stop": trailing_last,
        "long_regime": long_regime,
        "buy_signal": buy_last,
        # actual number of bars used in the MA
        "ma_length": effective_ma_len,
        "long_sma": sma_last,
        "price_minus_ma": price_minus_ma,
        "pct_diff": pct_diff,
        "above_buy_zone": above_zone,
        "num_bars": n,
        "had_enough_history": had_enough_history,
    }


# -----------------------------
# Alpaca data
# -----------------------------

def get_alpaca_clients():
    api_key = get_env("ALPACA_API_KEY", required=True)
    api_secret = get_env("ALPACA_API_SECRET", required=True)

    stock_client = StockHistoricalDataClient(api_key=api_key, secret_key=api_secret)
    # crypto data does not need keys, but we can still pass them
    crypto_client = CryptoHistoricalDataClient()

    return stock_client, crypto_client


def fetch_alpaca_bars(symbol: str, is_crypto: bool, stock_client, crypto_client, max_days: int):
    """
    Fetch daily bars from Alpaca (stocks vs crypto).
    Returns aligned lists: times, opens, highs, lows, closes, volumes
    """
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=max_days)

    if is_crypto:
        req = CryptoBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Day,
            start=start,
            end=now,
            limit=max_days,
        )
        bars = crypto_client.get_crypto_bars(req)
        # alpaca-py: bars.data[symbol] is list[Bar]
        series = bars.data[symbol] if hasattr(bars, "data") else bars[symbol]
    else:
        req = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Day,
            start=start,
            end=now,
            limit=max_days,
            feed=DataFeed.IEX,  # force IEX, avoid SIP error
        )
        bars = stock_client.get_stock_bars(req)
        series = bars.data[symbol] if hasattr(bars, "data") else bars[symbol]

    times = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []

    for bar in series:
        # Bar objects have attributes, not dict keys
        t = getattr(bar, "timestamp", None)
        o = safe_float(getattr(bar, "open", None))
        h = safe_float(getattr(bar, "high", None))
        l = safe_float(getattr(bar, "low", None))
        c = safe_float(getattr(bar, "close", None))
        v = safe_float(getattr(bar, "volume", 0.0))

        times.append(str(t))
        opens.append(o)
        highs.append(h)
        lows.append(l)
        closes.append(c)
        volumes.append(v)

    return times, opens, highs, lows, closes, volumes


# -----------------------------
# Kraken data
# -----------------------------

def fetch_kraken_ohlc(pair: str, max_days: int):
    """
    Fetch daily OHLC from Kraken REST API.
    Kraken's OHLC interval 1440 = 1 day.
    The endpoint returns up to 720 entries; we just slice the last `max_days`.
    """
    url = "https://api.kraken.com/0/public/OHLC"
    params = {
        "pair": pair,
        "interval": 1440,
    }

    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if data.get("error"):
        raise RuntimeError(f"Kraken error for {pair}: {data['error']}")

    result = data["result"]
    # result keys: e.g. "XXBTZUSD" and "last"
    series_key = None
    for k in result.keys():
        if k != "last":
            series_key = k
            break

    if series_key is None:
        raise RuntimeError(f"No OHLC series in Kraken result for {pair}")

    raw_series = result[series_key]
    if len(raw_series) > max_days:
        raw_series = raw_series[-max_days:]

    times = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []

    # each entry: [time, open, high, low, close, vwap, volume, count]
    for row in raw_series:
        ts = datetime.fromtimestamp(row[0], tz=timezone.utc)
        o = safe_float(row[1])
        h = safe_float(row[2])
        l = safe_float(row[3])
        c = safe_float(row[4])
        v = safe_float(row[6])

        times.append(ts.isoformat())
        opens.append(o)
        highs.append(h)
        lows.append(l)
        closes.append(c)
        volumes.append(v)

    return times, opens, highs, lows, closes, volumes


# -----------------------------
# Google Sheets writing
# -----------------------------

def metrics_to_row(broker: str, symbol: str, is_crypto: bool, metrics: dict):
    """
    Flatten metrics dict into a single row for Google Sheets.
    """
    return [
        metrics.get("status", ""),
        broker,
        symbol,
        "TRUE" if is_crypto else "FALSE",
        metrics.get("last_time", ""),
        metrics.get("last_open", ""),
        metrics.get("last_high", ""),
        metrics.get("last_low", ""),
        metrics.get("last_close", ""),
        metrics.get("last_volume", ""),
        metrics.get("atr_10", ""),
        metrics.get("atr_entry_loss", ""),
        metrics.get("trailing_stop", ""),
        "TRUE" if metrics.get("long_regime") else "FALSE" if metrics.get("long_regime") is not None else "",
        "TRUE" if metrics.get("buy_signal") else "FALSE" if metrics.get("buy_signal") is not None else "",
        metrics.get("ma_length", ""),   # effective MA window size actually used
        metrics.get("long_sma", ""),
        metrics.get("price_minus_ma", ""),
        metrics.get("pct_diff", ""),
        "TRUE" if metrics.get("above_buy_zone") else "FALSE" if metrics.get("above_buy_zone") is not None else "",
        metrics.get("num_bars", ""),
        "TRUE" if metrics.get("had_enough_history") else "FALSE" if metrics.get("had_enough_history") is not None else "",
    ]


def write_rows_to_sheet(ws, rows):
    header = [
        "status",
        "broker",
        "symbol",
        "is_crypto",
        "last_bar_time",
        "last_open",
        "last_high",
        "last_low",
        "last_close",
        "last_volume",
        "atr_10",
        "atr_entry_loss",
        "atr_trailing_stop",
        "long_regime",
        "buy_signal",
        "ma_length",
        "long_sma",
        "price_minus_ma",
        "pct_diff",
        "above_buy_zone",
        "num_bars",
        "had_enough_history",
    ]

    data = [header] + rows

    # 🔧 IMPORTANT CHANGE:
    # Clear ONLY columns A:V so that W onward (external functions / formulas)
    # remain untouched.
    ws.batch_clear(['A:V'])

    # Now write our data starting at A1; this only fills columns A–V.
    ws.update("A1", data, value_input_option="RAW")


# -----------------------------
# Main orchestration
# -----------------------------

def main():
    alpaca_symbols = parse_symbol_list("ALPACA_WHITELIST")
    kraken_symbols = parse_symbol_list("KRAKEN_WHITELIST")

    if not alpaca_symbols and not kraken_symbols:
        print("No symbols configured. Set ALPACA_WHITELIST and/or KRAKEN_WHITELIST.", file=sys.stderr)
        sys.exit(1)

    alpaca_crypto_symbols = set(parse_symbol_list("ALPACA_CRYPTO_SYMBOLS"))

    # Clients
    gc = get_google_client()
    ws = open_target_worksheet(gc)

    stock_client, crypto_client = get_alpaca_clients()

    rows = []

    # Alpaca symbols
    for sym in alpaca_symbols:
        is_crypto = sym in alpaca_crypto_symbols
        try:
            max_days = 300 if is_crypto else 1200  # enough to warm ATR + up to 825 bars for stocks
            times, o, h, l, c, v = fetch_alpaca_bars(sym, is_crypto, stock_client, crypto_client, max_days=max_days)
            metrics = build_metrics_from_ohlcv(times, o, h, l, c, v, is_crypto=is_crypto)
        except Exception as e:
            metrics = {"status": f"ERROR: {e}"}
        row = metrics_to_row("ALPACA", sym, is_crypto, metrics)
        rows.append(row)

    # Kraken symbols (all crypto)
    for pair in kraken_symbols:
        try:
            times, o, h, l, c, v = fetch_kraken_ohlc(pair, max_days=300)
            metrics = build_metrics_from_ohlcv(times, o, h, l, c, v, is_crypto=True)
        except Exception as e:
            metrics = {"status": f"ERROR: {e}"}
            row = metrics_to_row("KRAKEN", pair, True, metrics)
        else:
            row = metrics_to_row("KRAKEN", pair, True, metrics)
        rows.append(row)

    write_rows_to_sheet(ws, rows)
    print(f"Wrote {len(rows)} rows to sheet.")


if __name__ == "__main__":
    main()
