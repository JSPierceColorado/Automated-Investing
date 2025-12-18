import os
import json
import sys
import math
import time
import logging
from datetime import datetime, timedelta, timezone

import gspread
from google.oauth2.service_account import Credentials

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed, Adjustment

# Used only for market clock check
from alpaca.trading.client import TradingClient


logger = logging.getLogger(__name__)


# -----------------------------
# Logging
# -----------------------------
def configure_logging():
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.setLevel(level)
    logger.info("Logging initialized at level %s", logging.getLevelName(level))


# -----------------------------
# Env / Helpers
# -----------------------------
def get_env(name: str, default=None, required: bool = False):
    value = os.getenv(name, default)
    if required and (value is None or str(value).strip() == ""):
        logger.error("Missing required environment variable: %s", name)
        sys.exit(1)
    return value


def get_env_int(name: str, default: int, *, min_value: int | None = None, max_value: int | None = None) -> int:
    raw = os.getenv(name, "")
    raw = str(raw).strip()
    if raw == "":
        val = default
    else:
        try:
            val = int(raw)
        except ValueError:
            logger.warning("Invalid %s=%r; using default %s", name, raw, default)
            val = default

    if min_value is not None and val < min_value:
        logger.warning("%s=%s < %s; clamping", name, val, min_value)
        val = min_value
    if max_value is not None and val > max_value:
        logger.warning("%s=%s > %s; clamping", name, val, max_value)
        val = max_value
    return val


def truthy(x) -> bool:
    return str(x).strip().lower() in ("1", "true", "yes", "y", "on")


def parse_symbol_list(env_var: str):
    raw = os.getenv(env_var, "")
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def is_bad_num(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def sheet_val(x):
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return x


def col_letter(n: int) -> str:
    """1 -> A, 26 -> Z, 27 -> AA"""
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def mean_ignore_none(vals):
    xs = [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return (sum(xs) / len(xs)) if xs else None


def drop_bad_rows(times, o, h, l, c, v, vwap, tc):
    """
    Drop rows where OHLC is missing.
    Keep volume/vwap/trade_count as None if missing (do NOT coerce to 0).
    """
    nt, no, nh, nl, nc, nv, nvw, ntc = [], [], [], [], [], [], [], []
    for t, oo, hh, ll, cc, vv, vw, tcc in zip(times, o, h, l, c, v, vwap, tc):
        if any(is_bad_num(x) for x in (oo, hh, ll, cc)):
            continue
        nt.append(t)
        no.append(oo)
        nh.append(hh)
        nl.append(ll)
        nc.append(cc)
        nv.append(None if is_bad_num(vv) else vv)
        nvw.append(None if is_bad_num(vw) else vw)
        ntc.append(None if is_bad_num(tcc) else tcc)
    return nt, no, nh, nl, nc, nv, nvw, ntc


def is_blank_or_zero(v) -> bool:
    # blank string
    if isinstance(v, str):
        return v.strip() == ""

    # None / NaN
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True

    # numeric 0
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return v == 0

    return False


def chunked(lst, size: int):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


# -----------------------------
# Google Sheets
# -----------------------------
def get_google_client():
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
    tab_name = get_env("GOOGLE_SHEETS_WORKSHEET_NAME", "Alpaca-Stocks")

    sh = gc.open(sheet_name)
    try:
        return sh.worksheet(tab_name)
    except gspread.exceptions.WorksheetNotFound:
        logger.warning("Worksheet '%s' not found; creating it", tab_name)
        return sh.add_worksheet(title=tab_name, rows=1000, cols=60)


def write_rows_to_sheet(ws, header, rows):
    data = [header] + rows
    last_col = col_letter(len(header))
    clear_range = f"A:{last_col}"
    logger.info("Clearing only %s then writing %d rows into %s", clear_range, len(rows), ws.title)

    ws.batch_clear([clear_range])
    ws.update("A1", data, value_input_option="RAW")
    logger.info("Sheet update complete.")


# -----------------------------
# Alpaca clients / market open check
# -----------------------------
def get_alpaca_data_client():
    api_key = get_env("ALPACA_API_KEY", required=True)
    api_secret = get_env("ALPACA_API_SECRET", required=True)
    return StockHistoricalDataClient(api_key=api_key, secret_key=api_secret)


def get_market_clock():
    api_key = get_env("ALPACA_API_KEY", required=True)
    api_secret = get_env("ALPACA_API_SECRET", required=True)
    paper = truthy(get_env("ALPACA_PAPER", "true"))
    tc = TradingClient(api_key=api_key, secret_key=api_secret, paper=paper)
    return tc.get_clock()


def market_is_open() -> bool:
    """
    Returns True only if we can confirm the market is open.
    If clock check fails, we treat it as closed (fail-closed).
    """
    if not truthy(get_env("RUN_ONLY_WHEN_MARKET_OPEN", "true")):
        return True

    try:
        clock = get_market_clock()
        is_open = bool(getattr(clock, "is_open", False))
        if not is_open:
            logger.info("Market CLOSED. next_open=%s next_close=%s",
                        getattr(clock, "next_open", None),
                        getattr(clock, "next_close", None))
        else:
            logger.info("Market OPEN. next_close=%s", getattr(clock, "next_close", None))
        return is_open
    except Exception:
        logger.exception("Failed to check market clock (treating as closed).")
        return False


# -----------------------------
# Batched Alpaca fetches (speed-up)
# -----------------------------
def fetch_stock_bars_batch(stock_client, symbols, lookback_days: int, bars_limit: int):
    """
    Fetch daily bars for MANY symbols in one call.

    IMPORTANT:
      The Alpaca `limit` is a TOTAL limit across all symbols in the request.
      To avoid starving later symbols, keep batch_size small enough that:
        batch_size * expected_bars_per_symbol <= bars_limit
    """
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=lookback_days)

    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start,
        end=now,
        limit=bars_limit,
        feed=DataFeed.IEX,
        adjustment=Adjustment.SPLIT,
    )

    bars = stock_client.get_stock_bars(req)
    data = bars.data if hasattr(bars, "data") else bars  # dict-like: symbol -> list[Bar]
    if not isinstance(data, dict):
        return {sym: [] for sym in symbols}

    return {sym: data.get(sym, []) for sym in symbols}


def fetch_latest_trade_and_quote_batch(stock_client, symbols):
    """
    Fetch latest trade + quote for MANY symbols in one call each.
    Missing is fine (allowed missing/0).
    """
    out = {s: {} for s in symbols}
    try:
        from alpaca.data.requests import StockLatestTradeRequest, StockLatestQuoteRequest
    except Exception:
        return out

    # Latest trades
    try:
        tr_req = StockLatestTradeRequest(symbol_or_symbols=symbols, feed=DataFeed.IEX)
        tr = stock_client.get_stock_latest_trade(tr_req)
        tr_data = tr.data if hasattr(tr, "data") else tr
        if isinstance(tr_data, dict):
            for sym, trade in tr_data.items():
                if trade:
                    out.setdefault(sym, {})["last_trade_price"] = safe_float(getattr(trade, "price", None))
    except Exception:
        logger.debug("Latest trade batch unavailable", exc_info=True)

    # Latest quotes
    try:
        q_req = StockLatestQuoteRequest(symbol_or_symbols=symbols, feed=DataFeed.IEX)
        q = stock_client.get_stock_latest_quote(q_req)
        q_data = q.data if hasattr(q, "data") else q
        if isinstance(q_data, dict):
            for sym, quote in q_data.items():
                if quote:
                    out.setdefault(sym, {})["bid"] = safe_float(getattr(quote, "bid_price", None))
                    out.setdefault(sym, {})["ask"] = safe_float(getattr(quote, "ask_price", None))
    except Exception:
        logger.debug("Latest quote batch unavailable", exc_info=True)

    return out


def bars_to_arrays(series):
    """
    Convert a list[Bar] to arrays for metric building.
    Drops rows where OHLC is missing.
    """
    times, o, h, l, c, v, vwap, tc = [], [], [], [], [], [], [], []
    for bar in series:
        t = getattr(bar, "timestamp", None)
        times.append(t.isoformat() if hasattr(t, "isoformat") else str(t))
        o.append(safe_float(getattr(bar, "open", None)))
        h.append(safe_float(getattr(bar, "high", None)))
        l.append(safe_float(getattr(bar, "low", None)))
        c.append(safe_float(getattr(bar, "close", None)))
        v.append(safe_float(getattr(bar, "volume", None)))
        vwap.append(safe_float(getattr(bar, "vwap", None)))
        tc.append(safe_float(getattr(bar, "trade_count", None)))

    times, o, h, l, c, v, vwap, tc = drop_bad_rows(times, o, h, l, c, v, vwap, tc)
    return o, h, l, c, v, vwap, tc


# -----------------------------
# Metrics
# -----------------------------
def sma_last(values, length: int):
    n = len(values)
    if n == 0:
        return None
    use = min(length, n)
    window = values[-use:]
    return sum(window) / use if use > 0 else None


def atr_wilder_last(highs, lows, closes, length: int = 14):
    n = len(closes)
    if n < 2:
        return None

    tr = []
    for i in range(n):
        if i == 0:
            tr.append(highs[i] - lows[i])
        else:
            tr_hl = highs[i] - lows[i]
            tr_hc = abs(highs[i] - closes[i - 1])
            tr_lc = abs(lows[i] - closes[i - 1])
            tr.append(max(tr_hl, tr_hc, tr_lc))

    if n < length:
        return None

    atr = sum(tr[:length]) / length
    for i in range(length, n):
        atr = (atr * (length - 1) + tr[i]) / length
    return atr


def build_metrics(opens, highs, lows, closes, volumes, vwaps, trade_counts):
    n = len(closes)
    if n == 0:
        return {"ok": False}

    last_close = closes[-1]
    prev_close = closes[-2] if n >= 2 else None

    sma50 = sma_last(closes, 50)
    sma200 = sma_last(closes, 200)
    atr14 = atr_wilder_last(highs, lows, closes, 14)

    use_252 = min(252, n)
    window = closes[-use_252:]
    high_52w = max(window) if window else None
    low_52w = min(window) if window else None

    use_20 = min(20, n)
    vol20 = volumes[-use_20:] if use_20 > 0 else []
    close20 = closes[-use_20:] if use_20 > 0 else []

    avg_vol_20d = mean_ignore_none(vol20)
    avg_close_20d = mean_ignore_none(close20)

    avg_dollar_vol_20d = None
    if avg_vol_20d is not None and avg_close_20d is not None:
        avg_dollar_vol_20d = avg_vol_20d * avg_close_20d

    return {
        "ok": True,
        "close": last_close,
        "prev_close": prev_close,
        "volume": volumes[-1],
        "vwap": vwaps[-1],
        "trade_count": trade_counts[-1],
        "sma_50": sma50,
        "sma_200": sma200,
        "atr_14": atr14,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "avg_volume_20d": avg_vol_20d,
        "avg_dollar_volume_20d": avg_dollar_vol_20d,
        "num_bars": n,
    }


def required_fields_for_drop(sym: str, metrics: dict):
    """
    Drop symbol if ANY of these are blank OR 0:
      symbol, close, prev_close, volume, vwap, trade_count, sma_50, sma_200,
      atr_14, high_52w, low_52w, avg_volume_20d, avg_dollar_volume_20d, num_bars

    NOTE: last_trade_price, bid, ask, spread_pct are allowed missing/0.
    """
    return {
        "symbol": sym,
        "close": metrics.get("close"),
        "prev_close": metrics.get("prev_close"),
        "volume": metrics.get("volume"),
        "vwap": metrics.get("vwap"),
        "trade_count": metrics.get("trade_count"),
        "sma_50": metrics.get("sma_50"),
        "sma_200": metrics.get("sma_200"),
        "atr_14": metrics.get("atr_14"),
        "high_52w": metrics.get("high_52w"),
        "low_52w": metrics.get("low_52w"),
        "avg_volume_20d": metrics.get("avg_volume_20d"),
        "avg_dollar_volume_20d": metrics.get("avg_dollar_volume_20d"),
        "num_bars": metrics.get("num_bars"),
    }


# -----------------------------
# One full run (single pass over whitelist) - batched safely
# -----------------------------
def run_once():
    symbols = parse_symbol_list("ALPACA_WHITELIST")
    if not symbols:
        logger.error("No symbols configured. Set ALPACA_WHITELIST.")
        return

    # Legacy setting (kept), but we *default* the actual bars lookback smaller
    # because your metrics only require ~252 trading days + buffers.
    calendar_days = get_env_int("ALPACA_STOCK_CALENDAR_DAYS", 2200, min_value=30, max_value=10000)

    # Bars lookback used for the request.
    # Default: min(calendar_days, 600) to preserve the option of larger lookbacks,
    # while making the default fast and still sufficient for SMA200 + 52w + ATR14 + 20d.
    lookback_days = get_env_int("ALPACA_BARS_LOOKBACK_DAYS", min(calendar_days, 600), min_value=30, max_value=10000)

    # Total limit across all symbols in a single bars request.
    bars_limit = get_env_int("ALPACA_BARS_LIMIT", 10000, min_value=500, max_value=10000)

    # Desired batch size; we'll clamp it down to a safe value automatically.
    desired_batch_size = get_env_int("ALPACA_BATCH_SIZE", 25, min_value=1, max_value=500)

    # Optional: sleep briefly between batches to be nice to APIs.
    batch_sleep = float(str(get_env("ALPACA_BATCH_SLEEP_SECONDS", "0")).strip() or "0")

    # Compute a "safe" batch size so we don't starve symbols due to the shared limit.
    # Daily bars: trading days are ~70-75% of calendar days. Use 0.80 as conservative.
    est_bars_per_symbol = max(1, int(math.ceil(lookback_days * 0.80)))
    safe_batch_size = max(1, bars_limit // est_bars_per_symbol)
    batch_size = min(desired_batch_size, safe_batch_size)

    if batch_size != desired_batch_size:
        logger.warning(
            "Reducing batch size from %d to %d to avoid limit starvation "
            "(lookback_days=%d, bars_limit=%d, est_bars_per_symbol=%d).",
            desired_batch_size, batch_size, lookback_days, bars_limit, est_bars_per_symbol
        )

    logger.info(
        "Config: symbols=%d lookback_days=%d bars_limit=%d batch_size=%d (desired=%d) batch_sleep=%s",
        len(symbols), lookback_days, bars_limit, batch_size, desired_batch_size, batch_sleep
    )

    gc = get_google_client()
    ws = open_target_worksheet(gc)
    stock_client = get_alpaca_data_client()

    header = [
        "symbol",
        "close",
        "prev_close",
        "volume",
        "vwap",
        "trade_count",
        "sma_50",
        "sma_200",
        "atr_14",
        "high_52w",
        "low_52w",
        "avg_volume_20d",
        "avg_dollar_volume_20d",
        "last_trade_price",
        "bid",
        "ask",
        "spread_pct",
        "num_bars",
    ]

    rows = []
    dropped = 0

    for batch in chunked(symbols, batch_size):
        logger.info("Processing batch of %d symbols", len(batch))

        # 1) Historical bars (ONE call per batch)
        try:
            bars_map = fetch_stock_bars_batch(stock_client, batch, lookback_days, bars_limit)
        except Exception:
            logger.exception("Bars batch failed; dropping these %d symbols", len(batch))
            dropped += len(batch)
            if batch_sleep > 0:
                time.sleep(batch_sleep)
            continue

        # 2) Latest trade/quote (TWO calls per batch). Optional fields only.
        live_map = fetch_latest_trade_and_quote_batch(stock_client, batch)

        # 3) Per-symbol metrics + drop logic (CPU-only)
        for sym in batch:
            try:
                series = bars_map.get(sym, []) or []
                if not series:
                    dropped += 1
                    logger.warning("Dropping %s (no bars returned)", sym)
                    continue

                o, h, l, c, v, vwap, tc = bars_to_arrays(series)
                metrics = build_metrics(o, h, l, c, v, vwap, tc)
                if not metrics.get("ok"):
                    dropped += 1
                    logger.warning("Dropping %s (no usable data after cleaning)", sym)
                    continue

                live = live_map.get(sym, {}) or {}
                bid = live.get("bid")
                ask = live.get("ask")
                spread_pct = None
                if bid is not None and ask is not None and ask != 0:
                    spread_pct = ((ask - bid) / ask) * 100.0

                req = required_fields_for_drop(sym, metrics)
                missing = [k for k, vv in req.items() if is_blank_or_zero(vv)]
                if missing:
                    dropped += 1
                    logger.warning("Dropping %s due to blank/0 in: %s", sym, ",".join(missing))
                    continue

                row = [
                    sym,
                    metrics.get("close"),
                    metrics.get("prev_close"),
                    metrics.get("volume"),
                    metrics.get("vwap"),
                    metrics.get("trade_count"),
                    metrics.get("sma_50"),
                    metrics.get("sma_200"),
                    metrics.get("atr_14"),
                    metrics.get("high_52w"),
                    metrics.get("low_52w"),
                    metrics.get("avg_volume_20d"),
                    metrics.get("avg_dollar_volume_20d"),
                    live.get("last_trade_price"),
                    bid,
                    ask,
                    spread_pct,
                    metrics.get("num_bars"),
                ]
                rows.append([sheet_val(x) for x in row])

            except Exception:
                dropped += 1
                logger.exception("Dropping %s due to exception", sym)
                continue

        if batch_sleep > 0:
            time.sleep(batch_sleep)

    rows.sort(key=lambda r: str(r[0]))
    write_rows_to_sheet(ws, header, rows)

    logger.info("Run complete. Wrote %d rows. Dropped: %d.", len(rows), dropped)


# -----------------------------
# Main loop (perpetual)
# -----------------------------
def main():
    configure_logging()
    logger.info("=== Alpaca Stocks Screener (perpetual) started ===")

    poll_seconds = get_env_int("MARKET_POLL_SECONDS", 60, min_value=5, max_value=3600)
    post_run_sleep = get_env_int("POST_RUN_SLEEP_SECONDS", 60, min_value=0, max_value=3600)

    while True:
        try:
            if not market_is_open():
                time.sleep(poll_seconds)
                continue

            started = datetime.now(timezone.utc)
            logger.info("Starting run_once() at %s", started.isoformat())
            run_once()
            ended = datetime.now(timezone.utc)
            dur = (ended - started).total_seconds()
            logger.info("run_once() finished at %s (duration %.1f sec)", ended.isoformat(), dur)

            time.sleep(post_run_sleep)

        except Exception:
            logger.exception("Top-level loop exception; sleeping then continuing.")
            time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
