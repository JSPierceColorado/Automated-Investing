import os
import json
import sys
import math
import logging
from datetime import datetime, timedelta, timezone

import gspread
from google.oauth2.service_account import Credentials

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed, Adjustment

# Trading (asset metadata)
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
# Helpers
# -----------------------------
def get_env(name: str, default=None, required: bool = False):
    value = os.getenv(name, default)
    if required and (value is None or str(value).strip() == ""):
        logger.error("Missing required environment variable: %s", name)
        sys.exit(1)
    return value


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


def drop_bad_rows(times, o, h, l, c, v, vwap, tc):
    nt, no, nh, nl, nc, nv, nvw, ntc = [], [], [], [], [], [], [], []
    for t, oo, hh, ll, cc, vv, vw, tcc in zip(times, o, h, l, c, v, vwap, tc):
        if any(is_bad_num(x) for x in (oo, hh, ll, cc)):
            continue
        nt.append(t)
        no.append(oo)
        nh.append(hh)
        nl.append(ll)
        nc.append(cc)
        nv.append(0.0 if is_bad_num(vv) else vv)
        nvw.append(None if is_bad_num(vw) else vw)
        ntc.append(None if is_bad_num(tcc) else tcc)
    return nt, no, nh, nl, nc, nv, nvw, ntc


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
# Alpaca fetch
# -----------------------------
def get_alpaca_clients():
    api_key = get_env("ALPACA_API_KEY", required=True)
    api_secret = get_env("ALPACA_API_SECRET", required=True)

    stock_data = StockHistoricalDataClient(api_key=api_key, secret_key=api_secret)
    trading = TradingClient(api_key=api_key, secret_key=api_secret, paper=True)  # paper flag is fine for metadata
    return stock_data, trading


def fetch_stock_bars(stock_client, symbol: str, calendar_days: int):
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=calendar_days)

    req = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Day,
        start=start,
        end=now,
        limit=calendar_days,          # limit is count, not days; this is fine
        feed=DataFeed.IEX,
        adjustment=Adjustment.SPLIT,
    )

    bars = stock_client.get_stock_bars(req)
    series = bars.data.get(symbol, []) if hasattr(bars, "data") else bars.get(symbol, [])

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
    return times, o, h, l, c, v, vwap, tc


def fetch_latest_trade_and_quote(stock_client, symbol: str):
    """
    Optional: latest trade & quote. If your alpaca-py version lacks these request classes
    or you don't have access, we just return {}.
    """
    out = {}
    try:
        from alpaca.data.requests import StockLatestTradeRequest, StockLatestQuoteRequest
    except Exception:
        return out

    try:
        tr_req = StockLatestTradeRequest(symbol_or_symbols=[symbol], feed=DataFeed.IEX)
        tr = stock_client.get_stock_latest_trade(tr_req)
        trade = tr.data.get(symbol) if hasattr(tr, "data") else tr.get(symbol)
        if trade:
            out["last_trade_price"] = safe_float(getattr(trade, "price", None))
            tt = getattr(trade, "timestamp", None)
            out["last_trade_time"] = tt.isoformat() if hasattr(tt, "isoformat") else str(tt)
    except Exception:
        logger.debug("Latest trade unavailable for %s", symbol, exc_info=True)

    try:
        q_req = StockLatestQuoteRequest(symbol_or_symbols=[symbol], feed=DataFeed.IEX)
        q = stock_client.get_stock_latest_quote(q_req)
        quote = q.data.get(symbol) if hasattr(q, "data") else q.get(symbol)
        if quote:
            out["bid"] = safe_float(getattr(quote, "bid_price", None))
            out["ask"] = safe_float(getattr(quote, "ask_price", None))
    except Exception:
        logger.debug("Latest quote unavailable for %s", symbol, exc_info=True)

    return out


def fetch_asset_meta(trading_client, symbol: str):
    out = {}
    try:
        a = trading_client.get_asset(symbol)
        out["asset_name"] = getattr(a, "name", "")
        out["exchange"] = getattr(a, "exchange", "")
        out["tradable"] = getattr(a, "tradable", None)
        out["marginable"] = getattr(a, "marginable", None)
        out["shortable"] = getattr(a, "shortable", None)
        out["fractionable"] = getattr(a, "fractionable", None)
        out["easy_to_borrow"] = getattr(a, "easy_to_borrow", None)
    except Exception as e:
        logger.debug("Asset meta unavailable for %s (%s)", symbol, e, exc_info=True)
    return out


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

    # True Range
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

    # Seed with SMA
    atr = sum(tr[:length]) / length
    for i in range(length, n):
        atr = (atr * (length - 1) + tr[i]) / length
    return atr


def volatility_annualized_pct(closes, length: int = 20):
    n = len(closes)
    if n < 2:
        return None
    use = min(length, n - 1)
    # simple returns over last `use` days
    rets = []
    for i in range(n - use, n):
        prev = closes[i - 1]
        cur = closes[i]
        if prev == 0:
            continue
        rets.append((cur / prev) - 1.0)
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    daily_sd = math.sqrt(var)
    ann = daily_sd * math.sqrt(252.0)
    return ann * 100.0


def build_metrics(times, opens, highs, lows, closes, volumes, vwaps, trade_counts):
    n = len(closes)
    if n == 0:
        return {"status": "NO_DATA"}

    last_close = closes[-1]
    prev_close = closes[-2] if n >= 2 else None
    chg_1d_pct = None
    if prev_close not in (None, 0):
        chg_1d_pct = ((last_close / prev_close) - 1.0) * 100.0

    sma50 = sma_last(closes, 50)
    sma200 = sma_last(closes, 200)

    pct_from_sma50 = None if (sma50 in (None, 0)) else ((last_close / sma50) - 1.0) * 100.0
    pct_from_sma200 = None if (sma200 in (None, 0)) else ((last_close / sma200) - 1.0) * 100.0

    atr14 = atr_wilder_last(highs, lows, closes, 14)

    # 52-week high/low ~ 252 trading days (use what we have)
    use_252 = min(252, n)
    window = closes[-use_252:]
    high_52w = max(window) if window else None
    low_52w = min(window) if window else None

    pct_off_52w_high = None if (high_52w in (None, 0)) else ((last_close / high_52w) - 1.0) * 100.0
    pct_above_52w_low = None if (low_52w in (None, 0)) else ((last_close / low_52w) - 1.0) * 100.0

    # avg vol / avg dollar vol (20d)
    use_20 = min(20, n)
    vol20 = volumes[-use_20:] if use_20 > 0 else []
    close20 = closes[-use_20:] if use_20 > 0 else []
    avg_vol_20d = (sum(vol20) / use_20) if use_20 > 0 else None
    avg_close_20d = (sum(close20) / use_20) if use_20 > 0 else None
    avg_dollar_vol_20d = None
    if avg_vol_20d is not None and avg_close_20d is not None:
        avg_dollar_vol_20d = avg_vol_20d * avg_close_20d

    vol_20d_ann = volatility_annualized_pct(closes, 20)

    return {
        "status": "OK",
        "last_bar_time": times[-1],
        "close": last_close,
        "prev_close": prev_close,
        "change_1d_pct": chg_1d_pct,
        "volume": volumes[-1],
        "vwap": vwaps[-1],
        "trade_count": trade_counts[-1],
        "sma_50": sma50,
        "sma_200": sma200,
        "pct_from_sma_50": pct_from_sma50,
        "pct_from_sma_200": pct_from_sma200,
        "atr_14": atr14,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "pct_off_52w_high": pct_off_52w_high,
        "pct_above_52w_low": pct_above_52w_low,
        "avg_volume_20d": avg_vol_20d,
        "avg_dollar_volume_20d": avg_dollar_vol_20d,
        "volatility_20d_ann_pct": vol_20d_ann,
        "num_bars": n,
    }


# -----------------------------
# Main
# -----------------------------
def main():
    configure_logging()
    logger.info("=== Alpaca Stocks Screener run started ===")

    symbols = parse_symbol_list("ALPACA_WHITELIST")
    if not symbols:
        logger.error("No symbols configured. Set ALPACA_WHITELIST.")
        sys.exit(1)

    # Pull enough calendar days to reliably cover 252 trading bars + SMAs
    calendar_days = int(get_env("ALPACA_STOCK_CALENDAR_DAYS", 2200))

    gc = get_google_client()
    ws = open_target_worksheet(gc)

    stock_client, trading_client = get_alpaca_clients()

    header = [
        "status",
        "symbol",
        "last_bar_time",
        "close",
        "prev_close",
        "change_1d_pct",
        "volume",
        "vwap",
        "trade_count",
        "sma_50",
        "sma_200",
        "pct_from_sma_50",
        "pct_from_sma_200",
        "atr_14",
        "high_52w",
        "low_52w",
        "pct_off_52w_high",
        "pct_above_52w_low",
        "avg_volume_20d",
        "avg_dollar_volume_20d",
        "volatility_20d_ann_pct",
        "last_trade_price",
        "last_trade_time",
        "bid",
        "ask",
        "spread_pct",
        "asset_name",
        "exchange",
        "tradable",
        "marginable",
        "shortable",
        "fractionable",
        "easy_to_borrow",
        "num_bars",
    ]

    rows = []
    errors = 0

    for sym in symbols:
        try:
            logger.info("Processing %s", sym)
            times, o, h, l, c, v, vwap, tc = fetch_stock_bars(stock_client, sym, calendar_days)
            metrics = build_metrics(times, o, h, l, c, v, vwap, tc)

            live = fetch_latest_trade_and_quote(stock_client, sym)
            meta = fetch_asset_meta(trading_client, sym)

            bid = live.get("bid")
            ask = live.get("ask")
            spread_pct = None
            if bid is not None and ask is not None and ask != 0:
                spread_pct = ((ask - bid) / ask) * 100.0

            row = [
                metrics.get("status", ""),
                sym,
                metrics.get("last_bar_time", ""),
                metrics.get("close"),
                metrics.get("prev_close"),
                metrics.get("change_1d_pct"),
                metrics.get("volume"),
                metrics.get("vwap"),
                metrics.get("trade_count"),
                metrics.get("sma_50"),
                metrics.get("sma_200"),
                metrics.get("pct_from_sma_50"),
                metrics.get("pct_from_sma_200"),
                metrics.get("atr_14"),
                metrics.get("high_52w"),
                metrics.get("low_52w"),
                metrics.get("pct_off_52w_high"),
                metrics.get("pct_above_52w_low"),
                metrics.get("avg_volume_20d"),
                metrics.get("avg_dollar_volume_20d"),
                metrics.get("volatility_20d_ann_pct"),
                live.get("last_trade_price"),
                live.get("last_trade_time"),
                bid,
                ask,
                spread_pct,
                meta.get("asset_name", ""),
                meta.get("exchange", ""),
                meta.get("tradable"),
                meta.get("marginable"),
                meta.get("shortable"),
                meta.get("fractionable"),
                meta.get("easy_to_borrow"),
                metrics.get("num_bars"),
            ]

            rows.append([sheet_val(x) for x in row])

        except Exception as e:
            errors += 1
            logger.exception("Error processing %s", sym)
            rows.append(["ERROR", sym] + [""] * (len(header) - 2))

    write_rows_to_sheet(ws, header, rows)
    logger.info("=== Complete. Wrote %d rows (errors: %d) ===", len(rows), errors)
    print(f"Wrote {len(rows)} rows to sheet. Errors: {errors}")


if __name__ == "__main__":
    main()
