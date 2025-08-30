import sqlite3
import pandas as pd
import os
import yfinance as yf
from datetime import datetime, timedelta, timezone
from filelock import FileLock

DB_PATH = "tickers_cache.db"
LOCK_PATH = DB_PATH + ".lock"

# Columns we expect from yfinance.history()
EXPECTED_COLS = ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]


def get_conn(db_path=DB_PATH):
    """Return a sqlite3 connection (ensure parent dir exists)."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    # Use detect_types to preserve datetimes if needed; we store dates as TEXT (ISO).
    return sqlite3.connect(db_path, timeout=30, detect_types=sqlite3.PARSE_DECLTYPES)


def init_schema(conn=None):
    """Create the prices table and a small metadata table if not exists."""
    close_conn = False
    if conn is None:
        conn = get_conn()
        close_conn = True

    create_prices = """
    CREATE TABLE IF NOT EXISTS prices (
        ticker TEXT NOT NULL,
        date TEXT NOT NULL,                -- stored as ISO 'YYYY-MM-DD' or full timestamp
        Open REAL, High REAL, Low REAL, Close REAL,
        Volume REAL, Dividends REAL, "Stock Splits" REAL,
        PRIMARY KEY (ticker, date)
    );
    """
    create_meta = """
    CREATE TABLE IF NOT EXISTS metadata (
        key TEXT PRIMARY KEY,
        value TEXT
    );
    """
    conn.execute(create_prices)
    conn.execute(create_meta)
    conn.commit()
    if close_conn:
        conn.close()


def _df_to_tuples_for_sql(df: pd.DataFrame, ticker: str):
    """
    Convert dataframe to list of tuples aligned to table columns.
    Expect df.index to be DatetimeIndex (or convertible).
    Returns list of tuples: (ticker, date_iso, Open, High, Low, Close, Volume, Dividends, Stock Splits)
    """
    df = df.copy()
    # Ensure those expected columns exist; fill missing with NaN
    for c in EXPECTED_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    # Normalize index to date string (ISO)
    # Use date part only to avoid timezone noise (you can keep timestamp if you prefer)
    if isinstance(df.index, pd.DatetimeIndex):
        df_index = df.index.tz_localize(None) if df.index.tzinfo is not None else df.index
        date_strs = df_index.strftime("%Y-%m-%d")
    else:
        date_strs = pd.to_datetime(df.index).strftime("%Y-%m-%d")

    rows = []
    for idx, row in df.iterrows():
        date_iso = pd.to_datetime(idx).strftime("%Y-%m-%d")
        vals = (
            ticker,
            date_iso,
            float(row.get("Open", float("nan")) if pd.notna(row.get("Open", None)) else None),
            float(row.get("High", float("nan")) if pd.notna(row.get("High", None)) else None),
            float(row.get("Low", float("nan")) if pd.notna(row.get("Low", None)) else None),
            float(row.get("Close", float("nan")) if pd.notna(row.get("Close", None)) else None),
            float(row.get("Volume", float("nan")) if pd.notna(row.get("Volume", None)) else None),
            float(row.get("Dividends", float("nan")) if pd.notna(row.get("Dividends", None)) else None),
            float(row.get("Stock Splits", float("nan")) if pd.notna(row.get("Stock Splits", None)) else None),
        )
        rows.append(vals)
    return rows


def insert_or_replace_rows(conn, rows):
    """
    rows: list of tuples matching the order:
    (ticker, date, Open, High, Low, Close, Volume, Dividends, Stock Splits)
    """
    if not rows:
        return 0
    placeholders = ",".join(["?"] * 9)
    stmt = f"""
      INSERT OR REPLACE INTO prices
      (ticker, date, Open, High, Low, Close, Volume, Dividends, "Stock Splits")
      VALUES ({placeholders})
    """
    cur = conn.executemany(stmt, rows)
    conn.commit()
    # executemany doesn't give rowcount reliably across adapters, so return len(rows)
    return len(rows)


def get_max_date_for_ticker(conn, ticker: str):
    q = "SELECT MAX(date) FROM prices WHERE ticker = ?"
    cur = conn.execute(q, (ticker,))
    row = cur.fetchone()
    if row and row[0]:
        # return a pandas Timestamp
        return pd.to_datetime(row[0])
    return None


def full_load(tickers, min_date="1990-01-01"):
    """
    Perform a full load (replace existing rows for tickers).
    This will fetch full history and write into the DB (replace any existing rows for those tickers).
    Use with care for many tickers.
    """
    init_schema()
    lock = FileLock(LOCK_PATH, timeout=600)
    inserted_total = 0
    with lock:
        conn = get_conn()
        try:
            for tkr in tickers:
                print(f"[full_load] fetching {tkr} ...")
                df = yf.Ticker(tkr).history(period="max")
                if df.empty:
                    print(f"  no data for {tkr}")
                    continue
                df.index = pd.to_datetime(df.index)
                df = df.loc[min_date:].copy()
                rows = _df_to_tuples_for_sql(df, tkr)
                # delete existing rows for ticker before inserting (so full load replaces)
                conn.execute("DELETE FROM prices WHERE ticker = ?", (tkr,))
                n = insert_or_replace_rows(conn, rows)
                print(f"  inserted {n} rows for {tkr}")
                inserted_total += n
        finally:
            conn.close()
    return inserted_total


def update_ticker(ticker: str, min_date="1990-01-01"):
    """
    Incremental update: fetch data starting the day AFTER last saved date (if exists).
    If ticker has no rows, fetch from min_date to present.
    Returns number of new rows inserted.
    """
    init_schema()
    lock = FileLock(LOCK_PATH, timeout=600)
    with lock:
        conn = get_conn()
        try:
            last = get_max_date_for_ticker(conn, ticker)  # returns pd.Timestamp or None
            if last is None:
                start_date = pd.to_datetime(min_date).date()
            else:
                # Use date() to avoid timezone/time component issues
                last_date_only = pd.to_datetime(last).date()
                start_date = last_date_only + timedelta(days=1)

            # compute "today" in UTC to avoid comparing naive vs aware datetimes
            today = datetime.now(timezone.utc).date()

            # If start is strictly after today, nothing to fetch
            if start_date > today:
                # nothing to do; we already have up-to-date data
                print(f"[update] {ticker} is up-to-date (last saved: {last_date_only if last is not None else 'N/A'})")
                return 0

            start_str = start_date.strftime("%Y-%m-%d")
            print(f"[update] fetching {ticker} from {start_str} ...")

            # call yfinance only when start <= today
            try:
                df = yf.Ticker(ticker).history(start=start_str)
            except Exception as e:
                print(f"  yfinance error when fetching {ticker} from {start_str}: {e}")
                return 0

            if df is None or df.empty:
                print("  no new rows")
                return 0

            df.index = pd.to_datetime(df.index)
            rows = _df_to_tuples_for_sql(df, ticker)
            n = insert_or_replace_rows(conn, rows)
            print(f"  inserted {n} rows for {ticker}")
            return n
        finally:
            conn.close()

#Reads data for a ticker from DB and return as DataFrame indexed by datetime.
def load_ticker(ticker):
    init_schema()
    conn = get_conn()
    try:
        q = "SELECT date, Open, High, Low, Close, Volume, Dividends, \"Stock Splits\" FROM prices WHERE ticker = ? ORDER BY date"
        df = pd.read_sql_query(q, conn, params=(ticker,), parse_dates=["date"])
        if df.empty:
            return df
        df = df.set_index("date").sort_index()
        return df
    finally:
        conn.close()


if __name__ == "__main__":
    TICKERS = ["^GSPC", "^DJI", "^IXIC", "^RUT"]
    print("Starting full load for tickers:", TICKERS)
    inserted = full_load(TICKERS)
    print("Full load complete — inserted rows:", inserted)

    for t in TICKERS:
        new_rows = update_ticker(t)
        print(f"Ticker {t} new rows after update: {new_rows}")

    print("Done.")