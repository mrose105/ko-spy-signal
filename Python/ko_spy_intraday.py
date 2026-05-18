"""
ko_spy_intraday.py
==================
Intraday KO/SPY divergence signal backtest.

Signal logic:
  - SPY drops >= DROP_THRESHOLD from its rolling intraday high
  - KO is flat or positive over the same lookback window
  - Divergence = fakeout, enter SPY calls
  - Measure SPY forward return to 30min, 60min, and session close

Setup:
    pip install alpaca-py pandas numpy tabulate

Usage:
    export ALPACA_API_KEY=your_key
    export ALPACA_SECRET_KEY=your_secret
    python ko_spy_intraday.py
"""

import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, date
from tabulate import tabulate

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# ─── CONFIG ──────────────────────────────────────────────────

API_KEY    = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

CACHE_DIR = Path("./ko_spy_cache")
CACHE_DIR.mkdir(exist_ok=True)

YEARS = 1   # last 365 days only

DROP_THRESHOLD  = 0.003   # SPY drops >= 0.3% from rolling high
KO_FLAT_MAX     = 0.001   # KO move must be >= -0.1% (flat or up)
LOOKBACK_BARS   = 6       # 30-min window (6 x 5-min bars)
MIN_SIGNAL_TIME = "09:50"
MAX_SIGNAL_TIME = "15:00"

SESSIONS = {
    "morning":   ("09:30", "11:30"),
    "midday":    ("11:30", "14:00"),
    "afternoon": ("14:00", "15:00"),
}

# ─── DATA ────────────────────────────────────────────────────

def fetch_paginated(client, symbol, start, end):
    all_bars = []
    chunk_start = start
    chunk_size  = timedelta(days=90)
    while chunk_start < end:
        chunk_end = min(chunk_start + chunk_size, end)
        print(f"  {symbol}: {chunk_start.date()} -> {chunk_end.date()}")
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=chunk_start,
            end=chunk_end,
        )
        try:
            bars = client.get_stock_bars(req).df
            if isinstance(bars.index, pd.MultiIndex):
                bars = bars.xs(symbol, level="symbol")
            if len(bars) > 0:
                all_bars.append(bars[["close"]])
        except Exception as e:
            print(f"  Warning: {e}")
        chunk_start = chunk_end
        time.sleep(0.3)

    if not all_bars:
        return pd.DataFrame()
    df = pd.concat(all_bars)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
    return df[~df.index.duplicated(keep="last")].sort_index()


def load(client, symbol):
    cache = CACHE_DIR / f"{symbol}_{YEARS}yr_intraday.parquet"
    if cache.exists():
        print(f"  Loading {symbol} from cache...")
        df = pd.read_parquet(cache)
        last = df.index[-1].date()
        if last < date.today() - timedelta(days=1):
            new_start = datetime.combine(last + timedelta(days=1), datetime.min.time())
            new = fetch_paginated(client, symbol, new_start, datetime.now())
            if len(new) > 0:
                df = pd.concat([df, new])
                df = df[~df.index.duplicated(keep="last")].sort_index()
                df.to_parquet(cache)
        return df
    start = datetime.now() - timedelta(days=int(YEARS * 365.25))
    df = fetch_paginated(client, symbol, start, datetime.now())
    if len(df) > 0:
        df.to_parquet(cache)
    return df


def resample_5min(df, col):
    df5 = df.resample("5T").last().dropna()
    df5 = df5.between_time("09:30", "16:00")
    df5.columns = [col]
    return df5


# ─── SIGNAL TYPES ────────────────────────────────────────────
#
# Three single-leg signals (tightened to 0.5% threshold):
#
# BULLISH (buy SPY calls):
#   S1  Both down together   -> fade, buy SPY calls
#   S2  SPY down, KO up      -> regime divergence, buy SPY calls
#
# BEARISH (buy SPY puts):
#   S4  SPY up, KO down      -> rally has no backing, buy SPY puts

DROP_THRESHOLD_FULL = 0.005   # 0.50% — tightened to reduce signal frequency
DROP_THRESHOLD_HALF = 0.003   # 0.30% — S2 divergence threshold (matches full threshold)
FLAT_THRESHOLD      = 0.001   # 0.10% — defines "flat"

SIGNAL_META = {
    "S1_both_down_fade": {"direction": "bullish", "size": "full", "structure": "SPY call outright"},
    "S2_spy_down_ko_up": {"direction": "bullish", "size": "full", "structure": "SPY call outright"},
    "S4_spy_up_ko_down": {"direction": "bearish", "size": "full", "structure": "SPY put outright"},
}

def detect_signals(df):
    spy = df["SPY"]
    ko  = df["KO"]

    # SPY and KO moves over lookback window
    spy_roll_high = spy.rolling(LOOKBACK_BARS).max()
    spy_roll_low  = spy.rolling(LOOKBACK_BARS).min()
    spy_drop      = (spy_roll_high - spy) / spy_roll_high   # positive = SPY fell from high
    spy_rise      = (spy - spy_roll_low)  / spy_roll_low    # positive = SPY rose from low

    ko_move = (ko - ko.shift(LOOKBACK_BARS)) / ko.shift(LOOKBACK_BARS)

    spy_abs = spy_move = (spy - spy.shift(LOOKBACK_BARS)) / spy.shift(LOOKBACK_BARS)

    time_str  = df.index.strftime("%H:%M")
    in_window = (time_str >= MIN_SIGNAL_TIME) & (time_str <= MAX_SIGNAL_TIME)

    df = df.copy()
    df["spy_drop"]  = spy_drop
    df["spy_rise"]  = spy_rise
    df["spy_move"]  = spy_move
    df["ko_move"]   = ko_move
    df["time_str"]  = time_str
    df["date"]      = df.index.date

    # ── Classify each bar into a signal type ─────────────────
    spy_down_full = spy_drop >= DROP_THRESHOLD_FULL
    spy_down_half = spy_drop >= DROP_THRESHOLD_HALF
    spy_up_full   = spy_rise >= DROP_THRESHOLD_FULL

    ko_up   = ko_move >=  DROP_THRESHOLD_HALF
    ko_down = ko_move <= -DROP_THRESHOLD_HALF

    # S1: Both down (full threshold) — bullish fade, buy SPY calls
    s1 = spy_down_full & (ko_move <= KO_FLAT_MAX) & in_window
    # S2: SPY down, KO up — regime divergence, buy SPY calls
    s2 = spy_down_half & ko_up & in_window & ~s1
    # S4: SPY up, KO down — bearish, buy SPY puts
    s4 = spy_up_full & ko_down & in_window

    df["signal_type"] = None
    df.loc[s4, "signal_type"] = "S4_spy_up_ko_down"
    df.loc[s2, "signal_type"] = "S2_spy_down_ko_up"
    df.loc[s1, "signal_type"] = "S1_both_down_fade"

    df["signal"] = df["signal_type"].notna()

    def get_session(t):
        for name, (s, e) in SESSIONS.items():
            if s <= t <= e:
                return name
        return "other"

    df["session"] = df["time_str"].map(get_session)
    return df


# ─── DTE PRICER ──────────────────────────────────────────────

def bs_call(S, K, T, sigma, r=0.05):
    from math import log, sqrt, exp
    from scipy.stats import norm
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    return S*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)

def bs_delta(S, K, T, sigma, r=0.05):
    from math import log, sqrt
    from scipy.stats import norm
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1 = (log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    return norm.cdf(d1)

def dte_cost_table(spy_price=739.0, ko_price=80.50,
                   spy_iv=0.18, ko_iv=0.22, target_delta=0.17):
    from scipy.optimize import brentq
    rows = []
    for dte in [0, 1, 2, 3]:
        T = max(dte, 0.5) / 252
        # SPY strike at target delta
        try:
            spy_k = brentq(lambda K: bs_delta(spy_price, K, T, spy_iv) - target_delta,
                           spy_price, spy_price*1.15)
        except Exception:
            spy_k = spy_price * 1.02
        spy_prem  = bs_call(spy_price, spy_k, T, spy_iv)
        spy_d     = bs_delta(spy_price, spy_k, T, spy_iv)
        # KO strike whose premium matches SPY call cost
        try:
            ko_k = brentq(lambda K: bs_call(ko_price, K, T, ko_iv) - spy_prem,
                          ko_price*0.90, ko_price*1.20)
            ko_prem = bs_call(ko_price, ko_k, T, ko_iv)
            ko_d    = bs_delta(ko_price, ko_k, T, ko_iv)
        except Exception:
            ko_k = ko_price * 1.02
            ko_prem = 0.0
            ko_d    = 0.17
        net_cost   = round(spy_prem - ko_prem, 2)
        net_dollar = round((spy_d*spy_price - ko_d*ko_price)*100)
        rows.append({
            "DTE":         dte,
            "SPY strike":  round(spy_k, 1),
            "SPY call $":  round(spy_prem, 2),
            "SPY delta":   round(spy_d, 2),
            "KO strike":   round(ko_k, 2),
            "KO call $":   round(ko_prem, 2),
            "KO delta":    round(ko_d, 2),
            "Net cost $":  net_cost,
            "Net $ delta": f"+${net_dollar:,}",
        })
    return pd.DataFrame(rows)


# ─── FORWARD RETURNS + DTE OPTIMIZER ─────────────────────────

def measure_forward_returns(df):
    all_dates   = sorted(df["date"].unique())
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    results = []

    for ts, row in df[df["signal"]].iterrows():
        d        = row["date"]
        day_bars = df[df["date"] == d]
        future   = day_bars[day_bars.index > ts]
        if len(future) == 0:
            continue

        entry = row["SPY"]
        b30   = future.iloc[:6]  if len(future) >= 6  else future
        b60   = future.iloc[:12] if len(future) >= 12 else future
        r30   = (b30["SPY"].iloc[-1] - entry) / entry if len(b30) > 0 else np.nan
        r60   = (b60["SPY"].iloc[-1] - entry) / entry if len(b60) > 0 else np.nan
        r0dte = (future["SPY"].iloc[-1] - entry) / entry

        dte_rets = {0: r0dte}
        d_idx = date_to_idx[d]
        for dte in [1, 2, 3]:
            fi = d_idx + dte
            if fi < len(all_dates):
                fb = df[df["date"] == all_dates[fi]]
                dte_rets[dte] = (fb["SPY"].iloc[-1] - entry) / entry if len(fb) > 0 else np.nan
            else:
                dte_rets[dte] = np.nan

        def win(r):
            return r > 0 if (r is not None and not np.isnan(r)) else None

        results.append({
            "date":       d,
            "time":       row["time_str"],
            "session":    row["session"],
            "signal_type":row["signal_type"],
            "spy_drop%":  round(row["spy_drop"] * 100, 3),
            "ko_move%":   round(row["ko_move"]  * 100, 3),
            "ret_30min%": round(r30  * 100, 3),
            "ret_60min%": round(r60  * 100, 3),
            "ret_0dte%":  round(dte_rets[0] * 100, 3),
            "ret_1dte%":  round(dte_rets[1] * 100, 3) if not np.isnan(dte_rets[1]) else np.nan,
            "ret_2dte%":  round(dte_rets[2] * 100, 3) if not np.isnan(dte_rets[2]) else np.nan,
            "ret_3dte%":  round(dte_rets[3] * 100, 3) if not np.isnan(dte_rets[3]) else np.nan,
            "win_30min":  win(r30),
            "win_60min":  win(r60),
            "win_0dte":   dte_rets[0] > 0,
            "win_1dte":   win(dte_rets[1]),
            "win_2dte":   win(dte_rets[2]),
            "win_3dte":   win(dte_rets[3]),
        })

    return pd.DataFrame(results)


# ─── SUMMARY ─────────────────────────────────────────────────

def print_summary(sig):
    n = len(sig)
    if n == 0:
        print("No signals detected.")
        return

    print(f"\nTotal signals: {n} | {sig['date'].min()} to {sig['date'].max()}\n")

    # ── By signal type ──────────────────────────────────────
    print("─── By signal type ────────────────────────────────────────")
    type_rows = []
    for stype, meta in SIGNAL_META.items():
        sub = sig[sig["signal_type"] == stype]
        if len(sub) == 0:
            continue
        valid_0 = sub.dropna(subset=["win_0dte"])
        valid_60 = sub.dropna(subset=["win_60min"])
        type_rows.append({
            "Signal":      stype.split("_",1)[1].replace("_"," "),
            "Direction":   meta["direction"],
            "Size":        meta["size"],
            "Count":       len(sub),
            "Win 60min":   f"{valid_60['win_60min'].mean()*100:.1f}%" if len(valid_60) > 0 else "—",
            "Win 0DTE":    f"{valid_0['win_0dte'].mean()*100:.1f}%"  if len(valid_0)  > 0 else "—",
            "Avg 0DTE ret":f"{sub['ret_0dte%'].mean():.3f}%",
            "Structure":   meta["structure"],
        })
    print(tabulate(type_rows, headers="keys", tablefmt="rounded_outline", showindex=False))

    # ── Overall by window ───────────────────────────────────
    print("\n─── Overall win rate by window ────────────────────────────")
    rows = []
    for label, win_col, ret_col in [
        ("30 min",  "win_30min", "ret_30min%"),
        ("60 min",  "win_60min", "ret_60min%"),
        ("0DTE",    "win_0dte",  "ret_0dte%"),
        ("1DTE",    "win_1dte",  "ret_1dte%"),
        ("2DTE",    "win_2dte",  "ret_2dte%"),
        ("3DTE",    "win_3dte",  "ret_3dte%"),
    ]:
        valid = sig.dropna(subset=[win_col])
        if len(valid) == 0:
            continue
        rows.append({
            "Window":     label,
            "Signals":    len(valid),
            "Win rate":   f"{valid[win_col].dropna().astype(int).mean()*100:.1f}%",
            "Avg ret":    f"{valid[ret_col].mean():.3f}%",
            "Median ret": f"{valid[ret_col].median():.3f}%",
        })
    print(tabulate(rows, headers="keys", tablefmt="rounded_outline", showindex=False))

    # ── By session ──────────────────────────────────────────
    print("\n─── By session ────────────────────────────────────────────")
    sess_rows = []
    for s in ["morning", "midday", "afternoon"]:
        sub = sig[sig["session"] == s]
        if len(sub) == 0:
            continue
        v60 = sub.dropna(subset=["win_60min"])
        sess_rows.append({
            "Session":     s.capitalize(),
            "Signals":     len(sub),
            "Win 60min":   f"{v60['win_60min'].mean()*100:.1f}%" if len(v60) > 0 else "—",
            "Win 0DTE":    f"{sub.dropna(subset=['win_0dte'])['win_0dte'].mean()*100:.1f}%",
            "Avg 0DTE ret":f"{sub['ret_0dte%'].mean():.3f}%",
            "Top signal":  sub["signal_type"].value_counts().index[0].split("_",1)[1].replace("_"," "),
        })
    print(tabulate(sess_rows, headers="keys", tablefmt="rounded_outline", showindex=False))

    # ── By month ────────────────────────────────────────────
    print("\n─── Signal frequency by month ─────────────────────────────")
    sig["month"] = pd.to_datetime(sig["date"]).dt.to_period("M").astype(str)
    freq = sig.groupby("month").agg(
        total        = ("signal_type", "count"),
        bullish      = ("signal_type", lambda x: (x.isin(["S1_both_down_fade","S2_spy_down_ko_up"])).sum()),
        bearish      = ("signal_type", lambda x: (x == "S4_spy_up_ko_down").sum()),
        win_rate_0dte= ("win_0dte",    lambda x: f"{x.dropna().mean()*100:.0f}%"),
    ).reset_index()
    print(tabulate(freq, headers="keys", tablefmt="rounded_outline", showindex=False))

    # ── Most recent signals ─────────────────────────────────
    print("\n─── 15 most recent signals ────────────────────────────────")
    cols = ["date","time","session","signal_type","spy_drop%","ko_move%",
            "ret_60min%","ret_0dte%","ret_1dte%","win_0dte"]
    print(tabulate(sig.tail(15)[cols], headers="keys",
                   tablefmt="rounded_outline", showindex=False))

    sig.to_csv("ko_spy_intraday_signals.csv", index=False)
    print("\nExported -> ko_spy_intraday_signals.csv")


# ─── MAIN ────────────────────────────────────────────────────

def main():
    print("\n=== KO/SPY Intraday Divergence Signal ===")
    print(f"SPY drop >= {DROP_THRESHOLD_FULL*100:.1f}% from {LOOKBACK_BARS*5}-min high | 3 signals (S1/S2/S4)")
    print(f"Last 365 days\n")

    client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

    print("Loading SPY...")
    spy_raw = load(client, "SPY")
    print("Loading KO...")
    ko_raw  = load(client, "KO")

    spy_5m = resample_5min(spy_raw, "SPY")
    ko_5m  = resample_5min(ko_raw,  "KO")

    df = spy_5m.join(ko_5m, how="inner").dropna()
    df = df.between_time("09:30", "15:55")
    df = df[df.index >= pd.Timestamp.now(tz='America/New_York') - pd.Timedelta(days=365)]

    print(f"\n{len(df):,} 5-min bars loaded | {df.index.date[0]} to {df.index.date[-1]}\n")

    df      = detect_signals(df)
    results = measure_forward_returns(df)
    print_summary(results)
    print_dte_summary(results)

# ─── DTE SUMMARY (appended to print_summary output) ──────────

def print_dte_summary(sig):
    print("\n─── DTE optimization — win rate and avg SPY return ────────")
    rows = []
    for dte, win_col, ret_col in [
        ("0DTE (same day)",  "win_0dte", "ret_0dte%"),
        ("1DTE",             "win_1dte", "ret_1dte%"),
        ("2DTE",             "win_2dte", "ret_2dte%"),
        ("3DTE",             "win_3dte", "ret_3dte%"),
    ]:
        valid = sig.dropna(subset=[win_col])
        if len(valid) == 0:
            continue
        avg_ret = valid[ret_col].mean()
        win_rt  = valid[win_col].dropna().astype(int).mean()
        wins   = valid[valid[win_col] == True][ret_col]
        losses = valid[valid[win_col] == False][ret_col]
        edge   = (win_rt * wins.mean()) + ((1-win_rt) * losses.mean()) if len(losses) > 0 else win_rt * wins.mean()
        rows.append({
            "DTE window":   dte,
            "Signals":      len(valid),
            "Win rate":     f"{win_rt*100:.1f}%",
            "Avg SPY ret":  f"{avg_ret:.3f}%",
            "Avg win ret":  f"{wins.mean():.3f}%" if len(wins) > 0 else "—",
            "Avg loss ret": f"{losses.mean():.3f}%" if len(losses) > 0 else "—",
            "Edge score":   f"{edge:.4f}%",
        })
    print(tabulate(rows, headers="keys", tablefmt="rounded_outline", showindex=False))

    print("\n─── Option cost structure (SPY long / KO short, 0.17 delta) ─")
    try:
        cost_tbl = dte_cost_table()
        print(tabulate(cost_tbl, headers="keys", tablefmt="rounded_outline", showindex=False))
        print("\n  Net $ delta shows the residual long SPY exposure per 1:1 contract.")
        print("  Net cost shows premium saved by selling the matched KO call.")
    except Exception as e:
        print(f"  Pricer error: {e}")


if __name__ == "__main__":
    main()
