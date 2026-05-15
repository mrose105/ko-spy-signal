#!/usr/bin/env python3
"""
ko_spy_intraday.py — KO/SPY intraday fakeout backtest across historical sessions

Fetches N months of 5-min bars, runs fakeout detection on every session,
tracks same-session forward returns, and prints three tables:

  1. Overall win rate by signal type
  2. Win rate by session period  (AM 9:30–11 / MID 11–14 / PM 14–16)
  3. Win rate by calendar month
"""

import os, sys, argparse
from datetime import datetime, date, timedelta
from collections import defaultdict

import pandas as pd
import numpy as np
from tabulate import tabulate

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# ── Config ────────────────────────────────────────────────────────────────────
OR_MINUTES   = 30       # opening-range window
FAKEOUT_POKE = 0.0005   # min poke beyond level (0.05%)
VWAP_BAND    = 0.002    # SPY neutral zone around VWAP

SESSION_PERIODS = {
    "AM  (09:30–11:00)": ("09:30", "11:00"),
    "MID (11:00–14:00)": ("11:00", "14:00"),
    "PM  (14:00–16:00)": ("14:00", "16:00"),
}

# ── Auth ──────────────────────────────────────────────────────────────────────

def _client():
    key = (os.environ.get("ALPACA_API_KEY")
           or os.environ.get("APCA_API_KEY_ID", ""))
    sec = (os.environ.get("ALPACA_SECRET_KEY")
           or os.environ.get("APCA_API_SECRET_KEY", ""))
    if not key or not sec:
        sys.exit("ERROR: set ALPACA_API_KEY / APCA_API_KEY_ID and secret.")
    return StockHistoricalDataClient(key, sec)


# ── Data ──────────────────────────────────────────────────────────────────────

def fetch_intraday(client, symbol, start, end):
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=start, end=end,
        feed="iex",
    )
    df = client.get_stock_bars(req).df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("US/Eastern")
    return df.between_time("09:30", "16:00").sort_index()


def split_sessions(df):
    for d, grp in df.groupby(df.index.date):
        if len(grp) >= 10:   # skip truncated sessions
            yield d, grp


# ── Indicators ────────────────────────────────────────────────────────────────

def add_vwap(df):
    df = df.copy()
    tp = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (tp * df["volume"]).cumsum() / df["volume"].cumsum()
    return df


def opening_range(df):
    start   = df.index[0].normalize().replace(hour=9, minute=30)
    cutoff  = start + timedelta(minutes=OR_MINUTES)
    or_bars = df[df.index <= cutoff]
    return or_bars["high"].max(), or_bars["low"].min(), cutoff


def spy_regime_series(spy_df):
    spy_df = add_vwap(spy_df)
    reg = np.where(spy_df["close"] > spy_df["vwap"] * (1 + VWAP_BAND),  1,
           np.where(spy_df["close"] < spy_df["vwap"] * (1 - VWAP_BAND), -1, 0))
    spy_df["regime"] = reg.astype(int)
    return spy_df


# ── Fakeout detection (single session) ───────────────────────────────────────

def detect_session_fakeouts(ko_df, spy_df, session_date):
    ko_df  = add_vwap(ko_df)
    spy_df = spy_regime_series(spy_df)

    or_high, or_low, or_cutoff = opening_range(ko_df)
    reg_s = spy_df["regime"].reindex(ko_df.index, method="ffill").fillna(0).astype(int)

    _tag = {1: "SPY_BULL", -1: "SPY_BEAR", 0: "SPY_NEUT"}

    signals = []
    closes  = ko_df["close"].values
    times   = ko_df.index

    for i in range(2, len(ko_df) - 5):
        ts    = times[i]
        cur   = ko_df.iloc[i]
        prev  = ko_df.iloc[i - 1]
        prev2 = ko_df.iloc[i - 2]

        if ts <= or_cutoff:
            continue

        reg     = reg_s.iloc[i]
        vwap_now = cur["vwap"]

        # Forward returns — same session only, cap at remaining bars
        def fwd(n):
            j = i + n
            if j >= len(ko_df):
                return np.nan
            return closes[j] / closes[i] - 1

        base = {
            "date":     session_date,
            "time":     ts.strftime("%H:%M"),
            "month":    session_date.strftime("%Y-%m"),
            "spy":      _tag[int(reg)],
            "fwd_1":    fwd(1),
            "fwd_3":    fwd(3),
            "fwd_5":    fwd(5),
        }

        # Session period
        t = ts.strftime("%H:%M")
        if   t < "11:00": base["period"] = "AM  (09:30–11:00)"
        elif t < "14:00": base["period"] = "MID (11:00–14:00)"
        else:             base["period"] = "PM  (14:00–16:00)"

        # ── OR high upside fakeout ──
        poke_up = (prev["high"] - or_high) / or_high
        if poke_up > FAKEOUT_POKE and cur["close"] < or_high:
            signals.append({**base, "type": "UP_FAKE",  "trade_sign": -1})

        # ── OR low downside fakeout ──
        poke_dn = (or_low - prev["low"]) / or_low
        if poke_dn > FAKEOUT_POKE and cur["close"] > or_low:
            signals.append({**base, "type": "DN_FAKE",  "trade_sign": +1})

        # ── VWAP cross fakeouts ──
        if (prev2["close"] < prev2["vwap"]
                and prev["close"] > prev["vwap"]
                and cur["close"] < vwap_now):
            signals.append({**base, "type": "VWAP↓FAKE", "trade_sign": -1})

        elif (prev2["close"] > prev2["vwap"]
                and prev["close"] < prev["vwap"]
                and cur["close"] > vwap_now):
            signals.append({**base, "type": "VWAP↑FAKE", "trade_sign": +1})

    return signals


# ── Statistics ────────────────────────────────────────────────────────────────

def _win_row(label, subset, horizons=(1, 3, 5)):
    if not subset:
        return None
    row = {"group": label, "n": len(subset)}
    for h in horizons:
        key   = f"fwd_{h}"
        vals  = [(s[key], s["trade_sign"]) for s in subset
                 if not np.isnan(s[key])]
        if not vals:
            row[f"n_{h}d"]   = 0
            row[f"win_{h}d"] = "—"
            row[f"avg_{h}d"] = "—"
            continue
        wins = sum(1 for r, sign in vals if sign * r > 0)
        avgs = np.mean([r for r, _ in vals])
        row[f"n_{h}d"]   = len(vals)
        row[f"win_{h}d"] = f"{wins/len(vals)*100:.0f}%"
        row[f"avg_{h}d"] = f"{avgs*100:+.3f}%"
    return row


def build_tables(all_sigs):
    # ── Table 1: overall by signal type ──────────────────────────────────
    t1 = []
    for sig_type in ("UP_FAKE", "DN_FAKE", "VWAP↑FAKE", "VWAP↓FAKE"):
        sub = [s for s in all_sigs if s["type"] == sig_type]
        row = _win_row(sig_type, sub)
        if row:
            t1.append(row)
    total = _win_row("ALL", all_sigs)
    if total:
        t1.append(total)

    # ── Table 2: by session period ────────────────────────────────────────
    t2 = []
    for period in SESSION_PERIODS:
        sub = [s for s in all_sigs if s["period"] == period]
        row = _win_row(period, sub)
        if row:
            t2.append(row)

    # ── Table 3: by calendar month ────────────────────────────────────────
    months = sorted({s["month"] for s in all_sigs})
    t3 = []
    for m in months:
        sub = [s for s in all_sigs if s["month"] == m]
        row = _win_row(m, sub)
        if row:
            t3.append(row)

    return t1, t2, t3


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--months", type=int, default=6,
                   help="Months of history to backtest (default 6)")
    args = p.parse_args()

    client = _client()
    today  = date.today()
    start  = datetime.combine(today - timedelta(days=args.months * 31),
                              datetime.min.time())
    end    = datetime.combine(today, datetime.max.time())

    W = 66
    print(f"\n{'═'*W}")
    print(f"  KO / SPY  INTRADAY FAKEOUT BACKTEST  ({args.months} months, 5-min bars)")
    print(f"{'═'*W}")
    print(f"  Window: {start.date()} → {today}\n")

    print("  Fetching KO  bars … ", end="", flush=True)
    ko_raw = fetch_intraday(client, "KO", start, end)
    print(f"✓  {len(ko_raw):,} bars")

    print("  Fetching SPY bars … ", end="", flush=True)
    spy_raw = fetch_intraday(client, "SPY", start, end)
    print(f"✓  {len(spy_raw):,} bars\n")

    # ── Per-session detection ─────────────────────────────────────────────
    all_sigs = []
    ko_sessions  = list(split_sessions(ko_raw))
    spy_sessions = dict(split_sessions(spy_raw))

    print(f"  Processing {len(ko_sessions)} sessions …", end="", flush=True)
    skipped = 0
    for session_date, ko_df in ko_sessions:
        if session_date not in spy_sessions:
            skipped += 1
            continue
        spy_df = spy_sessions[session_date]
        sigs   = detect_session_fakeouts(ko_df, spy_df, session_date)
        all_sigs.extend(sigs)

    print(f"  done  ({skipped} skipped — no SPY data)\n")

    if not all_sigs:
        print("  No signals found. Try --months with a larger value.")
        return

    print(f"  Total signals: {len(all_sigs)}")
    types = defaultdict(int)
    for s in all_sigs:
        types[s["type"]] += 1
    for t, n in sorted(types.items()):
        print(f"    {t}: {n}")
    print()

    t1, t2, t3 = build_tables(all_sigs)

    print(f"{'─'*W}")
    print(f"  TABLE 1 — OVERALL WIN RATE BY SIGNAL TYPE")
    print(f"  (win = price moves in expected direction; UP_FAKE/VWAP↓FAKE = short, DN_FAKE/VWAP↑FAKE = long)")
    print(f"{'─'*W}")
    print(tabulate(t1, headers="keys", tablefmt="simple"))

    print(f"\n{'─'*W}")
    print(f"  TABLE 2 — WIN RATE BY SESSION PERIOD")
    print(f"{'─'*W}")
    print(tabulate(t2, headers="keys", tablefmt="simple"))

    print(f"\n{'─'*W}")
    print(f"  TABLE 3 — WIN RATE BY CALENDAR MONTH")
    print(f"{'─'*W}")
    print(tabulate(t3, headers="keys", tablefmt="simple"))

    print()


if __name__ == "__main__":
    main()
