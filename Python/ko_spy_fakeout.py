#!/usr/bin/env python3
"""
ko_spy_fakeout.py — KO/SPY intraday fakeout signal analysis

Detects opening-range and VWAP fakeout patterns in KO,
filtered and quality-graded by live SPY regime.

Usage:
    export ALPACA_API_KEY=...
    export ALPACA_SECRET_KEY=...
    python ko_spy_fakeout.py [--days N]  # default: most recent session
"""

import os
import sys
import argparse
from datetime import datetime, date, timedelta

import pandas as pd
import numpy as np
from scipy import stats
from tabulate import tabulate

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# ── Config ─────────────────────────────────────────────────────────────────
OR_MINUTES   = 30       # opening-range window
VWAP_BAND    = 0.002    # ±0.2% band around VWAP defines "neutral" regime
FAKEOUT_POKE = 0.0005   # bar must poke at least 0.05% beyond level to count
MARKET_OPEN  = "09:30"
MARKET_CLOSE = "16:00"


# ── Data fetching ───────────────────────────────────────────────────────────

def _get_client():
    key    = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_SECRET_KEY", "")
    if not key or not secret:
        sys.exit("ERROR: Set ALPACA_API_KEY and ALPACA_SECRET_KEY env vars.")
    return StockHistoricalDataClient(key, secret)


def fetch_bars(client, symbol, start, end):
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=start,
        end=end,
        feed="iex",
    )
    df = client.get_stock_bars(req).df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("US/Eastern")
    return df.between_time(MARKET_OPEN, MARKET_CLOSE)


def latest_session(df):
    last_date = df.index.date[-1]
    return df[df.index.date == last_date], last_date


# ── Indicators ──────────────────────────────────────────────────────────────

def add_vwap(df):
    df = df.copy()
    tp = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (tp * df["volume"]).cumsum() / df["volume"].cumsum()
    return df


def opening_range(df):
    session_start = df.index[0].normalize().replace(
        hour=9, minute=30, second=0, microsecond=0
    )
    cutoff = session_start + timedelta(minutes=OR_MINUTES)
    or_bars = df[df.index <= cutoff]
    return or_bars["high"].max(), or_bars["low"].min(), cutoff


def spy_regime_series(spy_df):
    """Per-bar regime: +1 bull / -1 bear / 0 neutral relative to VWAP."""
    spy_df = add_vwap(spy_df)
    regime = np.where(
        spy_df["close"] > spy_df["vwap"] * (1 + VWAP_BAND), 1,
        np.where(spy_df["close"] < spy_df["vwap"] * (1 - VWAP_BAND), -1, 0),
    )
    spy_df["regime"] = regime.astype(int)
    return spy_df


# ── Fakeout detection ───────────────────────────────────────────────────────

def _quality(regime, fakeout_dir):
    """
    Signal quality based on SPY regime vs fakeout direction.
    UP_FAKE + SPY_BULL = strong (KO can't hold breakout even with tailwind).
    DN_FAKE + SPY_BEAR = strong (KO holds up despite headwind).
    """
    if fakeout_dir == "UP_FAKE" and regime == 1:
        return "STRONG"
    if fakeout_dir == "DN_FAKE" and regime == -1:
        return "STRONG"
    if regime == 0:
        return "MOD"
    return "WEAK"


def detect_fakeouts(ko_df, spy_df, or_high, or_low, or_cutoff):
    ko_df  = add_vwap(ko_df)
    spy_df = spy_regime_series(spy_df)

    regime_aligned = spy_df["regime"].reindex(ko_df.index, method="ffill").fillna(0)

    signals = []

    for i in range(1, len(ko_df)):
        ts   = ko_df.index[i]
        prev = ko_df.iloc[i - 1]
        cur  = ko_df.iloc[i]

        if ts <= or_cutoff:
            continue  # opening range still forming

        regime   = int(regime_aligned.iloc[i])
        spy_tag  = {1: "SPY_BULL", -1: "SPY_BEAR", 0: "SPY_NEUT"}[regime]
        vwap_now = cur["vwap"]

        # ── OR high upside fakeout ──────────────────────────────────────────
        poke_up = (prev["high"] - or_high) / or_high
        if poke_up > FAKEOUT_POKE and cur["close"] < or_high:
            signals.append({
                "time":        ts.strftime("%H:%M"),
                "type":        "UP_FAKE",
                "level":       f"OR_H {or_high:.2f}",
                "poke%":       f"+{poke_up*100:.3f}%",
                "reversal%":   f"-{(or_high - cur['close']) / or_high * 100:.3f}%",
                "ko_close":    f"{cur['close']:.2f}",
                "vs_vwap":     f"{'▲' if cur['close']>vwap_now else '▼'}{abs(cur['close']-vwap_now):.2f}",
                "spy":         spy_tag,
                "quality":     _quality(regime, "UP_FAKE"),
            })

        # ── OR low downside fakeout ────────────────────────────────────────
        poke_dn = (or_low - prev["low"]) / or_low
        if poke_dn > FAKEOUT_POKE and cur["close"] > or_low:
            signals.append({
                "time":        ts.strftime("%H:%M"),
                "type":        "DN_FAKE",
                "level":       f"OR_L {or_low:.2f}",
                "poke%":       f"-{poke_dn*100:.3f}%",
                "reversal%":   f"+{(cur['close'] - or_low) / or_low * 100:.3f}%",
                "ko_close":    f"{cur['close']:.2f}",
                "vs_vwap":     f"{'▲' if cur['close']>vwap_now else '▼'}{abs(cur['close']-vwap_now):.2f}",
                "spy":         spy_tag,
                "quality":     _quality(regime, "DN_FAKE"),
            })

        # ── VWAP cross fakeout (cross + immediate reversal) ────────────────
        if i >= 2:
            prev2 = ko_df.iloc[i - 2]
            # crossed VWAP upward two bars ago, now back below
            if prev2["close"] < prev2["vwap"] and prev["close"] > prev["vwap"] and cur["close"] < vwap_now:
                mag = (prev["close"] - cur["close"]) / prev["close"] * 100
                signals.append({
                    "time":       ts.strftime("%H:%M"),
                    "type":       "VWAP_XFAKE↓",
                    "level":      f"VWAP {vwap_now:.2f}",
                    "poke%":      f"+{(prev['close']-prev['vwap'])/prev['vwap']*100:.3f}%",
                    "reversal%":  f"-{mag:.3f}%",
                    "ko_close":   f"{cur['close']:.2f}",
                    "vs_vwap":    f"▼{abs(cur['close']-vwap_now):.2f}",
                    "spy":        spy_tag,
                    "quality":    "STRONG" if regime == 1 else ("MOD" if regime == 0 else "WEAK"),
                })
            # crossed VWAP downward two bars ago, now back above
            elif prev2["close"] > prev2["vwap"] and prev["close"] < prev["vwap"] and cur["close"] > vwap_now:
                mag = (cur["close"] - prev["close"]) / prev["close"] * 100
                signals.append({
                    "time":       ts.strftime("%H:%M"),
                    "type":       "VWAP_XFAKE↑",
                    "level":      f"VWAP {vwap_now:.2f}",
                    "poke%":      f"-{(prev['vwap']-prev['close'])/prev['vwap']*100:.3f}%",
                    "reversal%":  f"+{mag:.3f}%",
                    "ko_close":   f"{cur['close']:.2f}",
                    "vs_vwap":    f"▲{abs(cur['close']-vwap_now):.2f}",
                    "spy":        spy_tag,
                    "quality":    "STRONG" if regime == -1 else ("MOD" if regime == 0 else "WEAK"),
                })

    return signals


# ── Analytics ────────────────────────────────────────────────────────────────

def correlation_report(ko_df, spy_df):
    ko_ret  = ko_df["close"].pct_change().dropna()
    spy_ret = spy_df["close"].pct_change().reindex(ko_ret.index, method="ffill").dropna()
    aligned = pd.concat([ko_ret, spy_ret], axis=1).dropna()
    aligned.columns = ["ko", "spy"]

    r, pval  = stats.pearsonr(aligned["ko"], aligned["spy"])
    roll_r   = aligned["ko"].rolling(12).corr(aligned["spy"]).dropna()

    # beta
    cov  = aligned.cov().iloc[0, 1]
    beta = cov / aligned["spy"].var()

    return r, pval, roll_r, beta


def price_summary(ko_df, spy_df, or_high, or_low):
    ko  = ko_df.iloc[-1]
    spy = spy_df.iloc[-1]
    ts  = ko_df.index[-1].strftime("%H:%M")

    ko_pos = (
        "ABOVE OR" if ko["close"] > or_high
        else "BELOW OR" if ko["close"] < or_low
        else "INSIDE OR"
    )
    ko_vs_vwap  = "▲ VWAP" if ko["close"]  > ko["vwap"]  else "▼ VWAP"
    spy_vs_vwap = "▲ VWAP" if spy["close"] > spy["vwap"] else "▼ VWAP"

    return ts, ko, spy, ko_pos, ko_vs_vwap, spy_vs_vwap


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KO/SPY intraday fakeout analysis")
    parser.add_argument("--days", type=int, default=4,
                        help="Look-back window in calendar days (default: 4)")
    args = parser.parse_args()

    client = _get_client()
    today  = date.today()
    start  = datetime.combine(today - timedelta(days=args.days), datetime.min.time())
    end    = datetime.combine(today, datetime.max.time())

    print(f"\n{'═'*62}")
    print(f"  KO / SPY  INTRADAY FAKEOUT SIGNAL ANALYSIS")
    print(f"{'═'*62}")

    print("  Fetching KO  bars … ", end="", flush=True)
    ko_raw = fetch_bars(client, "KO", start, end)
    ko_df, session_date = latest_session(ko_raw)
    print(f"✓  {len(ko_df)} bars  ({session_date})")

    print("  Fetching SPY bars … ", end="", flush=True)
    spy_raw = fetch_bars(client, "SPY", start, end)
    spy_df, _ = latest_session(spy_raw)
    print(f"✓  {len(spy_df)} bars\n")

    if ko_df.empty or spy_df.empty:
        sys.exit("  No data — market closed or IEX feed unavailable.")

    # ── Opening range ─────────────────────────────────────────────────────
    or_high, or_low, or_cutoff = opening_range(ko_df)
    or_range_pct = (or_high - or_low) / or_low * 100
    print(f"  Opening Range ({OR_MINUTES} min)  "
          f"High={or_high:.2f}  Low={or_low:.2f}  "
          f"Range={or_high-or_low:.3f}  ({or_range_pct:.3f}%)")

    # ── Correlation / beta ────────────────────────────────────────────────
    r, pval, roll_r, beta = correlation_report(ko_df, spy_df)
    print(f"  KO/SPY corr (session)  r={r:.3f}  p={pval:.4f}  "
          f"beta={beta:.3f}")
    print(f"  Rolling 12-bar corr    "
          f"mean={roll_r.mean():.3f}  "
          f"min={roll_r.min():.3f}  "
          f"max={roll_r.max():.3f}\n")

    # ── Fakeout signals ───────────────────────────────────────────────────
    signals = detect_fakeouts(ko_df, spy_df, or_high, or_low, or_cutoff)

    print(f"{'─'*62}")
    if not signals:
        print("  No fakeout signals detected this session.")
    else:
        strong = [s for s in signals if s["quality"] == "STRONG"]
        print(f"  FAKEOUT SIGNALS  ({len(signals)} total  |  {len(strong)} STRONG)")
        print(f"{'─'*62}")
        print(tabulate(signals, headers="keys", tablefmt="simple"))
        if strong:
            print(f"\n  ★  {len(strong)} STRONG signal(s) — SPY regime confirms fakeout direction")

    # ── Current state ─────────────────────────────────────────────────────
    spy_df = add_vwap(spy_df)
    ts, ko, spy, ko_pos, ko_vs_vwap, spy_vs_vwap = price_summary(
        add_vwap(ko_df), spy_df, or_high, or_low
    )
    print(f"\n{'─'*62}")
    print(f"  CURRENT STATE  {ts} ET")
    print(f"{'─'*62}")
    print(f"  KO   close={ko['close']:.2f}   vwap={ko['vwap']:.2f}   {ko_vs_vwap}   {ko_pos}")
    print(f"  SPY  close={spy['close']:.2f}  vwap={spy['vwap']:.2f}  {spy_vs_vwap}")
    print()


if __name__ == "__main__":
    main()
