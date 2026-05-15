#!/usr/bin/env python3
"""
ko_spy_merged.py — KO/SPY multi-mode signal analysis

Modes:
    historical  — multi-year daily analysis: perf, corr, beta, ratio, fakeout backtest
    intraday    — current session fakeout signals (5-min bars)

Usage:
    python ko_spy_merged.py --mode historical --years 5
    python ko_spy_merged.py --mode intraday
"""

import os, sys, argparse
from datetime import datetime, date, timedelta

import pandas as pd
import numpy as np
from scipy import stats
from tabulate import tabulate

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# ── Config ──────────────────────────────────────────────────────────────────
OR_MINUTES      = 30       # intraday opening-range window (minutes)
FAKEOUT_WINDOW  = 20       # rolling N-day high/low for daily fakeout level
FAKEOUT_POKE    = 0.001    # min breach beyond level to qualify (0.1%)
VWAP_BAND       = 0.002    # ±0.2% band for SPY neutral regime
RISK_FREE       = 0.045    # annualized risk-free rate for Sharpe


# ── Client ───────────────────────────────────────────────────────────────────

def _client():
    key    = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_SECRET_KEY", "")
    if not key or not secret:
        sys.exit("ERROR: Set ALPACA_API_KEY and ALPACA_SECRET_KEY env vars.")
    return StockHistoricalDataClient(key, secret)


# ── Data fetching ─────────────────────────────────────────────────────────────

def _extract(raw, symbol):
    df = raw.df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")
    return pd.to_datetime(df.index), df


def fetch_daily(client, symbol, start, end):
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start, end=end,
        feed="iex",
        adjustment="split",
    )
    raw = client.get_stock_bars(req)
    idx, df = _extract(raw, symbol)
    df.index = idx.tz_localize(None).normalize()
    return df.sort_index()


def fetch_intraday(client, symbol, start, end):
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=start, end=end,
        feed="iex",
    )
    raw = client.get_stock_bars(req)
    idx, df = _extract(raw, symbol)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    df.index = idx.tz_convert("US/Eastern")
    return df.between_time("09:30", "16:00")


# ── Core metrics ──────────────────────────────────────────────────────────────

def ann_ret(r):   return (1 + r).prod() ** (252 / len(r)) - 1
def ann_vol(r):   return r.std() * np.sqrt(252)
def sharpe(r):
    v = ann_vol(r)
    return (ann_ret(r) - RISK_FREE) / v if v > 0 else np.nan
def max_dd(prices):
    return ((prices - prices.cummax()) / prices.cummax()).min()
def beta_ols(ko_r, spy_r):
    return np.cov(ko_r, spy_r)[0, 1] / spy_r.var()


def add_vwap(df):
    df = df.copy()
    tp = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (tp * df["volume"]).cumsum() / df["volume"].cumsum()
    return df


# ── HISTORICAL MODE ───────────────────────────────────────────────────────────

def align(ko, spy):
    idx = ko.index.intersection(spy.index)
    return ko.loc[idx], spy.loc[idx]


def perf_table(ko_df, spy_df):
    ko_r  = ko_df["close"].pct_change().dropna()
    spy_r = spy_df["close"].pct_change().reindex(ko_r.index).dropna()
    al    = pd.concat([ko_r, spy_r], axis=1).dropna()
    al.columns = ["ko", "spy"]
    b = beta_ols(al["ko"], al["spy"])

    rows = []
    for sym, r, px in [("KO", al["ko"], ko_df["close"]), ("SPY", al["spy"], spy_df["close"])]:
        px = px.reindex(al.index)
        rows.append({
            "symbol":  sym,
            "ann_ret": f"{ann_ret(r)*100:+.1f}%",
            "ann_vol": f"{ann_vol(r)*100:.1f}%",
            "sharpe":  f"{sharpe(r):.2f}",
            "max_dd":  f"{max_dd(px)*100:.1f}%",
            "beta":    f"{b:.3f}" if sym == "KO" else "1.000",
        })
    return rows


def corr_table(ko_df, spy_df, windows=(20, 60, 252)):
    ko_r  = ko_df["close"].pct_change().dropna()
    spy_r = spy_df["close"].pct_change().reindex(ko_r.index).dropna()
    al    = pd.concat([ko_r, spy_r], axis=1).dropna()
    al.columns = ["ko", "spy"]

    rows = []
    for w in windows:
        roll = al["ko"].rolling(w).corr(al["spy"]).dropna()
        n = min(w, len(al))
        cur_r, _ = stats.pearsonr(al["ko"].iloc[-n:], al["spy"].iloc[-n:])
        rows.append({
            "window":  f"{w}-day",
            "mean":    f"{roll.mean():.3f}",
            "min":     f"{roll.min():.3f}",
            "max":     f"{roll.max():.3f}",
            "current": f"{cur_r:.3f}",
        })
    return rows


def ratio_stats(ko_df, spy_df):
    ratio = (ko_df["close"] / spy_df["close"].reindex(ko_df.index)).dropna()
    cur   = ratio.iloc[-1]
    mu    = ratio.mean()
    sig   = ratio.std()
    z     = (cur - mu) / sig
    pct   = stats.percentileofscore(ratio.values, cur)

    if   z >  2: tag = "EXTREME HIGH  → mean-reversion: short KO / long SPY"
    elif z >  1: tag = "HIGH"
    elif z < -2: tag = "EXTREME LOW   → mean-reversion: long KO / short SPY"
    elif z < -1: tag = "LOW"
    else:         tag = "NEUTRAL (within 1σ)"

    return cur, mu, sig, z, pct, tag, ratio


def daily_fakeouts(ko_df, spy_df):
    df = ko_df[["open", "high", "low", "close"]].copy()

    # Breakout levels formed strictly from bars before each candidate bar (no look-ahead)
    df["lvl_high"] = df["close"].shift(1).rolling(FAKEOUT_WINDOW).max()
    df["lvl_low"]  = df["close"].shift(1).rolling(FAKEOUT_WINDOW).min()

    # SPY regime: rolling 20-day annualised return
    spy_ma = spy_df["close"].pct_change().reindex(df.index).rolling(20).mean() * 252

    sigs = []
    for i in range(FAKEOUT_WINDOW + 2, len(df) - 5):
        cur  = df.iloc[i]
        prev = df.iloc[i - 1]
        ts   = df.index[i]

        lh = prev["lvl_high"]
        ll = prev["lvl_low"]
        if pd.isna(lh) or pd.isna(ll):
            continue

        sm = spy_ma.iloc[i]
        reg = "BULL" if sm > 0.05 else ("BEAR" if sm < -0.05 else "NEUT")

        def _fwd(n):
            return df["close"].iloc[i + n] / cur["close"] - 1 if i + n < len(df) else np.nan

        # Upside fakeout: prev bar's high poked above level, current bar closes back below
        if prev["high"] > lh * (1 + FAKEOUT_POKE) and cur["close"] < lh:
            sigs.append({"date": ts.date(), "type": "UP_FAKE",
                         "level": round(lh, 2), "poke%": round((prev["high"] / lh - 1) * 100, 3),
                         "close": round(cur["close"], 2),
                         "fwd_1d": _fwd(1), "fwd_3d": _fwd(3), "fwd_5d": _fwd(5),
                         "spy_reg": reg})

        # Downside fakeout: prev bar's low poked below level, current bar closes back above
        elif prev["low"] < ll * (1 - FAKEOUT_POKE) and cur["close"] > ll:
            sigs.append({"date": ts.date(), "type": "DN_FAKE",
                         "level": round(ll, 2), "poke%": round((1 - prev["low"] / ll) * 100, 3),
                         "close": round(cur["close"], 2),
                         "fwd_1d": _fwd(1), "fwd_3d": _fwd(3), "fwd_5d": _fwd(5),
                         "spy_reg": reg})
    return sigs


def fakeout_summary(sigs):
    rows = []
    for direction, trade_sign in [("UP_FAKE", -1), ("DN_FAKE", +1)]:
        sub = [s for s in sigs if s["type"] == direction]
        if not sub:
            continue
        f1 = [s["fwd_1d"] for s in sub if not np.isnan(s["fwd_1d"])]
        f3 = [s["fwd_3d"] for s in sub if not np.isnan(s["fwd_3d"])]
        f5 = [s["fwd_5d"] for s in sub if not np.isnan(s["fwd_5d"])]
        # Win = price moves in the expected direction after fakeout
        win1 = sum(1 for r in f1 if trade_sign * r > 0) / len(f1) if f1 else np.nan
        rows.append({
            "type":    direction,
            "n":       len(sub),
            "win_1d":  f"{win1*100:.0f}%" if not np.isnan(win1) else "—",
            "avg_1d":  f"{np.mean(f1)*100:+.2f}%" if f1 else "—",
            "avg_3d":  f"{np.mean(f3)*100:+.2f}%" if f3 else "—",
            "avg_5d":  f"{np.mean(f5)*100:+.2f}%" if f5 else "—",
            "note":    "short signal" if direction == "UP_FAKE" else "long signal",
        })
    return rows


def regime_breakdown(ko_df, spy_df):
    ko_r    = ko_df["close"].pct_change()
    spy_ma  = spy_df["close"].pct_change().reindex(ko_r.index).rolling(20).mean() * 252
    regime  = pd.cut(spy_ma, bins=[-np.inf, -0.05, 0.05, np.inf],
                     labels=["BEAR (<-5%)", "NEUT", "BULL (>+5%)"])

    rows = []
    for reg in ["BULL (>+5%)", "NEUT", "BEAR (<-5%)"]:
        sub = ko_r[regime == reg].dropna()
        if len(sub) < 10:
            continue
        rows.append({
            "spy_regime": reg,
            "n_days":     len(sub),
            "ko_ann_ret": f"{ann_ret(sub)*100:+.1f}%",
            "ko_ann_vol": f"{ann_vol(sub)*100:.1f}%",
            "ko_sharpe":  f"{sharpe(sub):.2f}",
            "hit_rate":   f"{(sub > 0).mean()*100:.0f}%",
        })
    return rows


def run_historical(client, years):
    today = date.today()
    start = datetime.combine(today - timedelta(days=int(years * 365.25)), datetime.min.time())
    end   = datetime.combine(today, datetime.max.time())

    W = 64
    print(f"\n{'═'*W}")
    print(f"  KO / SPY  HISTORICAL ANALYSIS  ({years:.0f}-year daily bars)")
    print(f"{'═'*W}")

    print("  Fetching KO  daily … ", end="", flush=True)
    ko_df = fetch_daily(client, "KO", start, end)
    print(f"✓  {len(ko_df)} bars  ({ko_df.index[0].date()} → {ko_df.index[-1].date()})")

    print("  Fetching SPY daily … ", end="", flush=True)
    spy_df = fetch_daily(client, "SPY", start, end)
    print(f"✓  {len(spy_df)} bars\n")

    ko_df, spy_df = align(ko_df, spy_df)

    def sec(title):
        print(f"\n{'─'*W}")
        print(f"  {title}")
        print(f"{'─'*W}")

    # ── Performance ───────────────────────────────────────────────────────
    sec("PERFORMANCE SUMMARY")
    print(tabulate(perf_table(ko_df, spy_df), headers="keys", tablefmt="simple"))

    # ── Correlation ───────────────────────────────────────────────────────
    sec("ROLLING CORRELATION  (KO vs SPY daily returns)")
    print(tabulate(corr_table(ko_df, spy_df), headers="keys", tablefmt="simple"))

    # ── Ratio ─────────────────────────────────────────────────────────────
    sec("KO / SPY PRICE RATIO  (mean-reversion positioning)")
    cur, mu, sig, z, pct, tag, ratio = ratio_stats(ko_df, spy_df)
    print(f"  Current ratio  : {cur:.4f}")
    print(f"  {years:.0f}yr mean      : {mu:.4f}  ±{sig:.4f}")
    print(f"  Z-score        : {z:+.2f}   ({pct:.0f}th percentile of {years:.0f}yr range)")
    print(f"  Signal         : {tag}")

    # Yearly ratio range
    ratio_df = ratio.to_frame("ratio")
    ratio_df["year"] = ratio_df.index.year
    yr_tbl = ratio_df.groupby("year")["ratio"].agg(["min", "mean", "max"])
    yr_tbl.columns = ["yr_low", "yr_mean", "yr_high"]
    yr_tbl = yr_tbl.round(4).reset_index()
    yr_tbl["year"] = yr_tbl["year"].astype(str)
    print(f"\n  Year-by-year ratio:")
    print(tabulate(yr_tbl.to_dict("records"), headers="keys", tablefmt="simple"))

    # ── Fakeout backtest ──────────────────────────────────────────────────
    sec(f"DAILY FAKEOUT BACKTEST  ({FAKEOUT_WINDOW}-day rolling range, ≥{FAKEOUT_POKE*100:.1f}% poke)")
    sigs = daily_fakeouts(ko_df, spy_df)

    if not sigs:
        print("  No signals found.")
    else:
        summ = fakeout_summary(sigs)
        print(tabulate(summ, headers="keys", tablefmt="simple"))

        # Last 12 signals
        print(f"\n  Most recent signals (last 12 of {len(sigs)} total):")
        recent = []
        for s in sigs[-12:]:
            f1 = s["fwd_1d"]
            f5 = s["fwd_5d"]
            recent.append({
                "date":    str(s["date"]),
                "type":    s["type"],
                "level":   s["level"],
                "poke%":   f"{s['poke%']:+.3f}%",
                "close":   s["close"],
                "fwd_1d":  f"{f1*100:+.2f}%" if not np.isnan(f1) else "pending",
                "fwd_5d":  f"{f5*100:+.2f}%" if not np.isnan(f5) else "pending",
                "spy_reg": s["spy_reg"],
            })
        print(tabulate(recent, headers="keys", tablefmt="simple"))

        # Regime-filtered stats
        print(f"\n  Win rate by SPY regime (1-day, direction-adjusted):")
        for direction, trade_sign in [("UP_FAKE", -1), ("DN_FAKE", +1)]:
            for reg in ["BULL", "NEUT", "BEAR"]:
                sub = [s for s in sigs if s["type"] == direction and s["spy_reg"] == reg]
                if len(sub) < 3:
                    continue
                f1  = [s["fwd_1d"] for s in sub if not np.isnan(s["fwd_1d"])]
                win = sum(1 for r in f1 if trade_sign * r > 0) / len(f1) if f1 else np.nan
                print(f"    {direction}  SPY={reg:4s}  n={len(sub):3d}  "
                      f"win={win*100:.0f}%  avg={np.mean(f1)*100:+.2f}%" if f1 else "")

    # ── Regime breakdown ──────────────────────────────────────────────────
    sec("KO PERFORMANCE BY SPY REGIME")
    print(tabulate(regime_breakdown(ko_df, spy_df), headers="keys", tablefmt="simple"))

    print()


# ── INTRADAY MODE ─────────────────────────────────────────────────────────────

def intraday_spy_regime(spy_df):
    spy_df = add_vwap(spy_df)
    regime = np.where(spy_df["close"] > spy_df["vwap"] * (1 + VWAP_BAND), 1,
              np.where(spy_df["close"] < spy_df["vwap"] * (1 - VWAP_BAND), -1, 0))
    spy_df["regime"] = regime.astype(int)
    return spy_df


def opening_range(df):
    start   = df.index[0].normalize().replace(hour=9, minute=30)
    cutoff  = start + timedelta(minutes=OR_MINUTES)
    or_bars = df[df.index <= cutoff]
    return or_bars["high"].max(), or_bars["low"].min(), cutoff


def intraday_fakeouts(ko_df, spy_df, or_high, or_low, or_cutoff):
    ko_df  = add_vwap(ko_df)
    spy_df = intraday_spy_regime(spy_df)
    reg_s  = spy_df["regime"].reindex(ko_df.index, method="ffill").fillna(0).astype(int)
    _tag   = {1: "SPY_BULL", -1: "SPY_BEAR", 0: "SPY_NEUT"}

    def quality(regime, direction):
        if direction == "UP_FAKE" and regime ==  1: return "STRONG"
        if direction == "DN_FAKE" and regime == -1: return "STRONG"
        return "MOD" if regime == 0 else "WEAK"

    sigs = []
    for i in range(2, len(ko_df)):
        ts    = ko_df.index[i]
        cur   = ko_df.iloc[i]
        prev  = ko_df.iloc[i - 1]
        prev2 = ko_df.iloc[i - 2]
        if ts <= or_cutoff:
            continue

        reg      = reg_s.iloc[i]
        spy_tag  = _tag[int(reg)]
        vwap_now = cur["vwap"]
        vs_vwap  = f"{'▲' if cur['close']>vwap_now else '▼'}{abs(cur['close']-vwap_now):.2f}"

        # OR high upside fakeout
        poke_up = (prev["high"] - or_high) / or_high
        if poke_up > 0.0005 and cur["close"] < or_high:
            sigs.append({"time": ts.strftime("%H:%M"), "type": "UP_FAKE",
                         "level": f"OR_H {or_high:.2f}",
                         "poke%": f"+{poke_up*100:.3f}%",
                         "rev%":  f"-{(or_high-cur['close'])/or_high*100:.3f}%",
                         "close": f"{cur['close']:.2f}", "vs_vwap": vs_vwap,
                         "spy": spy_tag, "quality": quality(reg, "UP_FAKE")})

        # OR low downside fakeout
        poke_dn = (or_low - prev["low"]) / or_low
        if poke_dn > 0.0005 and cur["close"] > or_low:
            sigs.append({"time": ts.strftime("%H:%M"), "type": "DN_FAKE",
                         "level": f"OR_L {or_low:.2f}",
                         "poke%": f"-{poke_dn*100:.3f}%",
                         "rev%":  f"+{(cur['close']-or_low)/or_low*100:.3f}%",
                         "close": f"{cur['close']:.2f}", "vs_vwap": vs_vwap,
                         "spy": spy_tag, "quality": quality(reg, "DN_FAKE")})

        # VWAP cross fakeout
        if prev2["close"] < prev2["vwap"] and prev["close"] > prev["vwap"] and cur["close"] < vwap_now:
            mag = (prev["close"] - cur["close"]) / prev["close"] * 100
            sigs.append({"time": ts.strftime("%H:%M"), "type": "VWAP↓FAKE",
                         "level": f"VWAP {vwap_now:.2f}",
                         "poke%": f"+{(prev['close']-prev['vwap'])/prev['vwap']*100:.3f}%",
                         "rev%":  f"-{mag:.3f}%",
                         "close": f"{cur['close']:.2f}", "vs_vwap": f"▼{abs(cur['close']-vwap_now):.2f}",
                         "spy": spy_tag,
                         "quality": "STRONG" if reg == 1 else ("MOD" if reg == 0 else "WEAK")})

        elif prev2["close"] > prev2["vwap"] and prev["close"] < prev["vwap"] and cur["close"] > vwap_now:
            mag = (cur["close"] - prev["close"]) / prev["close"] * 100
            sigs.append({"time": ts.strftime("%H:%M"), "type": "VWAP↑FAKE",
                         "level": f"VWAP {vwap_now:.2f}",
                         "poke%": f"-{(prev['vwap']-prev['close'])/prev['vwap']*100:.3f}%",
                         "rev%":  f"+{mag:.3f}%",
                         "close": f"{cur['close']:.2f}", "vs_vwap": f"▲{abs(cur['close']-vwap_now):.2f}",
                         "spy": spy_tag,
                         "quality": "STRONG" if reg == -1 else ("MOD" if reg == 0 else "WEAK")})
    return sigs


def run_intraday(client):
    today = date.today()
    start = datetime.combine(today - timedelta(days=4), datetime.min.time())
    end   = datetime.combine(today, datetime.max.time())

    W = 62
    print(f"\n{'═'*W}")
    print(f"  KO / SPY  INTRADAY FAKEOUT SIGNALS")
    print(f"{'═'*W}")

    print("  Fetching KO  5min … ", end="", flush=True)
    ko_raw = fetch_intraday(client, "KO", start, end)
    last   = ko_raw.index.date[-1]
    ko_df  = ko_raw[ko_raw.index.date == last]
    print(f"✓  {len(ko_df)} bars  ({last})")

    print("  Fetching SPY 5min … ", end="", flush=True)
    spy_raw = fetch_intraday(client, "SPY", start, end)
    spy_df  = spy_raw[spy_raw.index.date == last]
    print(f"✓  {len(spy_df)} bars\n")

    if ko_df.empty or spy_df.empty:
        sys.exit("  No intraday data — market closed or feed unavailable.")

    or_high, or_low, or_cutoff = opening_range(ko_df)
    print(f"  OR ({OR_MINUTES}min)  H={or_high:.2f}  L={or_low:.2f}  "
          f"Rng={or_high-or_low:.3f}  ({(or_high-or_low)/or_low*100:.3f}%)")

    ko_r  = ko_df["close"].pct_change().dropna()
    spy_r = spy_df["close"].pct_change().reindex(ko_r.index).dropna()
    al    = pd.concat([ko_r, spy_r], axis=1).dropna()
    al.columns = ["ko", "spy"]
    r, p  = stats.pearsonr(al["ko"], al["spy"])
    b     = beta_ols(al["ko"], al["spy"])
    print(f"  Session  corr={r:.3f}  p={p:.4f}  beta={b:.3f}\n")

    sigs = intraday_fakeouts(ko_df, spy_df, or_high, or_low, or_cutoff)
    print(f"{'─'*W}")
    print(f"  FAKEOUT SIGNALS  ({len(sigs)} detected)")
    print(f"{'─'*W}")

    if sigs:
        print(tabulate(sigs, headers="keys", tablefmt="simple"))
        strong = [s for s in sigs if s["quality"] == "STRONG"]
        if strong:
            print(f"\n  ★  {len(strong)} STRONG — SPY regime confirms fakeout direction")
    else:
        print("  None this session.")

    ko_df  = add_vwap(ko_df)
    spy_df = add_vwap(spy_df)
    kl, sl = ko_df.iloc[-1], spy_df.iloc[-1]
    ts     = ko_df.index[-1].strftime("%H:%M")
    ko_pos = ("ABOVE OR" if kl["close"] > or_high
              else "BELOW OR" if kl["close"] < or_low else "INSIDE OR")

    print(f"\n{'─'*W}")
    print(f"  STATE  {ts} ET")
    print(f"{'─'*W}")
    print(f"  KO   {kl['close']:.2f}  vwap={kl['vwap']:.2f}  "
          f"{'▲' if kl['close']>kl['vwap'] else '▼'} VWAP  {ko_pos}")
    print(f"  SPY  {sl['close']:.2f}  vwap={sl['vwap']:.2f}  "
          f"{'▲' if sl['close']>sl['vwap'] else '▼'} VWAP")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="KO/SPY multi-mode signal analysis")
    p.add_argument("--mode",  choices=["historical", "intraday"], default="intraday")
    p.add_argument("--years", type=float, default=5,
                   help="Years of history for --mode historical (default 5)")
    args = p.parse_args()

    client = _client()
    if args.mode == "historical":
        run_historical(client, args.years)
    else:
        run_intraday(client)


if __name__ == "__main__":
    main()
