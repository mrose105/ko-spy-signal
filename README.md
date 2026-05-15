# ko-spy-signal

KO / SPY intraday and historical signal analysis.

## Structure

```
ko-spy-signal/
├── Python/
│   ├── ko_spy_fakeout.py   # intraday 5-min fakeout signals (today's session)
│   └── ko_spy_merged.py    # multi-mode: historical backtest + intraday signals
├── R/
│   └── ko_spyRegression.R  # regression analysis and statistical modeling
└── output/
    └── ko_spy_charts.pdf   # charts generated from R analysis
```

## Python scripts

### Dependencies

```bash
pip install alpaca-py pandas numpy scipy tabulate
```

### Environment

```bash
export ALPACA_API_KEY=your_key
export ALPACA_SECRET_KEY=your_secret
```

### Usage

```bash
# Historical 5-year daily backtest
python Python/ko_spy_merged.py --mode historical --years 5

# Today's intraday fakeout signals (5-min bars)
python Python/ko_spy_merged.py --mode intraday

# Standalone intraday fakeout scanner
python Python/ko_spy_fakeout.py
```

## Signals

**Fakeout types detected:**

| Type | Logic |
|------|-------|
| `UP_FAKE` | Bar pokes above rolling high ≥0.1%, next bar closes back below |
| `DN_FAKE` | Bar pokes below rolling low ≥0.1%, next bar closes back above |
| `VWAP↑/↓FAKE` | KO crosses VWAP then reverses back within one bar |

**Quality grading** (STRONG / MOD / WEAK) is based on SPY regime confirmation:
- `UP_FAKE` + SPY bullish = STRONG (KO can't hold breakout despite tailwind → relative weakness)
- `DN_FAKE` + SPY bearish = STRONG (KO holds despite headwind → relative strength)

## Historical output sections

1. **Performance** — KO vs SPY: ann. return, vol, Sharpe, max drawdown, beta
2. **Rolling correlation** — 20 / 60 / 252-day windows with current reading
3. **KO/SPY ratio** — Z-score vs 5-year distribution, year-by-year range
4. **Fakeout backtest** — win rate, avg 1d/3d/5d forward returns, regime filter
5. **Regime breakdown** — KO performance in SPY bull / neutral / bear environments
