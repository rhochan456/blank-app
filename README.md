# 🎯 Tight-Stop Trade Ideas

A Streamlit trading-idea scanner. It scans a watchlist (stocks, ETFs, crypto via Yahoo Finance) for
setups whose **stop loss is tight**, meaning no wider than a set number of ATRs, and gives you entry, stop,
target and position size for each one.

**Setups:** trend pullback to the 20 EMA, volume breakout/breakdown, and inside bars (long and short).

**How a trade is built:**
- Entry: a stop order just beyond the signal bar, valid for the next bar only
- Stop: just beyond the other side of the signal bar. The idea is **dropped if the stop is wider than your max ATR**
- Target: your reward:risk multiple (default 2R)
- Size: shares so a stop-out costs your chosen % of the account
- Past stats: the same rules replayed on that ticker's history (win rate and average R)

> ⚠️ Educational tool, not financial advice. It does not place orders.

## Run it

```
$ curl -LsSf https://astral.sh/uv/install.sh | sh   # if you don't have uv
$ uv sync
$ uv run streamlit run streamlit_app.py
```

Tests: `uv run pytest`
