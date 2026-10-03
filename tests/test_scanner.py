import numpy as np
import pandas as pd

from scanner import Settings, _levels, add_indicators, backtest, position_size, scan_ticker


def make_prices(n=400, drift=0.001, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(drift + 0.015 * rng.standard_normal(n)))
    open_ = close * (1 + 0.003 * rng.standard_normal(n))
    high = np.maximum(open_, close) * (1 + np.abs(0.006 * rng.standard_normal(n)))
    low = np.minimum(open_, close) * (1 - np.abs(0.006 * rng.standard_normal(n)))
    vol = rng.integers(1_000_000, 3_000_000, n).astype(float)
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)


def test_levels_respect_tight_stop_cap():
    s = Settings(max_stop_atr=1.0, reward_risk=2.0)
    row = pd.Series({"High": 101.0, "Low": 100.0, "ATR": 2.0})
    entry, stop, target = _levels(row, "LONG", s)
    assert entry > row.High and stop < row.Low
    assert np.isclose(target - entry, 2 * (entry - stop))
    # A bar wider than the cap is rejected.
    assert _levels(pd.Series({"High": 105.0, "Low": 100.0, "ATR": 2.0}), "LONG", s) is None
    entry, stop, target = _levels(row, "SHORT", s)
    assert entry < row.Low and stop > row.High and target < entry


def test_scan_produces_valid_ideas():
    s = Settings()
    ideas = []
    for seed in range(15):
        ideas += scan_ticker(f"T{seed}", make_prices(seed=seed), s, lookback_bars=5)
    assert ideas, "expected at least one idea across synthetic tickers"
    for i in ideas:
        assert i.stop_atr <= s.max_stop_atr + 1e-9
        if i.direction == "LONG":
            assert i.stop < i.entry < i.target
        else:
            assert i.target < i.entry < i.stop


def test_backtest_r_values_bounded():
    s = Settings()
    res = backtest(add_indicators(make_prices(n=800, seed=3)), s)
    for rs in res.values():
        for r in rs:
            assert r >= -1.0 - 1e-9 and r <= s.reward_risk + 1e-9


def test_short_history_returns_nothing():
    assert scan_ticker("X", make_prices(n=100), Settings()) == []


def test_position_size():
    assert position_size(10_000, 1, 2.0, 50) == 50  # $100 risk / $2
    assert position_size(10_000, 1, 0.01, 50) == 200  # capped by cash
    assert position_size(10_000, 1, 0, 50) == 0


def test_status_filters_stale_ideas():
    from scanner import _status
    bars = lambda rows: pd.DataFrame(rows, columns=["High", "Low"])
    assert _status(bars([]), "LONG", 10, 9, 12) == "Pending"
    assert _status(bars([[9.8, 9.5]]), "LONG", 10, 9, 12) is None  # never filled
    assert _status(bars([[10.5, 9.5]]), "LONG", 10, 9, 12) == "Active"
    assert _status(bars([[10.5, 9.5], [12.5, 11]]), "LONG", 10, 9, 12) is None  # target hit
    assert _status(bars([[10.5, 8.5]]), "LONG", 10, 9, 12) is None  # stopped out
    assert _status(bars([[10.5, 9.5]]), "SHORT", 10, 11, 8) == "Active"
