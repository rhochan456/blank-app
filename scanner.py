"""Trade idea scanner: setups with tight, ATR-capped stop losses.

Every idea is built the same way:
  * entry  = a buy-stop just above (or sell-stop just below) the signal bar
  * stop   = just beyond the signal bar's opposite extreme (structure stop)
  * the idea is rejected if that stop is wider than ``max_stop_atr`` x ATR,
    so only setups with a genuinely tight stop survive
  * target = entry +/- ``reward_risk`` x risk
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class Settings:
    max_stop_atr: float = 1.0  # widest stop allowed, in ATRs
    reward_risk: float = 2.0  # target distance as a multiple of risk
    buffer_atr: float = 0.1  # padding beyond the signal bar for entry/stop
    breakout_lookback: int = 20
    volume_mult: float = 1.5  # breakout volume vs 20-bar average
    allow_shorts: bool = True
    max_hold_bars: int = 20  # backtest: give up after this many bars


@dataclass
class Idea:
    ticker: str
    setup: str
    direction: str  # "LONG" or "SHORT"
    date: pd.Timestamp
    last_close: float
    entry: float
    stop: float
    target: float
    risk_per_share: float
    risk_pct: float  # risk as % of entry
    stop_atr: float  # stop distance in ATRs
    rsi: float
    status: str = "Pending"  # "Pending" (order for next bar) or "Active" (already triggered)
    hist_trades: int = 0
    hist_win_rate: float = float("nan")
    hist_expectancy_r: float = float("nan")
    score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA, ATR, RSI and volume averages. Expects Open/High/Low/Close/Volume."""
    df = df.copy()
    c = df["Close"]
    df["EMA20"] = c.ewm(span=20, adjust=False).mean()
    df["EMA50"] = c.ewm(span=50, adjust=False).mean()
    df["EMA200"] = c.ewm(span=200, adjust=False).mean()

    prev_close = c.shift(1)
    tr = pd.concat(
        [df["High"] - df["Low"], (df["High"] - prev_close).abs(), (df["Low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = (100 - 100 / (1 + rs)).fillna(100)

    df["VolAvg20"] = df["Volume"].rolling(20).mean()
    return df


def _signals(df: pd.DataFrame, s: Settings) -> dict[tuple[str, str], pd.Series]:
    """Boolean series per (setup name, direction), True on signal bars."""
    c, o, h, l = df["Close"], df["Open"], df["High"], df["Low"]
    e20, e50, e200 = df["EMA20"], df["EMA50"], df["EMA200"]
    rsi, atr = df["RSI"], df["ATR"]
    hh = df["High"].rolling(s.breakout_lookback).max().shift(1)
    ll = df["Low"].rolling(s.breakout_lookback).min().shift(1)
    vol_ok = df["Volume"] > s.volume_mult * df["VolAvg20"]

    uptrend = (c > e50) & (e50 > e200) & (e50 > e50.shift(5))
    downtrend = (c < e50) & (e50 < e200) & (e50 < e50.shift(5))

    sig = {
        # Pullback into the 20 EMA within an uptrend, closing back up.
        ("Trend pullback", "LONG"): uptrend
        & (l <= e20 + 0.25 * atr)
        & (c > e20)
        & (c > o)
        & rsi.between(40, 65),
        # Fresh close above the prior N-bar high on expanding volume.
        ("Breakout", "LONG"): (c > hh) & (c.shift(1) <= hh.shift(1)) & vol_ok & (c > e50),
        # Inside bar (range contraction) in an uptrend: tightest stops of all.
        ("Inside bar", "LONG"): uptrend & (h < h.shift(1)) & (l > l.shift(1)),
    }
    if s.allow_shorts:
        sig[("Trend pullback", "SHORT")] = (
            downtrend & (h >= e20 - 0.25 * atr) & (c < e20) & (c < o) & rsi.between(35, 60)
        )
        sig[("Breakdown", "SHORT")] = (c < ll) & (c.shift(1) >= ll.shift(1)) & vol_ok & (c < e50)
        sig[("Inside bar", "SHORT")] = downtrend & (h < h.shift(1)) & (l > l.shift(1))
    return sig


def _levels(row: pd.Series, direction: str, s: Settings) -> tuple[float, float, float] | None:
    """Entry/stop/target for a signal bar, or None if the stop isn't tight enough."""
    atr = row["ATR"]
    if not np.isfinite(atr) or atr <= 0:
        return None
    pad = s.buffer_atr * atr
    if direction == "LONG":
        entry, stop = row["High"] + pad, row["Low"] - pad
        risk = entry - stop
        target = entry + s.reward_risk * risk
    else:
        entry, stop = row["Low"] - pad, row["High"] + pad
        risk = stop - entry
        target = entry - s.reward_risk * risk
    if risk <= 0 or risk > s.max_stop_atr * atr:
        return None
    return entry, stop, target


def _simulate(df: pd.DataFrame, i: int, direction: str, levels, s: Settings) -> float | None:
    """R multiple of a trade signalled on bar i, or None if never triggered.

    Conservative: if stop and target are both touched in the same bar, it's a loss.
    """
    entry, stop, target = levels
    risk = abs(entry - stop)
    highs, lows, closes = df["High"].to_numpy(), df["Low"].to_numpy(), df["Close"].to_numpy()
    end = min(len(df), i + 1 + s.max_hold_bars)
    j = i + 1
    # Entry order is valid for the next bar only.
    if j >= end:
        return None
    if direction == "LONG" and highs[j] < entry:
        return None
    if direction == "SHORT" and lows[j] > entry:
        return None
    for k in range(j, end):
        if direction == "LONG":
            if lows[k] <= stop:
                return -1.0
            if highs[k] >= target:
                return s.reward_risk
        else:
            if highs[k] >= stop:
                return -1.0
            if lows[k] <= target:
                return s.reward_risk
    exit_px = closes[end - 1]
    pnl = exit_px - entry if direction == "LONG" else entry - exit_px
    return pnl / risk


def backtest(df: pd.DataFrame, s: Settings) -> dict[tuple[str, str], list[float]]:
    """Historical R results per setup on this ticker (excludes the latest bar)."""
    results: dict[tuple[str, str], list[float]] = {}
    for key, mask in _signals(df, s).items():
        rs = []
        for i in np.flatnonzero(mask.to_numpy()[:-1]):
            if i < 200:  # need EMA200 warm-up
                continue
            lv = _levels(df.iloc[i], key[1], s)
            if lv is None:
                continue
            r = _simulate(df, i, key[1], lv, s)
            if r is not None:
                rs.append(r)
        results[key] = rs
    return results


def scan_ticker(ticker: str, raw: pd.DataFrame, s: Settings, lookback_bars: int = 1) -> list[Idea]:
    """Ideas from the most recent ``lookback_bars`` bars of one ticker."""
    if raw is None or len(raw) < 210:
        return []
    df = add_indicators(raw)
    sigs = _signals(df, s)
    hist = backtest(df, s)
    last_close = float(df["Close"].iloc[-1])
    ideas: list[Idea] = []
    n = len(df)
    for (setup, direction), mask in sigs.items():
        for i in range(max(0, n - lookback_bars), n):
            if not mask.iloc[i]:
                continue
            row = df.iloc[i]
            lv = _levels(row, direction, s)
            if lv is None:
                continue
            entry, stop, target = lv
            status = _status(df.iloc[i + 1 :], direction, entry, stop, target)
            if status is None:
                continue
            risk = abs(entry - stop)
            rs = hist.get((setup, direction), [])
            wins = sum(r > 0 for r in rs)
            idea = Idea(
                ticker=ticker,
                setup=setup,
                direction=direction,
                date=df.index[i],
                last_close=last_close,
                entry=round(entry, 4),
                stop=round(stop, 4),
                target=round(target, 4),
                risk_per_share=round(risk, 4),
                risk_pct=round(100 * risk / entry, 2),
                stop_atr=round(risk / row["ATR"], 2),
                rsi=round(float(row["RSI"]), 1),
                status=status,
                hist_trades=len(rs),
                hist_win_rate=round(100 * wins / len(rs), 1) if rs else float("nan"),
                hist_expectancy_r=round(float(np.mean(rs)), 2) if rs else float("nan"),
            )
            idea.score = _score(idea)
            ideas.append(idea)
    return ideas


def _status(later: pd.DataFrame, direction: str, entry: float, stop: float, target: float) -> str | None:
    """Status of an idea given the bars after its signal; None if it's no longer actionable."""
    if later.empty:
        return "Pending"
    first = later.iloc[0]
    triggered = first["High"] >= entry if direction == "LONG" else first["Low"] <= entry
    if not triggered:
        return None  # entry order expired unfilled
    if direction == "LONG":
        done = (later["Low"] <= stop).any() or (later["High"] >= target).any()
    else:
        done = (later["High"] >= stop).any() or (later["Low"] <= target).any()
    return None if done else "Active"


def _score(idea: Idea) -> float:
    """Higher is better: tighter stops plus a positive historical edge."""
    score = 50.0 - 20.0 * idea.stop_atr  # tighter stop = better
    if idea.hist_trades >= 5 and np.isfinite(idea.hist_expectancy_r):
        score += 30.0 * idea.hist_expectancy_r
    return round(score, 1)


def position_size(account: float, risk_pct: float, risk_per_share: float, entry: float) -> int:
    """Shares to buy so a stop-out loses ``risk_pct`` % of the account (never exceeding the account)."""
    if risk_per_share <= 0 or entry <= 0:
        return 0
    by_risk = account * risk_pct / 100 / risk_per_share
    by_cash = account / entry
    return int(max(0, min(by_risk, by_cash)))
