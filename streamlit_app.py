import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from scanner import Settings, add_indicators, position_size, scan_ticker

st.set_page_config(page_title="Tight-Stop Trade Ideas", page_icon="🎯", layout="wide")

DEFAULT_WATCHLIST = (
    "AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, AMD, AVGO, NFLX, "
    "JPM, V, XOM, LLY, COST, SPY, QQQ, IWM, GLD, BTC-USD"
)
INTERVALS = {
    "Daily (swing)": ("1d", "3y"),
    "1 hour (short-term)": ("1h", "730d"),
}


@st.cache_data(ttl=900, show_spinner=False)
def load_prices(tickers: tuple[str, ...], interval: str, period: str) -> dict[str, pd.DataFrame]:
    data = yf.download(
        list(tickers),
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    out = {}
    for t in tickers:
        try:
            multi = isinstance(data.columns, pd.MultiIndex)
            df = data[t] if multi else data
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            if not df.empty:
                out[t] = df
        except KeyError:
            pass
    return out


def chart(ticker: str, df: pd.DataFrame, idea: dict) -> go.Figure:
    df = add_indicators(df).tail(120)
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name=ticker
        )
    )
    for col, color in (("EMA20", "#f4a261"), ("EMA50", "#2a9d8f"), ("EMA200", "#8d99ae")):
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(width=1, color=color)))
    for label, key, color in (("Entry", "entry", "#3a86ff"), ("Stop", "stop", "#e63946"), ("Target", "target", "#2dc653")):
        fig.add_hline(
            y=idea[key], line_dash="dash", line_color=color,
            annotation_text=f"{label} {idea[key]:.2f}", annotation_position="right",
        )
    fig.update_layout(
        height=520, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.05),
    )
    return fig


st.title("🎯 Tight-Stop Trade Ideas")
st.caption(
    "Scans your watchlist for trend pullbacks, breakouts and inside bars, and only keeps setups "
    "whose structural stop is within your max-ATR limit. Each idea is backtested on the same ticker."
)

with st.sidebar:
    st.header("Scan settings")
    tickers_raw = st.text_area("Watchlist (comma separated)", DEFAULT_WATCHLIST, height=120)
    tf_label = st.selectbox("Timeframe", list(INTERVALS))
    lookback = st.slider("Look for signals in last N bars", 1, 5, 2)
    allow_shorts = st.checkbox("Include short ideas", True)

    st.header("Risk")
    max_stop_atr = st.slider("Max stop distance (× ATR)", 0.3, 2.0, 1.0, 0.1,
                             help="Lower = tighter stops, fewer ideas.")
    reward_risk = st.slider("Reward : risk target", 1.0, 5.0, 2.0, 0.5)
    account = st.number_input("Account size ($)", 100.0, 1e9, 10_000.0, 500.0)
    risk_pct = st.slider("Risk per trade (% of account)", 0.25, 3.0, 1.0, 0.25)

    run = st.button("🔍 Scan now", type="primary", width="stretch")

tickers = tuple(dict.fromkeys(t.strip().upper() for t in tickers_raw.replace("\n", ",").split(",") if t.strip()))
settings = Settings(max_stop_atr=max_stop_atr, reward_risk=reward_risk, allow_shorts=allow_shorts)

if run or "prices" not in st.session_state:
    if not tickers:
        st.warning("Add at least one ticker.")
        st.stop()
    interval, period = INTERVALS[tf_label]
    with st.spinner(f"Downloading {len(tickers)} tickers…"):
        try:
            st.session_state.prices = load_prices(tickers, interval, period)
        except Exception as e:  # network / Yahoo hiccups
            st.error(f"Could not download prices: {e}")
            st.stop()

prices: dict[str, pd.DataFrame] = st.session_state.prices
missing = [t for t in tickers if t not in prices]
if missing:
    st.info(f"No data for: {', '.join(missing)}")

ideas = []
for t, df in prices.items():
    ideas.extend(i.to_dict() for i in scan_ticker(t, df, settings, lookback_bars=lookback))

if not ideas:
    st.warning("No setups match right now. Try a longer watchlist, more lookback bars, or a wider max stop.")
    st.stop()

table = pd.DataFrame(ideas).sort_values("score", ascending=False).reset_index(drop=True)
table["shares"] = [position_size(account, risk_pct, r, e) for r, e in zip(table.risk_per_share, table.entry)]
table["$ at risk"] = (table.shares * table.risk_per_share).round(2)

c1, c2, c3 = st.columns(3)
c1.metric("Ideas found", len(table))
c2.metric("Long / Short", f"{(table.direction == 'LONG').sum()} / {(table.direction == 'SHORT').sum()}")
c3.metric("Avg stop", f"{table.stop_atr.mean():.2f} ATR · {table.risk_pct.mean():.2f}%")

st.subheader("Trade ideas (best first)")
st.dataframe(
    table[["ticker", "direction", "setup", "status", "date", "last_close", "entry", "stop", "target",
           "risk_pct", "stop_atr", "rsi", "hist_trades", "hist_win_rate", "hist_expectancy_r",
           "score", "shares", "$ at risk"]],
    hide_index=True,
    width="stretch",
    column_config={
        "date": st.column_config.DatetimeColumn("signal bar", format="YYYY-MM-DD HH:mm"),
        "risk_pct": st.column_config.NumberColumn("stop %", format="%.2f%%"),
        "stop_atr": st.column_config.NumberColumn("stop ATR"),
        "hist_trades": st.column_config.NumberColumn("past trades"),
        "hist_win_rate": st.column_config.NumberColumn("past win %", format="%.0f%%"),
        "hist_expectancy_r": st.column_config.NumberColumn("past avg R", format="%.2f"),
    },
)

st.subheader("Chart")
labels = [f"{r.ticker} · {r.direction} · {r.setup}" for r in table.itertuples()]
choice = st.selectbox("Idea", range(len(labels)), format_func=lambda i: labels[i])
sel = table.iloc[choice].to_dict()
st.plotly_chart(chart(sel["ticker"], prices[sel["ticker"]], sel), width="stretch")
verb = "Buy stop" if sel["direction"] == "LONG" else "Sell stop"
when = "next bar only" if sel["status"] == "Pending" else "already triggered, trade is live"
st.markdown(
    f"**Plan:** {verb} **{sel['shares']} {sel['ticker']}** at **{sel['entry']:.2f}** "
    f"({when}) · stop **{sel['stop']:.2f}** · target **{sel['target']:.2f}** · "
    f"max loss ≈ **${sel['$ at risk']:.2f}**"
)

with st.expander("How the ideas are built"):
    st.markdown(f"""
- **Trend pullback** – price above EMA50 > EMA200 (rising), dips to the 20 EMA and closes back up (mirrored for shorts).
- **Breakout / Breakdown** – first close beyond the prior {settings.breakout_lookback}-bar high/low on ≥{settings.volume_mult}× average volume.
- **Inside bar** – range contraction inside a trend; usually the tightest stops.
- **Entry** – stop order {settings.buffer_atr} ATR beyond the signal bar, valid for the next bar only.
- **Stop** – {settings.buffer_atr} ATR beyond the other side of the signal bar. Ideas whose stop is wider than **{max_stop_atr} ATR** are discarded.
- **Target** – {reward_risk}R. **Past stats** replay the same rules on this ticker's history (same-bar stop+target counts as a loss; open after {settings.max_hold_bars} bars exits at close).
- **Score** – rewards tighter stops and positive historical expectancy.
""")

st.caption(
    "⚠️ Educational tool, not financial advice. Tight stops get hit more often, gaps can skip past stops, "
    "and past results don't guarantee future ones. Data from Yahoo Finance may be delayed."
)
