"""Streamlit equity research dashboard for NYSE/Nasdaq long and short ideas.

The app deliberately separates market data from filing data. SEC Company Facts
and SEC submissions are used for reported financials when a CIK is available;
Yahoo Finance supplies price history and optional analyst/holder enrichment.
Missing data is shown as ``NA`` and does not silently become a passing signal.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf


st.set_page_config(page_title="Equity Signal Lab", page_icon="◒", layout="wide")
SEC_HEADERS = {"User-Agent": "Equity Signal Lab research app contact@example.com"}
DEFAULT_UNIVERSE = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AVGO", "JPM", "LLY", "TSLA", "AMD", "NFLX"]


def pct(value: Any) -> float | None:
    try:
        value = float(value)
        return value * 100 if abs(value) <= 2 else value
    except (TypeError, ValueError):
        return None


def fmt(value: Any, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{value:,.2f}{suffix}"


@st.cache_data(ttl=86400, show_spinner=False)
def sec_ticker_map() -> pd.DataFrame:
    response = requests.get("https://www.sec.gov/files/company_tickers.json", headers=SEC_HEADERS, timeout=20)
    response.raise_for_status()
    rows = list(response.json().values())
    return pd.DataFrame(rows).rename(columns={"ticker": "symbol", "title": "name", "cik_str": "cik"})


@st.cache_data(ttl=86400, show_spinner=False)
def listed_universe() -> list[str]:
    """Return Nasdaq and NYSE symbols; keep a small fallback if the listing feed is unavailable."""
    try:
        nasdaq = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt", sep="|")
        nasdaq = nasdaq[nasdaq["Test Issue"] == "N"]["NASDAQ Symbol"].dropna().astype(str)
        other = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", sep="|")
        other = other[other["Test Issue"] == "N"]["ACT Symbol"].dropna().astype(str)
        return sorted(set(nasdaq.tolist() + other.tolist()))
    except Exception:
        return DEFAULT_UNIVERSE


@st.cache_data(ttl=3600, show_spinner=False)
def prices(symbol: str, period: str = "5y") -> pd.DataFrame:
    frame = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False, threads=False)
    if frame.empty:
        return pd.DataFrame()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    return frame[[c for c in ["Open", "High", "Low", "Close", "Volume"] if c in frame]].dropna()


@st.cache_data(ttl=86400, show_spinner=False)
def sec_facts(symbol: str) -> dict[str, Any]:
    try:
        mapping = sec_ticker_map()
        row = mapping[mapping.symbol.str.upper() == symbol.upper()].iloc[0]
        cik = f"{int(row.cik):010d}"
        facts = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", headers=SEC_HEADERS, timeout=20).json()
        return {"cik": cik, "name": facts.get("entityName", symbol), "facts": facts.get("facts", {})}
    except Exception:
        return {"cik": None, "name": symbol, "facts": {}}


def fact_series(bundle: dict, names: list[str], unit: str = "USD") -> pd.Series:
    for taxonomy in ["us-gaap", "dei"]:
        for name in names:
            node = bundle.get("facts", {}).get(taxonomy, {}).get(name, {})
            units = node.get("units", {})
            values = units.get(unit) or next(iter(units.values()), [])
            rows = []
            for item in values:
                if item.get("form") in {"10-K", "10-Q", "8-K"} and item.get("fy") is not None:
                    rows.append({"end": item.get("end"), "value": item.get("val"), "form": item.get("form"), "fp": item.get("fp")})
            if rows:
                frame = pd.DataFrame(rows).drop_duplicates("end").sort_values("end")
                return pd.Series(frame.value.astype(float).values, index=pd.to_datetime(frame.end), name=name)
    return pd.Series(dtype=float)


def latest_yoy(series: pd.Series) -> float | None:
    if len(series) < 5:
        return None
    current, prior = float(series.iloc[-1]), float(series.iloc[-5])
    return None if prior == 0 else (current / prior - 1) * 100


def filing_metrics(symbol: str) -> dict[str, Any]:
    bundle = sec_facts(symbol)
    revenue = fact_series(bundle, ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"])
    eps = fact_series(bundle, ["EarningsPerShareDiluted", "EarningsPerShareBasic"], "USD/shares")
    net = fact_series(bundle, ["NetIncomeLoss", "ProfitLoss"])
    gross = fact_series(bundle, ["GrossProfit"])
    operating = fact_series(bundle, ["OperatingIncomeLoss"])
    cfo = fact_series(bundle, ["NetCashProvidedByUsedInOperatingActivities"])
    result = {"cik": bundle.get("cik"), "name": bundle.get("name", symbol), "revenue": revenue, "eps": eps, "net": net, "gross": gross, "operating": operating, "cfo": cfo}
    result["revenue_yoy"] = latest_yoy(revenue)
    result["eps_yoy"] = latest_yoy(eps)
    result["fcf_yoy"] = latest_yoy(cfo)
    result["net_margin"] = (float(net.iloc[-1] / revenue.iloc[-1] * 100) if len(net) and len(revenue) and revenue.iloc[-1] else None)
    result["gross_margin"] = (float(gross.iloc[-1] / revenue.iloc[-1] * 100) if len(gross) and len(revenue) and revenue.iloc[-1] else None)
    result["operating_margin"] = (float(operating.iloc[-1] / revenue.iloc[-1] * 100) if len(operating) and len(revenue) and revenue.iloc[-1] else None)
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def enrichment(symbol: str) -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        ticker = yf.Ticker(symbol)
        raw = ticker.info
        info.update({"price": raw.get("currentPrice") or raw.get("regularMarketPrice"), "target": raw.get("targetMeanPrice"), "recommendation": raw.get("recommendationKey"), "sector": raw.get("sector"), "industry": raw.get("industry"), "market_cap": raw.get("marketCap"), "summary": raw.get("longBusinessSummary")})
        rec = ticker.recommendations
        info["buy_share"] = ((rec["strongBuy"] + rec["buy"]).tail(4).sum() / rec.tail(4)[["strongBuy", "buy", "hold", "sell", "strongSell"]].sum().sum() * 100) if rec is not None and not rec.empty and "strongBuy" in rec else None
        ins = ticker.insider_transactions
        if ins is not None and not ins.empty:
            info["insider_90d"] = float(ins[ins.index >= pd.Timestamp.today() - pd.Timedelta(days=90)].get("Value", pd.Series(dtype=float)).fillna(0).sum())
        else:
            info["insider_90d"] = None
    except Exception:
        pass
    return info


def score_record(symbol: str, direction: str, thresholds: dict[str, float]) -> dict[str, Any]:
    metrics = filing_metrics(symbol)
    extra = enrichment(symbol)
    price = extra.get("price")
    target = extra.get("target")
    target_gap = (target / price - 1) * 100 if price and target else None
    is_long = direction == "Long"
    tests = {
        "Revenue growth": metrics.get("revenue_yoy"), "EPS growth": metrics.get("eps_yoy"), "Earnings beats": None,
        "FCF growth": metrics.get("fcf_yoy"), "Margins positive": min([x for x in [metrics.get("gross_margin"), metrics.get("operating_margin"), metrics.get("net_margin")] if x is not None], default=None),
        "Margins expanding": None, "Analyst consensus": 1 if extra.get("recommendation") in ({"buy", "strong_buy", "hold"} if is_long else {"hold", "sell", "strong_sell"}) else 0,
        "Buy share trend": None, "Target trend": None, "Price target gap": target_gap, "Insider cluster": extra.get("insider_90d"), "Institution net flow": None,
    }
    pass_map = {}
    for key, value in tests.items():
        threshold = thresholds.get(key, 0)
        if key == "Analyst consensus": pass_map[key] = bool(value)
        elif key in {"Insider cluster", "Institution net flow"}: pass_map[key] = value is not None and (value >= threshold if is_long else value <= -threshold)
        elif key in {"Margins positive", "Margins expanding"}: pass_map[key] = value is not None and value >= threshold
        else: pass_map[key] = value is not None and (value >= threshold if is_long else value <= -threshold)
    score = sum(10 for ok in pass_map.values() if ok)
    return {"Symbol": symbol, "Score": score, "Max": 120, "Price": price, "Target gap %": target_gap, "Revenue yoy %": metrics.get("revenue_yoy"), "EPS yoy %": metrics.get("eps_yoy"), "FCF yoy %": metrics.get("fcf_yoy"), "Margins %": metrics.get("net_margin"), "Insider 90d $": tests["Insider cluster"], "Recommendation": extra.get("recommendation", "NA"), "Direction": direction, "Metrics": metrics, "Extra": extra, "Criteria": pass_map}


def tech_flags(frame: pd.DataFrame, direction: str) -> dict[str, Any]:
    if frame.empty:
        return {}
    close, high, low = frame.Close, frame.High, frame.Low
    ema20 = close.ewm(span=20, adjust=False).mean(); ema200 = close.ewm(span=200, adjust=False).mean(); ema_week = close.resample("W").last().ewm(span=200, adjust=False).mean().reindex(close.index, method="ffill")
    price = float(close.iloc[-1]); avgvol = frame.Volume.rolling(20).mean().iloc[-1]
    supports = float(low.tail(60).min()) if direction == "Long" else float(high.tail(60).max())
    near = any(abs(price / level - 1) <= .01 for level in [ema20.iloc[-1], ema_week.iloc[-1], supports] if pd.notna(level) and level)
    slope = close.tail(40).iloc[-1] - close.tail(40).iloc[0]
    shape = ("channel up" if slope > 0 else "channel down")
    return {"near_level": near, "trend": shape, "high_volume": bool(avgvol and frame.Volume.iloc[-1] >= 2 * avgvol), "EMA20": ema20, "EMA200": ema200, "EMA200W": ema_week, "support_resistance": supports}


def chart(symbol: str, frame: pd.DataFrame, direction: str, insider_value: Any) -> go.Figure:
    t = tech_flags(frame, direction); fig = go.Figure()
    fig.add_trace(go.Candlestick(x=frame.index, open=frame.Open, high=frame.High, low=frame.Low, close=frame.Close, name=symbol))
    for key, color, width in [("EMA20", "cyan", 1), ("EMA200", "orange", 1), ("EMA200W", "yellow", 3)]:
        fig.add_trace(go.Scatter(x=frame.index, y=t[key], name=key, line={"color": color, "width": width}))
    if insider_value and float(insider_value) != 0:
        fig.add_trace(go.Scatter(x=[frame.index[-1]], y=[frame.Close.iloc[-1]], name="Insider purchase", mode="markers", marker={"color": "purple", "size": 12}, hovertemplate=f"Insider $ volume: {float(insider_value):,.0f}<extra></extra>"))
    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False, legend_orientation="h")
    return fig


def financial_table(metrics: dict[str, Any]) -> pd.DataFrame:
    series = {"Revenue": metrics.get("revenue"), "Net income": metrics.get("net"), "Operating income": metrics.get("operating"), "EPS": metrics.get("eps"), "Cash flow from operations": metrics.get("cfo")}
    return pd.DataFrame({k: s for k, s in series.items() if isinstance(s, pd.Series)}).tail(20).sort_index(ascending=False)


def main() -> None:
    st.title("Equity Signal Lab")
    st.caption("A filing-aware NYSE/Nasdaq research workbench — signals are screening heuristics, not investment advice.")
    with st.sidebar:
        st.header("Universe & setup")
        use_demo = st.checkbox("Use compact demo universe", value=True)
        universe = DEFAULT_UNIVERSE if use_demo else listed_universe()
        selected = st.multiselect("Symbols", universe, default=universe[:8])
        direction = st.radio("Rank", ["Long", "Short"], horizontal=True)
        st.caption("Primary sources: SEC Company Facts and filings. Market/analyst enrichment: Yahoo Finance.")
        st.divider(); st.header("Screen thresholds")
        thresholds = {"Revenue growth": st.slider("Revenue growth YoY %", -50, 100, 15 if direction == "Long" else -15), "EPS growth": st.slider("EPS growth YoY %", -100, 100, 15 if direction == "Long" else -15), "Earnings beats": st.slider("Earnings beats, past year", 0, 4, 3), "FCF growth": st.slider("FCF growth YoY %", -100, 100, 15 if direction == "Long" else -15), "Margins positive": st.slider("Minimum margin %", -50, 50, 0), "Margins expanding": st.slider("Margin expansion YoY pts", -50, 50, 0), "Buy share trend": st.slider("Buy recommendation change pts", -100, 100, 0), "Target trend": st.slider("Target change YoY %", -100, 200, 0), "Price target gap": st.slider("Price target gap %", -100, 200, 15 if direction == "Long" else -15), "Insider cluster": st.slider("Insider cluster / 90d $", 0, 10_000_000, 0), "Institution net flow": st.slider("Institution net flow $", -10_000_000, 10_000_000, 0)}
    if not selected:
        st.info("Choose at least one symbol in the sidebar."); return

    tab1, tab2, tab3, tab4 = st.tabs(["1 · Screener", "2 · Technicals", "3 · Company deep dive", "4 · Daily chart"])
    with st.spinner("Reading filing and market data…"):
        records = [score_record(symbol, direction, thresholds) for symbol in selected]
    table = pd.DataFrame(records).sort_values("Score", ascending=False)
    with tab1:
        st.subheader(f"Default {direction.lower()} candidates")
        st.dataframe(table[["Symbol", "Score", "Max", "Price", "Target gap %", "Revenue yoy %", "EPS yoy %", "FCF yoy %", "Margins %", "Insider 90d $", "Recommendation"]], use_container_width=True, hide_index=True)
        st.caption("Each of 12 criteria contributes 10 points. NA data is neutral; inspect source coverage before acting. Earnings surprises, Form 4 clusters, and 13F flow require a filing parser or licensed feed when Yahoo does not expose them.")
        cols = st.columns(3)
        for col, title, keys in zip(cols, ["Fundamentals", "Valuation", "Insider activity"], [["Revenue yoy %", "EPS yoy %", "FCF yoy %", "Margins %"], ["Recommendation", "Target gap %"], ["Insider 90d $"]]):
            with col:
                st.markdown(f"**{title}**")
                st.write(", ".join(keys))
                st.progress(min(float(table.Score.max()) / max(float(table.Max.max()), 1), 1))
    with tab2:
        st.subheader("Technical confirmation")
        tech_rows = []
        for row in records:
            flags = tech_flags(prices(row["Symbol"], "2y"), direction)
            tech_rows.append({"Symbol": row["Symbol"], "At EMA/support": flags.get("near_level", False), "Trend": flags.get("trend", "NA"), "2× volume": flags.get("high_volume", False), "Level": flags.get("support_resistance")})
        st.dataframe(pd.DataFrame(tech_rows), use_container_width=True, hide_index=True)
        st.caption("Longs seek EMA/support, rising structure, or 2× volume. Shorts invert the level and trend interpretation.")
    with tab3:
        symbol = st.selectbox("Company", selected, key="deep_company")
        row = next(r for r in records if r["Symbol"] == symbol); metrics = row["Metrics"]; extra = row["Extra"]
        st.subheader(f"{symbol} · {extra.get('sector', 'Sector unavailable')}")
        st.write(extra.get("summary") or "Business summary unavailable from the current provider.")
        a, b, c, d = st.columns(4); a.metric("Market cap", fmt(extra.get("market_cap"))); b.metric("Revenue YoY", fmt(metrics.get("revenue_yoy"), "%")); c.metric("Net margin", fmt(metrics.get("net_margin"), "%")); d.metric("SEC CIK", metrics.get("cik") or "NA")
        st.markdown("**Reported financials (SEC XBRL facts; latest 20 periods)**")
        st.dataframe(financial_table(metrics), use_container_width=True)
        if any(isinstance(metrics.get(k), pd.Series) and not metrics[k].empty for k in ["revenue", "net", "operating", "eps", "cfo"]):
            trend = pd.DataFrame({k: metrics[k] for k in ["revenue", "net", "operating", "eps", "cfo"] if isinstance(metrics.get(k), pd.Series) and not metrics[k].empty})
            st.line_chart(trend, height=260)
        peers = [p.strip().upper() for p in st.text_input("Three closest competitors (comma-separated)", value="MSFT, GOOGL, AMZN" if symbol == "AAPL" else "AAPL, MSFT, GOOGL").split(",") if p.strip() and p.strip().upper() != symbol]
        peer_rows = []
        for peer in peers[:3]:
            peer_metrics = filing_metrics(peer); peer_info = enrichment(peer)
            peer_rows.append({"Company": peer, "Price": peer_info.get("price"), "Market cap": peer_info.get("market_cap"), "Revenue YoY %": peer_metrics.get("revenue_yoy"), "Net margin %": peer_metrics.get("net_margin"), "Industry": peer_info.get("industry", "NA")})
        if peer_rows:
            st.markdown("**Peer comparison**")
            st.dataframe(pd.DataFrame(peer_rows), use_container_width=True, hide_index=True)
        st.markdown("**DCF / comparables workspace**")
        assumptions = st.columns(4); revenue_growth = assumptions[0].number_input("Forward revenue growth %", value=12.0); margin = assumptions[1].number_input("Forward operating margin %", value=20.0); wacc = assumptions[2].number_input("WACC %", value=9.0); terminal = assumptions[3].number_input("Terminal growth %", value=3.0)
        base = float(metrics["revenue"].iloc[-1]) if len(metrics.get("revenue", [])) else 0
        forecast = pd.DataFrame({"Year": ["Y1", "Y2", "Y3", "Y4", "Y5"], "Revenue": [base * (1 + revenue_growth / 100) ** i for i in range(1, 6)]}); forecast["Operating income"] = forecast.Revenue * margin / 100
        st.dataframe(forecast, use_container_width=True, hide_index=True)
        pv = float((forecast["Operating income"] * .75 / (1 + wacc / 100) ** np.arange(1, 6)).sum()) if base else 0
        st.metric("Illustrative enterprise value from forecast", fmt(pv), "Replace with company-specific debt, cash, shares and terminal value before use")
        implied = pv / float(extra.get("market_cap")) - 1 if pv and extra.get("market_cap") else None
        verdict = "Buy" if implied is not None and implied >= .15 else "Sell" if implied is not None and implied <= -.15 else "Hold"
        st.metric("Illustrative verdict", verdict, f"{implied:.1%} upside/downside vs market cap" if implied is not None else "Needs market cap and complete DCF inputs")
    with tab4:
        symbol = st.selectbox("Chart company", selected, key="chart_company"); row = next(r for r in records if r["Symbol"] == symbol); frame = prices(symbol, "2y")
        if frame.empty: st.warning("No daily price history available.")
        else: st.plotly_chart(chart(symbol, frame, direction, row.get("Insider 90d $")), use_container_width=True)
        st.caption("Daily OHLCV from Yahoo Finance. EMA colors: 20-day cyan, 200-day orange, 200-week thick yellow; purple marks the latest available insider activity.")


if __name__ == "__main__":
    main()
