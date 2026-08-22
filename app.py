"""Streamlit equity research dashboard for NYSE/Nasdaq long and short ideas.

The app deliberately separates market data from filing data. SEC Company Facts
and SEC submissions are used for reported financials when a CIK is available;
Yahoo Finance supplies price history and optional analyst/holder enrichment.
Missing data is shown as ``NA`` and does not silently become a passing signal.
"""

from __future__ import annotations

from datetime import date, timedelta
import re
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
import yfinance as yf


st.set_page_config(page_title="Equity Signal Lab", page_icon="◒", layout="wide")
SEC_HEADERS = {"User-Agent": "Equity Signal Lab research app contact@example.com"}
DEFAULT_UNIVERSE = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AVGO", "JPM", "LLY", "TSLA", "AMD", "NFLX"]
TOP_MARKET_CAP_DEFAULT = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "BRK-B", "LLY"]


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


def parse_symbol_list(text: str) -> list[str]:
    """Accept comma/space/newline/tab-separated symbols and pasted table text."""
    tokens = re.findall(r"(?<![A-Z0-9])([A-Z]{1,5}(?:[-.][A-Z]{1,2})?)(?![A-Z0-9])", text.upper())
    ignored = {"SYMBOL", "TICKER", "NAME", "PRICE", "MARKET", "CAP", "NONE", "NA", "BUY", "SELL", "HOLD"}
    return list(dict.fromkeys(token for token in tokens if token not in ignored))


def rounded_percent_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if "%" in str(column) or "Momentum" in str(column):
            result[column] = result[column].map(lambda value: round(float(str(value).replace("%", "").replace(",", "")), 1) if str(value).strip().replace("%", "").replace(",", "").replace(".", "", 1).replace("-", "", 1).isdigit() else value)
    return result


def clenow_momentum(frame: pd.DataFrame, window: int = 90) -> float | None:
    """Annualized exponential regression slope multiplied by regression R²."""
    if frame.empty or "Close" not in frame or len(frame) < window:
        return None
    close = pd.to_numeric(frame["Close"].tail(window), errors="coerce").dropna()
    if len(close) < window or (close <= 0).any():
        return None
    x = np.arange(len(close), dtype=float)
    y = np.log(close.to_numpy(dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    ss_res = float(np.square(y - fitted).sum())
    ss_tot = float(np.square(y - y.mean()).sum())
    r_squared = 1 - ss_res / ss_tot if ss_tot else 0.0
    return float(100 * (np.exp(slope * 250) - 1) * max(0.0, r_squared))


@st.cache_data(ttl=86400, show_spinner=False)
def top_market_cap_universe() -> list[str]:
    """Rank the default large-cap candidates by the latest available market cap."""
    rows = []
    for symbol in TOP_MARKET_CAP_DEFAULT:
        market_cap = enrichment(symbol).get("market_cap")
        rows.append((symbol, market_cap or 0))
    ranked = [symbol for symbol, _ in sorted(rows, key=lambda item: item[1], reverse=True)]
    return ranked or TOP_MARKET_CAP_DEFAULT


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
    result.update({
        "cogs": fact_series(bundle, ["CostOfRevenue", "CostOfGoodsAndServicesSold"]),
        "rd": fact_series(bundle, ["ResearchAndDevelopmentExpense"]),
        "ga": fact_series(bundle, ["SellingGeneralAndAdministrativeExpense"]),
        "opex": fact_series(bundle, ["OperatingExpenses"]),
        "interest": fact_series(bundle, ["InterestIncomeExpenseNonOperatingNet", "InterestExpenseNonOperating"]),
        "pretax": fact_series(bundle, ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest", "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"]),
        "taxes": fact_series(bundle, ["IncomeTaxExpenseBenefit"]),
        "shares": fact_series(bundle, ["WeightedAverageNumberOfDilutedSharesOutstanding", "EntityCommonStockSharesOutstanding"], "shares"),
        "assets": fact_series(bundle, ["Assets"]),
        "liabilities": fact_series(bundle, ["Liabilities"]),
        "equity": fact_series(bundle, ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]),
        "capex": fact_series(bundle, ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"]),
        "cash": fact_series(bundle, ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"]),
        "receivables": fact_series(bundle, ["AccountsReceivableNetCurrent"]), "inventory": fact_series(bundle, ["InventoryNet"]), "prepaids": fact_series(bundle, ["PrepaidExpenseAndOtherAssetsCurrent"]), "current_assets": fact_series(bundle, ["AssetsCurrent"]),
        "ppe": fact_series(bundle, ["PropertyPlantAndEquipmentNet"]), "lease_asset": fact_series(bundle, ["OperatingLeaseRightOfUseAsset"]), "deferred_tax_asset": fact_series(bundle, ["DeferredTaxAssetsNet"]), "other_assets": fact_series(bundle, ["OtherAssetsNoncurrent"]),
        "payables": fact_series(bundle, ["AccountsPayableCurrent"]), "other_current_liabilities": fact_series(bundle, ["OtherLiabilitiesCurrent"]), "deferred_revenue": fact_series(bundle, ["ContractWithCustomerLiabilityCurrent"]), "tax_receivables": fact_series(bundle, ["IncomeTaxesReceivable"]), "current_lease": fact_series(bundle, ["OperatingLeaseLiabilityCurrent"]), "current_debt": fact_series(bundle, ["LongTermDebtCurrent"]), "current_liabilities": fact_series(bundle, ["LiabilitiesCurrent"]),
        "long_term_lease": fact_series(bundle, ["OperatingLeaseLiabilityNoncurrent"]), "long_term_debt": fact_series(bundle, ["LongTermDebtNoncurrent"]), "other_liabilities": fact_series(bundle, ["OtherLiabilitiesNoncurrent"]), "preferred_stock": fact_series(bundle, ["PreferredStocksIncludingAdditionalPaidInCapital"]), "common_stock": fact_series(bundle, ["CommonStocksIncludingAdditionalPaidInCapital"]), "apic": fact_series(bundle, ["AdditionalPaidInCapital"]), "aoci": fact_series(bundle, ["AccumulatedOtherComprehensiveIncomeLossNetOfTax"]), "retained_earnings": fact_series(bundle, ["RetainedEarningsAccumulatedDeficit"]), "noncontrolling": fact_series(bundle, ["MinorityInterest"]),
        "depreciation": fact_series(bundle, ["DepreciationDepletionAndAmortization"]), "equity_comp": fact_series(bundle, ["ShareBasedCompensation"]), "deferred_taxes": fact_series(bundle, ["DeferredIncomeTaxExpenseBenefit"]), "change_receivables": fact_series(bundle, ["IncreaseDecreaseInAccountsReceivable"]), "change_inventory": fact_series(bundle, ["IncreaseDecreaseInInventories"]), "change_payables": fact_series(bundle, ["IncreaseDecreaseInAccountsPayable"]), "change_other_current": fact_series(bundle, ["IncreaseDecreaseInOtherOperatingAssets"]), "change_other_liabilities": fact_series(bundle, ["IncreaseDecreaseInOtherOperatingLiabilities"]), "proceeds_debt": fact_series(bundle, ["ProceedsFromIssuanceOfLongTermDebt"]), "payments_debt": fact_series(bundle, ["RepaymentsOfLongTermDebt"]), "cash_begin": fact_series(bundle, ["CashAndCashEquivalentsAtCarryingValue"]),
    })
    result["revenue_yoy"] = latest_yoy(revenue)
    result["eps_yoy"] = latest_yoy(eps)
    result["fcf_yoy"] = latest_yoy(cfo)
    result["net_margin"] = (float(net.iloc[-1] / revenue.iloc[-1] * 100) if len(net) and len(revenue) and revenue.iloc[-1] else None)
    result["gross_margin"] = (float(gross.iloc[-1] / revenue.iloc[-1] * 100) if len(gross) and len(revenue) and revenue.iloc[-1] else None)
    result["operating_margin"] = (float(operating.iloc[-1] / revenue.iloc[-1] * 100) if len(operating) and len(revenue) and revenue.iloc[-1] else None)
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def insider_trades(symbol: str) -> pd.DataFrame:
    try:
        raw = yf.Ticker(symbol).insider_transactions
        if raw is None or raw.empty:
            return pd.DataFrame(columns=["Date", "Buy volume", "Sell volume", "Buy shares", "Sell shares"])
        frame = raw.copy()
        date_col = next((c for c in ["Start Date", "Date", "startDate"] if c in frame.columns), None)
        frame["Date"] = pd.to_datetime(frame[date_col] if date_col else frame.index, errors="coerce")
        frame["Date"] = frame["Date"].dt.tz_localize(None)
        text = frame.astype(str).agg(" ".join, axis=1).str.lower()
        value = pd.to_numeric(frame.get("Value", pd.Series(index=frame.index, dtype=float)), errors="coerce").fillna(0)
        shares = pd.to_numeric(frame.get("Shares", pd.Series(index=frame.index, dtype=float)), errors="coerce").fillna(0)
        is_buy = text.str.contains("purchase|buy|acquisition", regex=True, na=False)
        is_sell = text.str.contains("sale|sell|disposition", regex=True, na=False)
        output = pd.DataFrame({"Date": frame["Date"], "Buy volume": value.where(is_buy, 0), "Sell volume": value.where(is_sell, 0), "Buy shares": shares.where(is_buy, 0), "Sell shares": shares.where(is_sell, 0)})
        return output.dropna(subset=["Date"]).groupby("Date", as_index=False).sum().sort_values("Date")
    except Exception:
        return pd.DataFrame(columns=["Date", "Buy volume", "Sell volume", "Buy shares", "Sell shares"])


@st.cache_data(ttl=3600, show_spinner=False)
def finviz_snapshot(symbol: str) -> dict[str, Any]:
    """Best-effort fallback for exchange, analyst recommendation, valuation and insider-transmission fields."""
    try:
        html = requests.get(f"https://finviz.com/quote.ashx?t={symbol}", headers={"User-Agent": "Mozilla/5.0"}, timeout=15).text
        tables = pd.read_html(html)
        pairs: dict[str, Any] = {}
        for table in tables:
            flat = table.astype(str).to_numpy().ravel().tolist()
            for i, value in enumerate(flat[:-1]):
                if value in {"Recom", "Insider Trans", "Target Price", "Exchange", "Earnings", "P/E", "P/S", "PEG", "P/FCF", "P/B"}:
                    pairs[value] = flat[i + 1]
        peer_section = re.search(r">Peers<.*?(?:Held by|Scroll to Statements)", html, flags=re.I | re.S)
        peer_html = peer_section.group(0) if peer_section else ""
        pairs["Peers"] = list(dict.fromkeys(re.findall(r"quote\.ashx\?t=([A-Z0-9.-]+)", peer_html, flags=re.I)))[:3]
        return pairs
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def finviz_peers(symbol: str) -> list[str]:
    peers = finviz_snapshot(symbol).get("Peers", [])
    return [str(peer).upper() for peer in peers if str(peer).upper() != symbol.upper()][:3]


@st.cache_data(ttl=3600, show_spinner=False)
def enrichment(symbol: str) -> dict[str, Any]:
    info: dict[str, Any] = {}
    ticker = yf.Ticker(symbol)
    try:
        raw = ticker.info or {}
    except Exception:
        raw = {}
    fallback = finviz_snapshot(symbol)
    recommendation = raw.get("recommendationKey") or raw.get("recommendationMean")
    recommendation = recommendation or fallback.get("Recom")
    try:
        rec_number = float(recommendation)
        consensus_key = "buy" if rec_number <= 2.0 else "hold" if rec_number <= 3.0 else "sell"
    except (TypeError, ValueError):
        consensus_key = str(recommendation).lower().replace(" ", "_") if recommendation else None
    info.update({"price": raw.get("currentPrice") or raw.get("regularMarketPrice"), "target": raw.get("targetMeanPrice"), "recommendation": recommendation, "consensus_key": consensus_key, "sector": raw.get("sector"), "industry": raw.get("industry"), "gics_industry": raw.get("industry"), "marketplace": raw.get("fullExchangeName") or raw.get("exchange") or fallback.get("Exchange"), "market_cap": raw.get("marketCap"), "summary": raw.get("longBusinessSummary"), "analyst_growth": raw.get("revenueGrowth") or raw.get("earningsGrowth"), "guidance": None, "last_earnings": None, "next_earnings": None, "cash": raw.get("totalCash"), "debt": raw.get("totalDebt"), "finviz_insider_trans": fallback.get("Insider Trans"), "valuation": {"P/E": raw.get("trailingPE") or fallback.get("P/E"), "P/S": raw.get("priceToSalesTrailing12Months") or fallback.get("P/S"), "PEG": raw.get("pegRatio") or fallback.get("PEG"), "P/OCF": None, "P/FCF": raw.get("priceToFreeCashflow") or fallback.get("P/FCF"), "P/B": raw.get("priceToBook") or fallback.get("P/B"), "P/TBV": None}})
    try:
        quote = prices(symbol, "1mo")
        # Use the latest completed daily close for the screener snapshot.
        info["price"] = float(quote["Close"].iloc[-1]) if not quote.empty else info.get("price")
    except Exception:
        info["price"] = info.get("price")
    if not info.get("market_cap") and info.get("price"):
        try:
            info["market_cap"] = float(ticker.fast_info.get("market_cap"))
        except Exception:
            pass
    try:
        rec = ticker.recommendations
        info["buy_share"] = ((rec["strongBuy"] + rec["buy"]).tail(4).sum() / rec.tail(4)[["strongBuy", "buy", "hold", "sell", "strongSell"]].sum().sum() * 100) if rec is not None and not rec.empty and "strongBuy" in rec else None
    except Exception:
        info["buy_share"] = None
    try:
        earnings_dates = pd.to_datetime(ticker.get_earnings_dates(limit=12).index, errors="coerce").tz_localize(None)
        today = pd.Timestamp.today().normalize()
        past = earnings_dates[earnings_dates <= today]
        future = earnings_dates[earnings_dates > today]
        info["last_earnings"] = past.max().date().isoformat() if len(past) else None
        info["next_earnings"] = future.min().date().isoformat() if len(future) else None
    except Exception:
        try:
            calendar = ticker.calendar
            dates = calendar.get("Earnings Date") if isinstance(calendar, dict) else None
            if dates:
                dates = pd.to_datetime(dates, errors="coerce")
                info["next_earnings"] = dates.min().date().isoformat()
        except Exception:
            pass
    try:
        ins = insider_trades(symbol)
        if ins is not None and not ins.empty:
            recent = ins[ins["Date"] >= pd.Timestamp.today() - pd.Timedelta(days=90)]
            info["insider_90d"] = float(recent["Buy volume"].sum())
        else:
            info["insider_90d"] = None
    except Exception:
        info["insider_90d"] = None
    return info


def score_record(symbol: str, direction: str, thresholds: dict[str, float]) -> dict[str, Any]:
    metrics = filing_metrics(symbol)
    extra = enrichment(symbol)
    momentum = clenow_momentum(prices(symbol, "1y"))
    price = extra.get("price")
    target = extra.get("target")
    target_gap = (target / price - 1) * 100 if price and target else None
    is_long = direction == "Long"
    tests = {
        "Revenue growth": metrics.get("revenue_yoy"), "EPS growth": metrics.get("eps_yoy"), "Earnings beats": None,
        "FCF growth": metrics.get("fcf_yoy"), "Margins positive": min([x for x in [metrics.get("gross_margin"), metrics.get("operating_margin"), metrics.get("net_margin")] if x is not None], default=None),
        "Margins expanding": None, "Analyst consensus": 1 if extra.get("consensus_key") in ({"buy", "strong_buy", "hold"} if is_long else {"hold", "sell", "strong_sell"}) else 0,
        "Buy share trend": None, "Target trend": None, "Price target gap": target_gap, "Insider cluster": extra.get("insider_90d"), "Institution net flow": None, "Clenow momentum": momentum,
    }
    pass_map = {}
    for key, value in tests.items():
        threshold = thresholds.get(key, 0)
        if key == "Analyst consensus": pass_map[key] = bool(value)
        elif key in {"Insider cluster", "Institution net flow"}: pass_map[key] = value is not None and (value >= threshold if is_long else value <= -threshold)
        elif key in {"Margins positive", "Margins expanding"}: pass_map[key] = value is not None and value >= threshold
        else: pass_map[key] = value is not None and (value >= threshold if is_long else value <= -threshold)
    score = sum(10 for ok in pass_map.values() if ok)
    return {"Symbol": symbol, "Score": score, "Max": 130, "Price": price, "Target gap %": target_gap, "Revenue yoy %": metrics.get("revenue_yoy"), "EPS yoy %": metrics.get("eps_yoy"), "FCF growth %": metrics.get("fcf_yoy"), "FCF yoy %": metrics.get("fcf_yoy"), "Operating margin %": metrics.get("operating_margin"), "Margins %": metrics.get("operating_margin"), "Insider 90d $": tests["Insider cluster"], "Finviz Insider Trans %": extra.get("finviz_insider_trans"), "Clenow Momentum": momentum, "Recommendation": extra.get("consensus_key") or extra.get("recommendation", "NA"), "Direction": direction, "Metrics": metrics, "Extra": extra, "Criteria": pass_map}


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
    t = tech_flags(frame, direction); fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Candlestick(x=frame.index, open=frame.Open, high=frame.High, low=frame.Low, close=frame.Close, name=symbol))
    for key, color, width in [("EMA20", "cyan", 1), ("EMA200", "orange", 1), ("EMA200W", "yellow", 3)]:
        fig.add_trace(go.Scatter(x=frame.index, y=t[key], name=key, line={"color": color, "width": width}))
    trades = insider_trades(symbol)
    if not trades.empty:
        visible = trades[trades.Date >= frame.index.min()]
        fig.add_trace(go.Bar(x=visible.Date, y=visible["Buy volume"], name="Insider purchases", marker_color="#2ca02c", opacity=.55, hovertemplate="Buy volume: $%{y:,.0f}<extra></extra>"), secondary_y=True)
        fig.add_trace(go.Bar(x=visible.Date, y=-visible["Sell volume"], name="Insider sales", marker_color="#d62728", opacity=.55, hovertemplate="Sale volume: $%{customdata:,.0f}<extra></extra>", customdata=visible["Sell volume"]), secondary_y=True)
    fig.update_yaxes(title_text="Close", secondary_y=False); fig.update_yaxes(title_text="Insider trade volume ($)", secondary_y=True, zeroline=True)
    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False, legend_orientation="h", barmode="relative")
    return fig


def financial_table(metrics: dict[str, Any]) -> pd.DataFrame:
    series = {"Revenue": metrics.get("revenue"), "Net income": metrics.get("net"), "Operating income": metrics.get("operating"), "EPS": metrics.get("eps"), "Cash flow from operations": metrics.get("cfo")}
    return pd.DataFrame({k: s for k, s in series.items() if isinstance(s, pd.Series)}).tail(20).sort_index()


STATEMENT_LINES = [
    ("Income statement", "Revenue", "revenue"), ("Income statement", "COGS", "cogs"),
    ("Income statement", "Gross Profit", "gross"), ("Income statement", "Gross Margin %", "gross_margin_series"),
    ("Income statement", "R&D", "rd"), ("Income statement", "G&A", "ga"),
    ("Income statement", "Total Operating Expenses", "opex"), ("Income statement", "Operating Income", "operating"),
    ("Income statement", "Operating Margin %", "operating_margin_series"), ("Income statement", "Interest Income", "interest"),
    ("Income statement", "Pretax Income", "pretax"), ("Income statement", "Taxes", "taxes"),
    ("Income statement", "Net Income", "net"), ("Income statement", "EPS", "eps"), ("Income statement", "Shares", "shares"),
    ("Balance sheet", "Cash", "cash"), ("Balance sheet", "Accounts receivable", "receivables"), ("Balance sheet", "Inventories", "inventory"), ("Balance sheet", "Prepaid expenses", "prepaids"), ("Balance sheet", "Total Current Assets", "current_assets"), ("Balance sheet", "Property and equipment", "ppe"), ("Balance sheet", "Lease asset", "lease_asset"), ("Balance sheet", "Deferred income tax asset", "deferred_tax_asset"), ("Balance sheet", "Other long-term assets", "other_assets"), ("Balance sheet", "Total Assets", "assets"),
    ("Balance sheet", "Accounts payable", "payables"), ("Balance sheet", "Other current liabilities", "other_current_liabilities"), ("Balance sheet", "Deferred revenue", "deferred_revenue"), ("Balance sheet", "Tax receivables", "tax_receivables"), ("Balance sheet", "Current lease liabilities", "current_lease"), ("Balance sheet", "Current portion of long-term debt", "current_debt"), ("Balance sheet", "Total Current Liabilities", "current_liabilities"), ("Balance sheet", "Lease liabilities", "long_term_lease"), ("Balance sheet", "Long-term debt", "long_term_debt"), ("Balance sheet", "Other long-term liabilities", "other_liabilities"), ("Balance sheet", "Total Liabilities", "liabilities"),
    ("Balance sheet", "Preferred stock", "preferred_stock"), ("Balance sheet", "Common stock", "common_stock"), ("Balance sheet", "Additional paid-in capital", "apic"), ("Balance sheet", "Accumulated other comprehensive income", "aoci"), ("Balance sheet", "Retained earnings", "retained_earnings"), ("Balance sheet", "Stockholders' Equity", "equity"), ("Balance sheet", "Non-controlling interest", "noncontrolling"), ("Balance sheet", "Total equity", "equity"), ("Balance sheet", "Total liabilities and equity", "equity"),
    ("Statement of cash flow", "Net income", "net"), ("Statement of cash flow", "Depreciation and amortization", "depreciation"), ("Statement of cash flow", "Equity-based compensation", "equity_comp"), ("Statement of cash flow", "Deferred income taxes", "deferred_taxes"), ("Statement of cash flow", "Change in accounts receivable", "change_receivables"), ("Statement of cash flow", "Change in inventories", "change_inventory"), ("Statement of cash flow", "Change in accounts payable", "change_payables"), ("Statement of cash flow", "Change in other operating assets", "change_other_current"), ("Statement of cash flow", "Change in other operating liabilities", "change_other_liabilities"), ("Statement of cash flow", "Cash Flow from Operations", "cfo"), ("Statement of cash flow", "Capital Expenditures", "capex"), ("Statement of cash flow", "Proceeds from debt", "proceeds_debt"), ("Statement of cash flow", "Payments of debt", "payments_debt"), ("Statement of cash flow", "Free Cash Flow", "fcf"),
]


def statement_frame(metrics: dict[str, Any], annual: bool = True) -> pd.DataFrame:
    """Build a date-across statement with a YoY row after every financial line."""
    derived = dict(metrics)
    revenue = metrics.get("revenue", pd.Series(dtype=float))
    if isinstance(revenue, pd.Series) and not revenue.empty:
        for key, numerator in [("gross_margin_series", metrics.get("gross")), ("operating_margin_series", metrics.get("operating"))]:
            if isinstance(numerator, pd.Series):
                derived[key] = numerator.divide(revenue.reindex(numerator.index), fill_value=np.nan) * 100
    for key in ["assets", "liabilities", "equity", "capex"]:
        derived.setdefault(key, pd.Series(dtype=float))
    if isinstance(derived.get("cfo"), pd.Series) and isinstance(derived.get("capex"), pd.Series):
        derived["fcf"] = derived["cfo"].sub(derived["capex"], fill_value=0)
    grouped: dict[str, pd.Series] = {}
    for _, label, key in STATEMENT_LINES:
        series = derived.get(key)
        if not isinstance(series, pd.Series) or series.empty: continue
        series = series[~series.index.duplicated(keep="last")].sort_index()
        grouped[label] = series.groupby(series.index.year).last() if annual else series.groupby(series.index.to_period("Q")).last()
    if not grouped: return pd.DataFrame()
    values = pd.DataFrame(grouped).T.reindex(sorted(set().union(*(s.index for s in grouped.values())), key=str), axis=1)
    output: dict[str, pd.Series] = {}
    for section, label, _ in STATEMENT_LINES:
        line = values.loc[label] if label in values.index else pd.Series(np.nan, index=values.columns, dtype=float)
        if label.endswith("Margin %") or label == "EPS":
            display_label = label
        elif label == "Shares":
            display_label = "Shares (mm)"
            line = line / 1_000_000
        else:
            display_label = f"{label} ($mm)"
            line = line / 1_000_000
        output[f"{section} · {display_label}"] = line
        output[f"{section} · {display_label} YoY %"] = line.pct_change(periods=1 if annual else 4) * 100
    result = pd.DataFrame(output).T
    result.columns = [str(c) for c in result.columns]
    return result


def peer_metric_frame(symbols: list[str], line: str, annual: bool = True) -> pd.DataFrame:
    key = next((key for section, label, key in STATEMENT_LINES if label == line), "revenue")
    rows = {}
    for symbol in symbols:
        metrics = filing_metrics(symbol)
        if key == "gross_margin_series":
            series = metrics.get("gross").divide(metrics.get("revenue").reindex(metrics.get("gross").index), fill_value=np.nan) * 100 if isinstance(metrics.get("gross"), pd.Series) else pd.Series(dtype=float)
        elif key == "operating_margin_series":
            series = metrics.get("operating").divide(metrics.get("revenue").reindex(metrics.get("operating").index), fill_value=np.nan) * 100 if isinstance(metrics.get("operating"), pd.Series) else pd.Series(dtype=float)
        else:
            series = metrics.get(key)
        if isinstance(series, pd.Series) and not series.empty:
            rows[symbol] = series.groupby(series.index.year).last() if annual else series.groupby(series.index.to_period("Q")).last()
    if not rows: return pd.DataFrame()
    frame = pd.DataFrame(rows).T
    frame.loc["Industry average"] = frame.mean(axis=0)
    return frame


@st.cache_data(ttl=86400, show_spinner=False)
def closest_peers(symbol: str) -> list[str]:
    """Use the provider industry label as a GICS-subindustry fallback, ranked by market cap."""
    target = enrichment(symbol)
    candidates = list(dict.fromkeys(TOP_MARKET_CAP_DEFAULT + DEFAULT_UNIVERSE))
    rows = []
    for candidate in candidates:
        if candidate == symbol: continue
        info = enrichment(candidate)
        if target.get("gics_industry") and info.get("gics_industry") != target.get("gics_industry"): continue
        rows.append((candidate, info.get("market_cap") or 0))
    if len(rows) < 3:
        rows = []
        for candidate in candidates:
            if candidate == symbol: continue
            info = enrichment(candidate)
            rows.append((candidate, info.get("market_cap") or 0))
    return [candidate for candidate, _ in sorted(rows, key=lambda x: x[1], reverse=True)[:4]]


def style_financial_table(frame: pd.DataFrame, peer_frames: list[pd.DataFrame] | None = None):
    if frame.empty: return frame
    peer_frames = peer_frames or []
    valid_peer_frames = [p for p in peer_frames if isinstance(p, pd.DataFrame) and not p.empty]
    peer_yoy = pd.concat(valid_peer_frames, axis=0) if valid_peer_frames else pd.DataFrame()
    def style(data: pd.DataFrame):
        out = pd.DataFrame("", index=data.index, columns=data.columns)
        for idx in data.index:
            if "YoY %" not in str(idx): continue
            for col in data.columns:
                value = pd.to_numeric(data.loc[idx, col], errors="coerce")
                peers = pd.to_numeric(peer_yoy.loc[idx, col], errors="coerce") if not peer_yoy.empty and idx in peer_yoy.index and col in peer_yoy.columns else pd.Series(dtype=float)
                peers = peers.dropna()
                if pd.notna(value) and len(peers) and value >= peers.mean():
                    out.loc[idx, col] = "background-color: #effaf0"
                if pd.notna(value) and len(peers) and value >= peers.max():
                    out.loc[idx, col] = "background-color: #d8f3dc"
        return out
    return frame.style.apply(style, axis=None).format(lambda value: "NA" if pd.isna(value) else (f"{value:,.1f}" if isinstance(value, (float, np.floating)) else value))


def style_screener_table(frame: pd.DataFrame, records: list[dict[str, Any]]):
    if frame.empty: return frame
    by_symbol = {record["Symbol"]: record for record in records}
    green = "background-color: #effaf0"
    mapping = {"Revenue growth": "Revenue yoy %", "EPS growth": "EPS yoy %", "FCF growth": "FCF yoy %", "Margins positive": "Operating margin %", "Analyst consensus": "Analyst Consensus", "Price target gap": "Target gap %", "Insider cluster": "Insider 90d $", "Clenow momentum": "Clenow Momentum"}
    def style(data: pd.DataFrame):
        out = pd.DataFrame("", index=data.index, columns=data.columns)
        for row_index, row in data.iterrows():
            record = by_symbol.get(row.get("Symbol"), {})
            for criterion, column in mapping.items():
                if column in out.columns and record.get("Criteria", {}).get(criterion):
                    out.loc[row_index, column] = green
        return out
    formatted = rounded_percent_columns(frame)
    def safe_one_decimal(value: Any) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)): return "NA"
        try: return f"{float(str(value).replace('%', '').replace(',', '')):.1f}"
        except (TypeError, ValueError): return str(value)
    formatters = {column: safe_one_decimal for column in formatted.columns if "%" in str(column) or "Momentum" in str(column)}
    return formatted.style.apply(style, axis=None).format(formatters)


def default_dcf_price(metrics: dict[str, Any], extra: dict[str, Any]) -> float | None:
    revenue = metrics.get("revenue"); cfo = metrics.get("cfo"); capex = metrics.get("capex"); shares = metrics.get("shares")
    if not isinstance(revenue, pd.Series) or revenue.empty or not isinstance(shares, pd.Series) or shares.empty: return None
    growth = pct(extra.get("analyst_growth")) or 12.0; base_fcf = ttm(cfo) - ttm(capex)
    if base_fcf <= 0: base_fcf = ttm(revenue) * (metrics.get("operating_margin") or 20) / 100 * .79
    forecast_fcf = np.array([base_fcf * (1 + growth / 100) ** i for i in range(1, 6)])
    discount = 1.09 ** np.arange(1, 6); terminal_value = forecast_fcf[-1] * 1.03 / (.09 - .03)
    equity_value = float((forecast_fcf / discount).sum() + terminal_value / discount[-1] + (extra.get("cash") or 0) - (extra.get("debt") or 0))
    return equity_value / float(shares.iloc[-1])


def ttm(series: pd.Series | None) -> float:
    if not isinstance(series, pd.Series) or series.empty: return 0.0
    return float(series.tail(4).sum())


def valuation_frame(symbols: list[str]) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        metrics = filing_metrics(symbol); extra = enrichment(symbol); cap = extra.get("market_cap") or 0
        revenue = ttm(metrics.get("revenue")); net = ttm(metrics.get("net")); cfo = ttm(metrics.get("cfo")); fcf = cfo - ttm(metrics.get("capex")); equity = float(metrics["equity"].iloc[-1]) if isinstance(metrics.get("equity"), pd.Series) and not metrics["equity"].empty else 0
        pe = cap / net if cap and net > 0 else None; growth = pct(extra.get("analyst_growth")) or metrics.get("eps_yoy")
        rows.append({"Company": symbol, "P/S": cap / revenue if cap and revenue else None, "P/E": pe, "PEG": pe / growth if pe and growth and growth > 0 else None, "P/OCF": cap / cfo if cap and cfo else None, "P/FCF": cap / fcf if cap and fcf > 0 else None, "P/B": cap / equity if cap and equity else None, "P/TBV": cap / equity if cap and equity else None})
    frame = pd.DataFrame(rows)
    if not frame.empty:
        numeric = frame.drop(columns="Company").apply(pd.to_numeric, errors="coerce")
        frame.loc[len(frame)] = ["Peer average"] + numeric.mean().tolist()
    return frame


def main() -> None:
    st.title("Equity Signal Lab")
    st.caption("A filing-aware NYSE/Nasdaq research workbench — signals are screening heuristics, not investment advice.")
    with st.sidebar:
        st.header("Universe & setup")
        use_demo = st.checkbox("Use compact demo universe", value=False)
        universe = DEFAULT_UNIVERSE if use_demo else top_market_cap_universe()
        selected = st.multiselect("Symbols", universe, default=universe[:8])
        scan_all = st.checkbox("Scan full listed universe", value=False, disabled=use_demo, help="Uses Nasdaq Trader's Nasdaq and other-listed symbol files. This can take time and is rate-limited by data providers.")
        use_manual = st.checkbox("Use manual list", value=False)
        manual_text = st.text_area("Paste symbols or a table", placeholder="AAPL, MSFT, SOFI\nPaste a column with a Symbol/Ticker header", disabled=not use_manual, height=90)
        direction = st.radio("Rank", ["Long", "Short"], horizontal=True)
        st.caption("Primary sources: SEC Company Facts and filings. Market/analyst enrichment: Yahoo Finance.")
        st.divider(); st.header("Screen thresholds")
        with st.form("screen_form"):
            thresholds = {"Revenue growth": st.slider("Revenue growth YoY %", -50, 100, 15 if direction == "Long" else -15), "EPS growth": st.slider("EPS growth YoY %", -100, 100, 15 if direction == "Long" else -15), "Earnings beats": st.slider("Earnings beats, past year", 0, 4, 3), "FCF growth": st.slider("FCF growth YoY %", -100, 100, 15 if direction == "Long" else -15), "Margins positive": st.slider("Operating margin %", -50, 50, 0), "Margins expanding": st.slider("Margin expansion YoY pts", -50, 50, 0), "Buy share trend": st.slider("Buy recommendation change pts", -100, 100, 0), "Target trend": st.slider("Target change YoY %", -100, 200, 0), "Price target gap": st.slider("Price target gap %", -100, 200, 15 if direction == "Long" else -15), "Insider cluster": st.slider("Insider cluster / 90d $", 0, 10_000_000, 0), "Institution net flow": st.slider("Institution net flow $", -10_000_000, 10_000_000, 0), "Clenow momentum": st.slider("Clenow Momentum %", -100, 500, 0 if direction == "Long" else 0)}
            min_score = st.slider("Minimum score to show", 0, 130, 60, step=10)
            scan = st.form_submit_button("Scan universe", type="primary", use_container_width=True)
    if scan:
        manual_symbols = parse_symbol_list(manual_text) if use_manual else []
        if not selected and not manual_symbols and not scan_all:
            st.warning("Choose at least one symbol before scanning.")
        else:
            scan_symbols = manual_symbols if use_manual else (listed_universe() if scan_all else selected)
            with st.spinner(f"Scanning {len(scan_symbols)} symbols…"):
                st.session_state["screen_records"] = [score_record(symbol, direction, thresholds) for symbol in scan_symbols]
            st.session_state["screen_config"] = {"direction": direction, "thresholds": thresholds, "min_score": min_score, "selected": scan_symbols}
    config = st.session_state.get("screen_config")
    records = st.session_state.get("screen_records", [])
    if not config or not records:
        st.info("Choose the universe and thresholds, then click **Scan universe**.")
        return
    direction = config["direction"]
    table = pd.DataFrame(records)
    filtered_records = [r for r in records if r["Score"] >= config["min_score"]]
    filtered_table = pd.DataFrame(filtered_records).sort_values("Score", ascending=False) if filtered_records else pd.DataFrame(columns=table.columns)
    candidate_symbols = [r["Symbol"] for r in filtered_records] or [r["Symbol"] for r in records]

    tab1, tab3 = st.tabs(["1 · Screener", "2 · Company deep dive"])
    with tab1:
        st.subheader(f"Scanned {direction.lower()} candidates")
        display = filtered_table.copy()
        if not display.empty:
            display["GICS Industry"] = display.Symbol.map({r["Symbol"]: r["Extra"].get("gics_industry", "NA") for r in filtered_records})
            display["Marketplace"] = display.Symbol.map({r["Symbol"]: r["Extra"].get("marketplace", "NA") for r in filtered_records})
            display["Current Price"] = display["Price"]
            display["Analyst Consensus"] = display["Recommendation"]
        for row in filtered_records:
            row["Technical"] = tech_flags(prices(row["Symbol"], "2y"), direction)
        if not display.empty:
            display["At EMA/support"] = display.Symbol.map({r["Symbol"]: r["Technical"].get("near_level", False) for r in filtered_records})
            display["Trend"] = display.Symbol.map({r["Symbol"]: r["Technical"].get("trend", "NA") for r in filtered_records})
            display["2× volume"] = display.Symbol.map({r["Symbol"]: r["Technical"].get("high_volume", False) for r in filtered_records})
        columns = ["Symbol", "Score", "Max", "GICS Industry", "Marketplace", "Current Price", "Analyst Consensus", "Revenue yoy %", "EPS yoy %", "FCF yoy %", "Operating margin %", "Target gap %", "Insider 90d $", "Finviz Insider Trans %", "Clenow Momentum", "At EMA/support", "Trend", "2× volume"]
        st.dataframe(style_screener_table(display[columns] if not display.empty else pd.DataFrame(columns=columns), filtered_records), use_container_width=True, hide_index=True)
        st.caption(f"{len(filtered_records)} of {len(records)} scanned symbols meet the {config['min_score']}-point minimum. Margins % is operating margin. Each satisfied criterion contributes 10 points; NA data is neutral.")
        st.caption("Each of 12 criteria contributes 10 points. NA data is neutral; inspect source coverage before acting. Earnings surprises, Form 4 clusters, and 13F flow require a filing parser or licensed feed when Yahoo does not expose them.")
        cols = st.columns(3)
        for col, title, keys in zip(cols, ["Fundamentals", "Valuation", "Insider activity"], [["Revenue yoy %", "EPS yoy %", "FCF yoy %", "Operating margin %"], ["Analyst Consensus", "Target gap %"], ["Insider 90d $", "Clenow Momentum"]]):
            with col:
                st.markdown(f"**{title}**")
                st.write(", ".join(keys))
                st.progress(min(float(filtered_table.Score.max()) / max(float(filtered_table.Max.max()), 1), 1) if not filtered_table.empty else 0)
    with tab3:
        symbol = st.selectbox("Company", candidate_symbols, key="deep_company")
        row = next(r for r in records if r["Symbol"] == symbol); metrics = row["Metrics"]; extra = row["Extra"]
        st.subheader(f"{symbol} · {extra.get('sector', 'Sector unavailable')}")
        st.write(extra.get("summary") or "Business summary unavailable from the current provider.")
        peer_source = st.radio("Peer source", ["Industry / market-cap", "Finviz first 3"], horizontal=True, key="peer_source")
        default_peer_list = finviz_peers(symbol) if peer_source == "Finviz first 3" else closest_peers(symbol)
        if not default_peer_list: default_peer_list = closest_peers(symbol)
        peer_valuation = valuation_frame(default_peer_list)
        peer_average_summary = "NA"
        if not peer_valuation.empty and "Peer average" in peer_valuation.Company.values:
            avg = peer_valuation[peer_valuation.Company == "Peer average"].iloc[0]
            peer_average_summary = " · ".join(f"{key} {avg[key]:.1f}x" for key in ["P/S", "P/E", "PEG", "P/OCF", "P/FCF", "P/B", "P/TBV"] if pd.notna(avg[key]))
        dcf_default = default_dcf_price(metrics, extra); consensus_target = extra.get("target"); current_price = extra.get("price")
        dcf_upside = dcf_default / current_price - 1 if dcf_default and current_price else None
        consensus_upside = consensus_target / current_price - 1 if consensus_target and current_price else None
        a, b, c, d, e, f, g = st.columns(7)
        cap_mm = extra.get("market_cap") / 1_000_000 if extra.get("market_cap") else None
        a.metric("Market cap ($mm)", f"{cap_mm:,.1f}" if cap_mm is not None else "NA"); b.metric("Stock price", fmt(current_price)); c.metric("DCF stock price", fmt(dcf_default), f"{dcf_upside:.1%} vs current" if dcf_upside is not None else "NA"); d.metric("Analyst consensus", fmt(consensus_target), f"{consensus_upside:.1%} vs current" if consensus_upside is not None else "NA"); e.metric("Last earnings", extra.get("last_earnings") or "NA"); f.metric("Next earnings", extra.get("next_earnings") or "NA"); g.metric("Peer average valuation", peer_average_summary)
        chart_frame = prices(symbol, "2y")
        if not chart_frame.empty:
            st.plotly_chart(chart(symbol, chart_frame, direction, row.get("Insider 90d $")), use_container_width=True)
        st.caption("Daily OHLCV from Yahoo Finance. EMA colors: 20-day cyan, 200-day orange, 200-week thick yellow; purple marks the latest available insider activity.")
        st.markdown("**Reported financials (SEC XBRL facts)**")
        annual_view = st.radio("Reporting period", ["Annual", "Quarterly"], horizontal=True, key="reporting_period") == "Annual"
        company_table = statement_frame(metrics, annual=annual_view)
        peers = [p.strip().upper() for p in st.text_input("Closest competitors (comma-separated)", value=", ".join(default_peer_list)).split(",") if p.strip() and p.strip().upper() != symbol][:4]
        peer_tables = [statement_frame(filing_metrics(peer), annual=annual_view) for peer in peers]
        st.markdown("**Reported financials — selected stock**")
        st.dataframe(style_financial_table(company_table, peer_tables), use_container_width=True)
        peer_label = st.selectbox("Comparison table", ["Industry average"] + [f"Peer {i + 1} · {p}" for i, p in enumerate(peers)], key="peer_table")
        if peer_label == "Industry average":
            aligned = [p for p in peer_tables if not p.empty]
            comparison_table = pd.concat(aligned).groupby(level=0).mean() if aligned else pd.DataFrame()
        else:
            peer_symbol = peers[int(peer_label.split("·")[0].split()[-1]) - 1]
            comparison_table = statement_frame(filing_metrics(peer_symbol), annual=annual_view)
        st.markdown(f"**Reported financials — {peer_label}**")
        st.dataframe(style_financial_table(comparison_table), use_container_width=True)
        st.markdown("**Comparable valuation multiples**")
        st.dataframe(valuation_frame([symbol] + peers), use_container_width=True, hide_index=True)
        available_lines = [label for _, label, _ in STATEMENT_LINES if any(str(index).endswith(f"· {label} ($mm)") or str(index).endswith(f"· {label}") for index in company_table.index)]
        if available_lines:
            selected_line = st.selectbox("Financial line for peer chart", available_lines, key="financial_line")
            comparison = peer_metric_frame([symbol] + peers, selected_line, annual=annual_view)
            if not comparison.empty:
                latest_column = comparison.columns[-1]
                chart_values = comparison[latest_column].rename("Latest reported value").to_frame()
                st.markdown(f"**{selected_line}: selected stock vs Peer 1/2/3 and industry average**")
                st.bar_chart(chart_values)
        st.markdown("**DCF workspace**")
        analyst_growth = pct(extra.get("analyst_growth")) or 12.0
        consensus_target = extra.get("target")
        assumptions = st.columns(5)
        revenue_growth = assumptions[0].number_input("Revenue growth %", value=float(analyst_growth), key="dcf_growth")
        margin = assumptions[1].number_input("Operating margin %", value=float(metrics.get("operating_margin") or 20), key="dcf_margin")
        wacc = assumptions[2].number_input("WACC %", value=9.0, key="dcf_wacc")
        terminal = assumptions[3].number_input("Terminal growth %", value=3.0, key="dcf_terminal")
        shares = assumptions[4].number_input("Shares (mm)", value=float(metrics.get("shares").iloc[-1] / 1_000_000) if isinstance(metrics.get("shares"), pd.Series) and not metrics["shares"].empty else 1.0, key="dcf_shares")
        st.caption(f"Analyst consensus growth used to prefill: {analyst_growth:.1f}%. Company guidance: {extra.get('guidance') or 'Not available from the current source.'}")
        base = ttm(metrics.get("revenue")); base_cogs = ttm(metrics.get("cogs")); base_rd = ttm(metrics.get("rd")); base_ga = ttm(metrics.get("ga")); base_interest = ttm(metrics.get("interest")); base_cfo = ttm(metrics.get("cfo")); base_capex = ttm(metrics.get("capex")); years = np.arange(0, 6)
        forecast = pd.DataFrame({"Year": ["Y0 Actuals"] + [f"Y{i}" for i in range(1, 6)]})
        forecast["Revenue"] = [base * (1 + revenue_growth / 100) ** i for i in years]
        forecast["COGS"] = forecast.Revenue * (base_cogs / base if base else (1 - (metrics.get("gross_margin") or 50) / 100))
        forecast["Gross Profit"] = forecast.Revenue - forecast.COGS
        forecast["R&D"] = forecast.Revenue * (base_rd / base if base and base_rd else .08); forecast["G&A"] = forecast.Revenue * (base_ga / base if base and base_ga else .10)
        forecast["Total Operating Expenses"] = forecast["R&D"] + forecast["G&A"]
        forecast["Operating Income"] = forecast.Revenue * margin / 100
        forecast["Interest Income"] = forecast.Revenue * (base_interest / base if base and base_interest else .005); forecast["Pretax Income"] = forecast["Operating Income"] + forecast["Interest Income"]
        forecast["Taxes"] = forecast["Pretax Income"] * .21; forecast["Net Income"] = forecast["Pretax Income"] - forecast["Taxes"]
        forecast["EPS"] = forecast["Net Income"] / max(shares, 1e-9) / 1_000_000
        forecast["Free Cash Flow"] = [base_cfo - base_capex] + [max(float(forecast.loc[i, "Net Income"]), 0) for i in range(1, 6)]
        amount_columns = [column for column in forecast.columns if column not in {"Year", "EPS"}]
        forecast_display = forecast.copy(); forecast_display[amount_columns] = forecast_display[amount_columns] / 1_000_000
        st.dataframe(forecast_display.set_index("Year").T, use_container_width=True)
        fcf_forecast = forecast.loc[1:, "Free Cash Flow"].to_numpy(dtype=float); discount_rate = max(wacc / 100, terminal / 100 + .01); terminal_value = fcf_forecast[-1] * (1 + terminal / 100) / (discount_rate - terminal / 100) if fcf_forecast.size else 0
        pv_equity = float((fcf_forecast / (1 + discount_rate) ** np.arange(1, 6)).sum() + terminal_value / (1 + discount_rate) ** 5 + (extra.get("cash") or 0) - (extra.get("debt") or 0)) if base and shares else 0
        dcf_price = pv_equity / (shares * 1_000_000) if shares else None
        dcf_upside = dcf_price / current_price - 1 if dcf_price and current_price else None
        consensus_upside = consensus_target / current_price - 1 if consensus_target and current_price else None
        st.caption(f"Editable DCF output: {fmt(dcf_price)} per share. The headline DCF and analyst consensus metrics remain at the top of this tab.")


if __name__ == "__main__":
    main()
