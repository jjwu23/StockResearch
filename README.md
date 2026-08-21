# Equity Signal Lab

A Streamlit equity research dashboard for NYSE/Nasdaq long and short candidates. It includes configurable fundamental, valuation, and insider-activity screening; technical confirmation; SEC reported financial tables; a lightweight DCF forecast workspace; and a daily EMA/candlestick chart.

The app is explicit about provenance. SEC Company Facts is used for reported financial metrics when the ticker can be mapped to a CIK. Yahoo Finance is used for daily OHLCV and optional analyst/holder enrichment. Missing data is shown as `NA` and is neutral in the score. The dashboard is an educational research tool, not investment advice.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown by Streamlit. Choose symbols and long/short thresholds in the sidebar. Use the four tabs to inspect ranking, technical confirmation, company financials/DCF assumptions, and the chart.

For production use, replace the optional Yahoo Finance analyst/insider enrichment with a licensed feed or a filing parser that directly extracts Form 4 and 13F transactions. The current app intentionally does not treat missing insider or analyst data as a passing signal.
