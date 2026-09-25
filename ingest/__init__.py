"""Data ingestion. Every module exposes ``fetch() -> DataFrame`` (CLAUDE.md section 10).

    fred.py            macro series, first publication (+ pre-ALFRED history, derived dates)
    macro_calendar.py  upcoming CPI / PCE / payrolls / FOMC -> events
    prices.py          raw OHLCV + corporate actions (yfinance, un-adjusted)
    ibkr_flex.py       the real account, read-only (Flex Web Service)
    sec_xbrl.py        audited fundamentals, point-in-time (SEC companyfacts)
    sec_filings.py     filing history, amendments, earnings calendar, companies registry
    short_interest.py  FINRA short interest (derived publication date)
    universe.py        S&P 500 membership history + member closes, for breadth
    local_fx.py        official local-currency rate (configured only locally, section 11)
"""
