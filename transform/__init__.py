"""Derived indicators. Pure functions, no network (CLAUDE.md section 10).

    macro.py              level 1: point-in-time readings
    breadth.py, regime.py level 2: breadth, RSP/SPY, rotation, the regime light
    fundamentals.py       level 3: TTM, derived Q4 and YTD quarters, margins
    value_accrual.py      level 3: dilution, SBC, net buybacks, ROIC
    thesis.py             level 3: the invalidation board
    portfolio.py          level 4: valuation, FIFO, reconciliation, performance vs market
    adjustments.py        adjusted series at a cut-off date, total return (section 9.1)
    corporate_actions.py  IBKR vs yfinance, the needs_review flag (section 9.2)
"""
