"""The behaviour mirror (CLAUDE.md §15.4), on a synthetic account with known answers."""

from __future__ import annotations

import pandas as pd
import pytest

from transform import behavior as bh

END = "2026-09-25"


def trade(tid, day, ticker, side, qty, price):
    return {"trade_id": tid, "conid": "1", "cik": None, "ticker": ticker,
            "ts": f"{day}T15:00:00+00:00", "side": side, "quantity": qty, "price": price,
            "currency": "USD", "commission": -0.35, "fx_rate": 1.0}


def daily(values, start="2025-10-01"):
    days = pd.bdate_range(start, periods=len(values))
    return pd.DataFrame({"ts": [d.date().isoformat() for d in days], "value": values})


TRADES = pd.DataFrame([
    trade("1", "2025-10-01", "WIN", "buy", 1.0, 100.0),
    trade("2", "2025-10-21", "WIN", "sell", 1.0, 110.0),     # sold at a gain after 20 days
    trade("3", "2025-10-01", "LOSE", "buy", 1.0, 100.0),     # kept, at a loss
])


def test_the_mirror_measures_turnover_holding_and_the_disposition_pattern():
    nav = daily([200.0] * 250)
    cash = daily([100.0] * 250)
    valued = pd.DataFrame({"ticker": ["LOSE"], "unrealized_pnl": [-20.0]})
    after = pd.Series([110.0, 132.0], index=pd.to_datetime(["2025-10-21", END]))
    spy = pd.Series([100.0, 110.0], index=pd.to_datetime(["2025-10-21", END]))
    m = bh.mirror(TRADES, nav, cash, valued, {"WIN": after, "SPY": spy}, END,
                  min_holding_days=90)
    assert m.turnover == pytest.approx(110.0 / 200.0), "min(purchases, sales) / average NAV"
    assert m.holding_median_days == pytest.approx(20.0)
    assert m.sold_before_horizon == pytest.approx(1.0)
    assert (m.sells_at_gain, m.sells_at_loss, m.open_at_loss) == (1, 0, 1)
    assert m.after_sale == pytest.approx(110.0 * 0.20), "what the sold share did afterwards"
    assert m.after_sale_spy == pytest.approx(110.0 * 0.10)
    assert m.cash_share_average == pytest.approx(0.5)
    assert m.commission_drag == pytest.approx(1.05 / 200.0)


def test_an_account_that_only_buys_turns_over_nothing():
    buys = TRADES[TRADES["side"] == "buy"]
    m = bh.mirror(buys, daily([200.0] * 250), daily([0.0] * 250), pd.DataFrame(), {}, END)
    assert m.turnover == 0.0 and m.holding_median_days is None
    assert m.after_sale is None, "nothing sold, nothing to measure — not zero"


def test_without_the_sold_stock_price_the_cost_of_selling_is_unknown():
    m = bh.mirror(TRADES, daily([200.0] * 250), daily([0.0] * 250), pd.DataFrame(), {}, END)
    assert m.after_sale is None
