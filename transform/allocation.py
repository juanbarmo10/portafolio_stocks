"""Core and satellite, and the rule for idle cash (📋 Desplegar; CLAUDE.md §15.5, point 4).

The written investment policy (§15.4 point 8) splits the account in two: a **core** of index
funds that captures the market's return at almost no cost and no decision, and a
**satellite** of a few written theses where the edge — and the risk — is sought. This
module checks the account against that policy. It never decides the policy: every
parameter is the user's (``portfolio.core`` and the cash limits), ``None`` until written,
and without them it says so instead of assuming one.

Three questions:

- **How is the account split today?** Core funds, satellite positions and cash, over the
  whole account.
- **Where does the next tranche go?** To the core while it is below its target by more than
  the written band, to the satellite otherwise — and then, first, to reinforcing a thesis
  already held (``transform.discipline``). **Rebalancing is done with contributions, never
  with sales**: selling a winner to feed the core is the disposition effect with a better
  excuse, and costs a commission and possibly a tax event.
- **Has cash stayed above its limit for too long?** Cash that waits is a position nobody
  decided (the account measured it: 3,6 of −11,6 points against SPY in a year).

Pure functions; no network, no database (section 10).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class Policy:
    """The written policy. ``None`` = not written, never a default."""

    core_tickers: list[str]
    core_target: float | None          # share of the whole account in the core
    band: float | None                 # tolerance around the target before acting
    max_satellites: int | None
    cash_max: float | None             # share of the account
    cash_grace_days: int | None

    @classmethod
    def from_config(cls, portfolio: Mapping[str, Any] | None) -> "Policy":
        cfg = dict(portfolio or {})
        core = dict(cfg.get("core") or {})
        return cls(
            core_tickers=[str(t).upper() for t in core.get("tickers") or []],
            core_target=core.get("target_share"),
            band=core.get("band"),
            max_satellites=cfg.get("max_satellites"),
            cash_max=cfg.get("cash_max"),
            cash_grace_days=cfg.get("cash_grace_days"),
        )

    @property
    def written(self) -> bool:
        return bool(self.core_tickers) and self.core_target is not None


@dataclass(frozen=True)
class Allocation:
    """The account against the policy. Shares are over the whole account (cash included)."""

    nav: float | None
    core_value: float
    satellite_value: float
    cash: float | None
    core_share: float | None
    satellite_share: float | None
    cash_share: float | None
    satellites: list[str]
    destination: str | None            # "core" | "satellite" | None (no policy)
    to_target: float | None            # contributions, all to the core, to reach the target
    notes: list[str] = field(default_factory=list)


def allocate(valued: pd.DataFrame | None, nav: float | None, cash: float | None,
             policy: Policy) -> Allocation:
    """See :class:`Allocation`. ``valued`` is ``portfolio.valuation`` of every position."""
    rows = valued if valued is not None and not valued.empty else pd.DataFrame(
        columns=["ticker", "market_value"])
    values = pd.to_numeric(rows["market_value"], errors="coerce").fillna(0.0)
    is_core = rows["ticker"].astype(str).str.upper().isin(policy.core_tickers)
    core, satellite = float(values[is_core].sum()), float(values[~is_core].sum())
    satellites = sorted(str(t) for t, v in zip(rows["ticker"], values) if v > 0
                        and str(t).upper() not in policy.core_tickers)

    def share(part: float | None) -> float | None:
        return None if part is None or not nav or nav <= 0 else part / nav

    notes: list[str] = []
    destination = to_target = None
    core_share = share(core)
    if not policy.written:
        notes.append("Sin política escrita (`portfolio.core.tickers` y `target_share`): el "
                     "panel no sabe qué parte es núcleo ni a dónde debe ir el tramo.")
    elif core_share is not None:
        target, band = float(policy.core_target), float(policy.band or 0.0)
        if core_share < target - band:
            destination = "core"
            # All new money to the core: target·(NAV + x) = core + x.
            to_target = max(0.0, (target * nav - core) / (1 - target)) if target < 1 else None
        else:
            destination = "satellite"
    if policy.max_satellites is not None and len(satellites) >= int(policy.max_satellites):
        notes.append(f"Ya tienes {len(satellites)} posiciones satélite y tu máximo es "
                     f"{int(policy.max_satellites)}: una idea nueva entra solo si sale otra.")
    return Allocation(
        nav=nav, core_value=core, satellite_value=satellite, cash=cash,
        core_share=core_share, satellite_share=share(satellite), cash_share=share(cash),
        satellites=satellites, destination=destination, to_target=to_target, notes=notes,
    )


def cash_spell(nav: pd.DataFrame, cash: pd.DataFrame, limit: float | None,
               ) -> tuple[int | None, str | None]:
    """``(days, since)``: how long cash has been above ``limit`` of the account, without a
    break, up to the latest statement. ``(0, None)`` when it is not above it now; ``(None,
    None)`` without a limit or without data."""
    if limit is None or nav is None or nav.empty or cash is None or cash.empty:
        return None, None

    def series(frame: pd.DataFrame) -> pd.Series:
        out = pd.Series(frame["value"].astype(float).to_numpy(),
                        index=pd.to_datetime(frame["ts"].astype(str).str[:10]))
        return out[~out.index.duplicated(keep="last")].sort_index()

    total, idle = series(nav), series(cash)
    shares = (idle / total.where(total > 0)).dropna()
    if shares.empty or shares.iloc[-1] <= limit:
        return 0, None
    below = shares[shares <= limit]
    start = shares[shares.index > below.index[-1]].index[0] if not below.empty \
        else shares.index[0]
    return int((shares.index[-1] - start).days), start.date().isoformat()
