"""Display formatting, with the public-mode rule enforced in one place.

Public mode is a deployment of this panel that anyone can open (CLAUDE.md sections 1,
11). **It renders no absolute money figure**: not the NAV, not a position's value, not a
commission in dollars. It renders the same information in relative form instead — an
index rebased to 100, a weight, a percentage — which carries the method without carrying
the balance. That is the whole public argument of the project anyway: how decisions are
made, not how much is at stake.

Two alternatives were rejected on 2026-09-17; the reasoning is here so it is not
re-litigated later (RESEARCH.md section 2.8):

- **Scaling every amount by a secret factor.** Commissions do not scale with portfolio
  size — IBKR tiered charges a per-order minimum — so one published commission recovers
  the factor, and with it every real figure. It also fails *open*: a deployment that
  forgets the env var publishes the real numbers. And it puts fabricated figures on
  screen, which is exactly what section 12 forbids.
- **A per-widget ``if public_mode``.** Remembered until the day it is not, and by then
  the number is published and cached by someone else.

So ``money()`` raises in public mode. A widget that tries to print an amount breaks
loudly in the deployment where breaking is harmless, instead of leaking in the one where
it is not. ``tests/test_public_mode.py`` renders the whole app publicly and fails on any
currency-shaped string, which covers widgets that never call in here.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

MISSING = "—"
"""Rendered for an absent value. A hole stays visible; it never becomes a zero (section 12)."""

PUBLIC_PAGES = frozenset({"Hoy", "Mercado", "Empresa", "Cartera"})
"""Page titles that may ship in a public deployment.

An allow-list, not a deny-list: a page added later is private until someone decides
otherwise, so the way this fails is a missing page rather than a published one.

**Only pages that exist are listed.** "Mercado", "Desplegar capital" and "Método" were
here before they were built, which quietly broke the rule above: the day phase 3 created
a page titled "Mercado" it would have shipped publicly without anyone deciding so. A name
earns its place here when the page exists and someone has looked at what it renders.
"Mercado" re-entered that way on 2026-09-24, by the user's decision, after its public
render was checked: market data only, and the short-interest block — which names
positions — left out.

The fiscal page is absent deliberately and must stay absent — its configuration alone
reveals the jurisdiction (section 11).
"""

PUBLIC_NOTE = (
    "**Vista pública.** Todas las cifras de cartera son **relativas**: índice base 100, "
    "pesos y porcentajes. Los importes absolutos no se publican — y no están escalados "
    "por un factor, que es lo mismo que publicarlos para quien sepa mirar."
)


def _spanish(text: str) -> str:
    """Swap an English-formatted number to Spanish notation: ``1,138.97`` -> ``1.138,97``.

    Via a placeholder, because replacing ``,`` and ``.`` in sequence makes the second
    replacement eat the output of the first — ``1,234.5678`` comes out as ``1,234,5678``,
    which reads as a different number and is exactly the bug this helper exists to stop
    being rewritten by hand at each call site.
    """
    return text.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


class PublicModeViolation(RuntimeError):
    """An absolute money figure was requested in a deployment that must not show one."""


def money(value: float | None, *, public: bool, currency: str = "USD") -> str:
    """Format an absolute amount, or refuse to in public mode.

    Args:
        value: Amount. ``None`` and ``NaN`` render as a visible hole.
        public: Whether this render is the public deployment.
        currency: ISO code appended to the figure.

    Returns:
        The amount in Spanish notation, e.g. ``'1.138,97 USD'``.

    Raises:
        PublicModeViolation: Always, when ``public`` is true. The caller is asking for the
            one thing the public deployment must not show; the fix is a relative metric
            (``rebase_100``, ``pct``, ``share``), not a quieter amount.
    """
    if public:
        raise PublicModeViolation(
            "Public mode renders no absolute amount. Use rebase_100(), share() or pct(), "
            "or drop the figure from the public view."
        )
    if value is None or pd.isna(value):
        return MISSING
    return f"{_spanish(f'{value:,.2f}')} {currency}"


def reported_amount(value: float | None, *, unit: str = "USD") -> str:
    """Format a figure taken from a public filing. **Not gated, on purpose.**

    Microsoft's revenue is public information: it is in a 10-K anyone can download, and
    hiding it publicly would be theatre rather than privacy. What :func:`money` gates is
    the *account* — the user's NAV, positions, commissions and dividends — because that is
    the part a reader has no business knowing (RESEARCH.md section 2.8).

    ⚠️ Using this function for an account figure would walk straight around that gate. If
    the number came from IBKR, it goes through :func:`money`.

    Large amounts are abbreviated, because a revenue line reading 331.839.000.000 is a
    number nobody parses at a glance. The scale words are spelled out in Spanish on
    purpose: "MM" means millions to some readers and thousands of millions to others, and
    the English "billion" is not the Spanish "billón" — an abbreviation that needs a
    convention to disambiguate is an abbreviation that will eventually be misread.
    """
    if value is None or pd.isna(value):
        return MISSING
    magnitude = abs(value)
    for limit, scale in ((1e12, "billones"), (1e9, "mil millones"), (1e6, "millones")):
        if magnitude >= limit:
            return f"{_spanish(f'{value / limit:,.2f}')} {scale} {unit}"
    return f"{_spanish(f'{value:,.2f}')} {unit}"


def number(value: float | None, *, decimals: int = 2) -> str:
    """A plain figure in Spanish notation: ``14.21`` -> ``14,21``. Safe in both modes.

    For index levels, ratios and spreads — anything that is neither money nor a share
    count. Exists so no page formats a number with a bare f-string and ends up mixing
    ``14.21`` next to ``24,8 %``.
    """
    if value is None or pd.isna(value):
        return MISSING
    return _spanish(f"{value:,.{decimals}f}")


def quantity(value: float | None, *, decimals: int = 4) -> str:
    """Format a share count in Spanish notation. Safe in both modes — a count is not money.

    Fractional shares are enabled on this account, so the decimals are not decoration.
    """
    if value is None or pd.isna(value):
        return MISSING
    return _spanish(f"{value:,.{decimals}f}")


def pct(fraction: float | None, *, decimals: int = 1) -> str:
    """Format a fraction as a percentage. Safe in both modes — a ratio hides the size.

    Args:
        fraction: Ratio, not percent points: ``0.0827`` renders as ``'8,3 %'``.
        decimals: Decimal places.
    """
    if fraction is None or pd.isna(fraction):
        return MISSING
    return f"{fraction * 100:.{decimals}f}".replace(".", ",") + " %"


def share(part: float | None, whole: float | None) -> float | None:
    """Fraction ``part / whole``, or ``None`` when it is not defined.

    A zero or missing denominator returns ``None`` rather than raising or defaulting to
    zero: a weight nobody can compute is a hole, not 0 % (section 12).
    """
    if part is None or whole is None or pd.isna(part) or pd.isna(whole) or whole == 0:
        return None
    return float(part) / float(whole)


def rebase_100(values: pd.Series | Sequence[float]) -> pd.Series:
    """Rebase a series to 100 at its first present, non-zero value.

    Scale-invariant by construction: ``rebase_100(s)`` equals ``rebase_100(k * s)`` for
    any non-zero ``k``. That is the point — the published curve is identical whatever the
    account is worth, so it reveals nothing about size. A scaled amount reveals
    everything the moment one anchor leaks; a rebased index has no anchor to leak.

    Args:
        values: Numeric series, in chronological order. Non-numeric entries become NaN.

    Returns:
        Series on the same index, ``NaN`` where the input had no value and throughout
        when the input never had a usable base.
    """
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    usable = series.dropna()
    usable = usable[usable != 0]
    if usable.empty:
        return pd.Series([float("nan")] * len(series), index=series.index, dtype="float64")
    return series / float(usable.iloc[0]) * 100.0


def drop_absolute_columns(
    frame: pd.DataFrame, columns: Sequence[str], *, public: bool
) -> pd.DataFrame:
    """Remove the columns holding absolute amounts when rendering publicly.

    The column is dropped, not blanked: an empty column invites the reader to wonder what
    was in it and invites the next developer to fill it back in.
    """
    if not public:
        return frame
    return frame.drop(columns=[column for column in columns if column in frame.columns])
