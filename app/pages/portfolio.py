"""🔬 Cartera — level 4: what you actually hold (CLAUDE.md section 2, phase 1 point 4).

The real portfolio is read from the IBKR Activity Flex Query and nothing else. It is
never typed in by hand (section 5.1), so until the Flex credentials exist this page has
nothing true to show — and says so, rather than rendering an empty table that looks like
"you hold nothing".
"""

from __future__ import annotations

import streamlit as st

from core.config import load_settings

settings = load_settings()

st.title("🔬 Cartera")

token = settings.secret("IBKR_FLEX_TOKEN")
query_id = settings.secret("IBKR_FLEX_QUERY_ID")

if not (token and query_id):
    st.warning("La cuenta de IBKR no está conectada todavía.")
    st.markdown(
        """
Esta página lee la cuenta **en solo lectura**, vía el *Activity Flex Query* del Flex Web
Service. Ese token es estructuralmente incapaz de colocar órdenes o mover fondos: es una
limitación del sistema de IBKR, no una configuración que haya que recordar mantener.

**Para desbloquearla** (hazlo primero contra la cuenta *paper*, que tiene su propio token
y su propio query id):

1. Crea un **Activity Flex Query** en formato XML con las secciones `AccountInformation`,
   `OpenPositions`, `Trades`, `CashTransactions` y `ChangeInDividendAccruals`.
2. En la sección *Trades*, **no** actives *Symbol Summary* ni *Orders*: rompen el parser
   de `ibflex`.
3. Genera el token del **Flex Web Service** y anota su fecha de caducidad — caduca, y
   cuando lo haga el fallo tiene que ser ruidoso.
4. Escribe `IBKR_FLEX_TOKEN` e `IBKR_FLEX_QUERY_ID` en `config/.env`.

Verifica la ruta exacta del menú en el portal de IBKR: cambia con cada rediseño.
"""
    )
    st.stop()

st.info(
    "Credenciales detectadas, pero `ingest/ibkr_flex.py` todavía no está implementado "
    "(fase 1, punto 3). La valoración, el drift, el PnL FIFO y la reconciliación contra "
    "el NAV que reporta IBKR llegan con él."
)
