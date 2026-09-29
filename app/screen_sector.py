"""🔎 Cribado, "Grupos de sector" mode (CLAUDE.md §15.5, point 14).

Rendered by ``app/pages/screen.py``. A group the user names in ``universe.sector_groups``
(silver miners first), each company on its latest fiscal year, US GAAP or IFRS
(``ingest/sector_groups``, ``transform/sector_health``): the balance sheet, the cash it
generates and the dilution, side by side. Nothing is ranked or scored (section 12).
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd
import streamlit as st

from app import data as app_data
from app.format import color_by_sign
from ingest.sector_groups import SOURCE
from transform import sector_health as sh
from transform.adjustments import split_adjusted


@st.cache_data(show_spinner="Leyendo el grupo…")
def group_view(tickers: tuple[str, ...], as_of_iso: str, mtime: float) -> pd.DataFrame:
    observations = app_data.observations(SOURCE, mtime)
    registry = app_data.companies()
    names = {str(r["cik"]): (str(r["ticker"]), r.get("name"))
             for r in registry.to_dict("records")}
    rows = app_data.prices_for(list(tickers))
    actions = app_data.corporate_actions()
    prices: dict[str, float | None] = {}
    for t in tickers:
        own = rows[rows["series_id"] == f"{t}:close_raw"]
        adjusted = split_adjusted(own.assign(ts=own["ts"].str[:10]), actions, t, as_of_iso) \
            if not own.empty else own
        prices[t] = float(adjusted["value"].iloc[-1]) if len(adjusted) else None
    return sh.health_table(observations, list(tickers), names, prices, as_of_iso)


def render(settings: Any, status: Callable[[str], str]) -> None:
    groups = settings.sector_groups
    if not groups:
        st.info("Sin grupos escritos. En `universe.sector_groups` de `settings.local.yaml`, "
                "un nombre y sus tickers (por ejemplo, mineras de plata); después "
                "`python run_ingest.py --only sector_groups` y `--only prices`.")
        return
    name = st.selectbox("Grupo", list(groups))
    tickers = tuple(groups[name])
    as_of_iso = pd.Timestamp.today().date().isoformat()
    table = group_view(tickers, as_of_iso, app_data.db_mtime())
    if table["fy_end"].isna().all():
        st.info("El grupo todavía no se ha descargado: `python run_ingest.py --only "
                "sector_groups` y `--only prices`.")
        return
    ingested = app_data.source_last_ingest(SOURCE, app_data.db_mtime())
    st.markdown(f"**{name}** · {len(tickers)} empresas · cifras del último ejercicio anual "
                f"presentado · descargado el {ingested} · precios al último cierre")

    millions = 1e6
    display = pd.DataFrame({
        "Ticker": table["ticker"],
        "Empresa": table["name"],
        "Estado": table["ticker"].map(lambda t: status(str(t))),
        "Normas": table["basis"],
        "Ejercicio": table["fy_end"],
        "Ingresos (M)": table["revenue"] / millions,
        "Crec. ingresos": table["revenue_growth"],
        "Margen operativo": table["operating_margin"],
        "FCF (M)": table["fcf"] / millions,
        "Margen FCF": table["fcf_margin"],
        "Caja (M)": table["cash"] / millions,
        "Deuda (M)": table["debt"] / millions,
        "Caja neta (M)": table["net_cash"] / millions,
        "Deuda neta / EBITDA": table["net_debt_to_ebitda"],
        "Dilución": table["dilution"],
        "Capitalización (M)": table["market_cap"] / millions,
        "Rend. FCF": table["fcf_yield"],
        "VE / EBITDA": table["ev_to_ebitda"],
        "Notas": table["notes"],
    })
    percent = st.column_config.NumberColumn(format="percent")
    money = st.column_config.NumberColumn(format="%.0f")
    st.dataframe(
        # Signs that are facts for the holder: cash generated and net cash up are good,
        # dilution is not. Growth and margins too. Nothing is a score.
        color_by_sign(display, {"Crec. ingresos": True, "Margen operativo": True,
                                "FCF (M)": True, "Margen FCF": True, "Caja neta (M)": True,
                                "Dilución": False}),
        hide_index=True, width="stretch",
        column_config={
            "Ingresos (M)": money, "FCF (M)": money, "Caja (M)": money, "Deuda (M)": money,
            "Caja neta (M)": st.column_config.NumberColumn(
                format="%.0f", help="Caja − deuda al cierre del ejercicio. Positiva = puede "
                                    "pagar toda su deuda con la caja que tiene."),
            "Capitalización (M)": money, "Crec. ingresos": percent,
            "Margen operativo": percent, "Margen FCF": percent, "Dilución": percent,
            "Rend. FCF": percent,
            "Deuda neta / EBITDA": st.column_config.NumberColumn(
                format="%.1f×", help="Años de EBITDA para pagar la deuda neta. Vacío con caja "
                                     "neta (no hay deuda neta que pagar) o sin EBITDA."),
            "VE / EBITDA": st.column_config.NumberColumn(format="%.1f×"),
        },
    )
    st.caption(
        "**Cómo leerla.** Una minera sana es la que aguanta una caída del metal **sin emitir "
        "acciones ni pedir prestado**: caja neta positiva (o deuda neta baja frente al EBITDA), "
        "FCF positivo y poca dilución. Todo sale del último informe anual, a los precios del "
        "metal de ese año: con la plata más cara hoy, el ejercicio siguiente será mejor, y al "
        "revés. **Límites:** solo cuenta la deuda con etiqueta estándar (un préstamo o un "
        "convertible con etiqueta propia no se ve; confírmalo en el 10-K o el 40-F); la mezcla "
        "plata / oro de cada una y su coste total por onza (AISC) no están en el XBRL. "
        "Crecimientos del 50-120 % y acciones +55 % o más vienen de compras de otras mineras. "
        "Las que presentan 40-F (IFRS) solo tienen cifras anuales: la página de Empresa no las "
        "lee trimestre a trimestre.")
    missing = [t for t, fy in zip(table["ticker"], table["fy_end"]) if fy is None or
               pd.isna(fy)]
    if missing:
        st.caption(f"Sin cifras en la SEC: {', '.join(missing)}.")
