"""🔎 Cribado, "Crecimiento" mode (CLAUDE.md §15.5, point 2).

Rendered by ``app/pages/screen.py`` when the mode is chosen. Every SEC filer at its latest
reported quarter (``transform/growth_screen``): growth, its acceleration and its quality.
Every limit is editable on the page and starts at the value written in
``screen.growth.defaults`` (decided by the user on 2026-09-28: 300 M market cap, 1 M daily
dollar volume, financials apart).
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd
import streamlit as st

from app import data as app_data
from app.format import MISSING, color_by_sign
from ingest.growth_screen import SOURCE
from transform import growth_screen as gs
from transform import screen as sc


def render(settings: Any, status: Callable[[str], str], researched: set[str]) -> None:
    """The growth mode. ``status(ticker)`` marks held / with thesis / under study;
    ``researched`` are the tickers already in ``tracked`` or ``watchlist``."""
    growth_cfg = dict(settings.raw.get("screen", {}).get("growth") or {})
    defaults = dict(growth_cfg.get("defaults") or {})
    table = app_data.growth_view(app_data.db_mtime())
    if table.empty:
        st.info("El cribado de crecimiento todavía no se ha descargado. Ejecuta "
                "`python run_ingest.py --only growth_screen --force` (unos 15-20 minutos la "
                f"primera vez); después corre solo cada {growth_cfg.get('run_every_days', 90)} "
                "días.")
        return

    ingested = app_data.source_last_ingest(SOURCE, app_data.db_mtime())
    newest = str(table.loc[~table["stale"], "quarter"].iloc[0]) if (~table["stale"]).any() \
        else MISSING
    st.markdown(
        f"**Último trimestre: {newest.replace('CY', '').replace('Q', ' T')}** · "
        f"{len(table)} empresas con un trimestre reciente · {int(table['stale'].sum())} de "
        f"ellas con un trimestre anterior · descargado el {ingested} · precios al "
        f"{table['price_date'].dropna().max() or MISSING}")

    st.subheader("Filtros")
    st.caption("Empiezan en los valores escritos en `settings.yaml` (`screen.growth.defaults`) "
               "y aquí puedes cambiarlos o apagarlos. No se optimizaron contra nada.")
    cols = st.columns(6)

    def limit(column: Any, label: str, key: str, *, scale: float, step: float,
              help_text: str) -> float | None:
        on = column.checkbox(label, value=defaults.get(key) is not None, key=f"g_on_{key}")
        value = column.number_input(label, value=float(defaults.get(key) or 0.0) * scale,
                                    step=step, key=f"g_{key}", label_visibility="collapsed",
                                    disabled=not on, help=help_text)
        return value / scale if on else None

    limits = {
        "min_market_cap": limit(cols[0], "Capitalización mín. (M USD)", "min_market_cap",
                                scale=1e-6, step=50.0,
                                help_text="Último cierre × acciones básicas del trimestre."),
        "min_dollar_volume": limit(cols[1], "Volumen diario mín. (M USD)", "min_dollar_volume",
                                   scale=1e-6, step=0.5,
                                   help_text="Mediana de precio × volumen en las últimas "
                                             f"{growth_cfg.get('volume_sessions', 60)} "
                                             "sesiones: lo que se puede comprar y vender sin "
                                             "mover el precio."),
        "min_revenue_q": limit(cols[2], "Ingresos trim. mín. (M USD)", "min_revenue_q",
                               scale=1e-6, step=10.0,
                               help_text="Un crecimiento sobre una base minúscula (una "
                                         "empresa que empieza a facturar) encabeza cualquier "
                                         "orden por crecimiento."),
        "min_revenue_growth": limit(cols[3], "Crec. ingresos mín. (%)", "min_revenue_growth",
                                    scale=100, step=5.0,
                                    help_text="Trimestre frente al mismo del año anterior."),
        "min_gross_margin": limit(cols[4], "Margen bruto mín. (%)", "min_gross_margin",
                                  scale=100, step=5.0,
                                  help_text="Beneficio bruto / ingresos del trimestre."),
        "max_dilution": limit(cols[5], "Dilución máx. (%)", "max_dilution", scale=100,
                              step=1.0, help_text="Acciones básicas frente al año anterior."),
    }
    c1, c2 = st.columns(2)
    keep_unknown = c1.checkbox(
        "Conservar las que un filtro no puede juzgar (dato faltante)", value=False,
        key="g_keep_unknown",
        help="Desconocido no es aprobar ni suspender. Por defecto quedan fuera, pero se cuentan.")
    exclude_financials = c2.checkbox(
        "Apartar bancos, aseguradoras y REIT (SIC 6000-6799)",
        value=bool(defaults.get("exclude_financials", True)), key="g_exclude_financials",
        help="Su flujo de caja y su margen bruto no significan lo mismo que en una industrial.")
    result = sc.apply_filters(table, limits, keep_unknown=keep_unknown,
                              exclude_financials=exclude_financials, filters=gs.FILTERS)
    labels = {"market_cap": "capitalización", "dollar_volume": "volumen",
              "revenue_q": "ingresos",
              "revenue_growth": "crecimiento", "gross_margin": "margen bruto",
              "dilution": "dilución"}
    line = f"**{len(result.table)}** pasan · {result.failed} no pasan"
    if result.excluded_financials:
        line += f" · {result.excluded_financials} financieras aparte"
    unknown = {k: v for k, v in result.unknown.items() if v}
    if unknown and not keep_unknown:
        line += (" · fuera por falta de dato: "
                 + ", ".join(f"{v} ({labels[k]})" for k, v in unknown.items()))
    st.markdown(line)

    shown = result.table.sort_values("rule_of_40", ascending=False, na_position="last")
    display = pd.DataFrame({
        "Ticker": shown["ticker"],
        "Empresa": shown["name"],
        "Estado": shown["ticker"].map(lambda t: status(str(t))),
        "Trimestre": shown["quarter"].str.replace("CY", "").str.replace("Q", " T"),
        "Ingresos trim. (M)": shown["revenue_q"] / 1e6,
        "Crec. ingresos": shown["revenue_growth"],
        "Aceleración (pp)": shown["acceleration"] * 100,
        "Margen bruto": shown["gross_margin"],
        "Δ margen bruto (pp)": shown["gross_margin_change"] * 100,
        "Margen operativo": shown["operating_margin"],
        "Margen op. incremental": shown["incremental_margin"],
        "Regla del 40": shown["rule_of_40"],
        "Margen FCF (año)": shown["fcf_margin"],
        "SBC / ingresos (año)": shown["sbc_over_revenue"],
        "Dilución": shown["dilution"],
        "Autonomía (años)": shown["runway_years"],
        "Capitalización (M)": shown["market_cap"] / 1e6,
        "P / ventas": shown["price_to_sales"],
        "Volumen diario (M)": shown["dollar_volume"] / 1e6,
        "Recuento dudoso": shown["cap_suspect"],
    })
    percent = st.column_config.NumberColumn(format="percent")
    st.dataframe(
        # Signs that are facts: growth, its acceleration and margins up favour the holder;
        # dilution up does not.
        color_by_sign(display, {"Crec. ingresos": True, "Aceleración (pp)": True,
                                "Δ margen bruto (pp)": True, "Margen operativo": True,
                                "Margen FCF (año)": True, "Dilución": False}),
        hide_index=True, width="stretch", height=560,
        column_config={
            "Ingresos trim. (M)": st.column_config.NumberColumn(format="%.1f"),
            "Crec. ingresos": percent, "Margen bruto": percent, "Margen operativo": percent,
            "Margen FCF (año)": percent, "SBC / ingresos (año)": percent, "Dilución": percent,
            "Aceleración (pp)": st.column_config.NumberColumn(
                format="%.1f", help="Crecimiento de este trimestre menos el del trimestre "
                                    "presentado anterior, en puntos. Positivo = acelera."),
            "Δ margen bruto (pp)": st.column_config.NumberColumn(
                format="%.1f", help="Margen bruto frente al mismo trimestre del año anterior."),
            "Margen op. incremental": st.column_config.NumberColumn(
                format="percent", help="Beneficio operativo adicional / ingresos adicionales "
                                       "frente al año anterior. Por encima del margen actual = "
                                       "apalancamiento operativo. Vacío si los ingresos no "
                                       "crecieron."),
            "Regla del 40": st.column_config.NumberColumn(
                format="percent", help="Crecimiento de ingresos del trimestre + margen FCF del "
                                       "último año. Por encima del 40 %, el crecimiento no se "
                                       "compra quemando caja sin medida."),
            "Autonomía (años)": st.column_config.NumberColumn(
                format="%.1f", help="Caja e inversiones a corto al cierre del trimestre / "
                                    "quema anual de caja libre. Vacío si genera caja (mira el "
                                    "margen FCF) o si falta un dato."),
            "Capitalización (M)": st.column_config.NumberColumn(format="%.0f"),
            "P / ventas": st.column_config.NumberColumn(format="%.1f×"),
            "Volumen diario (M)": st.column_config.NumberColumn(format="%.1f"),
            "Recuento dudoso": st.column_config.CheckboxColumn(
                help="El volumen de un día supera la capitalización: la empresa etiquetó sus "
                     "acciones en otra escala (miles o millones). Capitalización vacía."),
        },
    )
    st.caption(
        "Ordenada por la regla del 40; pulsa una cabecera para reordenar. **El orden es para "
        "leer, no una señal validada**: ni el crecimiento ni la regla del 40 se han probado "
        "como predictores de la rentabilidad (§9.7). Una casilla vacía es un dato que no "
        "existe o un cociente sobre una base negativa, nunca una estimación. Un crecimiento "
        "sobre una base minúscula (una empresa que empieza a facturar) sube arriba del todo: "
        "el filtro de ingresos mínimos lo aparta.")

    st.subheader("Pasar una candidata a estudio")
    options = [t for t in shown["ticker"].dropna() if t not in researched]
    if options:
        pick = st.selectbox("Candidata", options, index=None, placeholder="Elige un ticker",
                            key="g_pick")
        if pick:
            cik = str(shown.loc[shown["ticker"] == pick, "cik"].iloc[0])
            st.markdown("Añádela a `universe.watchlist` en `config/settings.local.yaml` — "
                        "ticker y CIK, **nada más** (§5.1), y el CIK **entre comillas** "
                        "(§9.13):")
            st.code(f'- ticker: {pick}\n  cik: "{cik}"', language="yaml")

    with st.expander("Límites del cribado de crecimiento"):
        st.markdown("""
- **Cada empresa en su último trimestre presentado**, de los tres más recientes, frente al
  mismo trimestre del año anterior. El cuarto trimestre natural casi no existe como cifra
  de 3 meses en la SEC (va dentro del informe anual, §9.12), y una empresa con otro cierre
  fiscal pierde otro trimestre distinto: la columna «Trimestre» dice cuál se usó.
- **Flujo de caja del último año, no del trimestre.** Casi nadie presenta el flujo de caja
  de 3 meses: la mayoría da acumulados del ejercicio (§9.14). Por eso el margen FCF, el SBC y
  la regla del 40 mezclan periodos: crecimiento del trimestre, caja del año.
- **Sin fecha de presentación.** Como el cribado anual: sirve para hoy, **nunca** para un
  backtest (§9.4).
- **Precios de yfinance** (una fuente frágil, §4.3): solo el último cierre, el volumen y los
  splits. Sin precio, la capitalización queda vacía. Varias clases de acción: se usa el
  precio de la principal y, si no presenta el recuento básico total, el diluido.
- **Un tag de ingresos por empresa.** Algunas presentan el total con un concepto y una partida
  con otro, o cambian de concepto entre periodos. Se usa el mayor donde coinciden (un total
  nunca es menor que una de sus partes) y el mismo en todos los periodos.
- **Emisores extranjeros que presentan en IFRS** (20-F, como NU) no aparecen.
- **Recuento en otra escala.** Algún emisor etiqueta sus acciones en miles (Iovance: 450.189
  por ~450 M). Si el volumen de un día supera la capitalización, la capitalización queda
  vacía en vez de corregirse a ojo.
- **Dilución** con acciones **básicas** (§9.15). Un salto de más del 50 % es una salida a
  bolsa, un SPAC o una fusión y queda vacío.
- Las microcaps ilíquidas se cuelan si apagas los filtros de capitalización y volumen: su
  precio puede moverse un 10 % con una sola orden tuya.
""")
