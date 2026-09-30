# %% [markdown]
# # Guía interactiva del panel
#
# Un recorrido por **todo lo que usa el panel**: cada concepto de `THEORY.md` y cada decisión de
# `RESEARCH.md`, con los datos reales de la base, en tablas de pandas y gráficas de matplotlib.
#
# **Cómo usarla.** Ábrela en VS Code y ejecuta celda a celda (`Shift+Enter` sobre cada `# %%`).
# Cada idea tiene **una función sencilla** con parámetros: cambia el ticker, las fechas o el
# umbral en la llamada de debajo y vuelve a ejecutar. No hace falta tocar nada más.
#
# **De dónde salen los datos.** Todo sale de la base local y de las cachés, sin red, a través
# del laboratorio (`lab/`), que es el mismo código de `run_lab.py`. Las reglas de la casa se
# aplican solas:
# - nada se ve antes de publicarse;
# - los precios se reconstruyen crudos y la rentabilidad es total (con dividendos);
# - el universo es el que tenía el S&P 500 cada día.
#
# Requisitos (una vez): `pip install -e ".[guia]"` (matplotlib e ipykernel).
#
# Mapa:
# 1. El dato y el tiempo (publicación, revisiones, crudo frente a ajustado, medias, supervivencia)
# 2. ¿Hay apetito por riesgo? — la macro
# 3. ¿Está sano el mercado? — el semáforo y la estructura
# 4. Sectores, índices, temas y tus acciones frente a ellos
# 5. ¿Sigue viva la tesis? — la empresa
# 6. ¿Toca ejecutar? — tu cartera y su riesgo
# 7. Compras de directivos
# 8. Validación estadística: cómo se prueba una regla sin engañarse
# 9. Qué datos hay
# 10. Glosario

# %%
import dataclasses
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db.database import open_connection, read_observations, read_table  # noqa: E402
from lab import Lab, evaluate, signals, summary, test  # noqa: E402
from lab.data import known_as_of  # noqa: E402
from transform import breadth as br  # noqa: E402
from transform import entry_context as ec  # noqa: E402
from transform import fundamentals as fun  # noqa: E402
from transform import insider_activity as ia  # noqa: E402
from transform import ownership as own  # noqa: E402
from transform import portfolio as port  # noqa: E402
from transform import price_action as pa  # noqa: E402
from transform import quarterly as qt  # noqa: E402
from transform import risk  # noqa: E402
from transform import themes as th  # noqa: E402
from transform import valuation as val  # noqa: E402
from transform import value_accrual as va  # noqa: E402

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
plt.rcParams.update({"figure.figsize": (11, 4), "axes.grid": True, "grid.alpha": 0.3})

lab = Lab()                       # los datos, cacheados en .cache/lab la primera vez
S = lab.settings
HOY = pd.Timestamp.today().normalize()
DESDE = "2019-01-01"              # cambia aquí la fecha de inicio por defecto de las gráficas

MIS_ACCIONES = lab.held()         # lo que dice tu último extracto de IBKR (nunca escrito a mano)
EN_ESTUDIO = [str(c["ticker"]) for c in S.watchlist_companies]
CON_TESIS = [str(c["ticker"]) for c in S.tracked_companies]
SECTORES = S.raw["panel"]["level2"]["sector_breadth"]["sectors"]
print("En cartera:", MIS_ACCIONES)
print("En estudio:", EN_ESTUDIO, "· con tesis:", CON_TESIS)

# %% [markdown]
# ### Herramientas comunes
# Funciones pequeñas que usan todas las secciones. Leerlas ayuda a entender de dónde sale cada
# número.

# %%
_SEC = None


def obs(source=None, series_ids=None):
    """Observaciones en formato largo ``[source, series_id, ts, ts_release, value]``."""
    conn = open_connection(S.db_path)
    try:
        return read_observations(conn, source=source, series_ids=series_ids)
    finally:
        conn.close()


def tabla(nombre):
    conn = open_connection(S.db_path)
    try:
        return read_table(conn, nombre)
    finally:
        conn.close()


def sec():
    """Los fundamentales auditados (SEC XBRL) de las empresas en estudio y en cartera."""
    global _SEC
    if _SEC is None:
        _SEC = obs(source="sec")
    return _SEC


def cik(ticker):
    """El CIK (la clave de la SEC; el ticker no es clave, §9.3)."""
    for c in S.researched_companies:
        if str(c.get("ticker")) == ticker and c.get("cik"):
            return str(c["cik"]).zfill(10)
    return lab.cik_of().get(ticker)


def precio(ticker, total=True):
    """Serie diaria: rentabilidad total (con dividendos, §9.1) o precio ajustado por splits."""
    frame = lab.total_return([ticker]) if total else lab.prices([ticker])
    return frame[ticker].dropna()


def base100(tickers, desde=DESDE, total=True):
    """Varias series llevadas a 100 en ``desde``: se comparan trayectorias, no niveles."""
    out = {}
    for t in tickers:
        s = precio(t, total)
        s = s[s.index >= pd.Timestamp(desde)]
        if len(s):
            out[t] = s / s.iloc[0] * 100
    return pd.DataFrame(out)


def rentabilidad(serie, dias):
    """De la sesión más cercana a hoy − ``dias`` hasta la última; None si no llega."""
    serie = serie.dropna()
    if serie.empty:
        return None
    antes = serie[serie.index <= serie.index[-1] - pd.Timedelta(days=dias)]
    return None if antes.empty else float(serie.iloc[-1] / antes.iloc[-1] - 1)


def caida_maxima(serie):
    """La peor caída desde un máximo previo (drawdown), en fracción negativa."""
    serie = serie.dropna()
    return float((serie / serie.cummax() - 1).min()) if len(serie) else None


# %% [markdown]
# ---
# ## 1. El dato y el tiempo
#
# ### 1.1 Fecha de referencia y fecha de publicación
# Todo dato tiene **dos fechas**: el periodo que describe (`ts`, *referencia*) y el día en que se
# supo (`ts_release`, *publicación*). El IPC de agosto se publica a mediados de septiembre. Si
# reconstruyes el pasado con la fecha de referencia, usas información que nadie tenía ese día:
# es el **sesgo de anticipación**, y siempre favorece a la estrategia simulada (§9.4).
#
# La tabla mide el **rezago** de cada serie: cuántos días pasan entre el final del periodo y su
# publicación.

# %%
def rezago_publicacion(series=("CPIAUCSL", "PCEPILFE", "PAYEMS", "NFCI", "DTWEXBGS", "DGS10",
                               "VIXCLS", "DCOILWTICO")):
    rows = []
    for sid in series:
        o = obs(series_ids=[sid])
        o = o[o["ts_release"].astype(str).str.len() > 0]
        lag = (pd.to_datetime(o["ts_release"].str[:10]) - pd.to_datetime(o["ts"].str[:10])).dt.days
        rows.append({"serie": sid, "observaciones": len(o), "rezago mediano (días)": lag.median(),
                     "mínimo": lag.min(), "máximo": lag.max(),
                     "desde": o["ts"].min()[:10] if len(o) else None})
    t = pd.DataFrame(rows)
    t.plot.barh(x="serie", y="rezago mediano (días)", legend=False,
                title="Días entre el final del periodo y su publicación (mediana)")
    plt.show()
    return t


rezago_publicacion()

# %% [markdown]
# ### 1.2 Lo que se sabía cada día
# Una serie *point-in-time* no dice «cuánto valió el IPC de enero», sino «qué IPC se conocía el
# 15 de febrero». La gráfica pone las dos lecturas juntas: la línea por fecha de publicación
# va **siempre por detrás**. El semáforo, las alertas y el laboratorio usan solo la segunda.

# %%
def publicado_vs_referencia(serie="CPIAUCSL", desde="2021-01-01"):
    o = obs(series_ids=[serie])
    por_referencia = pd.Series(o["value"].to_numpy(),
                               index=pd.to_datetime(o["ts"].str[:10])).sort_index()
    por_referencia = por_referencia[~por_referencia.index.duplicated(keep="first")]
    calendario = pd.bdate_range(desde, HOY)
    conocido = known_as_of(o, calendario)
    ax = por_referencia[por_referencia.index >= desde].plot(
        label="por fecha de referencia (lo que ves hoy)", drawstyle="steps-post")
    conocido.plot(ax=ax, label="lo que se sabía ese día (point-in-time)", alpha=0.8)
    ax.set_title(f"{serie}: el mismo dato, dos fechas")
    ax.legend()
    plt.show()


publicado_vs_referencia("CPIAUCSL")

# %% [markdown]
# ### 1.3 Revisiones y reexpresiones
# Una empresa puede volver a presentar una cifra ya publicada (un 10-K/A, o la del año anterior
# que viene como comparativo en el informe siguiente). La base **guarda todas las versiones**
# (`ts_release` está en la clave, §6), y se usa la vigente en cada fecha. La tabla enseña, para
# una empresa, los periodos que tienen más de una versión.

# %%
def versiones(ticker, metrica="revenue", periodo="q"):
    c = cik(ticker)
    if not c:
        print(f"{ticker}: sin CIK")
        return None
    s = sec()[sec()["series_id"] == f"{c}:{metrica}:{periodo}"]
    if s.empty:
        print(f"{ticker}: sin {metrica} en la SEC (¿emisor IFRS o sin ficha/estudio?)")
        return None
    g = s.sort_values("ts_release").groupby("ts").agg(
        versiones=("value", "size"), primera=("value", "first"), ultima=("value", "last"),
        publicada_1=("ts_release", "min"), publicada_ultima=("ts_release", "max"))
    g["cambio"] = g["ultima"] / g["primera"] - 1
    varias = g[g["versiones"] > 1]
    cambiadas = varias[varias["cambio"].abs() > 1e-9]
    print(f"{ticker} · {metrica}: {len(g)} periodos, {len(varias)} con más de una versión "
          f"(casi siempre el comparativo del informe siguiente, con la misma cifra), "
          f"{len(cambiadas)} con la cifra cambiada.")
    return (cambiadas if len(cambiadas) else varias).sort_values(
        "cambio", key=lambda x: x.abs(), ascending=False).head(12)


versiones(EN_ESTUDIO[0] if EN_ESTUDIO else "MSFT")

# %% [markdown]
# ### 1.4 Precio crudo, ajustado y rentabilidad total (§9.1)
# - **Crudo:** lo que marcó la pantalla ese día. Un *split* 10:1 lo divide entre 10 de golpe,
#   sin que nadie pierda nada.
# - **Ajustado por splits:** el pasado dividido por los splits posteriores, para que la serie no
#   salte. ⚠️ Cambia cada vez que hay un split nuevo, así que «el ajustado de un día» depende de
#   cuándo lo mires. Por eso el panel **guarda el crudo** y ajusta a una fecha de corte.
# - **Rentabilidad total:** suma además cada dividendo en su fecha. Es lo que ganó quien tenía
#   la acción.
#
# Prueba con una acción que haya hecho un split (NVDA, 10:1 en junio de 2024) o una que pague
# muchos dividendos (XLU, T, KO).

# %%
def crudo_vs_ajustado(ticker="NVDA", desde="2023-06-01"):
    raw = obs(series_ids=[f"{ticker}:close_raw"])
    crudo = pd.Series(raw["value"].to_numpy(), index=pd.to_datetime(raw["ts"].str[:10])).sort_index()
    ajustado = precio(ticker, total=False)
    total = precio(ticker, total=True)
    crudo, ajustado, total = (s[s.index >= desde] for s in (crudo, ajustado, total))
    fig, ax = plt.subplots()
    crudo.plot(ax=ax, label="crudo (lo que marcó la pantalla)")
    ajustado.plot(ax=ax, label="ajustado por splits")
    (total / total.iloc[0] * ajustado.iloc[0]).plot(ax=ax, label="rentabilidad total (con dividendos)",
                                                    linestyle="--")
    ax.set_title(f"{ticker}: tres formas de contar el mismo precio")
    ax.legend()
    plt.show()
    splits = lab.actions()
    splits = splits[(splits["ticker"] == ticker) & (splits["kind"] == "split")]
    return pd.DataFrame({
        "precio (sin dividendos)": [rentabilidad(ajustado, (ajustado.index[-1] - ajustado.index[0]).days)],
        "total (con dividendos)": [rentabilidad(total, (total.index[-1] - total.index[0]).days)],
    }, index=[f"{ticker} desde {desde}"]), splits[["ex_date", "ratio", "source"]].tail(5)


crudo_vs_ajustado("NVDA")

# %% [markdown]
# ### 1.5 Media aritmética frente a media geométrica (§9.10)
# +100 % y después −50 % te deja donde empezaste: 0 % real. La media aritmética de esas dos
# cifras dice **+25 %**. Cuanto más volátil es la serie, más exagera. Para «crecimiento medio»
# el panel usa siempre la **tasa compuesta (CAGR)**: `(final / inicial)^(1/años) − 1`.

# %%
def aritmetica_vs_geometrica(tickers=("SPY", "QQQ", "IWM"), anos=10):
    rows = []
    for t in tickers:
        s = precio(t)
        s = s[s.index >= HOY - pd.DateOffset(years=anos)]
        anuales = s.resample("YE").last().pct_change().dropna()
        cagr = (s.iloc[-1] / s.iloc[0]) ** (365.25 / (s.index[-1] - s.index[0]).days) - 1
        rows.append({"activo": t, "media aritmética anual": anuales.mean(), "CAGR (real)": cagr,
                     "diferencia": anuales.mean() - cagr, "volatilidad anual": anuales.std()})
    juguete = pd.Series([1.0, -0.5])
    print("Ejemplo: +100 % y −50 % → media aritmética",
          f"{juguete.mean():+.0%}, compuesta {(np.prod(1 + juguete)) ** 0.5 - 1:+.0%}")
    return pd.DataFrame(rows).set_index("activo")


aritmetica_vs_geometrica()

# %% [markdown]
# ### 1.6 Sesgo de supervivencia (§9.5)
# Si calculas «cuántas empresas del S&P 500 suben» con la lista de **hoy**, faltan las que
# quebraron o salieron: el pasado parece más sano de lo que fue. El panel usa la lista de cada
# día, pero yfinance ya no tiene precios de muchas empresas que salieron. La gráfica enseña qué
# parte de los miembros de cada fecha tiene precio. Por debajo del 85 %, la amplitud por
# empresas no se usa.

# %%
def cobertura_supervivencia(paso="QE", tolerancia_dias=10):
    mask = lab.members()
    tr = lab.total_return()
    ultimo = tr.apply(lambda s: s.last_valid_index())      # último cierre de cada ticker
    primero = tr.apply(lambda s: s.first_valid_index())
    fechas = [d for d in mask.resample(paso).last().index if d <= HOY]
    rows = []
    for d in fechas:
        on = mask.index[mask.index <= d]
        if not len(on):
            continue
        miembros = list(mask.columns[mask.loc[on[-1]].to_numpy()])
        # Con precio = cotizaba ese día: su historia empieza antes y no terminó antes (con unos
        # días de margen: algunos precios se actualizan una vez por semana).
        con_precio = [t for t in miembros if t in tr.columns and pd.notna(primero.get(t))
                      and primero[t] <= d
                      and ultimo[t] >= min(d, HOY) - pd.Timedelta(days=tolerancia_dias)]
        rows.append({"fecha": d, "miembros": len(miembros), "con precio": len(con_precio)})
    t = pd.DataFrame(rows).set_index("fecha")
    t["cobertura"] = t["con precio"] / t["miembros"]
    ax = t["cobertura"].plot(title="Miembros del S&P 500 con precio en la base")
    ax.axhline(0.85, color="grey", linestyle="--", label="umbral del panel (85 %)")
    ax.legend()
    plt.show()
    return t.tail(8)


cobertura_supervivencia()

# %% [markdown]
# ---
# ## 2. ¿Hay apetito por riesgo? — la macro (nivel 1)
#
# Cada serie, **tal como se publicó cada día**. La tabla explica qué es cada una y en qué
# sentido suele favorecer a las acciones. Lo ambiguo (el tipo a 10 años, la inflación, el
# petróleo) no lleva sentido: depende del momento.

# %%
EXPLICA = {
    "BAA10Y": ("Prima de crédito Baa − 10 años", "lo que cobran de más las empresas medianas "
               "por prestarles; si sube, el mercado teme impagos", "abajo"),
    "BAMLH0A0HYM2": ("Spread high yield", "lo mismo para empresas de peor calidad; solo desde 2023",
                     "abajo"),
    "T10Y2Y": ("Curva 10a − 2a", "negativa = invertida: históricamente precede recesiones, con "
               "mucho retraso", "arriba"),
    "NFCI": ("Condiciones financieras (Chicago Fed)", "negativo = crédito fácil", "abajo"),
    "DTWEXBGS": ("Dólar amplio", "un dólar fuerte endurece las condiciones en todo el mundo",
                 "abajo"),
    "VIXCLS": ("VIX", "volatilidad esperada a 30 días; el «índice del miedo»", "abajo"),
    "VXVCLS": ("VIX3M", "lo mismo a 3 meses; si el VIX lo supera, estrés presente", "abajo"),
    "DGS10": ("Tesoro 10 años", "coste del dinero a largo; ambiguo", None),
    "DFII10": ("Tipo real 10 años (TIPS)", "lo que paga el bono protegido de inflación; compite "
               "con el oro y con las acciones de crecimiento", None),
    "T10YIE": ("Inflación implícita 10 años", "la inflación que descuenta el mercado", None),
    "DFF": ("Fed funds efectiva", "el tipo de la Fed", None),
    "CPIAUCSL": ("IPC", "inflación general, nivel del índice", None),
    "PCEPILFE": ("PCE subyacente", "la inflación que mira la Fed", None),
    "PAYEMS": ("Nóminas no agrícolas", "empleo; se revisa mucho", None),
    "DCOILWTICO": ("Petróleo WTI", "presión de inflación; ambiguo", None),
    "WALCL": ("Balance de la Fed", "parte de la liquidez neta (balance − Tesoro − repos)", None),
}


def glosario_macro():
    return pd.DataFrame([{"serie": k, "nombre": v[0], "qué es": v[1],
                          "favorece a las acciones si va": v[2] or "ambiguo"}
                         for k, v in EXPLICA.items()]).set_index("serie")


glosario_macro()

# %%
def macro(series=("BAA10Y", "T10Y2Y", "NFCI", "DTWEXBGS", "VIXCLS", "DFII10", "DCOILWTICO"),
          desde=DESDE):
    fig, axes = plt.subplots(len(series), 1, figsize=(11, 2.2 * len(series)), sharex=True)
    for ax, sid in zip(np.atleast_1d(axes), series):
        s = lab.fred(sid)
        s = s[s.index >= desde]
        s.plot(ax=ax)
        nombre = EXPLICA.get(sid, (sid,))[0]
        ax.set_title(f"{nombre} ({sid})", fontsize=10, loc="left")
    plt.tight_layout()
    plt.show()


def tabla_macro(series=tuple(EXPLICA)):
    rows = []
    for sid in series:
        s = lab.fred(sid).dropna()
        if s.empty:
            continue
        hace = s[s.index <= s.index[-1] - pd.Timedelta(days=30)]
        historia = s[s.index >= s.index[-1] - pd.DateOffset(years=10)]
        rows.append({"serie": sid, "nombre": EXPLICA[sid][0], "hoy": s.iloc[-1],
                     "cambio 30 d": s.iloc[-1] - hace.iloc[-1] if len(hace) else None,
                     "percentil 10 años": (historia <= s.iloc[-1]).mean()})
    return pd.DataFrame(rows).set_index("serie")


macro()
tabla_macro()

# %% [markdown]
# ### 2.1 Una serie macro frente a las acciones
# ¿Se movían juntas? La correlación de los **cambios mensuales** dice si la acción tiende a subir
# o bajar cuando la serie sube. Es relación pasada, no causa ni predicción. Prueba con el dólar
# y NU (que gana en reales), con el petróleo y una aerolínea, o con el tipo real y el oro (GLD).

# %%
def macro_vs_acciones(serie="DTWEXBGS", tickers=("SPY", "QQQ"), desde=DESDE):
    m = lab.fred(serie)
    m = m[m.index >= desde]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    m.plot(ax=a1, title=f"{EXPLICA.get(serie, (serie,))[0]}")
    base100(tickers, desde).plot(ax=a2, title="Acciones, base 100")
    plt.tight_layout()
    plt.show()
    mensual = m.resample("ME").last()
    cambios = mensual.pct_change() if mensual.min() > 0 else mensual.diff()
    rows = []
    for t in tickers:
        r = precio(t).resample("ME").last().pct_change()
        both = pd.concat([cambios, r], axis=1).dropna()
        rows.append({"acción": t, f"correlación mensual con {serie}": both.corr().iloc[0, 1],
                     "meses": len(both)})
    return pd.DataFrame(rows).set_index("acción")


macro_vs_acciones("DTWEXBGS", tuple(["SPY"] + MIS_ACCIONES[:3]))

# %% [markdown]
# ---
# ## 3. ¿Está sano el mercado? — el semáforo y la estructura (nivel 2)
#
# ### 3.1 El semáforo
# Ocho componentes votan +1 (a favor), 0 o −1 (en contra), cada uno con una regla escrita antes
# de mirar ningún resultado. **Mayoría simple**: rojo si más de la mitad de los que votan dice
# risk-off. En la fase 4 se midió que **no predice la rentabilidad**, pero el rojo sí precedió
# caídas más hondas el mes siguiente: es un freno de conducta, no un indicador de compra.

# %%
COLORES = {"risk_on": "#3a9d5d", "neutral": "#e0a526", "risk_off": "#d64545"}


def semaforo(desde="2015-01-01"):
    f = lab.regime().frame
    f = f[f.index >= desde]
    spy = precio("SPY")
    spy = spy[spy.index >= desde]
    fig, ax = plt.subplots()
    ax.plot(spy.index, spy.to_numpy(), color="black", linewidth=1)
    ax.set_yscale("log")
    for estado, color in COLORES.items():
        dias = f.index[f["verdict"] == estado]
        ax.scatter(dias, [spy.min() * 0.97] * len(dias), color=color, s=4, label=estado)
    ax.set_title("SPY (rentabilidad total, escala log) y el veredicto del semáforo debajo")
    ax.legend(loc="upper left")
    plt.show()
    etiquetas = {c["key"]: c["label"] for c in S.raw["panel"]["level2"]["regime"]["components"]}
    hoy = f.iloc[-1]
    votos = pd.DataFrame({"componente": [etiquetas[k] for k in etiquetas],
                          "voto hoy": [hoy.get(k) for k in etiquetas]})
    print("Veredicto hoy:", hoy["verdict"], "· días por veredicto desde", desde)
    print(f["verdict"].value_counts().to_string())
    return votos


semaforo()

# %%
def votos_historicos(desde="2015-01-01"):
    """Qué parte del tiempo votó cada componente en contra, neutro o a favor."""
    f = lab.regime().frame
    f = f[f.index >= desde]
    keys = [c["key"] for c in S.raw["panel"]["level2"]["regime"]["components"]]
    t = pd.DataFrame({k: f[k].value_counts(normalize=True) for k in keys}).T.fillna(0)
    t = t.rename(columns={-1: "en contra", 0: "neutro", 1: "a favor"})
    t.plot.barh(stacked=True, title="Reparto de votos de cada componente", color=["#d64545", "#bbb", "#3a9d5d"])
    plt.show()
    return t


votos_historicos()

# %% [markdown]
# ### 3.2 Estructura: amplitud, estrechez, rotación y volatilidad
# - **Amplitud sectorial:** qué parte de los 9 sectores cotiza sobre su media de 200 sesiones.
#   ¿Sube el índice porque suben todos o porque suben cinco?
# - **RSP / SPY:** el S&P 500 con todas las empresas pesando lo mismo, frente al ponderado por
#   tamaño. Si cae, el rally es estrecho (lo llevan las grandes).
# - **Rotación defensiva:** servicios públicos y consumo básico frente a tecnología y consumo
#   discrecional. Si sube, el mercado descuenta una desaceleración.
# - **VIX / VIX3M:** por encima de 1 = miedo a corto mayor que a largo: estrés presente.

# %%
def estructura_mercado(desde=DESDE):
    raw = br.wide_closes(lab._raw_closes())
    splits = br.splits_by_ticker(lab.actions())
    L2 = S.raw["panel"]["level2"]
    amplitud = br.sector_breadth(raw, splits, L2["sector_breadth"]["sectors"]).set_index("date")
    estrechez = br.equal_weight_ratio(raw, splits).set_index("date")
    rotacion = br.defensive_rotation(raw, splits).set_index("date")
    vix = lab.fred("VIXCLS") / lab.fred("VXVCLS")
    series = {"Amplitud sectorial (fracción sobre su media 200)": amplitud["share"],
              "RSP / SPY (estrechez)": estrechez["ratio"],
              "Rotación defensiva (sube = defensa)": rotacion["rotation"],
              "VIX / VIX3M (>1 = estrés)": vix}
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for ax, (titulo, s) in zip(axes, series.items()):
        s.index = pd.to_datetime(s.index)
        s[s.index >= desde].plot(ax=ax)
        ax.set_title(titulo, fontsize=10, loc="left")
    axes[0].axhline(0.5, color="grey", linestyle="--")
    axes[3].axhline(1.0, color="grey", linestyle="--")
    plt.tight_layout()
    plt.show()


estructura_mercado()

# %% [markdown]
# ---
# ## 4. Sectores, índices, temas y tus acciones
#
# ### 4.1 Los sectores
# Rentabilidad total de los nueve SPDR sectoriales y su diferencia con SPY en varias ventanas.
# Sirve para ver **dónde** está el mercado. Cambiar de sector como *timing* no tiene evidencia
# en las pruebas del panel.

# %%
NOMBRES_SECTOR = {"XLK": "Tecnología", "XLU": "Serv. públicos", "XLP": "Consumo básico",
                  "XLF": "Financiero", "XLE": "Energía", "XLV": "Salud",
                  "XLY": "Consumo discrecional", "XLI": "Industria", "XLB": "Materiales"}


def sectores(ventanas=(21, 63, 126, 252), desde=DESDE):
    spy = precio("SPY")
    rows = []
    for t in SECTORES:
        s = precio(t)
        row = {"sector": f"{t} · {NOMBRES_SECTOR.get(t, t)}"}
        for v in ventanas:
            a, b = rentabilidad(s, int(v * 365 / 252)), rentabilidad(spy, int(v * 365 / 252))
            row[f"{v} ses."] = a
            row[f"{v} ses. vs SPY"] = None if a is None or b is None else (1 + a) / (1 + b) - 1
        rows.append(row)
    base100(SECTORES + ["SPY"], desde).plot(title="Sectores, base 100 (con dividendos)",
                                            figsize=(11, 5))
    plt.show()
    return pd.DataFrame(rows).set_index("sector").sort_values(f"{ventanas[-1]} ses. vs SPY",
                                                               ascending=False)


sectores()

# %% [markdown]
# ### 4.2 Tus acciones frente a los índices
# - **Rentabilidad frente a SPY:** `(1 + acción) / (1 + SPY) − 1`, compuesta, nunca una resta.
# - **Beta:** cuánto se mueve la acción por cada 1 % de SPY (2 = el doble).
# - **Volatilidad anual:** desviación de los rendimientos diarios × √252.
# - **Caída máxima:** la peor caída desde un máximo previo.
#
# Cambia `tickers` y `indices` (QQQ tecnología, IWM pequeñas, RSP equiponderado, o un sector).

# %%
def comparar(tickers=None, indices=("SPY", "QQQ", "IWM"), desde=DESDE, dias=365):
    tickers = list(tickers or MIS_ACCIONES)
    base100(list(indices) + tickers, desde).plot(title="Base 100, con dividendos", figsize=(11, 5))
    plt.show()
    spy = precio("SPY")
    rows = []
    for t in list(indices) + tickers:
        s = precio(t)
        b = pa.behaviour(s, spy, HOY, days=dias)
        rows.append({"ticker": t, f"rent. {dias} d": b.total_return,
                     "frente a SPY": b.excess, "beta": b.beta, "volatilidad": b.volatility,
                     "caída máx. del periodo": b.max_drawdown, "desde el máximo": b.drawdown_now})
    return pd.DataFrame(rows).set_index("ticker")


comparar()

# %%
def contra_indice(ticker=None, indice="SPY", ventana=63, desde=DESDE):
    """Fuerza relativa móvil (la acción entre el índice, base 100) y beta móvil."""
    ticker = ticker or (MIS_ACCIONES[0] if MIS_ACCIONES else "AAPL")
    a, b = precio(ticker), precio(indice)
    both = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    both = both[both.index >= desde]
    relativa = both["a"] / both["b"]
    r = both.pct_change()
    beta = r["a"].rolling(ventana).cov(r["b"]) / r["b"].rolling(ventana).var()
    fig, (x1, x2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    (relativa / relativa.iloc[0] * 100).plot(ax=x1, title=f"{ticker} / {indice} (sube = mejor que el índice)")
    beta.plot(ax=x2, title=f"Beta móvil de {ventana} sesiones")
    x2.axhline(1, color="grey", linestyle="--")
    plt.tight_layout()
    plt.show()


contra_indice()

# %% [markdown]
# ### 4.3 ¿La empresa o su tema?
# `(1 + acción)/(1 + SPY) = (1 + acción)/(1 + tema) × (1 + tema)/(1 + SPY)`: una parte la explica
# el tema entero (su ETF), la otra, la empresa. Los temas están en `universe.themes`
# (`settings.local.yaml`).

# %%
def empresa_vs_tema(ticker=None):
    ticker = ticker or (MIS_ACCIONES[0] if MIS_ACCIONES else None)
    temas = th.themes_of(ticker, S.themes) if ticker else []
    if not temas:
        print(f"{ticker}: sin tema escrito en universe.themes")
        return None
    rows = []
    for tema in temas:
        for p in th.split(precio(ticker), precio(tema.etf), precio("SPY"), HOY):
            rows.append({"tema": f"{tema.name} ({tema.etf})", "horizonte": p.horizon,
                         "acción vs SPY": p.company_vs_spy, "tema vs SPY": p.theme_vs_spy,
                         "acción vs su tema": p.company_vs_theme})
    return pd.DataFrame(rows)


empresa_vs_tema()

# %%
def correlaciones(tickers=None, dias=365):
    """Correlación de rendimientos diarios: cerca de 1 = la misma apuesta."""
    tickers = list(tickers or (MIS_ACCIONES + ["SPY", "GLD", "TLT"]))
    r = lab.total_return(tickers)
    r = r[r.index >= HOY - pd.Timedelta(days=dias)].pct_change(fill_method=None)
    c = r.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(c.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(c)), c.columns, rotation=90)
    ax.set_yticks(range(len(c)), c.columns)
    fig.colorbar(im)
    ax.set_title(f"Correlación diaria, {dias} días")
    plt.show()
    return c


correlaciones()

# %% [markdown]
# ### 4.4 Metales y petróleo
# El oro no paga nada: compite con el **tipo real**. La plata y las mineras son más cíclicas.
# La tabla enseña qué hizo cada uno en las caídas del S&P 500 del 10 % o más.

# %%
def metales_en_caidas(activos=("GLD", "SLV", "GDX", "SIL", "TLT")):
    from transform import metals
    spy = precio("SPY")
    episodios = metals.drawdown_episodes(spy[spy.index >= "2005-01-01"])
    tabla_ = metals.cushion(episodios, {a: precio(a) for a in activos})
    return tabla_.rename(columns={"peak": "máximo", "trough": "mínimo", "depth": "SPY"})


metales_en_caidas()

# %% [markdown]
# ### 4.5 Fuerza relativa dentro del universo invertible
# El cribado de crecimiento guarda la rentabilidad de precio a 6 y 12 meses de ~3.900 empresas.
# El **percentil** dice qué parte del universo invertible (capitalización ≥ 300 M y volumen ≥ 1 M
# USD/día) lo hizo peor. **Contexto, no señal:** el momentum cambió de signo entre periodos en
# el S&P 500.

# %%
def fuerza_relativa(tickers=None, horizonte="12m"):
    from ingest.growth_screen import PRICE_SOURCE, SOURCE
    from transform import growth_screen as gs
    registro = tabla("companies")
    nombres = {str(r["cik"]): (str(r["ticker"]), r.get("name")) for r in registro.to_dict("records")}
    defaults = S.raw["screen"]["growth"]["defaults"]
    t = gs.growth_table(obs(source=SOURCE), obs(source=PRICE_SOURCE), nombres,
                        dict(zip(registro["cik"], registro["sic"])), reference=defaults)
    if t.empty:
        print("Sin cribado de crecimiento: `python run_ingest.py --only growth_screen --force`")
        return None
    universo = t[t["investable"]][f"return_{horizonte}"].dropna()
    universo.clip(-1, 3).plot.hist(bins=80, color="#bbb",
                                   title=f"Rentabilidad {horizonte} del universo invertible")
    mias = t[t["ticker"].isin(list(tickers or (MIS_ACCIONES + EN_ESTUDIO)))]
    for _, r in mias.iterrows():
        if pd.notna(r[f"return_{horizonte}"]):
            plt.axvline(min(r[f"return_{horizonte}"], 3), color="red", alpha=0.6)
            plt.text(min(r[f"return_{horizonte}"], 3), plt.ylim()[1] * 0.9, r["ticker"], rotation=90)
    plt.show()
    return mias.set_index("ticker")[["price_date", "return_6m", "rs_6m", "return_12m", "rs_12m"]]


fuerza_relativa()

# %% [markdown]
# ---
# ## 5. ¿Sigue viva la tesis? — la empresa (nivel 3)
#
# Solo las empresas en estudio, con ficha o en cartera tienen fundamentales en la base (SEC
# XBRL, auditados, fechados por su presentación). Un emisor extranjero en IFRS (NU) no tiene
# trimestres en la SEC.
#
# ### 5.1 Trimestre a trimestre
# - **Crecimiento interanual:** el trimestre frente al mismo del año anterior (la
#   estacionalidad se anula).
# - **Márgenes:** bruto (lo que queda tras el coste de lo vendido), operativo (tras gastos),
#   neto (tras todo) y **FCF** (caja que genera de verdad).
# - **SBC:** sueldos pagados en acciones. Es un coste real que diluye.
# - El Q4 no existe como 10-Q: se deriva del anual (§9.12). Muchos flujos vienen acumulados
#   y también se derivan (§9.14).

# %%
def fundamentales(ticker=None, trimestres=8):
    ticker = ticker or (EN_ESTUDIO[0] if EN_ESTUDIO else "MSFT")
    c = cik(ticker)
    t = qt.quarter_table(sec(), c, HOY, quarters=trimestres) if c else pd.DataFrame()
    if t.empty:
        print(f"{ticker}: sin trimestres en la SEC")
        return None
    t = t.set_index(t.columns[0]) if not isinstance(t.index, pd.DatetimeIndex) else t
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    (t["revenue"] / 1e6).plot.bar(ax=a1, title=f"{ticker}: ingresos por trimestre (M USD)")
    margenes = [m for m in ("gross_margin", "operating_margin", "net_margin", "fcf_margin") if m in t]
    t[margenes].plot(ax=a2, marker="o", title="Márgenes")
    a2.axhline(0, color="grey")
    plt.tight_layout()
    plt.show()
    columnas = [x for x in ("revenue", "revenue_growth", "gross_margin", "operating_margin",
                            "net_margin", "fcf", "fcf_margin", "sbc_share") if x in t]
    return t[columnas]


fundamentales()

# %% [markdown]
# ### 5.1b TTM: los últimos doce meses
# Un trimestre suelto es ruidoso y estacional. El **TTM** suma los cuatro últimos trimestres
# **contiguos** (si falta uno, no hay TTM: tres trimestres no son un año) y avanza trimestre a
# trimestre. Cada punto está fechado cuando se **presentó** el último de sus cuatro trimestres.

# %%
def ttm(ticker=None, metricas=("revenue", "operating_income", "net_income")):
    ticker = ticker or (EN_ESTUDIO[0] if EN_ESTUDIO else "MSFT")
    c = cik(ticker)
    series = {}
    for m in metricas:
        t = fun.ttm_series(sec(), c, m, HOY) if c else pd.DataFrame()
        if len(t):
            series[m] = pd.Series(t["value"].to_numpy(), index=pd.to_datetime(t["ts"].astype(str).str[:10]))
    if not series:
        print(f"{ticker}: sin TTM")
        return None
    d = pd.DataFrame(series) / 1e6
    d.plot(marker="o", title=f"{ticker}: TTM (M USD)")
    plt.axhline(0, color="grey")
    plt.show()
    return d.tail(8)


ttm()

# %% [markdown]
# ### 5.2 ¿Cómo llega eso al accionista? (§2, el concepto rector)
# - **Dilución:** cuánto crece el número de acciones en un año. Si la empresa crece un 30 % y
#   diluye un 30 %, tu parte no crece.
# - **Recompras netas:** recompras − emisión. La cifra bruta engaña si el SBC la anula.
# - **Conversión a caja:** FCF / beneficio. Por debajo de 1, beneficio que no llega a la caja.
# - **ROIC:** beneficio operativo tras impuestos / capital invertido.
# - ⚠️ Con pérdidas, las acciones diluidas son iguales a las básicas (§9.15): para medir la
#   dilución a varios años, las básicas.

# %%
def captura_de_valor(ticker=None):
    ticker = ticker or (EN_ESTUDIO[0] if EN_ESTUDIO else "MSFT")
    c = cik(ticker)
    if not c:
        return None
    lectura = va.assess(sec(), c, HOY)
    tabla_ = pd.Series({k: v for k, v in dataclasses.asdict(lectura).items()
                        if not isinstance(v, (list, dict))}, name=ticker)
    for metrica in ("basic_shares", "diluted_shares"):
        s = sec()[sec()["series_id"] == f"{c}:{metrica}:q"]
        if len(s):
            serie = pd.Series(s["value"].to_numpy(), index=pd.to_datetime(s["ts"].str[:10]))
            serie = serie[~serie.index.duplicated(keep="last")].sort_index() / 1e6
            serie.plot(label=metrica, marker=".")
    plt.title(f"{ticker}: acciones medias por trimestre (millones)")
    plt.legend()
    plt.show()
    return tabla_.to_frame()


captura_de_valor()

# %% [markdown]
# ### 5.3 ¿A qué precio? Valoración y DCF inverso
# - **Capitalización:** precio × acciones. **VE** (valor de empresa): + deuda − caja.
# - **VE / ventas, VE / EBIT, P / FCF, rendimiento FCF** (FCF / capitalización: lo que rendiría
#   si repartiera toda la caja).
# - **Percentil en su historia:** ¿está más cara o más barata que de costumbre?
# - **DCF inverso:** en vez de adivinar el futuro, pregunta qué crecimiento del FCF durante 10
#   años justifica el precio de hoy, con una tasa de descuento dada. Si exige +30 % anual, la
#   tesis tiene que creérselo.

# %%
def valoracion(ticker=None, anos=5, descuentos=(0.08, 0.10, 0.12)):
    ticker = ticker or (EN_ESTUDIO[0] if EN_ESTUDIO else "MSFT")
    c = cik(ticker)
    precios = obs(series_ids=[f"{ticker}:close_raw"])
    if not c or precios.empty:
        return None
    ahora = val.assess(sec(), precios, lab.actions(), c, ticker, HOY)
    fechas = pd.date_range(HOY - pd.DateOffset(years=anos), HOY, freq="ME")
    historia = val.history(sec(), precios, lab.actions(), c, ticker, list(fechas))
    if len(historia):
        h = historia.set_index("date")
        cols = [x for x in ("fcf_yield", "ev_to_sales", "pe") if x in h]
        h[cols].plot(subplots=True, figsize=(11, 6), title=f"{ticker}: múltiplos a cada fecha")
        plt.show()
    dcf = val.reverse_dcf(ahora, list(descuentos), terminal_growth=0.025, years=10)
    inverso = pd.DataFrame({base: [getattr(x, "growth", None) for x in lista]
                            for base, lista in dcf.items()},
                           index=[f"descuento {d:.0%}" for d in descuentos])
    actual = pd.Series({k: v for k, v in dataclasses.asdict(ahora).items()
                        if isinstance(v, (int, float)) or v is None}, name=ticker)
    return actual.to_frame(), inverso.rename(columns={"fcf": "crecimiento implícito del FCF",
                                                      "fcf_after_sbc": "… del FCF tras SBC"})


valoracion()

# %% [markdown]
# ### 5.4 Quién compra y quién vende: directivos, fondos, retail y cortos
# - **Directivos (Form 4):** compras (código P) y ventas; las de plan 10b5-1 se programan meses
#   antes. En las pruebas del panel, las compras **no predijeron nada** (§2.74).
# - **Institucionales (13F):** acciones en manos de fondos, trimestral y con 45 días de retraso.
# - **Interés corto (FINRA):** acciones vendidas en corto; **días para cubrir** = corto / volumen
#   diario.

# %%
def directivos(ticker=None, meses=36):
    ticker = ticker or (EN_ESTUDIO[0] if EN_ESTUDIO else None)
    c = cik(ticker) if ticker else None
    lectura = ia.assess(obs(source="sec_form4"), c, HOY, months=meses) if c else None
    if lectura is None:
        print(f"{ticker}: sin compras ni ventas de directivos registradas")
        return None
    m = lectura.monthly.pivot_table(index="month", columns="label", values="value", aggfunc="sum")
    (m / 1e6).plot.bar(stacked=True, figsize=(12, 4),
                       title=f"{ticker}: compras (+) y ventas (−) de directivos, M USD")
    plt.show()
    return pd.Series({"compras 12 m": lectura.buy_12m, "ventas 12 m": lectura.sell_12m,
                      "de ellas discrecionales": lectura.discretionary_12m,
                      "última presentación": lectura.last_filed}, name=ticker).to_frame()


def cortos(ticker=None):
    ticker = ticker or (EN_ESTUDIO[0] if EN_ESTUDIO else None)
    f = obs(source="finra", series_ids=[f"{ticker}:short_interest", f"{ticker}:days_to_cover"])
    if f.empty:
        print(f"{ticker}: sin interés corto")
        return None
    w = f.pivot_table(index=pd.to_datetime(f["ts"].str[:10]), columns="series_id", values="value")
    w.columns = [x.split(":", 1)[1] for x in w.columns]
    w.plot(subplots=True, figsize=(11, 5), title=f"{ticker}: interés corto (FINRA, quincenal)")
    plt.show()
    return w.tail(6)


def institucionales(ticker=None):
    ticker = ticker or (EN_ESTUDIO[0] if EN_ESTUDIO else None)
    c = cik(ticker) if ticker else None
    basicas = sec()[sec()["series_id"] == f"{c}:basic_shares:q"] if c else pd.DataFrame()
    acciones = (pd.Series(basicas["value"].to_numpy(), index=pd.to_datetime(basicas["ts"].str[:10]))
                .sort_index() if len(basicas) else None)
    lectura = own.institutional(obs(source="sec_13f"), c, HOY,
                                shares_outstanding=acciones) if c else None
    if lectura is None or lectura.quarters.empty:
        print(f"{ticker}: sin 13F")
        return None
    q = lectura.quarters.set_index("quarter")
    (q["shares"] / 1e6).tail(12).plot.bar(
        title=f"{ticker}: acciones en manos de fondos (M), por trimestre")
    plt.show()
    return q.tail(8)


directivos()
cortos()
institucionales()

# %% [markdown]
# ### 5.5 Antes de comprar: dónde está el precio
# Distancia a la media de 200 sesiones y rentabilidad a 1, 3 y 12 meses. Las dos condiciones
# del estudio de §2.65 («cuchillo cayendo», «muy estirada») se enseñan como **descripción**:
# ninguna resultó regla.

# %%
def contexto_de_entrada(tickers=None):
    condiciones = S.raw["knife_study"]["conditions"]
    rows = []
    for t in list(tickers or (MIS_ACCIONES + EN_ESTUDIO)):
        e = ec.assess(precio(t, total=False), HOY, condiciones)
        if e is not None:
            rows.append({"ticker": t, "vs media 200": e.distance_to_sma, "1 mes": e.return_1m,
                         "3 meses": e.return_3m, "12 meses": e.return_12m,
                         "desde máx. 52 sem.": e.drawdown_52w, "cuchillo": e.knife,
                         "muy estirada": e.overextended})
    return pd.DataFrame(rows).drop_duplicates("ticker").set_index("ticker")


contexto_de_entrada()

# %% [markdown]
# ### 5.6 Crecimiento de la empresa frente a crecimiento por acción
# La brecha entre las dos es **lo que se llevaron los accionistas nuevos** (la dilución). Con
# acciones **básicas**, porque las diluidas saltan al pasar de pérdida a beneficio (§9.15).

# %%
def por_accion(ticker=None, horizontes=(3, 5)):
    from transform import per_share
    ticker = ticker or (EN_ESTUDIO[0] if EN_ESTUDIO else "MSFT")
    c = cik(ticker)
    lectura = per_share.assess(sec(), c, HOY, horizons=tuple(horizontes)) if c else None
    if lectura is None or lectura.table.empty:
        print(f"{ticker}: sin cifras suficientes")
        return None
    if not lectura.chart.empty:
        lectura.chart.plot(title=f"{ticker}: ingresos totales frente a ingresos por acción (índice 100)")
        plt.show()
    print(f"Base de acciones: {lectura.basis}; acciones, crecimiento anual: {lectura.shares}")
    return lectura.table


por_accion()

# %% [markdown]
# ### 5.7 Segmentos: dónde está el crecimiento
# Del XBRL de cada 10-Q/10-K (hechos con dimensión). Si la empresa cambia la medida de resultado
# de un segmento, se nombra y no se empalma.

# %%
def segmentos(ticker=None):
    """``ticker=None``: la primera de tus empresas (cartera y estudio) con segmentos."""
    from transform import segments
    datos = obs(source="sec_segments")
    candidatos = [ticker] if ticker else list(dict.fromkeys(MIS_ACCIONES + EN_ESTUDIO))
    vista = None
    for ticker in candidatos:
        c = cik(ticker)
        vista = segments.assess(datos, c, HOY, S.source("sec")["concepts"]["revenue"]["tags"],
                                S.source("segments")["profit_concepts"]) if c else None
        if vista is not None and len(vista.members) > 1:
            break
    if vista is None:
        print(f"{ticker}: sin segmentos etiquetados")
        return None
    h = vista.history.pivot_table(index="date", columns="member", values="revenue")
    (h / 1e6).plot(marker="o", title=f"{ticker}: ingresos por segmento (M USD)")
    plt.show()
    return vista.table


segmentos()

# %% [markdown]
# ### 5.8 Salud financiera de un grupo de sector (mineras de plata)
# Último ejercicio anual, US GAAP o IFRS (40-F): caja neta, deuda neta / EBITDA, FCF y dilución.
# Una minera sana aguanta una caída del metal sin emitir acciones ni pedir prestado. Una deuda
# vacía es **desconocida**, no cero.

# %%
def salud_sector(grupo=None):
    from transform import sector_health as sh
    grupos = S.sector_groups
    if not grupos:
        print("Sin grupos en universe.sector_groups")
        return None
    grupo = grupo or next(iter(grupos))
    registro = tabla("companies")
    nombres = {str(r["cik"]): (str(r["ticker"]), r.get("name")) for r in registro.to_dict("records")}
    precios = {t: (precio(t, total=False).iloc[-1] if len(precio(t, total=False)) else None)
               for t in grupos[grupo]}
    t = sh.health_table(obs(source="sec_sector"), grupos[grupo], nombres, precios, HOY)
    t = t.set_index("ticker")
    (t[["cash", "debt"]] / 1e6).plot.bar(title=f"{grupo}: caja y deuda (M USD, último ejercicio)")
    plt.show()
    return t[["basis", "fy_end", "revenue_growth", "fcf_margin", "net_cash",
              "net_debt_to_ebitda", "dilution", "fcf_yield", "notes"]]


salud_sector()

# %% [markdown]
# ---
# ## 6. ¿Toca ejecutar? — tu cartera y su riesgo (nivel 4)
#
# - **Peso:** valor de la posición / valor de las acciones.
# - **Contribución al riesgo:** qué parte de la volatilidad de la cartera aporta cada posición.
#   Pesa más quien más se mueve **y** más se mueve con las demás.
# - **Sensibilidades (betas a factores):** cuánto se movió cada acción por cada movimiento del
#   mercado, los tipos, el dólar, las pequeñas, la tecnología, el consumo cíclico y el petróleo.
#   Si varias caen juntas ante lo mismo, son **una sola apuesta** (§5.3).

# %%
def cartera():
    cuenta = obs(source="ibkr")
    posiciones = port.latest_positions(cuenta)
    precios = obs(series_ids=[f"{t}:close_raw" for t in posiciones["ticker"]])
    valorada = port.valuation(posiciones, port.latest_prices(precios))
    return valorada.set_index("ticker")


def riesgo_de_cartera(dias=365):
    valorada = cartera()
    pesos = (valorada["market_value"] / valorada["market_value"].sum()).dropna().to_dict()
    indices = {t: precio(t) for t in pesos}
    desglose = risk.risk_breakdown(pesos, risk.daily_returns(indices, HOY, dias))
    t = desglose.table.set_index("ticker")
    t[["weight", "share_of_risk"]].plot.bar(title="Peso frente a parte del riesgo")
    plt.show()
    print(f"Volatilidad anual de las acciones: {desglose.portfolio_volatility:.1%}")
    return t


def sensibilidades(tickers=None, anos=3):
    tickers = list(tickers or (MIS_ACCIONES + EN_ESTUDIO))
    fact = risk.factor_returns({t: precio(t) for t in ["SPY", "IWM", "XLK", "XLY", "XLP"]},
                               lab.fred("DGS10"), lab.fred("DTWEXBGS"), lab.fred("DCOILWTICO"))
    fact = fact[fact.index >= HOY - pd.DateOffset(years=anos)]
    rows = []
    for t in dict.fromkeys(tickers):
        s = risk.sensitivity(precio(t), fact, t)
        if s:
            rows.append({"ticker": t, **{k: v for k, v in s.betas.items()},
                         "R²": s.r2, "semanas": s.weeks})
    betas = pd.DataFrame(rows).set_index("ticker")
    escenarios = pd.DataFrame({esc: betas[f] * shock for f, (_, _, shock, esc) in risk.FACTORS.items()
                               if f in betas})
    return betas, escenarios


cartera()

# %%
riesgo_de_cartera()

# %%
betas, escenarios = sensibilidades()
print("Movimiento estimado de cada acción en cada escenario (pasado, no predicción):")
escenarios

# %% [markdown]
# ### 6.1 ¿Estás batiendo al mercado?
# - **TWR (rentabilidad ponderada por tiempo):** encadena los cambios diarios del patrimonio
#   quitando tus aportes. Mide tus **decisiones**, no tus depósitos.
# - **Cartera sombra:** lo que valdría si cada aporte hubiera ido a SPY el mismo día.

# %%
def frente_al_mercado(dias=365):
    cuenta = obs(source="ibkr")
    nav = port.nav_series(cuenta)
    nav = nav[pd.to_datetime(nav["ts"].str[:10]) >= HOY - pd.Timedelta(days=dias)]
    flujos = port.external_flows(tabla("cash_transactions"))
    resultado, curva = port.performance(nav, flujos, precio("SPY"))
    if resultado is None:
        print("Sin historia suficiente")
        return None
    if not curva.empty:
        curva.set_index(curva.columns[0]).plot(title="Tu cuenta frente a la cartera sombra en SPY")
        plt.show()
    return pd.Series(dataclasses.asdict(resultado)).to_frame("valor")


frente_al_mercado()

# %% [markdown]
# ### 6.2 El espejo de conducta
# El riesgo nº 1 del panel no es técnico, es conductual (§2):
# - **rotación:** cuánto operas frente a tu patrimonio;
# - **días de tenencia de lo vendido** frente a tu horizonte;
# - **efecto disposición:** vender ganadoras y aguantar perdedoras;
# - **qué hicieron las acciones después de venderlas.**

# %%
def espejo_de_conducta(ventana_dias=365, horizonte_minimo=90):
    from transform import behavior
    cuenta = obs(source="ibkr")
    operaciones = tabla("trades")
    posiciones = port.latest_positions(cuenta)
    precios = obs(series_ids=[f"{t}:close_raw" for t in posiciones["ticker"]])
    valorada = port.valuation(posiciones, port.latest_prices(precios))
    indices = {t: precio(t) for t in set(operaciones["ticker"]) | {"SPY"}
               if t in lab.total_return().columns}
    m = behavior.mirror(operaciones, port.nav_series(cuenta), port.nav_series(cuenta, port.NAV_CASH),
                        valorada, indices, HOY, window_days=ventana_dias,
                        min_holding_days=horizonte_minimo)
    return pd.Series(dataclasses.asdict(m)).to_frame("valor")


espejo_de_conducta()

# %% [markdown]
# ### 6.3 Cuánto poner en una idea: escenarios y Kelly
# Escribe escenarios (probabilidad y múltiplo de tu dinero a N años). La **rentabilidad
# esperada** se calcula sobre la riqueza esperada, no como media de rentabilidades (§9.10).
# **Kelly** da la fracción que maximiza el crecimiento a largo plazo si las probabilidades
# fueran ciertas. No lo son, así que el panel enseña ¼ de Kelly como **techo**, nunca como
# objetivo. Cambia los números y mira cómo se mueve.

# %%
def kelly(probabilidades=(0.3, 0.5, 0.2), multiplos=(0.2, 1.5, 4.0), anos=5, fraccion=0.25):
    from transform.scenarios import kelly_fraction
    p, m = np.array(probabilidades), np.array(multiplos)
    esperado = float(p @ m)
    f = kelly_fraction(list(p), list(m))
    fracciones = np.linspace(0, 1, 101)
    crecimiento = [float(p @ np.log(np.maximum(1 + x * (m - 1), 1e-9))) for x in fracciones]
    plt.plot(fracciones, crecimiento)
    plt.axvline(f, color="red", label=f"Kelly {f:.0%}")
    plt.axvline(f * fraccion, color="green", label=f"{fraccion:.0%} de Kelly {f * fraccion:.0%}")
    plt.title("Crecimiento logarítmico esperado según la fracción apostada")
    plt.legend()
    plt.show()
    return pd.Series({"múltiplo esperado": esperado,
                      "rentabilidad anual esperada": esperado ** (1 / anos) - 1,
                      "prob. de perder": float(p[m < 1].sum()),
                      "Kelly": f, f"techo ({fraccion:.0%} de Kelly)": f * fraccion})


kelly()

# %% [markdown]
# ---
# ## 7. Compras de directivos en el S&P 500 (los datos de los estudios)
# Todas las compras en mercado abierto de consejeros y directivos desde 2017 (Form 4, trimestral
# de la SEC), la base de los estudios de §2.42 y §2.62. Los **eventos** son grupos de 3 o más
# directivos distintos en 30 días, o una compra del CEO/CFO de más de 100.000 USD.

# %%
def compras_de_directivos(desde="2017-01-01", top=15):
    p = lab.insider_purchases()
    p = p[pd.to_datetime(p["filing_date"]) >= desde]
    mensual = p.assign(mes=pd.to_datetime(p["filing_date"]).dt.to_period("M")) \
        .groupby("mes")["value"].sum() / 1e6
    mensual.plot(title="Compras de directivos por mes (M USD, todas las empresas)")
    plt.show()
    return p.groupby("symbol")["value"].agg(["count", "sum"]).sort_values(
        "sum", ascending=False).head(top).rename(columns={"count": "compras", "sum": "USD"})


def eventos_de_directivos():
    """Tarda ~2 minutos: agrupa ~200.000 compras por empresa y ventana."""
    cl, _ = signals.build(lab, "insider_cluster")
    ex, _ = signals.build(lab, "insider_executive")
    return pd.DataFrame({"grupos (≥3 en 30 d)": cl.groupby(cl["date"].dt.year).size(),
                         "CEO/CFO ≥ 100.000 USD": ex.groupby(ex["date"].dt.year).size()})


compras_de_directivos()

# %%
eventos_de_directivos()

# %% [markdown]
# ---
# ## 8. Validación estadística: cómo se prueba una regla sin engañarse
#
# ### 8.1 Por qué fechas separadas (rejilla disjunta)
# Si una señal dura semanas y mides la rentabilidad a 90 días **cada día**, dos días seguidos
# comparten 89 de sus 90 días: no son dos observaciones, son casi la misma. La correlación
# entre rentabilidades de días seguidos lo demuestra. El panel toma una fecha cada 90 días.

# %%
def demo_solapamiento(horizonte=90, activo="SPY"):
    s = precio(activo)
    diario = s.shift(-int(horizonte * 252 / 365)) / s - 1
    rejilla = diario.iloc[::int(horizonte * 252 / 365)]
    return pd.DataFrame({"muestras": [diario.count(), rejilla.count()],
                         "correlación con la siguiente": [diario.autocorr(1), rejilla.autocorr(1)]},
                        index=["cada día (solapadas)", f"cada {horizonte} días (disjuntas)"])


demo_solapamiento()

# %% [markdown]
# ### 8.2 La prueba de permutación
# ¿La diferencia observada podría salir por azar? Se barajan las etiquetas (o se cambian signos
# al azar) miles de veces y se cuenta qué parte de esas barajas da una diferencia igual o
# mayor: eso es la **p**. El histograma es la distribución del azar; la línea roja, lo observado.

# %%
def demo_permutacion(regla="knife", horizonte=180, **parametros):
    resultados, por_fecha = test(lab, regla, horizons=(horizonte,), record=False, **parametros)
    v = por_fecha["value"].to_numpy()
    rng = np.random.default_rng(0)
    azar = [np.mean(v * rng.choice([-1, 1], len(v))) for _ in range(5000)]
    plt.hist(azar, bins=60, color="#bbb")
    plt.axvline(v.mean(), color="red", label=f"observado {v.mean():+.2%}")
    plt.title(f"{regla}, {horizonte} d: medias al azar (cambiando signos) frente a la observada")
    plt.legend()
    plt.show()
    print(resultados[0])
    return por_fecha.tail(10)


demo_permutacion("knife", 180)

# %% [markdown]
# ### 8.3 Muchas pruebas: Benjamini-Hochberg
# Con 20 reglas que no hacen nada, a p < 0,05 saldrá alguna «significativa» por azar. La
# corrección de Benjamini-Hochberg controla la proporción de falsos descubrimientos. El
# registro del laboratorio la aplica sobre **todo** lo que has probado.

# %%
def demo_bh(pruebas=20, repeticiones=2000, alfa=0.10):
    from validation.metrics import benjamini_hochberg
    rng = np.random.default_rng(1)
    sin_corregir, con_bh = 0, 0
    for _ in range(repeticiones):
        p = rng.uniform(size=pruebas)          # reglas que no hacen nada
        sin_corregir += (p < 0.05).any()
        con_bh += any(s for _, s in benjamini_hochberg(p.tolist(), alfa))
    return pd.Series({f"probar {pruebas} reglas inútiles: sale alguna p<0,05": sin_corregir / repeticiones,
                      f"sale alguna significativa tras BH (α={alfa})": con_bh / repeticiones})


demo_bh()

# %% [markdown]
# ### 8.4 La base: con qué se compara (§2.74)
# Un evento se compara con la misma acción en «días normales». Si excluyes de la base los días
# cercanos al evento, quitas justo las ventanas que contienen su propio movimiento, y aparece un
# efecto que no existe. Aquí, la reacción a resultados con las dos bases.

# %%
def demo_sesgo_de_base(horizonte=90, excluir_dias=90):
    """Tarda ~1 minuto: ~2.000 eventos y ~40.000 puntos de base, dos veces."""
    eventos, _ = signals.build(lab, "earnings_reaction")
    justa, _ = evaluate.events(lab, eventos, rule="demo", horizons=(horizonte,), record=False)
    sesgada, _ = evaluate.events(lab, eventos, rule="demo", horizons=(horizonte,), record=False,
                                 away_days=excluir_dias)
    return pd.DataFrame({"diferencia evento − base": [justa[0].mean, sesgada[0].mean],
                         "p": [justa[0].p, sesgada[0].p]},
                        index=["base: días cualesquiera", f"base: excluye ±{excluir_dias} d"])


demo_sesgo_de_base()

# %% [markdown]
# ### 8.5 Probar una regla, y la curva de una estrategia
# `probar` usa el catálogo del laboratorio (`python run_lab.py reglas`). Con `registrar=True`
# la prueba queda en el registro y cuenta para la corrección: úsalo cuando la pruebes **en
# serio**, con la hipótesis escrita antes. La **estrategia** es una simulación: una curva que
# bate a SPY no prueba nada sin la prueba y el periodo reservado (*holdout*).

# %%
def catalogo():
    return signals.catalogue()[["regla", "tipo", "hipótesis", "por defecto", "qué es"]]


def probar(regla="low_accruals", registrar=False, **parametros):
    salida = test(lab, regla, record=registrar, **parametros)
    resultados = salida[0] if isinstance(salida, tuple) else salida
    return evaluate.table(resultados)[["rule", "horizon", "n", "mean", "hit", "p", "half_1",
                                       "half_2", "note"]]


def estrategia(regla="low_accruals", top=30, rebalanceo=91, coste_pb=10, **parametros):
    puntuar, _ = signals.build(lab, regla, **parametros)
    bt = evaluate.strategy(lab, lambda d, m: puntuar(d, m).nlargest(top).index,
                           rebalance_days=rebalanceo, cost_bps=coste_pb)
    bt.curve.plot(title=f"{regla}: top {top}, rebalanceo cada {rebalanceo} d (SIMULACIÓN)")
    plt.show()
    print(bt)
    return bt


catalogo()

# %%
probar("low_accruals")

# %%
estrategia("low_accruals", top=30)

# %%
summary().head(15)            # todo lo registrado, con BH sobre el total

# %% [markdown]
# ---
# ## 9. Qué datos hay
# Cada fuente, cuántas series y filas, y de qué fecha a qué fecha.

# %%
def inventario():
    conn = open_connection(S.db_path)
    try:
        t = pd.read_sql_query(
            "SELECT source AS fuente, COUNT(DISTINCT series_id) AS series, COUNT(*) AS filas, "
            "MIN(ts) AS desde, MAX(ts) AS hasta FROM observations GROUP BY source "
            "ORDER BY filas DESC", conn)
    finally:
        conn.close()
    t["desde"], t["hasta"] = t["desde"].str[:10], t["hasta"].str[:10]
    return t.set_index("fuente")


inventario()

# %% [markdown]
# ---
# ## 10. Glosario
# Los términos del panel en una tabla. `glosario("margen")` filtra por palabra.

# %%
GLOSARIO = pd.DataFrame([
    ("Point-in-time", "El valor tal como se conocía en una fecha, sin revisiones posteriores.", "1.2"),
    ("ts / ts_release", "Fecha de referencia del dato / fecha en que se publicó.", "1.1"),
    ("Sesgo de anticipación", "Usar en el pasado información que aún no existía.", "1.1"),
    ("Reexpresión (10-K/A)", "Cifras ya publicadas que la empresa vuelve a presentar.", "1.3"),
    ("Precio crudo / ajustado", "Lo que marcó la pantalla / dividido por los splits posteriores.", "1.4"),
    ("Rentabilidad total", "Precio más dividendos reinvertidos en su fecha.", "1.4"),
    ("CAGR", "Crecimiento anual compuesto: (final/inicial)^(1/años) − 1.", "1.5"),
    ("Sesgo de supervivencia", "Mirar el pasado solo con las empresas que sobrevivieron.", "1.6"),
    ("Prima de crédito", "Lo que cobran de más las empresas por endeudarse frente al Tesoro.", "2"),
    ("Curva invertida", "Tipos a corto por encima de los de largo.", "2"),
    ("Tipo real", "Rendimiento del bono protegido de inflación.", "2"),
    ("Semáforo", "Mayoría de 8 votos con reglas fijas; freno de conducta, no predictor.", "3.1"),
    ("Amplitud", "Qué parte del mercado acompaña al índice.", "3.2"),
    ("RSP / SPY", "Equiponderado frente a ponderado por tamaño: estrechez del rally.", "3.2"),
    ("VIX / VIX3M", "Miedo a 30 días frente a 3 meses; >1 = estrés.", "3.2"),
    ("Beta", "Cuánto se mueve una acción por cada 1 % del índice.", "4.2"),
    ("Caída máxima (drawdown)", "La peor caída desde un máximo previo.", "4.2"),
    ("Fuerza relativa", "Rentabilidad frente a la de otros en el mismo periodo.", "4.2"),
    ("Tema / ETF de tema", "El grupo de empresas que comparten la apuesta.", "4.3"),
    ("Margen bruto / operativo / neto", "Lo que queda de cada dólar vendido tras cada capa de costes.", "5.1"),
    ("FCF", "Flujo de caja libre: flujo operativo − inversión.", "5.1"),
    ("SBC", "Sueldos pagados en acciones: coste real que diluye.", "5.1"),
    ("TTM", "Los últimos doce meses (cuatro trimestres contiguos).", "5.1"),
    ("Dilución", "Crecimiento del número de acciones.", "5.2"),
    ("Recompras netas", "Recompras − emisión de acciones.", "5.2"),
    ("Conversión a caja", "FCF / beneficio neto.", "5.2"),
    ("ROIC", "Beneficio operativo tras impuestos / capital invertido.", "5.2"),
    ("VE", "Capitalización + deuda − caja.", "5.3"),
    ("Rendimiento FCF", "FCF / capitalización.", "5.3"),
    ("DCF inverso", "El crecimiento que justifica el precio de hoy.", "5.3"),
    ("Form 4 / plan 10b5-1", "Declaración de operaciones de directivos / ventas programadas.", "5.4"),
    ("13F", "Posiciones trimestrales de gestores > 100 M USD.", "5.4"),
    ("Interés corto / días para cubrir", "Acciones vendidas en corto / corto entre volumen diario.", "5.4"),
    ("Contribución al riesgo", "Parte de la volatilidad de la cartera que aporta una posición.", "6"),
    ("Modo de fallo compartido", "Posiciones que caen juntas ante lo mismo.", "6"),
    ("Rejilla disjunta", "Fechas separadas por el horizonte: observaciones independientes.", "8.1"),
    ("p (permutación)", "Probabilidad de una diferencia así por azar.", "8.2"),
    ("q (Benjamini-Hochberg)", "p corregida por todas las pruebas hechas.", "8.3"),
    ("Placebo", "El mismo evento movido antes: el efecto no debe estar ya.", "8.4"),
    ("Holdout", "Periodo reservado que se mira una sola vez, con la regla fijada.", "8.5"),
], columns=["término", "qué es", "sección"])


def glosario(palabra=None):
    if not palabra:
        return GLOSARIO.set_index("término")
    m = GLOSARIO["término"].str.contains(palabra, case=False) | \
        GLOSARIO["qué es"].str.contains(palabra, case=False)
    return GLOSARIO[m].set_index("término")


glosario()
