# equitydash

Panel personal de análisis de renta variable estadounidense. Ingesta datos de fuentes
**gratuitas**, sincroniza una cuenta de Interactive Brokers en **solo lectura**, almacena
todo en base de datos local, calcula indicadores derivados y los presenta en una interfaz
web con alertas y validación estadística.

Horizonte: **mediano y largo plazo** (trimestres a años). **No es una herramienta de trading.**

> **Estado: fases 0, 1 y 2 implementadas.** El panel lee macro point-in-time, precios
> crudos, la cuenta real de IBKR y fundamentales auditados de la SEC. La cartera
> **reconcilia contra el NAV que reporta el bróker** (dos cálculos independientes del
> mismo número, diferencia −0,00%). Ver [Hoja de ruta](#hoja-de-ruta).

---

## Qué responde el panel

Cuatro preguntas, siempre en este orden. Todo widget, métrica y alerta mapea a una de ellas;
lo que no mapea, no se construye.

| Nivel | Pregunta | Con qué se responde |
|---|---|---|
| 1 | ¿Hay apetito por riesgo? | Macro: Fed, inflación, empleo, dólar, curva, crédito |
| 2 | ¿El mercado está sano o es un rally estrecho? | Amplitud, rotación sectorial, volatilidad, posicionamiento |
| 3 | ¿La tesis de la empresa sigue viva? | Fundamentales auditados (SEC XBRL) y captura de valor |
| 4 | ¿Toca ejecutar según el plan? | Cartera real, comisiones, eventos próximos |

**Regla de jerarquía:** los niveles 1 y 2 mandan sobre el 3. Una empresa excelente cae 40%
en un entorno *risk-off*. Si 1 y 2 están claramente en rojo, no se compra aunque el nivel 3
sea perfecto.

### Concepto rector: *shareholder value accrual*

La pregunta central no es "¿le va bien a la empresa?" sino **"¿cómo exactamente se beneficia
el accionista si a la empresa le va bien?"**. Una empresa puede crecer ingresos 30% anual y
destruir valor por acción si diluye 35% en el mismo periodo. El panel hace visible esa brecha
entre crecimiento del negocio y crecimiento *por acción*: acciones diluidas, *stock-based
compensation*, recompras netas de emisión, conversión de beneficio contable a caja y ROIC.

---

## Decisiones de diseño

Estas son deliberadas. Las explicaciones importan más que la lista.

### Diseñado contra el over-trading

El riesgo principal de un panel de inversión no es técnico, es conductual. Con horizonte de
años, la frecuencia óptima de consulta es semanal o mensual, así que el panel está construido
para **no premiar la consulta frecuente**:

- **Pull, no push.** Sin tickers en vivo, sin auto-refresh, sin animaciones de precio.
- **Resolución diaria.** Nunca intradía.
- Alertas solo por eventos accionables definidos por reglas escritas de antemano.

Lo que **no** se construye, por la misma razón: gráficos intradía, alertas por movimiento de
precio, pantallas de opciones, *screeners* de *earnings plays*, backtests de trading y
cualquier integración de ejecución de órdenes.

### IBKR en solo lectura, por construcción

La cuenta se lee con el **Flex Web Service** (Activity Flex Query, XML). Su token es
*estructuralmente* incapaz de colocar, modificar o cancelar órdenes o de mover fondos: es una
limitación del sistema de IBKR, no una configuración que haya que recordar mantener. Por eso
se prefiere a la API de TWS, que exige un proceso persistente con sesión iniciada y habilita
trading.

La **API de datos de mercado de IBKR no se usa como fuente del panel**: requiere suscripciones
por exchange y las licencias de esos feeds restringen redistribución y almacenamiento. Los
precios vienen de fuentes públicas gratuitas.

### Point-in-time de fábrica

Cada hecho XBRL de la SEC trae `end` (fecha de referencia del periodo) y `filed` (fecha de
presentación). El esquema los guarda como `ts` y `ts_release`, y **`ts_release` forma parte de
la clave primaria**: un mismo trimestre puede reportarse varias veces (original, enmienda,
reexpresión) y las tres versiones se conservan. El panel muestra la última; cualquier backtest
filtra por `ts_release <= fecha_simulada`, nunca por `ts`.

Sin esto, asignar el 10-K del año fiscal 2025 a "2025" usaría información del futuro durante
los casi dos meses que van hasta su presentación en febrero de 2026.

### Precios crudos, ajuste bajo demanda

Un precio "ajustado" se recalcula retroactivamente cada vez que hay un split o un dividendo:
la serie de cierres ajustados que se descarga hoy **no es la serie que existía el año pasado**.
Un backtest sobre cierre ajustado usa información del futuro en cada punto.

Por eso se almacena el **cierre crudo** (`TICKER:close_raw`) y las acciones corporativas por
separado, y la serie ajustada se construye para una fecha de corte concreta aplicando solo las
acciones con `ex_date <=` esa fecha. Un test bloqueante de CI verifica que la serie ajustada a
una fecha pasada no cambia al ingerir un dividendo posterior.

Esto obliga a un paso extra en la ingesta, porque los proveedores gratuitos no entregan el cierre
crudo. En yfinance, `auto_adjust=False` solo desactiva el ajuste por dividendos: la serie sigue
ajustada por splits. El ingester reconstruye el crudo multiplicando por los splits con fecha ex
posterior a cada barra, y un test lo verifica contra el cierre histórico real de AAPL alrededor de
su split 4:1 de 2020.

Almacenar el crudo tiene una segunda ventaja, esta de arquitectura: **mantiene la fuente
reemplazable**. Un cierre crudo es un hecho que no caduca; una serie ajustada solo tiene sentido
junto al proveedor que la calculó y a la fecha en que la calculó. El día que yfinance deje de
funcionar —es una librería no oficial sobre endpoints sin contrato— se cambia el ingester y el
histórico ya almacenado sigue siendo válido.

### El ticker no es una clave

Los tickers se reasignan (`FB` → `META`) y los de empresas quebradas se reutilizan. La clave
primaria de todo lo que viene de la SEC es el **CIK**. En los datos de IBKR, que llegan con
ticker, el CIK se resuelve durante la ingesta y se deja `NULL` con aviso si no resuelve —
nunca se adivina.

### Diversificación por modo de fallo, no por número de tickers

El sector GICS da falsa tranquilidad: ocho empresas de cinco sectores distintos pueden ser
todas la misma apuesta a tipos de interés bajos. El panel agrupa la cartera simultáneamente
por sector GICS, por `thesis_category` definida por el usuario y por exposición aproximada a
factores, y muestra una tabla explícita de **modos de fallo compartidos**.

### Ninguna empresa entra sin criterio de invalidación

La ficha de tesis de cada empresa incluye qué observación concreta, verificable y con umbral
mataría la tesis. No es una convención: `thesis_log.invalidation` es `NOT NULL` en el esquema
y la carga de configuración rechaza una ficha incompleta.

### La vista pública no enseña ninguna cifra absoluta

Un despliegue público de este panel muestra **pesos, porcentajes y una curva de patrimonio
en base 100**; nunca un importe. Lo que se publica es el método, no el saldo.

Se descartó la alternativa obvia —escalar los importes por un factor secreto— por tres
razones. Es **invertible**: las comisiones no escalan con el tamaño de la cartera (IBKR
cobra un mínimo por orden), así que un solo importe publicado delata el factor y con él
todas las cifras. **Falla abierto**: un despliegue que no herede la variable de entorno
publica lo real. Y pone **números fabricados con aspecto de reales** en pantalla.

La regla es código, no un recordatorio: la función que formatea importes **lanza una
excepción** en modo público, la lista de páginas publicables es blanca (una página nueva es
privada por defecto) y un test renderiza el panel entero y falla ante cualquier cadena con
forma de dinero.

### Nunca un valor estimado en silencio

Cuando un dato falta, se muestra `None`. Las empresas usan conceptos XBRL distintos para lo
mismo (`Revenues` frente a `RevenueFromContractWithCustomerExcludingAssessedTax`); hay un mapa
de sinónimos con orden de preferencia y registro de cuál se usó, pero si ninguno resuelve, el
valor queda vacío y visible. Un concepto mal mapeado produce una serie plausible y equivocada,
que es peor que un hueco.

---

## El universo: cuatro círculos

Una empresa no entra al panel de cualquier manera, y la diferencia entre los círculos es
una decisión de diseño, no una taxonomía.

| Círculo | Qué exige | Qué obtiene |
|---|---|---|
| **Referencias de mercado** | nada, van en `settings.yaml` | Alimentan los niveles 1 y 2. No son posiciones |
| **En estudio** (`watchlist`) | ticker y CIK | Datos completos: fundamentales, presentaciones, calendario |
| **Seguimiento** (`tracked`) | ficha de tesis completa | Lo anterior **más** el tablero de invalidación |
| **Cartera real** | nada: se lee de IBKR | Valoración, PnL, drift, reconciliación contra el NAV |

La ficha de tesis exige un **criterio de invalidación verificable y con umbral**, y el
esquema lo rechaza si falta. Sin criterio de falsación no hay forma de saber cuándo vender,
y una posición se sostiene sola por inercia.

**Por qué existe el círculo "en estudio".** Exigir la ficha para todo creaba un punto
muerto: el panel no bajaba una sola cifra hasta que la tesis estuviera escrita, así que la
herramienta construida para apoyar la investigación no se podía usar *durante* ella — y la
única salida era inventarse un criterio de invalidación, que es lo que la regla existe para
impedir. **La ficha se escribe después de mirar los números, no antes.** Un candidato
obtiene datos, no opinión: no entra al tablero ni cuenta como universo. Promocionarlo es
mover la entrada y completarla.

---

## Convenciones contables que el panel implementa

Tres reconstrucciones que un panel ingenuo hace mal, y en las tres el resultado equivocado
es **plausible a la vista** — que es lo que las hace peligrosas.

**No existe el 10-Q del cuarto trimestre.** El Q4 solo aparece dentro del 10-K, así que un
ejercicio llega como tres trimestres más una cifra anual. Pintar "revenue trimestral"
directo de XBRL se deja una barra de cada cuatro y la serie sigue pareciendo una serie. El
Q4 se deriva (`FY − Q1 − Q2 − Q3`) y **hereda la fecha del 10-K**, no la del cierre del
trimestre: fecharlo al cierre regala dos meses de futuro cada año, invisibles, porque el
número en sí es correcto.

**El estado de flujos suele presentarse como Q1 + acumulados.** Muchas empresas solo dan un
hecho de tres meses en el Q1; el Q2 y Q3 llegan como acumulados de 6 y 9 meses. Filar un
acumulado en la serie trimestral daría una cifra dos o tres veces mayor; descartarlos sin
más deja **un trimestre de flujo de caja por año** y por tanto ningún FCF. Se guardan
aparte y el trimestre se deriva (`Q2 = YTD2 − Q1`), heredando la fecha del documento que
trae el acumulado.

**Un recuento de acciones no es un flujo.** El número medio ponderado de acciones diluidas
llega como concepto de duración, pero es un promedio del periodo: sumar cuatro trimestres
lo multiplica por cuatro y derivar el Q4 restando da un número negativo. La configuración
las marca `kind: average` y solo se leen por una vía que se niega a sumarlas.

Cada derivación se valida contra la realidad, no contra sí misma: se deriva una métrica que
la empresa **sí** reporta suelta y se compara con lo reportado.

---

## Fuentes de datos

Todas gratuitas. Es una restricción del proyecto, no una circunstancia.

| Dato | Fuente | Acceso |
|---|---|---|
| Macro (CPI, PCE, NFP, fed funds, dólar, curva, spread HY, NFCI, VIX) | FRED | REST, clave gratuita |
| Fundamentales auditados | SEC EDGAR XBRL | `data.sec.gov`, sin clave |
| Fechas de presentación y enmiendas | SEC submissions | `data.sec.gov`, sin clave |
| Mapa ticker↔CIK | SEC `company_tickers.json` | sin clave |
| Precios diarios OHLC y acciones corporativas | yfinance | sin clave |
| Short interest | FINRA | API pública sin credenciales (quincenal; la fuente no publica fecha de difusión, así que se deriva — tarde a propósito) |
| VIX y VIX3M | FRED (`VIXCLS`, `VXVCLS`) | misma clave que el resto de la macro |
| Composición histórica del S&P 500 | [`fja05680/sp500`](https://github.com/fja05680/sp500) (MIT) | reconstrucción desde 1996; fiable desde 2001. Los precios de los miembros que salieron no son gratis, así que la amplitud muestra su **cobertura** y solo vale desde el 85 % (2019-09-23) |
| Put/call ratio | — | **fuera de alcance**: CBOE no ofrece ruta pública y su `robots.txt` prohíbe la sección |
| Cuenta real | IBKR Flex Web Service | token de solo lectura |

**Fuera de alcance** (de pago): datos de mercado de IBKR, Bloomberg/FactSet/Refinitiv,
estimaciones de consenso de analistas, datos de opciones por contrato y constituyentes
históricos de índices con fecha de efectividad.

Esa última ausencia tiene consecuencias: calcular amplitud con la lista de constituyentes de
**hoy** aplicada al pasado sobreestima la salud del mercado, porque las empresas que quebraron
o fueron expulsadas no están. El panel documenta esa limitación de forma visible, no en un
comentario del código.

---

## Instalación

Requiere Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,prices]"   # añade extras según la fase: ibkr, app, postgres

cp config/.env.example config/.env
cp config/settings.local.yaml.example config/settings.local.yaml
```

Rellenar `config/.env` con las claves (FRED, SEC User-Agent, token Flex de IBKR, Telegram) y
`config/settings.local.yaml` con las tesis y la cartera objetivo. Ambos están *gitignored*.

Para el token de IBKR conviene **empezar por la cuenta paper**, que tiene su propio token y
query id. La query debe incluir las secciones `AccountInformation`, `OpenPositions`, `Trades`,
`CashTransactions` y `ChangeInDividendAccruals`.

### Ejecución

```bash
python run_ingest.py --dry-run       # crea/verifica la base de datos, sin llamadas de red
python run_ingest.py                 # ejecuta la ingesta registrada
python run_ingest.py --only fred     # solo una fuente
```

Fuentes registradas: `fred`, `prices`, `ibkr`, `sec`, `sec_filings`. Una fuente sin sus
credenciales configuradas **se omite con aviso**, no rompe el pipeline; y una unidad rota
dentro de una fuente (una serie, un ticker, una empresa) se reporta y devuelve código de
salida ≠ 0 sin llevarse por delante a las demás.

El panel se abre con:

```bash
streamlit run app/main.py                      # vista local, con importes
PUBLIC_MODE=1 streamlit run app/main.py        # vista pública, sin ninguna cifra absoluta
```

Re-ejecutar es seguro por construcción: toda escritura pasa por un
`INSERT ... ON CONFLICT DO UPDATE`.

### Tests y lint

```bash
pytest -q
ruff check .
```

---

## Configuración

Nada está hardcodeado.

| Archivo | Contenido | Versionado |
|---|---|---|
| `config/settings.yaml` | Universo de referencia, fuentes, umbrales, ventanas | sí |
| `config/settings.local.yaml` | Tesis, pesos objetivo, parámetros fiscales | **no** |
| `config/.env` | Claves y tokens | **no** |

`settings.local.yaml` se fusiona encima de `settings.yaml` (*deep merge*), así que solo hace
falta escribir lo que se quiera sobrescribir. Las variables de entorno tienen prioridad sobre
el `.env`, que a su vez la tiene sobre los YAML.

Variables de runtime: `LOG_LEVEL`, `PUBLIC_MODE` (oculta la cuenta real y la capa fiscal) y
`DATABASE_URL` (cambia SQLite por PostgreSQL sin tocar código).

---

## Arquitectura

```
ingest/     fetch() -> DataFrame [source, series_id, ts, ts_release, value]
              │
db/loader   upsert idempotente (ON CONFLICT DO UPDATE)
              │
db/         SQLite (fases 0-4) o PostgreSQL (fase 5) vía DATABASE_URL
              │
transform/  funciones puras, sin red — entrada faltante -> None
              ajuste por acciones corporativas A UNA FECHA DE CORTE, nunca "la de hoy"
              │
app/        Streamlit multipágina (st.navigation)
              ├── 🏠 Hoy       nivel 1, macro point-in-time
              ├── 🏢 Empresa   nivel 3, una empresa a fondo contra su propia tesis
              └── 🔬 Cartera   nivel 4, posiciones y reconciliación contra el NAV
alerts/     Telegram, con deduplicación vía alerts_log
validation/ retornos forward + bootstrap por permutación + FDR Benjamini-Hochberg
```

El almacenamiento usa **formato largo**: `observations(source, series_id, ts, ts_release,
value)`. Añadir una serie nueva nunca requiere una migración.

Cada módulo de ingesta expone `fetch() -> DataFrame` con el mismo contrato de columnas, y el
loader es agnóstico a la fuente. Las transformaciones son funciones puras sin acceso a red, lo
que las hace testeables con fixtures congelados. Los parsers **fallan de forma ruidosa**: si
la estructura de una fuente cambia, salta una excepción; nunca se devuelve un resultado vacío
o a medias.

---

## Hoja de ruta

| Fase | Contenido | Estado |
|---|---|---|
| 0 | Andamiaje: esquema, adaptador de BD, loader idempotente, config, logging, CI | ✅ |
| 1 | Niveles 1 y 4: macro (FRED), precios crudos, cuenta IBKR, página de cartera | ✅ |
| 2 | Nivel 3: fundamentales SEC XBRL, normalización de taxonomía, dilución y captura de valor, página por empresa | ✅ |
| 3 | Nivel 2: amplitud, rotación sectorial, semáforo de régimen | en curso: VIX3M, interés corto y amplitud con cobertura hechos; faltan rotación, semáforo y la página |
| 4 | Alertas Telegram y validación estadística | pendiente |
| 5 | Capa fiscal, PostgreSQL, orquestación y despliegue | pendiente |

El semáforo de régimen de la fase 3 usa **pesos iguales y fijos**, no optimizados sobre el
histórico, con el voto de cada componente visible. Bloquea decisiones; nunca es un gatillo de
compra. Con suficientes indicadores siempre aparece una combinación que habría funcionado —
por eso la fase 4 aplica corrección FDR de Benjamini-Hochberg sobre la batería completa de
señales y documenta también **las que no funcionan**.

---

## Licencia

MIT.

Este proyecto es una herramienta de análisis personal. No es asesoría de inversión ni fiscal.
Los cálculos fiscales que incluye son estimaciones marcadas como tales y no sustituyen a un
contador.
