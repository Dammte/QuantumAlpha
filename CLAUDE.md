# QuantumAlpha

Gestor de cartera personal de estilo cuantitativo. Backend FastAPI/Python (`backend/`),
frontend React/Vite (`frontend/`), desplegado en Render (backend) + Vercel (frontend).
Cartera activa, horizonte corto/medio (días a semanas) — el sistema existe para no tener
que revisar cada activo a mano todos los días.

## Puesta en marcha

**Backend** (Python 3.12+, PostgreSQL 16 o Docker):

```bash
cd backend
python -m venv .venv
./.venv/Scripts/activate        # Windows; source .venv/bin/activate en Unix
pip install -r requirements-dev.txt
cp .env.example .env            # edita DATABASE_URL si hace falta
alembic upgrade head            # crea/actualiza las tablas
uvicorn app.main:app --reload
```

API en `http://localhost:8000`, docs interactivas en `/docs`. Las migraciones también se
aplican solas al arrancar la app (`app/infrastructure/db/migrate.py`) — útil en Render,
pero para desarrollo local es más rápido correr `alembic upgrade head` a mano una vez.

Con Docker: `cd backend && cp .env.example .env && docker compose up --build`, luego
`docker compose exec api alembic upgrade head` la primera vez.

**Frontend**:

```bash
cd frontend
npm install
npm run dev       # servidor de desarrollo
npm run build     # build de producción
npm run lint      # eslint
```

## Tests y linter

```bash
cd backend
pytest -q                    # suite completa: ~830 tests, unitarios + integración, ~2-3 min
ruff check app tests         # linter, debe quedar limpio siempre
```

Los tests de integración sustituyen PostgreSQL por SQLite en memoria y yfinance por un
proveedor falso y determinista (`tests/integration/conftest.py`) — no necesitan red ni
credenciales. `tests/unit/` es más rápido (segundos) y no toca BD ni red — usa
`pytest -q tests/unit` para iterar rápido durante el desarrollo antes de correr la suite
completa antes de un commit.

**Regla de trabajo**: cada cambio de lógica de decisión (nueva función en
`technical_analysis.py`, `exit_engine.py`, `trade_manager.py`, etc.) lleva su test unitario
con casos sintéticos construidos a mano donde la respuesta correcta es obvia —
sigue el estilo de `tests/unit/test_technical_analysis.py` y `tests/unit/test_exit_engine.py`
(funciones planas `test_<función>_<escenario>()`, sin clases, series/DataFrames construidos
inline, `pytest.approx` para floats, comentarios explicando la intuición numérica cuando el
valor esperado no es obvio a simple vista).

## Arquitectura

Capas: `domain/` (dataclasses puras + interfaces/puertos, sin FastAPI/SQLAlchemy/yfinance) →
`services/` (lógica de negocio, tampoco importa framework — solo domain/interfaces) →
`infrastructure/` (adaptadores concretos: ORM, repositorios, proveedor de datos) → `api/`
(FastAPI) → `schemas/` (DTOs Pydantic). Un servicio que necesita persistencia depende de un
puerto en `domain/interfaces/` (p. ej. `MarketDataProvider`, `TradePlanRepositoryPort`), nunca
de la clase concreta de `infrastructure/db/repositories/` — así `services/` se puede testear
sin BD ni red. Ver `backend/README.md` para el detalle completo.

## Filosofía del motor de decisión

Lee `backend/docs/quant_methodology.md` completo antes de tocar `levels_engine.py`,
`trade_geometry.py`, `exit_engine.py`, `technical_analysis.py` o cualquier script de ablación —
documenta qué hace cada pieza, qué evidencia la respalda (o no todavía), y cómo se recalibra.

Puntos que no son negociables:

- **Es un gate de reglas transparente, no una caja negra ni ML, y no una puntuación
  ponderada.** `levels_engine.evaluate_gate` decide con condiciones booleanas (todas deben
  cumplirse) en vez de sumar/restar puntos - un factor fuerte no puede compensar uno
  genuinamente descalificante, y qué condición falló es siempre visible. El checklist
  ponderado original (`recommendation_engine.build_recommendation`) se retiró por completo en
  la reconstrucción de 2026-09 una vez el gate lo sustituyó en todos los caminos en vivo - ver
  `levels_engine.py`/`recommendation_engine.py`, ambos con el porqué en su docstring, y
  `docs/quant_methodology.md` §1 y §6.6.
- **Ningún factor/condición nueva entra sin evidencia medida.** No añadas un indicador porque
  "suena bien" o tiene buena cita académica detrás — el filtro de régimen de Faber se
  implementó, se sometió al estudio de ablación (`scripts/factor_ablation_study.py`), y se
  retiró cuando la evidencia propia lo contradijo (§6.1). Ese es el estándar: mide antes de
  confiar. Cualquier condición nueva o cambio de una existente debe venir acompañado de correr
  (o actualizar) ese estudio, no de intuición.
- **Comprar y vender son preguntas distintas, con evidencia distinta.** `exit_engine.py`
  decide si una posición ya abierta debe cerrarse/recortarse/protegerse — y **nunca** importa
  `recommendation_engine.py` ni recibe RS Rating, fundamentales o el checklist de Minervini
  como parámetros (verificado por AST, no solo por convención - ver
  `test_exit_engine_never_imports_recommendation_engine`). Un buen fundamental no es razón
  para aguantar una ruptura técnica en una cartera gestionada a semanas. Ver
  `docs/quant_methodology.md` §8.
- **Gemini nunca puntúa ni decide nada.** La capa de Gemini (`infrastructure/llm/`) es
  exclusivamente una narrativa de solo lectura sobre una decisión que el gate ya tomó - nunca
  entra en `evaluate_gate`, `trade_geometry` ni `exit_engine.py`, nunca bloquea el flujo
  principal (sin `GEMINI_API_KEY`, o si la llamada falla por cualquier motivo, cada método
  devuelve `None` de inmediato), y nunca sale de este backend hacia Google salvo que el
  propietario active la clave explícitamente. Ver `GeminiNarrator`/`LLMNarrator`.
- **No inventes datos.** Si algo no se puede calcular con la información disponible (p. ej.
  un "máximo de 52 semanas" con solo 60 barras de histórico), la función devuelve `None` —
  nunca una aproximación silenciosa etiquetada como si fuera el dato real.
- **Todo cambio de lógica de decisión se documenta y versiona.** Un cambio material en las
  condiciones del gate bumpea `levels_engine.GATE_VERSION` (se graba en cada
  `PositionDailyState`/`TradePlan` persistido, para poder atribuir un veredicto pasado a la
  lógica exacta que lo produjo); `recommendation_engine.ENGINE_VERSION` marca, más en general,
  qué motor de decisión está en producción (`"2026-09-v6-levels"` desde que el gate sustituyó
  al checklist por completo). Todo se documenta en `docs/quant_methodology.md`. Un refactor
  puro (mover código sin cambiar ningún resultado, verificado contra los tests existentes) no
  necesita bump.
- **Los parámetros de trading viven en un solo sitio.** `app/core/trading_params.py` (tamaño de
  posición, techos de riesgo, calibración del Chandelier, umbrales del gate) - un servicio que
  necesita uno de estos números lo importa de ahí, nunca redefine su propia copia. Expuesto de
  solo lectura en `GET /system/params`.
- **No llamadas de red por ticker en los caminos calientes.** `PortfolioRiskService` ya sufrió
  un incidente de latencia en producción por esto (ver su docstring) — semanal/mensual se
  derivan del histórico diario ya descargado (`technical_analysis.resample_ohlcv`), nunca una
  llamada nueva por posición. Los endpoints de lectura del Radar/Hoy (`GET /portfolios/{id}/today`,
  `GET /radar`) van más lejos: leen tablas precomputadas por los jobs nocturnos
  (`scripts/daily_close.py`), sin calcular nada en el propio request.
- **No `ThreadPoolExecutor` en el backend.** Ya se probó y empeoró las cosas en la instancia
  de Render por sobresuscripción de BLAS/OpenMP (ver el mismo docstring). Si hace falta
  velocidad, cachear, no paralelizar.

## Estado de la reconstrucción del motor de entrada (septiembre 2026)

Rama `trigger-engine-rebuild`, en curso. Objetivo: sustituir el checklist ponderado de
`recommendation_engine.py` por un gate de reglas booleanas (`levels_engine.py`) con geometría de
entrada explícita (`trade_geometry.py`), respondiendo a las 4 preguntas del propietario ("¿qué
hago hoy con lo que tengo?", "¿qué está a punto de dar entrada?", "¿es buena entrada este activo
concreto?", "¿está funcionando mi sistema?") con datos precomputados por jobs nocturnos en vez de
cómputo en caliente por request. Ver `backend/docs/quant_methodology.md` §25 en adelante para el
detalle completo, con su propio historial de qué está hecho y qué no.

**El gate son 5 criterios eliminatorios literales** (`Eligibility`: `liquidity_ok`,
`data_quality_ok`, `weekly_not_stage4`, `no_fast_bearish_cross`, `no_event_risk`,
`GATE_VERSION="2026-09-levels-v2"`), no los 6 de una aproximación anterior (tendencia/parabólico/
sobrecompra/OBV/par rápido/R:R) escrita sin el texto original en contexto - ver
`docs/quant_methodology.md` §27.6 para el porqué completo de cada retiro y cada criterio nuevo.

**Completo y en producción**: `levels_engine.evaluate_gate` como único camino de decisión de
entrada (el checklist viejo, retirado); precompute diario (`scripts/daily_close.py`,
`ticker_daily_state`/`position_daily_state`/`daily_brief`) y refresco intradía
(`scripts/intraday_refresh.py`); lecturas puras `GET /portfolios/{id}/today`, `GET /radar`;
escalada de salida con coste incluido y excepción de posición pequeña
(`trade_manager.compute_scaled_exit_plan`); universo dinámico activado (Fase 10);
`app/core/trading_params.py` como fuente única de parámetros; capa Gemini de solo lectura
(`GeminiNarrator.explain_gate`), inerte sin clave.

**Resuelto (septiembre 2026)**: el "par rápido" ya está unificado en una sola definición
EMA21/EMA55 (`multi_timeframe.FAST_MA_PERIOD`/`SLOW_MA_PERIOD`) - `multi_timeframe.py`,
`market_screener_service.py` y las reglas duras de `exit_engine.py` (vía
`portfolio_risk_service.py`) leen todos el mismo cálculo; ya no hay una SMA21/50 independiente
compitiendo con el EMA21/55 real de `technical_analysis.detect_fast_pair_bearish_veto`. El cruce
dorado/de la muerte SMA50/SMA200 sigue siendo, a propósito, un concepto SMA estándar y separado -
Parte 3.2 nunca pidió tocar ese.

**Resuelto (septiembre 2026)**: `trade_geometry.py` implementa el diseño real de Parte 7 -
cascada de stop por tipo de entrada (`EntryType`: ruptura, rebote en soporte, retroceso a EMA21,
continuación sobre EMA55) con el techo duro de 2.0 ATR, techo de riesgo adaptativo por percentil
de ATR (`RISK_CEILING_*` de `trading_params.py`, verificado contra los ejemplos exactos de Parte
15/20), objetivo neto de costes con caída a 2:1 fijo, y tamaño de posición con sus tres límites
(riesgo fijo, techo de capital, mínimo viable) - **deliberadamente partido en dos funciones**, no
una sola: `compute_entry_geometry` (stop/objetivo/techo de riesgo, sin capital) es lo que
`evaluate_gate` llama - el gate de un ticker no pertenece a ninguna cartera en particular, no hay
capital que darle; `size_position` (acciones/valor de posición/% de cartera) solo tiene sentido
una vez se conoce el capital de una cartera *específica*. **Resuelto (septiembre 2026)**: conectado
en `GET /market/radar?portfolio_id=` (ver más abajo) - el candidato natural que este mismo párrafo
señalaba. `compute_trade_geometry` es un envoltorio de conveniencia sobre ambas, para quien ya
tiene los dos contextos de antemano (los tests, un cálculo puntual bajo demanda).

**Ya conectado**: `evaluate_gate` acepta `ema21`/`ema55` opcionales y, cuando se le dan, rellena
`GateResult.entry_geometry` (`None` si no) - `ticker_analysis_service.compute_core_signals` ya
calcula y pasa el EMA21/55 real, así que tanto "Analizar activo" como el `/risk` de cartera (que
reutiliza `compute_core_signals`) devuelven la geometría real de Parte 7 hoy mismo
(`GateResultResponse.entry_geometry` en la API). **Resuelto (septiembre 2026)**: `scripts/daily_close.py`
(`build_ticker_daily_state`) ahora calcula su propio EMA21/55 (mismo `close` ya en memoria, sin
llamada de red nueva) y se lo pasa a `evaluate_gate`; `TickerDailyState.entry_geometry` (columna
JSON nueva, migración `d3f7a2b8c1e4`) persiste el resultado sin sizing
(`trade_geometry.geometry_to_dict`) para cada ticker del universo. `GET /market/radar` lo expone
sin dimensionar por defecto y, con `?portfolio_id=`, lo dimensiona contra el capital real de esa
cartera (`trade_geometry.size_position`/`geometry_from_dict`) - la lectura del Radar para una
cartera que `trade_geometry.py` señalaba como el candidato natural para `size_position`, ahora sí
conectado. `trade_plan_service.py` sigue sin cambiar - llama a `compute_stop_and_target`
directamente, nunca a `evaluate_gate`: una posición reconstruida ya tiene una cantidad real y
fija, dimensionarla no tendría sentido (ver ese módulo, "Deliberadamente lazy").

**Resuelto (septiembre 2026)**: `size_position` por sí sola solo aplica los techos de *una*
posición (riesgo fijo, `MAX_POSITION_PCT`) - ciega, por diseño propio, a cuánto del presupuesto de
riesgo agregado (6%) o del techo de concentración por sector (30%) ya consume el resto de la
cartera. `portfolio_construction_service.final_position_size`/`max_shares_for_position_risk`
existían, probados, sin ningún llamador - la propia "capa por encima de una sola posición" que
`size_position` señalaba como pendiente. `apply_portfolio_limits` (nuevo, mismo módulo) es esa capa:
`GET /market/radar?portfolio_id=` la aplica después de `size_position` sobre cada candidato viable,
con el riesgo agregado ya comprometido (`trade_plan` abierto por posición, mismo patrón de lectura
que `/portfolios/{id}/construction`) y la concentración sectorial ya ocupada (`sector_of` +
`compute_sector_concentration`) - nunca una llamada de red nueva por candidato, solo aritmética
sobre datos ya en memoria.

**Resuelto (septiembre 2026)**: la extensión parabólica de `exit_engine.py` (REDUCE) se mide
ahora sobre `technical_analysis.atr_multiple_from_ema` (EMA21, `EXTENDED_ATR_MULTIPLE=3.0`, Parte
9) a través de su propio parámetro `atr_multiple_from_ema21` - una función y un parámetro
deliberadamente separados de `atr_multiple_from_sma`/`CoreTickerSignals.atr_multiple` (SMA50), que
`levels_engine.evaluate_gate`'s "sin extensión parabólica" y `market_screener_service.py` siguen
usando sin cambios, a propósito - no era el mismo campo con dos consumidores, era un umbral nuevo
que necesitaba su propia base, no la del gate.
