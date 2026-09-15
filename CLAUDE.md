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
pytest -q                    # suite completa: ~750 tests, unitarios + integración, ~2-3 min
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

**Deuda conocida, explícitamente no resuelta todavía** (no asumir que ya está hecho solo porque
el nombre del archivo sugiere que sí): `trade_geometry.py` todavía usa un stop de ATR fijo
(`ATR_STOP_MULTIPLE`) y un objetivo 2:1 simple, no la cascada de stop por tipo de entrada ni el
techo de riesgo adaptativo por percentil de ATR que el plan original describe; la extensión
parabólica de `exit_engine.py`/`levels_engine.py` (`EXTENDED_ATR_MULTIPLE`) sigue midiéndose
sobre `atr_multiple` (base SMA50, un campo ampliamente compartido) en vez de EMA21 como pide
Parte 9 - cambiar esa base es una decisión aparte, de mayor alcance, no un efecto secundario de
la unificación del par rápido de arriba.
