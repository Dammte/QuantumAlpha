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
pytest -q                    # suite completa: ~500 tests, unitarios + integración, ~10-11 min
                              # (GARCH/Monte Carlo por test la hace pesada — normal)
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

Lee `backend/docs/quant_methodology.md` completo antes de tocar `recommendation_engine.py`,
`exit_engine.py`, `technical_analysis.py` o cualquier script de ablación — documenta qué hace
cada pieza, qué evidencia la respalda (o no todavía), y cómo se recalibra.

Puntos que no son negociables:

- **Es un checklist de reglas transparente y ponderado, no una caja negra ni ML.** Cada señal
  que dispara suma o resta puntos, y el porqué es siempre visible. Ver
  `docs/quant_methodology.md` §1 y §6.6 (por qué se descartó meta-labeling).
- **Ningún factor nuevo entra sin evidencia medida.** No añadas un indicador porque "suena
  bien" o tiene buena cita académica detrás — el filtro de régimen de Faber se implementó,
  se sometió al estudio de ablación (`scripts/factor_ablation_study.py`), y se retiró cuando
  la evidencia propia lo contradijo (§6.1). Ese es el estándar: mide antes de confiar.
  Cualquier peso nuevo o cambio de peso existente debe venir acompañado de correr (o
  actualizar) ese estudio, no de intuición.
- **Comprar y vender son preguntas distintas, con evidencia distinta.** `exit_engine.py`
  decide si una posición ya abierta debe cerrarse/recortarse/protegerse — y **nunca** importa
  `recommendation_engine.py` ni recibe RS Rating, fundamentales o el checklist de Minervini
  como parámetros. Un buen fundamental no es razón para aguantar una ruptura técnica en una
  cartera gestionada a semanas. Ver `docs/quant_methodology.md` §8.
- **No inventes datos.** Si algo no se puede calcular con la información disponible (p. ej.
  un "máximo de 52 semanas" con solo 60 barras de histórico), la función devuelve `None` —
  nunca una aproximación silenciosa etiquetada como si fuera el dato real.
- **Todo cambio de puntuación se documenta y versiona.** Un cambio en qué factores entran o
  con qué peso en `recommendation_engine.py` bumpea `ENGINE_VERSION` (se graba en cada
  `RecommendationSnapshotORM`, para poder atribuir un veredicto pasado a la lógica exacta que
  lo produjo) y se documenta en `docs/quant_methodology.md`. Un refactor puro (mover código sin
  cambiar ningún resultado, verificado contra los tests existentes) no necesita bump.
- **No llamadas de red por ticker en los caminos calientes.** `PortfolioRiskService` ya sufrió
  un incidente de latencia en producción por esto (ver su docstring) — semanal/mensual se
  derivan del histórico diario ya descargado (`technical_analysis.resample_ohlcv`), nunca una
  llamada nueva por posición.
- **No `ThreadPoolExecutor` en el backend.** Ya se probó y empeoró las cosas en la instancia
  de Render por sobresuscripción de BLAS/OpenMP (ver el mismo docstring). Si hace falta
  velocidad, cachear, no paralelizar.

## Estado del refactor del motor de salida (agosto 2026)

Trabajo en curso en la rama `exit-engine-overhaul`, en fases (ver el plan original en el
historial de commits de esa rama). Completado: bugs D5/D10/D11, núcleo multi-timeframe
(`multi_timeframe.py`), motor de salida independiente (`exit_engine.py`) y persistencia de
`trade_plan` integrados en producción (`GET /portfolios/{id}/risk`). Pendiente: trailing stops
tipo Chandelier (`trade_manager.py`), instrumentación de rendimiento de señales, backtest
honesto con triple-barrera, recalibración de pesos con evidencia, riesgo a nivel de cartera,
y la interfaz (panel de acciones del día, semáforo multi-temporalidad). Ver
`docs/quant_methodology.md` §8 para el detalle de lo ya hecho.
