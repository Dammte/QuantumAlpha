# QuantumAlpha Backend

API en Python/FastAPI para la gestión de una cartera de inversión personal: carteras, transacciones,
posiciones y métricas de riesgo/rendimiento al estilo de un fondo cuantitativo.

## Arquitectura

El proyecto sigue una arquitectura por capas (inspirada en clean/hexagonal architecture, pero sin
sobre-ingeniería) para que la lógica de negocio no dependa de FastAPI, SQLAlchemy ni de un proveedor
de datos concreto:

```
app/
  domain/           # Entidades de negocio puras (dataclasses) e interfaces (puertos)
    models/          Asset, Portfolio, Position, Transaction, PriceBar
    interfaces/      MarketDataProvider (puerto que abstrae la fuente de datos de mercado)
  services/         # Casos de uso / lógica de negocio, sin dependencias de framework
    metrics_service.py     Métricas de riesgo y rendimiento (Sharpe, Sortino, VaR, drawdown...)
    portfolio_service.py   Reconstruye posiciones e historial de valor de una cartera
    market_data_service.py Wrapper agnóstico del proveedor de datos
  infrastructure/   # Adaptadores concretos (detalles de implementación)
    db/              Modelos ORM (SQLAlchemy) y repositorios
    market_data/      YFinanceProvider (implementación del puerto MarketDataProvider)
  api/              # Capa HTTP (FastAPI): routers, endpoints, dependencias
  schemas/          # DTOs de entrada/salida (Pydantic)
  core/             # Configuración, logging
alembic/            # Migraciones de base de datos
tests/
  unit/             # Tests de dominio/servicios, sin BD ni red
  integration/       # Tests de la API con SQLite en memoria y un proveedor de datos falso
```

**Por qué esta separación:** `domain` y `services` no importan nada de FastAPI/SQLAlchemy/yfinance,
así que las métricas y la lógica de cartera se pueden testear e incluso reutilizar (p. ej. en un script
de backtesting) sin levantar la API ni una base de datos. Cambiar de yfinance a otro proveedor de datos
en el futuro solo implica escribir un nuevo adaptador que implemente `MarketDataProvider`.

## Requisitos

- Python 3.12+ (probado también con 3.14)
- PostgreSQL 16 (o Docker, ver más abajo)

## Puesta en marcha (local, sin Docker)

```bash
cd backend
python -m venv .venv
./.venv/Scripts/activate        # Windows
pip install -r requirements-dev.txt

cp .env.example .env            # y edita DATABASE_URL si hace falta
alembic upgrade head            # crea las tablas
uvicorn app.main:app --reload
```

La API queda disponible en `http://localhost:8000`, con documentación interactiva en
`http://localhost:8000/docs`.

## Puesta en marcha con Docker

```bash
cd backend
cp .env.example .env
docker compose up --build
```

Esto levanta PostgreSQL y la API. La primera vez, ejecuta las migraciones dentro del contenedor:

```bash
docker compose exec api alembic upgrade head
```

> Nota: en este entorno de desarrollo no había Docker/PostgreSQL disponibles para probar `docker-compose`
> de punta a punta, así que verifica el `docker compose up` la primera vez que lo uses.

## Tests

```bash
pytest            # 17 tests: unitarios de métricas/dominio + integración de la API (SQLite en memoria)
ruff check app tests
```

Los tests de integración sustituyen la base de datos real por SQLite en memoria y el proveedor de
yfinance por uno falso y determinista (ver `tests/integration/conftest.py`), así que corren rápido y
sin red ni credenciales.

## Endpoints principales (v0.1)

| Método | Ruta                                          | Descripción                                   |
|--------|-----------------------------------------------|------------------------------------------------|
| POST   | `/api/v1/portfolios`                          | Crear una cartera                              |
| GET    | `/api/v1/portfolios`                          | Listar carteras                                |
| GET    | `/api/v1/portfolios/{id}`                     | Detalle de una cartera                         |
| DELETE | `/api/v1/portfolios/{id}`                     | Eliminar una cartera                           |
| POST   | `/api/v1/portfolios/{id}/transactions`        | Registrar compra/venta                         |
| GET    | `/api/v1/portfolios/{id}/transactions`        | Historial de transacciones                     |
| GET    | `/api/v1/portfolios/{id}/summary`              | Posiciones actuales, valor de mercado, P&L      |
| GET    | `/api/v1/portfolios/{id}/metrics`              | Métricas de riesgo/rendimiento (ver abajo)      |

## Métricas incluidas en esta primera versión

Basadas en las convenciones de librerías de referencia del ecosistema quant en Python
(PyPortfolioOpt, Riskfolio-Lib, QuantStats, empyrical):

- Retorno acumulado y CAGR
- Volatilidad anualizada
- Ratio de Sharpe y ratio de Sortino
- Máximo drawdown y ratio de Calmar
- VaR histórico y CVaR (95%)
- Win rate
- Beta y alfa de Jensen frente a un benchmark (opcional, vía `benchmark_ticker`, p. ej. `^GSPC`)

## Roadmap (siguientes pasos sugeridos)

1. **Persistencia de precios históricos**: cachear velas OHLCV en `price_bars` en vez de llamar a
   yfinance en cada request (ya está el modelo `PriceBarORM`, falta el repositorio + job de sincronización).
2. **Autenticación**: si el proyecto crece más allá de un único usuario, añadir JWT/OAuth2.
3. **Más métricas**: ratio de información, tracking error, exposición sectorial/geográfica, correlación
   entre activos, diversification ratio, factor exposure (tamaño/valor/momentum) si se añaden fundamentales.
4. **Optimización de cartera**: frontera eficiente, Hierarchical Risk Parity, Black-Litterman
   (posible integración de PyPortfolioOpt o Riskfolio-Lib como servicio adicional).
5. **Backtesting**: simular estrategias basadas en reglas técnicas sobre el historial ya cacheado.
6. **Alertas/rebalanceo**: notificar cuándo una posición se desvía de su peso objetivo.

Cada uno de estos puntos se puede abordar como una iteración independiente sobre la base actual.
