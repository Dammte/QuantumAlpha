from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True, slots=True)
class SetupTickerHistory:
    """Una fila de la tabla `setup_ticker_history` (Parte 11.2 del encargo:
    "del mismo replay de la Parte 10, filtrado por ticker" - "este valor ha
    formado 4 VCP en 5 años; 3 dispararon y 2 alcanzaron objetivo"). A
    diferencia de `setup_performance` (la fila cruzada, agregada sobre TODO
    el universo, con su propio umbral `MIN_SAMPLE_FOR_STATS` para hablar de
    una tasa medida con confianza), esta tabla guarda CONTEOS literales de
    un ticker concreto - un hecho histórico, no una probabilidad estimada,
    así que no hay campo `confidence` ni umbral de muestra mínima: "este
    valor lo ha hecho 2 veces" es información honesta con n=2, no algo que
    haya que ocultar hasta n=30 (Parte 15: "las cifras del setup son
    historia medida", nunca una probabilidad fabricada - pero tampoco un
    conteo real escondido tras un umbral que no le corresponde).

    `computed_at` es la misma foto completa que `setup_performance` - ver
    `SetupTickerHistoryRepository.replace_all`, mismo motivo (Postgres/NULL
    no es un problema aquí porque no hay columnas nulas de segmentación,
    pero el patrón de "reemplazar todo, nunca acumular" sigue siendo
    correcto: una fila vieja de un estudio anterior no debe sobrevivir junto
    a la nueva)."""

    id: int | None
    ticker: str
    region: str
    setup_name: str
    family: str
    n_observations: int  # veces que este setup llegó a READY en este ticker
    n_triggered: int  # de esas, cuántas confirmaron el nivel (disparo)
    n_target_hit: int  # de las disparadas, cuántas tocaron objetivo antes que stop
    first_ready_date: date
    last_ready_date: date
    computed_at: datetime
