# Metodología cuantitativa de QuantumAlpha

_Primera auditoría a fondo: agosto de 2026. Segunda auditoría (respuesta a una revisión externa): agosto de 2026, mismo mes. Este documento explica qué hace el motor de recomendación, por qué cada pieza está ahí, qué evidencia respalda (o no respalda todavía) cada una, y cómo se recalibra. Está escrito para que un inversor, un fondo, o cualquier tercero técnico pueda auditar el sistema sin tener que leer el código fuente primero._

## 1. Filosofía

El sistema **no** es una caja negra ni un modelo de machine learning entrenado sobre datos históricos para maximizar retorno pasado (lo que tendría un riesgo real de sobreajuste). Es un **checklist de reglas transparente y ponderado**: cada señal técnica, estadística o fundamental que dispara suma o resta puntos, y cada punto se muestra explícitamente al usuario con su etiqueta. La puntuación total decide el veredicto (`comprar` / `esperar` / `evitar`) contra dos umbrales fijos.

La pregunta que ha guiado esta auditoría no es "¿podemos añadir más indicadores?" sino **"¿qué combinación de señales, ninguna redundante entre sí, nos da la mejor lectura posible, y de cuáles tenemos evidencia real de que funcionan?"**

## 2. Inventario completo de factores, por nivel de confianza

### Nivel 1 — Validados empíricamente (evidencia estadística propia, no solo teoría)

| Factor | Fuente | Evidencia |
|---|---|---|
| **Rango de 52 semanas confirmado** (precio ≥25% sobre su mínimo anual y dentro del 25% de su máximo anual) | Minervini Trend Template (2 de sus 8 criterios, separados del resto) | Único factor significativo (p<0.01, test de permutación) **en dos horizontes independientes**: +1.04 p.p. a 3 meses, +2.67 p.p. a 6 meses, sobre ~217 tickers x 10 años. El factor individual mejor probado de todo el sistema. |
| **Extensión parabólica** (precio a más de 4x ATR de su media móvil de 50) | Indicador ATR propio | Significativo y con el signo correcto (-0.80 p.p., p<0.001) específicamente a 21 días - un movimiento ya muy extendido corrige a corto plazo, consistente con la literatura de "reversión tras sobreextensión". |
| **Backtest walk-forward propio del ticker** | `walk_forward_backtest.py` | Cada vez que analizas un ticker, el sistema revalida sus propias señales *en ese ticker específico* con test-t de Welch + test de permutación + corrección de Bonferroni, y te avisa si "comprar" no ha superado históricamente a "evitar" en ese nombre concreto. Es el único mecanismo que puede decirte "esta vez no funciona aquí". |

### Nivel 2 — Fundamento teórico sólido, backtesting cruzado no concluyente todavía

La mayoría de las señales de tendencia (MA20>MA50>MA200, Fase 2/4 de Weinstein) mostraron un patrón interesante y **coherente con la literatura académica**, no un fallo: a 21 días (≈1 mes) el efecto medido es de *reversión* (contrario al signo actual), pero el signo se corrige progresivamente hacia el sentido esperado a 63 y 126 días, sin llegar aún a significación estadística por tamaño de muestra en este periodo concreto. Esto coincide exactamente con dos hallazgos académicos bien documentados:

- **Reversión a corto plazo** (Jegadeesh, 1990): a 1 mes, las acciones tienden a revertir, no a continuar.
- **Momentum de 3-12 meses** (Jegadeesh & Titman, 1993): el efecto de continuación de tendencia solo aparece a horizontes más largos.

**Conclusión y acción tomada**: no se invirtió el signo de ningún factor de tendencia (sería sobreajustar a un solo horizonte y contradecir 30+ años de literatura), pero **el backtest walk-forward ahora usa el mismo horizonte que ya seleccionas para Monte Carlo (1m/3m/6m) en vez de un valor fijo de 21 días** — antes se validaba el sistema exactamente en la zona de reversión, no en la de momentum que el propio diseño (herencia Weinstein/Minervini/CANSLIM) asume.

| Factor | Peso actual | Estado |
|---|---|---|
| Tendencia (MA20>MA50>MA200) | +2 / -3 | Momentum clásico; efecto correcto a 63-126d, no aún significativo en esta muestra |
| Fase de Weinstein (2/4) | +2 / -3 | Igual que tendencia - lente distinta sobre el mismo hecho |
| RS Rating (IBD, percentil cruzado) | +2 / -1 | No incluido en el estudio de ablación (necesita snapshot cruzado que no existe en el pasado) - respaldado por décadas de literatura de IBD |
| ADX/DMI (fuerza de tendencia) | +1 | No significativo en esta muestra, dirección variable por horizonte |
| Cadena de Markov | +2 / -2 | Filtrada por un test de aleatoriedad propio (Wald-Wolfowitz) - solo cuenta cuando la secuencia del propio ticker no es estadísticamente indistinguible de ruido |
| GARCH (régimen de volatilidad) | -1 | Contexto de riesgo, no dirección - también alimenta el tamaño de posición (Kelly) por separado |

### Nivel 3 — Plausible, fundamento teórico sólido, aún sin muestra suficiente para significación

| Factor | Peso | Por qué se añadió |
|---|---|---|
| **Divergencia de volumen (OBV)** — "esfuerzo vs. resultado" de Wyckoff | +1 / -2 | Antes el volumen no influía en la puntuación en absoluto pese a calcularse. Es la única señal basada en participación real de compradores/vendedores, no solo en precio. No alcanzó significación en el estudio (pocos casos de divergencia clara en la muestra), pero es una fuente de información estructuralmente distinta de todo lo demás. |
| **Crecimiento de ingresos** (≥15% interanual) | +1 / -1 | CANSLIM (O'Neil) y el propio Minervini exigen crecimiento fundamental, no solo timing técnico. Verificado empíricamente que `revenueGrowth` de yfinance está disponible de forma fiable (100% de la muestra probada, incluyendo tickers europeos). |
| **Margen neto** (≥15% / negativo) | +1 / -1 | Factor de calidad (Fama-French "quality"). |
| **Apalancamiento elevado** (deuda/patrimonio >200%) | -1 | Factor de riesgo de solvencia. |
| **Estructura de reversión a la media** (exponente de Hurst < 0.45, ver §6) | -1 | Solo penaliza, no premia — evita contar la tendencia dos veces desde otro ángulo. Ver §6 para la metodología completa. |

Los fundamentales **solo se aplican en "Analizar activo"** (búsqueda individual), no en la cartera ni en la watchlist premium — necesitan una llamada de red adicional por ticker, y fue exactamente ese patrón (N llamadas por posición) lo que causó un cuelgue real en producción una vez ya corregido. Con datos reales, no se repite ese error para ganar un factor secundario. El exponente de Hurst y el test ADF, en cambio, se calculan siempre (son gratis en red, solo cuestan CPU) en las tres superficies.

### Excluidos deliberadamente del backtest de ablación (no del sistema en vivo)

- **RS Rating**: necesita un snapshot del universo completo en cada fecha histórica, no disponible.
- **Markov / GARCH**: reajustar estos modelos en cada punto histórico, para cientos de tickers, es computacionalmente inviable a este alcance.
- **Soporte/resistencia**: el escaneo de pivotes es O(n) por llamada; recalcularlo en cada barra histórica no es asumible.
- **Fundamentales**: yfinance no expone fundamentales históricos punto-en-el-tiempo de forma fiable y gratuita.

Estas limitaciones ya estaban documentadas en `walk_forward_backtest.py` antes de esta auditoría; se mantienen igual.

## 3. El estudio de ablación (cómo se generó la evidencia de la sección 2)

`scripts/factor_ablation_study.py` — reproducible, ver docstring del propio script para el comando exacto.

- **Universo**: ~217 tickers (US + Europa, el mismo universo curado que usa el screener).
- **Muestra**: ~22.800 observaciones a 21 días, ~7.500 a 63 días, ~3.650 a 126 días (muestreo no solapado: el paso entre observaciones es igual al horizonte, para evitar la pseudo-replicación clásica de ventanas solapadas).
- **Test**: para cada factor, se compara el retorno futuro medio cuando el factor se activa vs. cuando no, con un test-t de Welch **y** un test de permutación (5000 permutaciones) sin asumir normalidad - el mismo rigor que ya usaba `walk_forward_backtest.py` para un solo ticker, aplicado ahora de forma agrupada sobre todo el universo.
- **Por qué agrupado y no ticker a ticker**: un backtest de un solo ticker tiene, típicamente, 100-150 muestras - insuficiente para significación en la mayoría de los factores. Agrupar sobre 217 tickers da 3.000-23.000 muestras según el horizonte, el mismo principio que usa la investigación académica de factores (Fama-MacBeth).

**Repetir el estudio**: cualquier cambio de universo, periodo o factor debería ir acompañado de volver a correr este script antes de confiar en los pesos actuales - así lo indica el propio código.

Resultados completos (todas las filas, los 3 horizontes): `docs/factor_ablation_report_h21.csv`, `docs/factor_ablation_report_h63.csv`, `docs/factor_ablation_report_h126.csv`.

## 4. Investigación externa que informó esta auditoría

- **CANSLIM** (William O'Neil): crecimiento de beneficios/ventas ≥25% interanual, liderazgo relativo, patrocinio institucional, dirección del mercado general.
- **Trend Template de Minervini**: las 8 condiciones de tendencia que ya usábamos, con el hallazgo de que su parte de "rango de 52 semanas" es la única con evidencia estadística robusta e independiente.
- **Fases de Weinstein**: ya implementado (`classify_stage`).
- **Wyckoff** (esfuerzo vs. resultado, divergencia de volumen): factor nuevo (OBV).
- **Factor investing académico** (AQR, Fama-French): momentum, calidad, value - confirmó que "calidad" (margen, ROE) y "crecimiento" son factores independientes y complementarios al momentum de precio, no redundantes con él.
- **Jegadeesh (1990) / Jegadeesh & Titman (1993)**: reversión a 1 mes vs. momentum a 3-12 meses - explica el patrón por horizonte encontrado en el estudio de ablación.
- **Renaissance Technologies** (información pública, no hay acceso a su metodología real): validan señales por p-valor antes de incorporarlas a producción - el mismo principio que sigue este documento.
- **Kelly fraccionado** (ya implementado): medio-Kelly, ajustado por régimen de volatilidad GARCH, con techo duro del 25% del capital - la práctica documentada de gestión de tamaño de posición en fondos quant/CTA.

## 6. Segunda auditoría: respuesta a una revisión externa

El usuario compartió un informe generado por otra IA con acceso solo a `render.yaml` (sin lectura del código fuente, algo que el propio informe reconoce honestamente). Varias de sus "hipótesis muy probables" ya estaban resueltas (ajuste por dividendos/splits con `auto_adjust=True`, 27 archivos de test con 300+ tests, el motor de backtesting con test-t + permutación + Bonferroni que ya existía antes del informe). Pero acertó en varias cosas reales, cada una tratada por separado:

### 6.1 Régimen de mercado — implementado, probado, y retirado por evidencia contraria

El punto más fuerte del informe: el filtro táctico de SMA-200 de Meb Faber sobre el índice de referencia, y un gate de estrés por VIX, se calculaban perfectamente para el panel de "Contexto" pero nunca llegaban a la recomendación de cada ticker. Se implementó (`market_below_sma200`, `vix_stress`, -2 puntos cada uno) y se sometió al mismo estudio de ablación que todo lo demás — no se dio por bueno solo por tener buena cita académica detrás.

**Resultado, con corrección de Benjamini-Hochberg por comparaciones múltiples**: ambos factores fueron significativos (p<0.01 corregido) a 21 días **con el signo contrario al asumido** — comprar cuando el índice estaba bajo su SMA-200, o con el VIX en pánico/crisis, se asoció con retornos *mejores* a 21 y 126 días, no peores (+2.85 p.p. y +4.35 p.p. respectivamente a 21 días; +9.17 p.p. el VIX a 126 días). Un patrón de "comprar el pánico" sobre un universo ya curado de calidad, no la evitación de mercados bajistas plurianuales que mide la investigación original de Faber (una afirmación estructuralmente distinta, a una escala temporal mucho mayor, que este estudio no puso a prueba).

**Decisión**: no se dejó el factor con el signo original (la evidencia lo contradice) ni se invirtió a positivo (una sola muestra de 10 años dominada por un par de recuperaciones en V es poca base para apostar al contrario con dinero real). Se retiró de la puntuación. La infraestructura (`technical_analysis.market_regime_inputs()`, `vix_regime()`) se mantiene — la usa `MarketContextService` y el propio estudio de ablación — y el contexto (`market_trend`, `vix_regime`) se sigue calculando y mostrando en "Analizar activo" con fines informativos, simplemente ya no mueve el veredicto. Ver el comentario correspondiente en `recommendation_engine.py` para el detalle completo.

Esto es, en sí mismo, el ejemplo más claro de por qué existe el estudio de ablación: una sugerencia con buena literatura detrás resultó no sostenerse con los datos propios, y se descubrió *antes* de salir a producción con dinero real.

### 6.2 Corrección por comparaciones múltiples en el propio estudio

El primer estudio de ablación (agosto 2026, ver commits anteriores) probó ~13 factores de forma independiente al 1% de significación sin corregir por el número de pruebas — con 13 tests, la probabilidad de al menos un falso positivo por azar puro no es despreciable. `scripts/factor_ablation_study.py` ahora reporta también p-valores ajustados por **Benjamini-Hochberg** (control de tasa de falso descubrimiento) junto a los brutos — un factor debe superar el umbral ajustado, no solo el bruto, antes de confiar en él. Implementado en `benjamini_hochberg_adjust()`, con test dedicado.

### 6.3 Exponente de Hurst + test de Dickey-Fuller aumentado (ADF)

El informe señaló, correctamente, que mezclar factores de tendencia (que asumen persistencia) con osciladores de reversión (RSI) sin saber cuál describe realmente a cada ticker es una asunción implícita, no una decisión basada en evidencia. `app/services/statistical_structure.py` (nuevo módulo) calcula:

- **Exponente de Hurst** (método de rango reescalado, R/S, implementado desde cero — es una regresión simple, no requiere tablas de valores críticos): H > 0.55 → estructura tendencial/persistente; H < 0.45 → reversión a la media/anti-persistente; en medio → indistinguible de un paseo aleatorio.
- **Test ADF** (Augmented Dickey-Fuller, vía `statsmodels` — este sí es una dependencia nueva, porque sus valores críticos vienen de tablas de superficie de respuesta que no vale la pena rederivar a mano, a diferencia del resto de modelos estadísticos de esta app, que están implementados desde cero): ¿la serie de precios (en logaritmos) tiene una raíz unitaria, o es genuinamente estacionaria?

Verificado con datos sintéticos (paseo aleatorio → H≈0.51; serie tendencial → H≈0.86; proceso de Ornstein-Uhlenbeck mean-reverting → H≈0.43) y con tickers reales (AAPL/KO ≈ paseo aleatorio, NVDA/VSXY con H>0.55 tras años de tendencia genuina — coherente con lo que cualquiera que haya visto esos gráficos esperaría).

Se usa como un único factor de precaución **asimétrico**: solo resta punto cuando la estructura es claramente de reversión (H<0.45) — no hay bonus simétrico por tendencia, porque eso volvería a contar la tendencia/fase ya puntuadas desde otro ángulo. No se ha incluido todavía en el estudio de ablación cruzado (calcularlo en cada punto histórico, para 217 tickers, es la misma razón de coste computacional que excluye a Markov/GARCH de ese estudio) — nivel de confianza 3 (teoría sólida, sin validación cruzada propia).

### 6.4 Trazabilidad de señales (el otro punto fuerte del informe)

Antes de esta auditoría, no existía ningún registro histórico de qué decía el sistema sobre un ticker en un momento dado — la definición operativa de "alerta fundamentada" que pedía el informe requiere precisamente eso. Ahora, cada llamada real a "Analizar activo" persiste una fila inmutable en `recommendation_snapshots` (ver `RecommendationSnapshotORM`): ticker, timestamp, precio, veredicto, puntuación, el desglose completo de qué factores dispararon y con cuántos puntos, el horizonte usado, y `ENGINE_VERSION` (una cadena que se sube a mano cada vez que cambia la lógica de puntuación, para poder atribuir un veredicto pasado a la versión exacta del motor que lo produjo). Consultable vía `GET /api/v1/market/tickers/{ticker}/history`.

Es deliberadamente mínimo: guarda el snapshot, no compara automáticamente contra lo que pasó después (eso necesitaría un job programado que esta arquitectura — sin *worker* en segundo plano, todo se sirve bajo demanda — no tiene todavía). La comparación "¿acertó?" es, por ahora, un ejercicio manual con los datos ya guardados, no una funcionalidad automática.

### 6.5 Transparencia de vela en curso

yfinance incluye la barra de "hoy" en cuanto abre la sesión, con un cierre que es en realidad el último precio negociado, no un cierre confirmado. No se optó por descartar los datos del día en curso (el informe lo plantea como "no evalúes sobre velas no cerradas") — un usuario que consulta a media sesión quiere el estado real ahora mismo, no el cierre de ayer. En su lugar, `is_intraday_snapshot` (booleano, en cada respuesta de "Analizar activo") indica cuándo la última barra corresponde a la fecha de hoy, para que quede claro que esas cifras concretas pueden moverse antes del cierre real.

### 6.6 Lo que se evaluó y se descartó, con motivo

- **Migrar de yfinance a un proveedor de pago** (Tiingo, EOD Historical Data): riesgo real y ya documentado (yfinance es un scraper no oficial), pero es una decisión de coste recurrente que le corresponde al usuario, no algo para decidir unilateralmente.
- **Hierarchical Risk Parity / construcción de cartera avanzada**: con una cartera personal de 7-10 posiciones, la ventaja sobre un enfoque simple es limitada — no justifica la complejidad todavía.
- **Meta-labeling con LightGBM/XGBoost** (López de Prado): el propio informe advierte que es zona de riesgo hasta tener resuelto lo anterior, y contradice la filosofía de sistema transparente y auditable que ya tiene este proyecto (un checklist de reglas explicable, no una caja negra entrenada). No se ha tocado.
- **Bolsa de Valores de Colombia**: excluida explícitamente por decisión del usuario — mercado pequeño, fuera del alcance actual (EE.UU. y Europa desarrollada).

## 7. Limitaciones honestas

- El estudio de ablación cubre un único periodo histórico (~10 años, terminando en agosto de 2026) y un único universo curado (~217 tickers de gran/mediana capitalización, mayoritariamente EE.UU. y Europa desarrollada). No garantiza que los mismos pesos sean óptimos en small caps, mercados emergentes, o el próximo ciclo de mercado.
- "Estadísticamente significativo" no es lo mismo que "económicamente explotable después de costes de transacción" - ningún resultado de este documento incluye comisiones, slippage ni impuestos.
- Los factores de Nivel 3 (volumen, fundamentales) tienen fundamento teórico pero **no** evidencia estadística propia todavía - se mantienen con peso pequeño precisamente por eso, no porque se hayan probado y fallado.
- ROE se investigó y se descartó deliberadamente como factor de puntuación: en la verificación empírica, Apple mostró un ROE de ~148% (distorsión por recompras masivas de acciones), lo que lo hace poco fiable como señal lineal sin normalización adicional que este sistema no implementa todavía.
