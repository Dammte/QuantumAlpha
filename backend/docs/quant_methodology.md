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

## 8. Motor de salida independiente y lectura multi-temporalidad (auditoría de reglas de salida, agosto 2026)

Tercera auditoría, esta vez centrada en un problema reportado directamente por el usuario: el sistema seguía diciendo "mantener" sobre posiciones que, mirando el gráfico, ya mostraban una ruptura técnica clara — a veces en varias temporalidades a la vez. El diagnóstico (12 fallos, D1-D12) se verificó línea a línea contra el código antes de tocar nada; todos se confirmaron exactos. Los tres más importantes, y lo que se hizo con cada uno:

**D1 — el sistema solo veía barras diarias.** No existía ninguna lectura semanal en todo el sistema, así que "el cruce bajista que ya se ve en semanal" era, literalmente, algo que el sistema no podía calcular. `technical_analysis.resample_ohlcv()` deriva semanal/mensual del histórico diario ya descargado — sin ninguna llamada de red adicional — y `multi_timeframe.py` (nuevo) produce una lectura jerárquica: la semanal fija el sesgo, la diaria fija el momento de ejecución, nunca se promedian. Un conflicto (semanal bajista + diaria alcista, o viceversa) puntúa *en contra* del setup, no de forma neutra — un rebote diario dentro de una tendencia semanal bajista es peor que una lectura sin tendencia clara, no igual.

**D2 — el cruce inminente (`detect_imminent_cross`) se calculaba, se pintaba en la UI, y no influía en ninguna decisión.** No se resolvió subiendo su peso dentro del checklist de compra (`recommendation_engine.py`) — cambiar esos pesos sin el estudio de ablación correspondiente habría violado la propia regla de este documento ("ningún factor nuevo entra sin evidencia medida"). Se resolvió en el sitio correcto: como disparador de urgencia dentro del motor de salida nuevo (`exit_engine.py`), con umbrales fijos por R² y sesiones, no como puntos sumados a un score.

**D3 — no existía ningún motor de salida. "Vender" se deducía de un checklist de compra.** Esta es la corrección central. `portfolio_risk_service.py` decidía la señal de una posición ya abierta a partir del mismo veredicto comprar/esperar/evitar que decide si *comprar* algo nuevo — así que un death cross confirmado con buen RS Rating y buenos fundamentales podía seguir puntuando positivo y salir "mantener". `exit_engine.py` es un motor independiente que responde a una pregunta distinta con evidencia distinta: **nunca importa `recommendation_engine.py` ni recibe RS Rating, crecimiento de ingresos, margen o el checklist de Minervini como parámetros** — no por convención, sino porque su firma (`evaluate_exit`) no tiene esos parámetros en absoluto. Un buen fundamental deja de poder compensar una ruptura técnica.

Devuelve uno de cinco niveles de urgencia, no un binario vender/mantener:

| Urgencia | Significado | Ejemplos de disparadores |
|---|---|---|
| `exit_now` | Salir en la próxima sesión | Precio perfora el stop vigente; 2 cierres consecutivos (o 1 con volumen >1,5x) bajo la SMA50 diaria con semanal ya no alcista; alineación bajista total semanal+diaria; death cross SMA20/50 confirmado (calidad ≠ "ruido") con precio bajo ambas medias; rotura del último soporte relevante |
| `reduce` | Recortar la posición, no cerrarla | Cruce inminente SMA50/200 con R²≥0,7 a ≤5 sesiones y diaria ya no alcista; divergencia bajista de OBV con RSI cayendo desde sobrecompra y volumen creciente; objetivo original alcanzado; extensión >4 ATR en una posición ya en beneficio |
| `tighten_stop` | Subir el stop, mantener | Cruce inminente SMA20/50 con R²≥0,6; ADX cayendo por debajo de 20 tras haber superado 25; vela envolvente bajista en un nivel de resistencia; posición en beneficio >1,5R |
| `watch` | Vigilar, sin acción | El resto de lo que ya vigilaba `portfolio_risk_service.py` (proximidad a soporte/resistencia, cruces proyectados por debajo del umbral de confianza para actuar) |
| `hold` | Tesis técnica intacta | Nada de lo anterior se disparó |

Los umbrales concretos (R², múltiplos de ATR, sesiones) son de primer trazo — el mismo estado en que arrancaron `BUY_THRESHOLD`/`AVOID_THRESHOLD` antes de su propia auditoría — y deben recalibrarse con evidencia (Fase 4/5 del plan) antes de tratarlos como definitivos.

**D4 — el sistema no sabía qué stop se propuso al comprar, ni cuándo se abrió la posición.** `stop_loss`/`take_profit` en `build_recommendation` solo se calculan cuando `verdict == "comprar"` — para una posición ya abierta con veredicto "esperar" o "evitar" (el caso que importa) siempre eran `None`. Nueva tabla `trade_plans` (migraciones `81f9cb75933e` y `49ac08cef677`) persiste el plan de cada posición: precio/fecha/cantidad de entrada, stop y objetivo iniciales, stop vigente (trailing, gestionado por `trade_manager.py` — ver más abajo), y una tesis en texto libre. Para una posición abierta antes de que esta tabla existiera, `trade_plan_service.py` la reconstruye de forma perezosa (en la primera evaluación de riesgo, no en el momento de la compra — así el endpoint de transacciones sigue siendo una escritura rápida, sin coste de red añadido) usando el histórico de precio **tal como estaba en la fecha de entrada real**, nunca con datos de hoy, y lo marca explícitamente como reconstruido en el campo `thesis` para que quede claro en la UI que no es el plan que se habría mostrado en el momento real de la compra.

**Integración**: `portfolio_risk_service.py` sigue calculando el veredicto de compra (para responder "¿añadiría más aquí?" — una pregunta distinta) pero la señal de la posición (`signal`, el campo que ya existía) ahora se corrige con la urgencia de salida cuando esta es más severa: `exit_now`/`reduce` fuerzan `exit_warning` incluso sobre una posición que puntuaría "comprar" hoy. Los campos nuevos (`exit_urgency`, `exit_reasons`, `trade_plan`, `r_multiple`, `multi_timeframe`, `scaled_exit`) se añadieron sin tocar `signal`/`score`/`reasons` existentes — los tests de integración que fijan ese contrato siguen pasando sin modificarlos.

**Nota sobre `ENGINE_VERSION`**: no se cambió en esta fase. `ENGINE_VERSION` traza cambios en la *puntuación* de `recommendation_engine.py` (qué factores entran y con qué peso), y ninguno de estos cambios toca esa puntuación — el único cambio dentro de `recommendation_engine.py` es extraer el cálculo de stop/objetivo a una función reutilizable (`compute_stop_and_target`), verificado como un refactor de comportamiento idéntico contra la suite de tests existente. `exit_engine.py`/`trade_manager.py` son motores nuevos e independientes, sin versión propia todavía; `trade_plans.engine_version` reutiliza `ENGINE_VERSION` únicamente como referencia de trazabilidad de cuándo se generó el plan.

### 8.1 Gestión activa del trade: stop dinámico, salidas escalonadas y stop temporal (Fase 3, agosto 2026)

Una vez existe un plan de trade persistido (D4), hace falta algo que lo gestione en el tiempo — `trade_manager.py`, funciones puras sin I/O, misma disciplina que `technical_analysis.py`.

- **Chandelier Exit** (Chuck LeBeau): `stop = máximo de los máximos de 22 barras − multiplicador × ATR(14)`. El multiplicador depende del régimen de volatilidad GARCH ya estimado por el sistema (`baja`→2,5, `normal`→3,0, `elevada`→3,25, `alta`→3,5 — más margen en regímenes genuinamente volátiles para no saltar por ruido, menos en calma para proteger con más precisión) y se estrecha a 2,0 en cuanto la posición supera +2R, priorizando proteger la ganancia ya conseguida sobre dar espacio a la operación. Valores de primer trazo, mismo estado que `BUY_THRESHOLD`/`AVOID_THRESHOLD` antes de su propia auditoría — pendientes de calibrar con evidencia.
- **El stop dinámico solo sube, nunca baja** — testeado explícitamente (`test_trailing_stop_never_lowers` y el resto de la suite de `trade_manager.py`), y calculado siempre sobre barras cerradas (nunca la vela en curso), así que no repinta durante la sesión. Se recalcula y persiste en cada evaluación fresca (no cacheada) de riesgo de cartera, **después** de que `exit_engine.evaluate_exit` ya haya juzgado el cierre de hoy contra el stop que ya estaba vigente — nunca contra un stop recién ensanchado con el propio dato de hoy, que habría sido una forma sutil de mirar al futuro.
- **Límite de riesgo por posición**: `max_shares_for_position_risk()` calcula cuántas acciones mantienen el riesgo (precio de entrada − stop) dentro de `MAX_POSITION_RISK_PCT` (1% del capital de la cartera) — la respuesta correcta a un stop demasiado ancho es reducir el tamaño de la posición, nunca ensanchar el stop. El límite **agregado** (6% sumando todas las posiciones a la vez) queda para la Fase 6 (`portfolio_construction_service.py`), que necesita visibilidad de todas las posiciones a la vez — esta función es deliberadamente solo el guardarraíl por posición.
- **Stop temporal**: una posición que lleva ≥20 sesiones cerradas sin alcanzar +1R (y sin haber saltado el stop) genera un `REDUCE` con motivo explícito de capital inmovilizado — implementado directamente en `exit_engine.py` (reutiliza `PositionContext.bars_held`, ya existente) en vez de un módulo aparte, para no tener dos sitios calculando "urgencia" por separado.
- **Salidas escalonadas**: `compute_scaled_exit_plan()` sugiere (nunca ejecuta) vender un tercio de la posición **original** al alcanzar +1R (con el stop sugerido a break-even) y otro tercio al alcanzar +2R, dejando correr el resto con el Chandelier Exit. Qué milestone falta se lee directamente de cuánta cantidad *sigue mantenida* respecto a `initial_quantity` (cantidad al crear el plan, nueva columna en `trade_plans`) — no de un flag separado de "ya sugerido", que podría desincronizarse de lo que realmente se vendió. La propia posición es la fuente de verdad de lo que ya se ha hecho.
- **D10, coste de rotación** (`opportunity_cost.py`): además de la corrección de escalas de D10 (ver más abajo, ya hecha), se añadió `MIN_EXPECTED_EDGE_AFTER_COSTS` (+1 punto extra sobre `SWAP_SCORE_MARGIN`) para que una rotación no se sugiera por una ventaja de score que no compensaría la comisión/spread/impacto fiscal reales de hacerla. Sin conversión calibrada de puntos a coste real todavía (necesitaría el mismo tipo de evidencia — Information Coefficient, regresión — que el resto del sistema exige antes de confiar en un número) — es un búfer conservador explícito, no una estimación medida, y se documenta como tal en el propio código.

### 8.2 Instrumentación: ¿de verdad funciona esto? (Fase 0, agosto 2026)

No se optimiza lo que no se puede medir. `RecommendationSnapshotORM` existía desde la primera auditoría (§6.4) pero solo se leía una a una, por ticker (`GET /market/tickers/{ticker}/history`) — nadie las agregaba en un hit rate o un retorno medio. Y no existía ningún registro histórico del `signal`/`exit_urgency` de una *posición* de cartera en absoluto, así que "¿cuántas veces el sistema dijo mantener y el activo cayó?" no se podía responder ni en principio.

- **`PositionSignalSnapshotORM`** (nueva tabla, migración `9a3b5080be2d`): una fila por evaluación de riesgo genuinamente fresca (no servida desde caché) de una posición — `signal`, `exit_urgency`, score, precio, R múltiplo. Se escribe desde `portfolio_risk_service.assess_position_risk()` cada vez que de verdad se recalcula (no en cada carga del dashboard que sirve desde caché).
- **Limitación honesta, dicha sin adornos**: esta tabla empieza vacía. La pregunta "¿cuántas veces el sistema dijo mantener y el activo cayó?" solo se puede responder **hacia adelante** desde que se desplegó este cambio, nunca retroactivamente — ese dato nunca se guardó antes. Lo que sí hay desde ya es el historial completo de veredictos de compra (`comprar`/`esperar`/`evitar`) vía `RecommendationSnapshotORM`, que `signal_performance_service.py` sí puede agregar con datos históricos reales.
- **`signal_performance_service.py`**: funciones puras (`compute_verdict_outcomes`, `compute_signal_outcomes`, `find_false_negatives`, `forward_return`) que toman snapshots + una serie de precio y devuelven, por veredicto/señal y horizonte (5/10/21/63 sesiones): tamaño de muestra, hit rate (fracción con retorno futuro positivo — una lectura neutra, no "acertó", para no ocultar información volteando el criterio según la categoría), retorno medio y mediano. `find_false_negatives` lista, por ticker y fecha, cada `hold` seguido de una caída >5% en 10 sesiones — no solo los cuenta, los nombra. La única función de orquestación (`build_signal_performance_report`) hace una sola llamada batched a `get_bulk_ohlcv` por cada ticker distinto que aparece en el historial, nunca una llamada por ticker.
- **`GET /api/v1/system/signal-performance`**: expone el informe completo. Vista "Rendimiento del sistema" en el frontend pendiente (Fase 7).
- **`scripts/decision_journal_export.py`**: vuelca cada veredicto/señal de posición con su retorno futuro realizado a un CSV, usando exactamente `signal_performance_service.forward_return` (nunca una segunda implementación que pudiera discrepar silenciosamente) — para auditar el historial completo fuera de la app.

### 8.3 Backtest honesto con triple-barrera (Fase 4, agosto 2026)

**D7 — el backtest walk-forward no probaba la estrategia que el sistema recomienda.** `walk_forward_backtest.py:295` mide `close[i+h]/close[i] - 1`: comprar y mantener `h` sesiones pase lo que pase, ignorando por completo el `stop_loss`/`take_profit` que `build_recommendation` propone en esa misma barra. Validaba una estrategia que nadie ejecuta.

**`backtest_engine.py`** (nuevo, `walk_forward_backtest.py` se mantiene tal cual — sigue siendo lo que usa `compute_core_signals()` para el backtest por ticker que se muestra en "Analizar activo" a sus propios horizontes 1m/3m/6m; son motores separados y aditivos, no uno sustituye al otro todavía):

- **Etiquetado de triple barrera** (López de Prado, *Advances in Financial Machine Learning*, cap. 3): para cada señal "comprar" replayada con `walk_forward_backtest.replay_recommendation_at` (la misma función, ahora devuelve la `Recommendation` completa, no solo el veredicto — ver más abajo), se simula la operación barra a barra hasta que se toca el stop, el objetivo, o el horizonte máximo (10 y 21 sesiones — el horizonte real de esta cartera, no 63/126). Usa `high`/`low` intrabarra, no solo el cierre — un backtest que solo mira cierres subestima sistemáticamente los stops saltados. Si ambas barreras se tocan en la misma barra, gana el stop (convención conservadora, evita el sesgo optimista más común en backtests caseros).
- **El trailing stop de la Fase 3 se simula de verdad**, no solo el stop fijo: `label_triple_barrier(..., trailing=True)` corre `trade_manager.chandelier_multiplier`/`update_trailing_stop` — las mismas funciones que gestionan una posición real — barra a barra. El stop de cada barra se comprueba contra el nivel vigente *antes* de que la propia barra lo actualice (el mismo orden causal ya establecido en la integración en vivo de `portfolio_risk_service.py`) — actualizar primero y comprobar después habría sido una forma sutil de mirar al futuro dentro de una sola barra.
- **`replay_recommendation_at`**: `walk_forward_backtest._replay_verdict_at` (privada) ahora es un envoltorio fino sobre esta nueva función pública, que devuelve la `Recommendation` completa — verificado como cambio de comportamiento idéntico contra los tests existentes. Es la única forma en que `backtest_engine.py` sabe qué habría propuesto el sistema en cada punto; nunca hay una segunda reconstrucción del veredicto que pudiera discrepar.
- **Métricas de trading** (`compute_trading_metrics`), no solo un retorno medio: expectancy, profit factor, win rate, MAE/MFE (la métrica que dice si los stops están demasiado ceñidos o demasiado holgados y que el sistema no tenía), drawdown máximo de la curva de equity encadenando los retornos de cada operación, duración media de ganadoras vs perdedoras, y retorno neto de costes (`ROUND_TRIP_COST_PCT`, la misma cifra conservadora ya documentada en `opportunity_cost.py` — §7 ya admitía que ningún resultado del proyecto incluía comisiones; ahora todo backtest de este motor reporta bruto y neto).
- **Validación fuera de muestra**: `split_by_date()` (partición temporal simple, para el 2016-2022 vs 2023-2026 que pide el encargo) y `purged_kfold_splits()` (K-Fold con purga y embargo, López de Prado cap. 7 — descarta del entrenamiento una banda de muestras alrededor de cada fold de test para que la ventana de la barrera vertical de una etiqueta no se filtre al fold vecino).
- **Benchmarks honestos**: `buy_and_hold_labels()` (comprar y mantener sobre los mismos puntos de entrada, sin stop ni objetivo) y `random_entry_labels()` (entradas en barras aleatorias, con el mismo dimensionamiento ATR que una señal real — si la estrategia no supera esto, el momento de entrada no aporta ventaja alguna sobre elegir días al azar).
- **Limitación heredada, ya documentada**: una muestra por ticker sigue siendo fina (la misma razón por la que existe el estudio de ablación cruzado a escala de universo — ver §3). `n_signals_evaluated` se reporta siempre explícitamente, nunca oculto tras un resultado vacío por debajo de algún umbral arbitrario de tamaño de muestra.
- **Pendiente, explícitamente permitido por el propio encargo**: este motor todavía no sustituye a `walk_forward_backtest.py` en la vista "Analizar activo" ni en la watchlist premium — es la base que usará la reescritura de `scripts/factor_ablation_study.py` (Fase 5) y la futura vista "Rendimiento del sistema" (Fase 7). Migrar los consumidores en vivo es un paso de integración posterior, no una carencia de corrección de este módulo.
