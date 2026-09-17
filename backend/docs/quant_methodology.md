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
| `exit_now` | Salir en la próxima sesión | Precio perfora el stop vigente; 2 cierres consecutivos (o 1 con volumen >1,5x) bajo la SMA50 diaria con semanal ya no alcista; alineación bajista total semanal+diaria; death cross SMA21/50 confirmado (calidad ≠ "ruido") con precio bajo ambas medias; rotura del último soporte relevante |
| `reduce` | Recortar la posición, no cerrarla | Cruce inminente SMA50/200 con R²≥0,7 a ≤5 sesiones y diaria ya no alcista; divergencia bajista de OBV con RSI cayendo desde sobrecompra y volumen creciente; objetivo original alcanzado; extensión >4 ATR en una posición ya en beneficio; **cierre confirmado bajo la SMA21 diaria, el día de la rotura (§10)** |
| `tighten_stop` | Subir el stop, mantener | Cruce inminente SMA21/50 con R²≥0,6; ADX cayendo por debajo de 20 tras haber superado 25; vela envolvente bajista en un nivel de resistencia; posición en beneficio >1,5R; **cruce bajista de MACD confirmado en diario (§10)** |
| `watch` | Vigilar, sin acción | El resto de lo que ya vigilaba `portfolio_risk_service.py` (proximidad a soporte/resistencia, cruces proyectados por debajo del umbral de confianza para actuar); **precio ya lleva varias sesiones bajo la SMA21 sin que sea la rotura del día (§10)** |
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

**Actualización (septiembre 2026, reconstrucción de niveles/triggers, Fase 4)**: `walk_forward_backtest.py` se retira por completo — no queda ningún consumidor real (la vista "Analizar activo" ya usaba `triple_barrier_backtest` desde antes de esta ronda; el único uso que quedaba era el propio `find_triple_barrier_entries` de este módulo). `replay_recommendation_at` (rebuild point-in-time del veredicto/checklist antiguo) se sustituye por `levels_engine.replay_gate_at` — mismo contrato, mismas dos simplificaciones documentadas (sin RS Rating ni soporte/resistencia point-in-time), pero reconstruye el nuevo gate booleano en vez del checklist ponderado retirado. `_permutation_test` (rutina estadística genérica, sin dependencia real del checklist) se movió sin cambios a `scripts/factor_ablation_study.py`, su único otro consumidor. El backtest de triple-barrera valida ahora las entradas que el gate propondría, no las que proponía el checklist — coherente con que la propia "Recomendación" en vivo pasa a ser el gate en esta misma Fase.

### 8.4 Reescritura del estudio de ablación (Fase 5, agosto 2026) — script listo, todavía no ejecutado contra datos reales

`scripts/factor_ablation_study.py` reescrito para corregir cuatro huecos metodológicos reales del estudio original (los mismos que motivaron D8):

1. **Etiquetado de triple-barrera**, no retorno a horizonte fijo — cada muestra usa `backtest_engine.label_triple_barrier` con trailing stop real, dimensionado con las mismas constantes (`ATR_STOP_MULTIPLE`, `REWARD_RISK_RATIO`) que usa el propio sistema en vivo, aplicadas de forma uniforme a toda muestra sin importar qué factor se esté probando.
2. **Demediado transversal**: cada retorno se expresa relativo a la media de su propio cubo de mes calendario antes de calcular cualquier estadístico — esto es exactamente lo que faltaba para que `vix_stress` no siguiera midiendo beta de mercado (+4,35 p.p. en el estudio original, §6.1) disfrazado de señal.
3. **Regresión multivariante** (`run_multivariate_regression`, todos los factores como predictores simultáneos vía OLS) junto a la prueba univariante — la propia comprobación con datos sintéticos durante el desarrollo mostró exactamente el patrón que se esperaba: varios factores colineales (tendencia, fase, ADX) pierden toda su significación en conjunto salvo el que de verdad aporta información independiente.
4. **Information Coefficient** (`compute_information_coefficient`): correlación de Spearman entre el disparador de cada factor y el retorno demediado, calculada por cubo de mes y promediada (IC medio) con su propia dispersión (IC/std(IC) = "IC IR") — la métrica estándar de la industria, más informativa que un único test agregado porque muestra si un factor es consistente mes a mes o lo sostiene un par de periodos con suerte.

Además: segmentación por régimen (`segment_by_regime`, mercado sobre/bajo su SMA200 y VIX en calma/estrés — reejecuta el mismo análisis univariante por separado en cada segmento) y horizontes por defecto ahora en 5/10/21 sesiones (el horizonte real de esta cartera), con 63/126 todavía disponibles vía `--horizons` para comprobar sensibilidad al horizonte frente a la literatura de momentum.

**Validado con datos sintéticos** (no contra el universo real — ver más abajo): un universo de 25 tickers simulados con regímenes alcistas/bajistas alternos confirma que el pipeline completo corre sin errores de extremo a extremo — demediado, prueba univariante, regresión multivariante, IC, segmentación por régimen — y que el caso límite de "ningún factor con muestra suficiente" (encontrado durante esta misma validación, con un universo sintético demasiado pequeño) ya no hace fallar el script sino que devuelve un informe vacío pero bien formado.

**Deliberadamente NO hecho en esta fase**: no se ha ejecutado el script contra los ~217 tickers reales (tarda varios minutos contra la red y el propio encargo pide no precipitar ningún cambio de peso sin revisión humana de un resultado recién generado), y no se ha tocado ningún peso en `recommendation_engine.py`, ni implementado el umbral por percentil transversal de D9 (es un cambio de arquitectura en vivo — "en qué percentil está hoy el universo" necesita infraestructura de la que este script offline no dispone — y recalibrar el umbral de decisión es exactamente el tipo de paso que necesita una persona revisando un resultado real, no una acción automática). El propio script documenta, en su docstring, las reglas para cuando alguien sí recalibre a partir de sus resultados: ningún cambio de peso sin significación BH-ajustada fuera de muestra, ningún signo invertido sin evidencia fuerte en varios regímenes, pesos en enteros pequeños y redondos, y comparar siempre el coeficiente multivariante contra el univariante antes de confiar en un factor.

### 8.5 Riesgo a nivel de cartera (Fase 6, agosto 2026)

**D12 — riesgo de cartera y decisión estaban desconectados.** Cada posición se juzgaba como si la cartera no existiera: dos semiconductores con correlación 0,9 contaban como dos apuestas independientes, no había límite de concentración sectorial, ni una cifra del riesgo agregado real (cuánto dinero se pierde si todos los stops saltan a la vez), ni distinción entre el peso de capital de una posición y su verdadera contribución al riesgo.

`portfolio_construction_service.py` (nuevo, funciones puras sin I/O, misma disciplina que `technical_analysis.py`) — todavía no conectado a un endpoint en vivo, igual que `backtest_engine.py` en la Fase 4, por la misma razón: es la base, la integración en el dashboard es un paso posterior:

- **Matriz de correlación** (`compute_correlation_matrix`, 60 sesiones) y **`find_correlated_pairs`**: alerta cuando dos posiciones superan 0,8 de correlación — son una sola apuesta con dos nombres, no dos.
- **Concentración sectorial** (`compute_sector_concentration`, reutiliza `market_universe.sector_of` — sin mapeo propio nuevo) y **`flag_concentrated_sectors`**: alerta por encima del 30% del capital en un mismo sector (umbral de primer trazo, sin calibrar todavía).
- **Contribución al riesgo por posición** (`compute_risk_contributions`), no solo peso por capital: descomposición estándar RC_i = w_i·(Σw)_i / σ_p (teorema de Euler sobre la varianza de cartera, un resultado ya establecido, no una fórmula propia) — verificado con un caso analíticamente limpio: dos posiciones sin correlación, mismo peso de capital, pero una con el triple de volatilidad, contribuye más al riesgo que su peso de capital sugeriría.
- **Volatilidad realizada de la cartera** (`compute_portfolio_volatility`) frente a un objetivo del 15% anualizado (primer trazo, sin calibrar), y `suggest_volatility_reduction()` para señalar qué posiciones recortar primero cuando se excede — las que más contribuyen al riesgo, no necesariamente las de mayor peso de capital. Verificado con dos casos límite analíticamente exactos: dos posiciones idénticas (correlación 1) no dan ninguna diversificación (la volatilidad combinada iguala la de una sola), y dos posiciones exactamente opuestas dan volatilidad cero (cobertura perfecta).
- **Riesgo agregado en R** (`compute_aggregate_risk`): suma de (precio − stop) × tamaño de todas las posiciones — el dinero real que se pierde si todo salta a la vez, con límite del 6% del capital (frente al 1% por posición ya en `trade_manager.py`). Una posición ya cotizando bajo su propio stop contribuye 0, no un número negativo — esa pérdida es problema de `exit_engine.py` (ya debería estar marcada `exit_now`), no algo que compense el riesgo del resto.
- **Tamaño final de posición** (`final_position_size`): el tamaño final de una posición nueva es el más estricto entre el tamaño de riesgo fijo (`trade_geometry.py`, ver Fase 3/§10 — sustituye lo que hasta 2026-09 sugería `kelly_criterion.py`, retirado por falta de evidencia cross-sectional), el límite de riesgo por posición (`trade_manager.max_shares_for_position_risk`) y el límite de concentración sectorial — nunca más ancho que cualquiera de los tres.

## 9. Frontend: haciendo visible lo que ya calculaba el backend (Fase 7, agosto 2026)

Las Fases 0–6 añadieron `exit_urgency`, `trade_plan`, `r_multiple`, `multi_timeframe`, `scaled_exit` y (esta fase) `bars_held` a `PositionRiskResponse` — pero hasta ahora nada en el dashboard los leía: la tabla de posiciones seguía mostrando solo `signal`/`score`, el mismo contrato de antes de toda esta auditoría. Esta fase no cambia ninguna lógica de decisión (ningún D-número nuevo, `ENGINE_VERSION` no cambia) — es exclusivamente hacer visible lo que el motor de salida ya decidía.

- **Panel "Acciones requeridas hoy"** (`TodayActionsPanel.jsx`, parte superior del dashboard): cada posición cuyo `exit_urgency` es `exit_now`, `reduce` o `tighten_stop` — nunca `watch`, que es un estado de vigilancia pasiva ya visible en la insignia de la tabla, no una acción — ordenada por severidad exacta (`exit_engine._URGENCY_SEVERITY`), con todos los motivos que dispararon esa lectura, no solo el titular. Este es el fix visible del bug original: un death cross semanal con RS Rating 85 ya no puede quedar enterrado en un "mantener" de la tabla — aparece aquí, arriba de todo, con el motivo exacto.
- **Semáforo multi-temporalidad** (`MultiTimeframeSemaphore.jsx`): dos puntos (semanal, diario) coloreados por tendencia, con el detalle completo (fase, cruces, precio vs SMA50) en el tooltip — versión compacta inline en la tabla, versión completa (con `alignment` y `conflicts` en texto) en la ficha de posición. Hace visible el propio D1: por primera vez se puede ver, sin abrir nada, que la semanal ya está bajista aunque la diaria todavía no lo confirme.
- **Ficha de posición** (`PositionDetailPanel.jsx`, fila expandible en la tabla): precio y fecha de entrada, stop vigente frente al inicial, objetivo, R actual, sesiones mantenidas (`bars_held`, expuesto en esta fase — antes se calculaba en `assess_position_risk` y se descartaba), distancia al stop en % y en múltiplos de ATR, la tesis del plan (marcada honestamente cuando es una reconstrucción retroactiva, ver `trade_plan_service.RECONSTRUCTED_THESIS`), la sugerencia de salida escalonada si aplica, y todos los `exit_reasons` sin filtrar por nivel.
- **Vista "Rendimiento del sistema"** (`SystemPerformanceView.jsx`, sección propia en la barra lateral, consume `GET /api/v1/system/signal-performance`): hit rate y retorno medio/mediano por veredicto y por señal de posición a cada horizonte, y la lista nominal de falsos negativos ("mantener" seguido de una caída real). Traslada tal cual la limitación honesta del §8.2: la tabla por señal de posición solo cubre lo ocurrido desde que `PositionSignalSnapshotORM` empezó a grabar — no hay manera de reconstruir hacia atrás algo que nunca se guardó, y la vista lo dice explícitamente en vez de fingir un histórico que no existe.
- **Separación señal confirmada / sesión en curso**: no se construyó infraestructura nueva para esto — el motor de salida ya opera exclusivamente sobre `ta.closed_bars(df)` (ninguna lectura de `exit_urgency`/`trade_plan` usa la vela del día en curso), así que la distinción ya existe por construcción. Lo único añadido es la nota explícita en la ficha de posición ("basado en el cierre de la última sesión confirmada"), con el mismo tono que la insignia "● Sesión en curso" que "Analizar activo" ya usa para `is_intraday_snapshot`.

**Limitación honesta.** `backtest_engine.py` (Fase 4) y `portfolio_construction_service.py` (Fase 6) siguen sin un endpoint que los sirva — esta fase no los conecta, porque hacerlo bien (mostrar un backtest de triple-barrera junto al walk-forward actual, o un panel de riesgo agregado de cartera) es una superficie de UI nueva por derecho propio, no una extensión de un campo ya existente en `PositionRiskResponse` como el resto de esta fase.

## 10. Afinado del motor de salida con la cartera real en producción (agosto 2026, `ENGINE_VERSION` → v3)

Con el dashboard ya en marcha (Fase 7), una revisión contra la cartera real encontró un caso concreto: CRWD aparecía como `add_candidate` ("Aumentar") con puntuación 8, mientras el propietario, mirando el gráfico esa misma tarde, veía que el precio ya había roto claramente su media móvil rápida tras una caída fuerte de la sesión. Dos causas reales, no una — verificadas con los datos crudos de yfinance antes de tocar ningún umbral, no asumidas:

**Causa 1 — `closed_bars()` excluía la sesión de hoy incluso horas después del cierre real.** El corte era puramente por fecha de calendario (`última_barra.date() >= hoy`), heredado de la corrección D6 original. A las 21:56 UTC (~2 horas después del cierre real de EE.UU. a las 20:00 UTC en horario de verano), la barra de hoy ya era un cierre asentado, pero seguía descartándose hasta la medianoche UTC — un apagón de varias horas cada tarde/noche sobre toda señal discreta (cruces, precio-vs-media, patrones), justo la ventana en la que alguien revisando su cartera por la tarde más probablemente mira el dashboard. `closed_bars()` ahora compara contra `CLOSED_BAR_CUTOFF_UTC` (21:30 UTC, el cierre más tardío entre EE.UU. y Europa bajo cualquier horario de verano/invierno, con margen de asentamiento) en vez de solo la fecha — una barra de hoy se trata como cerrada en cuanto el mercado ya cerró de verdad, no al día siguiente. `now: datetime` sustituye a `today: date` en su firma (inyectable en tests).

**Causa 2 (la de fondo) — el motor de salida no tenía ningún disparador para "el precio acaba de romper su propia media rápida".** El propietario opera a corto/medio plazo con SMA21/50 en diario/horario, más RSI y MACD — así lo confirmó explícitamente. Verificado con los datos reales de CRWD (yfinance, cierre a cierre): la EMA21 seguía muy por encima de la EMA50 (diferencia ~14,7, sin cruce de medias entre sí todavía) — **no había ocurrido un cruce SMA21/50 real**, así que ampliar la detección de cruces no habría bastado. Lo que sí había ocurrido: el precio cerró por debajo de su propia SMA21 por primera vez en semanas, tras una caída de un día. `TimeframeRead.price_vs_sma20` ya calculaba exactamente este dato (comparación precio-vs-media rápida) desde la Fase 1, pero `exit_engine.py` nunca lo usaba en ningún disparador — ni `daily.macd_cross`, tampoco usado pese a estar ya calculado. Dos campos ya computados, cero disparadores conectados a ellos.

**Cambios:**

- **`multi_timeframe.FAST_MA_PERIOD = 21`** sustituye el `20` que usaba internamente `_read_timeframe()` para el par rápido. Cambio deliberadamente acotado: solo afecta el cálculo interno y propio de `multi_timeframe.py` (que ya duplicaba su propio cálculo de SMAs, independiente del `sma20`/`CoreSignalsResponse.sma20` que usa "Analizar activo" para gráficos/Bollinger/estadísticas generales — ese sigue en 20, el estándar de facto para Bollinger, sin tocar). Los nombres de campo (`ma_cross_20_50`, `cross_quality_20_50`, `imminent_cross_20_50`, `price_vs_sma20`) se mantienen por estabilidad de esquema/API — cada uno documentado en el propio código como "en realidad 21, no 20" en vez de renombrar 16 archivos por una cuestión de qué entero usar. Los textos que sí llegan al usuario (mensajes de `exit_engine.py`, esta tabla) dicen "SMA21/50", no "SMA20/50" — el otro camino independiente que sigue en 20 (`ticker_analysis_service.imminent_cross_short_term`, el badge de cruce inminente en "Analizar activo") sigue diciendo honestamente "SMA20/SMA50", porque genuinamente sigue siendo 20 ahí.
- **`consecutive_closes_below_daily_sma_fast`** (nuevo parámetro de `evaluate_exit`, mismo patrón que `consecutive_closes_below_daily_sma50` ya existente): `portfolio_risk_service.py` lo calcula con `ta.consecutive_closes_below(closed["close"], ta.sma(closed["close"], mtf.FAST_MA_PERIOD))`, reutilizando la función ya probada, no una nueva.
  - **`reduce`** cuando `== 1` (la rotura ocurre hoy, no una ya conocida) — deliberadamente en `reduce`, no en `tighten_stop`: es uno de los dos niveles que `portfolio_risk_service.py` deja sobrescribir un `add_candidate` a `exit_warning` (ver su comentario de precedencia), que es exactamente lo que hacía falta para que CRWD dejara de mostrar "Aumentar" el día de la rotura.
  - **`watch`** cuando `> 1` (ya se avisó, sigue por debajo, sin volver a escalar cada sesión).
- **Cruce bajista de MACD confirmado en diario** (`daily.macd_cross == "bearish"`) → `tighten_stop`. Mismo nivel que "ADX cayendo" y por el mismo motivo: un cruce de MACD aislado es propenso a whipsaw en mercados laterales, así que sube el stop y mantiene alerta en vez de forzar una salida.
- RSI ya estaba parcialmente cubierto (la regla `reduce` de divergencia OBV + RSI cayendo desde sobrecompra, Fase 2) — no se añadió un disparador de RSI aislado en esta pasada; queda como posible refinamiento futuro si la evidencia lo pide.
- **No se tocó el lado de compra.** "Similar para activos que tengan un cruce hacia arriba" (petición del propietario) no se implementó como un nuevo factor puntuado en `recommendation_engine.py` — habría violado la propia regla de este documento ("ningún peso nuevo sin el estudio de ablación correspondiente"), y ese estudio sigue sin ejecutarse contra el universo real por decisión explícita del propietario (§8.4). El checklist de compra ya premia parcialmente esto vía sus factores existentes de tendencia/fase/RS Rating.
- **Intradía/horario**: el propietario menciona operar también en gráficas de horas. Sigue sin construirse (Fase 1.2, deliberadamente mínima desde el plan original) — es una pieza de trabajo bastante mayor (parámetro `interval` en `MarketDataProvider`, un flujo de datos nuevo, el campo `intraday` de `MultiTimeframeRead` sigue siempre en `None`) y no estaba evidenciada por el caso concreto de CRWD (que era 100% diario), así que se deja fuera de esta pasada en vez de construirse a medias bajo presión de tiempo.

**`ENGINE_VERSION` → `"2026-08-audit-v3"`.** No cambió ningún factor/peso de `recommendation_engine.py` — se bumpea porque `TradePlan.engine_version`/`PositionSignalSnapshotORM.engine_version` usan esta misma constante como marca general de "qué versión de la lógica de decisión produjo esto" (compra *y* salida, ver Fase 0), y el conjunto de disparadores de `exit_engine.py` sí cambió materialmente.

**Verificado contra la cartera real** (no solo tests sintéticos): tras el fix, con la caché forzada a recalcular (`?refresh=true`), AVGO pasó a `exit_now` (precio ya perforando su stop reconstruido, -2,37R) y PANW/V a `reduce` (objetivo alcanzado, salida escalonada sugerida) - los tres visibles de inmediato en el panel "Acciones requeridas hoy" en vez de enterrados en la tabla.

## 11. Cierre de cabos sueltos del encargo original (agosto 2026)

Una revisión explícita de lo prometido en el plan original encontró tres puntos que seguían pendientes o incompletos - los tres cerrados en esta pasada, sin esperar a que se repitiera el patrón de "lo encuentra el propietario mirando el gráfico":

- **Badge de cruce inminente, de decorativo a elemento de primera clase.** `ImminentCrossBadge.jsx` mostraba dirección y sesiones, pero no el R² ni si esa lectura ya bastaría para que el motor de salida actuase - lo pedía explícitamente la Fase 7 original. Ahora muestra el R² siempre, y compara contra el mismo umbral que usaría `exit_engine.py` para esa misma señal (`IMMINENT_CROSS_50_200_MIN_R2`/`IMMINENT_CROSS_20_50_MIN_R2`, reflejados en el frontend con el mismo comentario "mirrors X" que ya usa `RecommendationCard.jsx` para `BUY_THRESHOLD`/`AVOID_THRESHOLD`) - honestamente, sin insinuar un efecto sobre la puntuación de compra que no existe (D2: esta señal nunca puntúa el lado de compra, solo dispara el motor de salida en una posición ya abierta).
- **Test explícito de "una señal en una vela no cerrada nunca se confirma", de punta a punta.** Existía a nivel de `closed_bars()` en aislamiento, pero `analyze_multi_timeframe()` - el camino real que usa `exit_engine.py` - nunca se probó con una barra fechada *hoy*, porque los tests existentes usan fechas sintéticas fijas en el pasado que nunca activan esa rama. Se añadió `now: datetime | None` a `analyze_multi_timeframe`/`_read_timeframe` (mismo patrón de inyección que ya tenía `closed_bars`) y un test que construye una ruptura de la SMA rápida que solo existe si se incluye la barra de hoy, verificando ambos lados del corte horario.
- **`signals_confirmed`/`live_snapshot` como campos separados en el esquema**: se planeó así originalmente, pero no se implementó como una separación estructural nueva - habría significado reestructurar `CoreSignalsResponse` (usado por "Analizar activo", Premium Watchlist, riesgo de cartera y sugerencias de intercambio a la vez), un cambio de alto impacto por una ganancia puramente organizativa, cuando la garantía real (ninguna señal discreta usa jamás una vela sin cerrar) ya existe por construcción vía `closed_bars()` en todo el sistema, y ahora también está probada de punta a punta (punto anterior). Decisión: no añadir el campo: verificar y documentar la garantía en su lugar.

**Un cabo que se revisó y se decidió NO cerrar, con motivo explícito**: el plan original (punto 2.10) daba por hecho que añadir un parámetro `interval` a `MarketDataProvider`/`YFinanceProvider` sería "un cambio de una línea, sin romper la interfaz". Al revisarlo para implementarlo, `PriceBar.trade_date` resultó ser un `date`, no un `datetime` - cualquier valor de `interval` distinto de `"1d"` (p. ej. `"1h"`) colapsaría varias barras del mismo día en la misma fecha de forma silenciosa, exactamente el tipo de dato corrompido sin avisar que este proyecto se prohíbe explícitamente. Añadir el parámetro tal cual se planeó habría sido peor que no añadirlo - un parámetro que aparenta funcionar y no funciona. Queda sin añadir hasta que el intradía real se construya (siempre condicionado, según el propio plan, a que se pida explícitamente) y venga acompañado del cambio de `trade_date` que de verdad hace falta.

## 12. El estudio de ablación, ejecutado de verdad contra el universo completo (agosto 2026)

Con el script ya reescrito y validado sintéticamente en la Fase 5 (§8.4), se ejecutó por fin contra los ~216 tickers reales (US+Europa), 10 años de histórico diario, en los horizontes reales de esta cartera (5/10/21 sesiones) y, como comprobación adicional, también a 63/126 sesiones (la escala de momentum clásica, para poner a prueba directamente la hipótesis de "se corrige a horizontes largos" que dejó abierta el estudio original). 24 CSV quedan en `backend/docs/factor_ablation_report_v2_h*.csv` (el informe agrupado y cada segmento de régimen) como registro de evidencia, junto a los `factor_ablation_report_h{21,63,126}.csv` del estudio original (metodología distinta, sin demediado transversal - se conservan como historial, no se sobrescriben).

**El hallazgo más importante, y el más delicado: `trend_up`/`trend_down`/`stage2`/`stage4`/`adx_strong_trend` (la familia de "tendencia confirmada", el núcleo del checklist) miden con el signo contrario al que puntúan hoy - de forma significativa (BH-ajustado), consistente en el coeficiente multivariante (no es solo colinealidad con otro factor) y consistente en los cinco horizontes probados (5/10/21/63/126 sesiones) y en cada segmento de régimen (mercado sobre/bajo su SMA200, VIX en calma/estrés) donde hay muestra suficiente para medir.** Con la metodología original (sin demediar, sin multivariante) ya se había visto este mismo patrón a 21 días y se documentó como reversión a corto plazo (Jegadeesh 1990) que se esperaba corregir hacia el momentum esperado (Jegadeesh & Titman 1993) a 63/126 días, sin llegar aún a significación en esa muestra más pequeña. **Esta vez, con demediado transversal y control multivariante, el patrón no se corrige a 63/126 días - se mantiene, y en `trend_up` incluso se vuelve más fuerte (diferencia de -0,65 p.p. a 63 sesiones, -1,30 p.p. a 126, ambas significativas tras BH, coeficiente multivariante también negativo y significativo: -0,60 p.p. p=0,005 a 63 sesiones; -1,02 p.p. p=0,0004 a 126).**

**Por qué no se traduce esto en un cambio de peso, a pesar de superar la propia barra de "evidencia fuerte y consistente en varios regímenes"**: esto es, casi con seguridad, el mismo fenómeno que ya documentó §6.1 con `market_below_sma200`/`vix_stress` - "comprar el pánico" dentro de un universo *ya curado de calidad* durante una década dominada por un mercado alcista secular no es lo mismo que "la tendencia no sirve como señal de entrada" en general. Un nombre de calidad que está temporalmente por debajo de su tendencia en este universo concreto tiene, en la práctica, más pinta de "oportunidad de compra en la caída" que de "empresa que se deteriora" - exactamente la misma distinción, a otra escala, que ya se hizo con el filtro de régimen. §6.1 decidió, ante un hallazgo de la misma forma (aunque de menor calado): ni mantener el signo original ni invertirlo - retirar el factor de la puntuación, porque "una sola muestra de 10 años dominada por un par de recuperaciones en V es poca base para apostar al contrario con dinero real". Esa razón se aplica aquí con más fuerza todavía, no menos: `trend_up`/`trend_down`/Fase de Weinstein no son un factor más del checklist, son su columna vertebral desde el §1 de este mismo documento - invertirlos, o incluso retirarlos, no sería "recalibrar un peso" en el sentido que cubren las reglas anti-sobreajuste de este proyecto, sería cambiar la filosofía de inversión completa de seguimiento de tendencia a reversión a la media. Es exactamente el tipo de decisión que el propio script pide no automatizar ("producir evidencia es un paso, actuar sobre ella es un paso deliberado y separado") y que, dado lo que está en juego, le corresponde al propietario revisar y decidir explícitamente - no algo para asumir en su nombre bajo un mandato general de "hazlo si hace falta". **Ningún peso de esta familia se ha tocado en esta pasada.**

**El resto de factores, revisados uno a uno** (univariante + multivariante + IC, en los cinco horizontes):

- **`golden_cross`/`death_cross`**: siguen sin significación BH en ningún horizonte (segunda confirmación independiente, tras el estudio original) - ni el coeficiente multivariante llega nunca a ser significativo. Se dejan sin tocar por la misma razón que el resto de la familia de tendencia (comparten el mismo posible sesgo de "universo de calidad"), no porque falte evidencia de que no funcionan.
- **`rsi_oversold_bounce` (+1): validación real, no solo ausencia de contradicción.** A 5 sesiones - el horizonte donde un rebote desde sobreventa debería jugarse y agotarse - el efecto es significativo, con el signo correcto, y sigue siendo significativo en el coeficiente multivariante (+0,53 p.p., p<0,001) una vez controlado por el resto de factores: no es colinealidad con la familia de tendencia, es una señal propia. Se desvanece a 10-21 sesiones exactamente como cabría esperar de un rebote táctico corto. Sin cambios - el peso actual ya está bien calibrado.
- **`atr_parabolic` (-2)**: signo correcto y significativo a 5 sesiones (una extensión parabólica corrige a corto plazo, coherente con la literatura de sobre-extensión), se desvanece a horizontes más largos según se diluye en colinealidad con ADX/tendencia. Comportamiento esperado, sin cambios.
- **`stage4` (-3)**: el coeficiente multivariante (una vez controlada la colinealidad con el resto de la familia de tendencia) es negativo y significativo a 5 y 10 sesiones - coincide con el signo actual, aunque el univariante bruto (contaminado por el mismo efecto que `trend_up`) diga lo contrario. Una confirmación tranquilizadora, no una razón para cambiar nada.
- **`minervini_range_position`/`rsi_overbought_outside_strong_trend`/`obv_bullish`/`obv_bearish`**: sin señal independiente significativa en ninguna dirección una vez controlada la colinealidad (o, en el caso de `rsi_overbought_outside_strong_trend`, muestra demasiado pequeña - 45-134 casos - para sacar ninguna conclusión). Sin cambios.

**`ENGINE_VERSION` no se ha movido en este paso** - no cambió ningún peso, así que no hay nada nuevo que atribuir a una versión distinta de la puntuación.

## 13. Segunda auditoría independiente — Bloque 1: bugs que producían decisiones falsas (agosto 2026, `ENGINE_VERSION` → v4)

Una auditoría independiente sobre el trabajo de las Fases 0-7 encontró ocho bugs concretos, todos
verificados línea a línea contra el código antes de tocar nada (ninguno estaba mal diagnosticado).
Los ocho se corrigieron el mismo día, cada uno con su test de regresión:

1. **`ADD_CANDIDATE` nunca se degradaba por `TIGHTEN_STOP`/`WATCH`, solo por `EXIT_NOW`/`REDUCE`
   o desde `HOLD`.** El resto vivo del bug original (D2/D3): un veredicto "comprar" con deterioro
   técnico real (p. ej. un cruce de medias proyectado con confianza suficiente) seguía mostrando
   "Añadir" en el badge. `portfolio_risk_service.py` ahora degrada a `WATCH` desde cualquier
   `signal` que no sea ya `EXIT_WARNING`, para `TIGHTEN_STOP` y `WATCH` por igual - `EXIT_NOW`/
   `REDUCE` siguen siendo los únicos que fuerzan `EXIT_WARNING`.
2. **El Chandelier Exit podía usar un máximo anterior a la entrada de la posición.**
   `trade_manager.chandelier_stop` tomaba el máximo de una ventana fija de 22 barras *del histórico
   completo que se le pasara* - si una posición se abrió tras un retroceso desde un máximo más alto
   anterior a la compra, ese máximo pre-entrada seguía dentro de la ventana. Combinado con que el
   stop solo puede subir nunca bajar, esto podía dejar el stop permanentemente por encima del precio
   vigente (`EXIT_NOW` irreversible). Dos cambios: `chandelier_stop` ya no exige una ventana
   completa de `window` barras (usa lo que haya disponible, igual que `detect_recent_cross`'s
   propio recorte de lookback) - `portfolio_risk_service.py` le pasa el `high` ya acotado a
   `>= plan.entry_date`, nunca el histórico completo; y `compute_trailing_stop` ahora recibe el
   precio vigente y descarta cualquier candidato que quede en o por encima de él, sea cual sea su
   origen.
3. **`TradePlanRepositoryPort.close()` estaba bien implementado y nunca se llamaba.** Una venta que
   llevaba la posición a 0 no cerraba el plan - una recompra posterior heredaba el `current_stop` ya
   traileado del lote muerto, a un precio de entrada completamente distinto. `add_transaction` ahora
   llama a `close()` cuando una venta deja la cantidad neta en 0 (o por debajo, margen de
   redondeo), y `ensure_trade_plan` añade un segundo guardarraíl independiente: si el plan abierto
   que devuelve `get_open` tiene un `entry_date` que no coincide con la entrada real del lote actual
   (`find_current_lot_entry`, ya existente), lo trata como obsoleto, lo cierra explícitamente, y
   reconstruye uno nuevo.
4. **Rotura de soporte con bases de precio mezcladas.** `nearest_support`/`nearest_resistance` que
   llegaban a `exit_engine.py` se calculaban sobre el precio **vivo** (`compute_core_signals` corre
   sobre el `df` crudo), mientras que el `price` que `evaluate_exit` compara es el cierre **cerrado**.
   Un hueco alcista de un día para otro podía dejar el soporte (calculado relativo al precio de hoy,
   más alto) por encima del cierre de ayer usado en la comparación, disparando "Vender ya" por el
   hueco, no por una rotura real. `portfolio_risk_service.py` ahora recalcula
   `support_resistance_levels` específicamente para el motor de salida usando los mismos datos
   **cerrados** que `exit_price` - misma base en ambos lados de la comparación, siempre.
5. **`detect_cross_with_quality` leía volumen de la barra equivocada.** `diff = (fast - slow).dropna()`
   elimina el warmup NaN de la media lenta (49 barras para 21/50, 199 para 50/200), así que la
   posición del cruce dentro de `diff` ya no coincide con su posición en `volume` (nunca truncado
   igual). El código reutilizaba esa posición desalineada para cortar `volume`, leyendo casi siempre
   el principio de la serie en vez del entorno real del cruce. Corregido a `.loc` por la fecha real
   de la barra del cruce, no `.iloc` por una posición reciclada de otra serie. Test nuevo con NaN de
   warmup real (200 barras) - los tests anteriores usaban series constantes sin NaN y no podían
   detectar esto.
6. **La regla de posición estancada no tenía techo.** Cualquier posición de años sin alcanzar +1R (y
   sin saltar el stop) disparaba `REDUCE` en cada evaluación, para siempre - "stop temporal" se
   pensó para capital atascado en una operación de corto plazo, no para recomendar recortar
   indefinidamente una posición de varios años cerca de breakeven. `STALLED_CEILING_BARS = 60`
   acota la regla.
7. **`signal_performance_service.py` sin deduplicar, y `hit_rate` con el signo equivocado para las
   etiquetas bajistas.** Cada evaluación fresca (cada recarga que cayera en un cache-miss, cada
   "Actualizar ahora") añadía otra observación para el mismo ticker/día - `n` medía la frecuencia de
   recarga, no el número de llamadas distintas del sistema. Deduplicado ahora por
   `(ticker, fecha calendario)`, quedándose con la más reciente de cada día, antes de agregar en
   `compute_verdict_outcomes`/`compute_signal_outcomes`/`find_false_negatives`. Y `hit_rate` para
   `evitar`/`exit_warning` (una llamada a evitar o vender) ahora cuenta un retorno **negativo** como
   acierto - la decisión de la ronda anterior de mantener una única lectura "neutra" (fracción
   positiva) para toda etiqueta era defendible en abstracto pero engañosa en la práctica: un 65% de
   `hit_rate` en "evitar" bajo esa definición significaba que el activo subió el 65% de las veces,
   justo lo contrario de una llamada acertada. `mean_return`/`median_return` siguen sin signo
   invertido - solo cambia qué lado de cero cuenta como acierto.
8. **`multi_timeframe._is_bullish`/`_is_bearish` y el `weekly_not_bullish` de `exit_engine.py`
   median "semanal alcista" con dos definiciones distintas que podían discrepar** - la primera
   (usada por `combine_timeframes`, lo que ve la UI como `alignment`) cuenta Fase 2 de Weinstein
   como alcista aunque `trend` por sí solo sea lateral; la segunda (el disparador EXIT_NOW más duro
   de `exit_engine.py`) solo miraba `trend`, más estricta - un mismo activo podía leer
   "bullish_aligned" en pantalla y a la vez armar el disparador más severo por debajo. Unificado en
   `multi_timeframe.timeframe_bias()`, una sola función que ambos importan, con un cuarto estado
   explícito - `"unknown"` (`price_vs_sma200 is None`, menos de ~200 barras semanales, ~3,85 años de
   historia) - que ya no cuenta silenciosamente como "no alcista" para ningún disparador que exija
   "semanal confirmado bajista": un ticker con poca historia semanal pierde ese disparador concreto
   en vez de heredar una lectura bajista que nadie confirmó.

**`ENGINE_VERSION` → `"2026-08-audit-v4"`.** Ningún peso de `recommendation_engine.py` cambió; se
bumpea porque el conjunto de disparadores de `exit_engine.py` sí cambió materialmente (puntos 6 y 8),
y esta constante traza también la lógica de salida, no solo la de compra (ver §10).

**Criterios de aceptación verificados en este bloque**: test de veredicto "comprar" forzado +
deterioro técnico (urgencia `tighten_stop`/`watch`) → `signal` nunca `add_candidate`; test de
Chandelier con un máximo pre-entrada que nunca deja el stop por encima del precio; test de venta
total + recompra que crea un plan con `entry_price`/`entry_date` nuevos y cierra el plan viejo
explícitamente; test de `detect_cross_with_quality` con warmup NaN real que lee la barra correcta.
(El de `label_triple_barrier` con hueco es del Bloque 2, todavía por hacer.) 666 tests en verde
(subieron de 640), `ruff check app tests` limpio.

## 14. Segunda auditoría independiente — Bloque 2: conectar lo que ya existía (agosto 2026)

El encargo era explícito: "casi todo lo que hace falta ya existe, el trabajo es conectar, corregir y
medir". Seis piezas, todas ya construidas y probadas en aislamiento en rondas anteriores, sin un
solo llamador real:

1. **`backtest_engine.label_triple_barrier` no era consciente de huecos de apertura.** Un stop en 95
   con la barra abriendo en 62 (un hueco bajista) se registraba como -5% (relleno al nivel del stop,
   que nunca estuvo realmente disponible para vender) en vez del -38% real (relleno al peor precio
   entre apertura y stop). Nuevo parámetro `open_` (opcional - `None` mantiene el comportamiento
   antiguo, consciente de sus propios límites, para quien no tenga la serie de apertura a mano):
   `fill = min(open, stop)` en un stop-out, `fill = max(open, target)` en un target alcanzado - una
   sola fórmula que no cambia nada cuando la barra no tuvo hueco real (`open` cae dentro de
   `[low, high]` por construcción, así que el `min`/`max` es un no-op salvo que de verdad haya
   hueco). `random_entry_labels` y `run_triple_barrier_backtest` (este último con `open_` obligatorio
   - todo caller real tiene la serie de apertura a mano) lo propagan a `label_triple_barrier`.
2. **`compute_trading_metrics` solo restaba costes al `net_return_pct` final.** `win_rate`,
   `avg_win_pct`/`avg_loss_pct`, `expectancy_pct`, `profit_factor` y `max_drawdown_pct` eran todos
   brutos - una operación de +0,10% bruto (menor que el coste de ida y vuelta, 0,2%) contaba como
   "ganadora" cuando en realidad fue una pérdida neta. Recalculado todo sobre retornos netos;
   `avg_mae_pct`/`avg_mfe_pct` se quedan brutos a propósito (describen recorrido intrabarra, no P&L -
   el coste no aplica a un nivel de precio que nunca se realizó como entrada/salida real).
3. **La curva de equity no anteponía una base de 1,0.** `np.cumprod([1+r for r in returns])` sin ese
   1,0 inicial deja invisible la pérdida de la primera operación al calcular el drawdown - corrijo
   aquí mi propio criterio de la ronda anterior: el test que yo mismo escribí
   (`test_trading_metrics_max_drawdown_on_a_losing_streak`) defendía la cifra subestimada como
   correcta ("normalizado al propio multiplicador de la primera operación"), y no lo era. Con base
   1,0 antepuesta, tres pérdidas del 10% seguidas (netas del coste) dan el drawdown compuesto real
   desde antes de la primera operación, no desde después.
4. **El trailing simulado nunca activaba el multiplicador de bloqueo de beneficio (+2R).**
   `tm.chandelier_multiplier(vol_regime, None)` - el segundo argumento, fijo en `None`, en cada barra
   del bucle. Ahora se recalcula en cada barra: `r_multiple = (close_barra - entry_price) /
   (entry_price - stop_loss)`, usando siempre el `stop_loss` **original** de la función (nunca el
   `current_stop` ya trailado) como denominador de riesgo - igual que hace el sistema en vivo. El
   multiplicador 2,0 (más ajustado que el 3,0 por defecto de régimen "normal") ya se activa a partir
   de +2R también en el backtest, no solo en producción.
5. **`ticker_analysis_service.py` no usaba `analyze_multi_timeframe`/`closed_bars` en absoluto.**
   Cada campo de "Analizar activo" (veredicto, stop, objetivo, cruces, patrones, Minervini, pivotes)
   se calculaba sobre el `df` crudo, vela en curso incluida. Dos añadidos, no una reescritura:
   - `multi_timeframe: MultiTimeframeRead`, siempre poblado - la misma lectura semanal/diaria sobre
     barras **cerradas** que `portfolio_risk_service.py` ya usa para el motor de salida, ahora
     también visible en `CoreTickerSignals`/`TickerAnalysis` (y por tanto en las tres superficies que
     comparten `compute_core_signals`: "Analizar activo", riesgo de cartera, watchlist premium).
     `portfolio_risk_service.py` dejó de calcularla por su cuenta una segunda vez - reutiliza
     `signals.multi_timeframe`, evitando repetir el resample semanal completo dos veces por posición
     en cada refresco de cartera.
   - `confirmed_recommendation: Recommendation | None` - el mismo veredicto/stop/objetivo, re-derivado
     con cada input técnico discreto (tendencia, stage, cruce, Minervini, soporte/resistencia)
     recalculado sobre `technical_analysis.closed_bars`, no el frame vivo. `None` cuando no hay nada
     que separar (la última barra ya está asentada, así que `recommendation` ya *es* la lectura
     confirmada) - nunca un duplicado inventado sin motivo. Decisión explícita de alcance: markov/
     GARCH/OBV/fundamentales **no** se recalculan (siguen siendo los del `recommendation` vivo) -
     son estimaciones estadísticas continuas, no señales discretas que repintan barra a barra como
     un cruce de medias, y recalcularlas exigiría rehacer el ajuste GARCH completo por cada consulta
     intradía.
6. **`run_triple_barrier_backtest` seguía sin un solo llamador fuera de sus propios tests.** Conectado
   a `compute_core_signals` como `triple_barrier_backtest: TripleBarrierBacktestResult | None`, a un
   horizonte **fijo de 21 sesiones** (`TRIPLE_BARRIER_HORIZON_DAYS`, del propio rango validado de
   `backtest_engine.py` - nunca el horizonte 1m/3m/6m que elige quien busca el ticker, ya que 63/126
   sesiones queda fuera de lo que este motor está pensado y documentado para medir). Decisión
   explícita distinta de la literalidad del encargo ("sustituir" el backtest legacy): en vez de
   retirar `backtest: WalkForwardBacktestResult`, lo mantengo **junto al nuevo campo**, porque
   `premium_watchlist_service._approval_score`'s `backtest_contradicts` todavía lee el backtest
   legacy (su reescritura para depender de `backtest_engine` con ≥30 operaciones/bucket es trabajo
   explícito del Bloque 3) - retirarlo ahora habría dejado esa puerta rota a mitad de camino en vez
   de en un punto de corte limpio. El Bloque 4 (etiquetar cada backtest por lo que es, con sus
   propias limitaciones) usa esta misma coexistencia.

   **Corrección medida tras el primer intento**: lo conecté inicialmente dentro de
   `compute_core_signals` sin condición, así que también corría para cada posición de
   `portfolio_risk_service.py` y cada candidato de `premium_watchlist_service.py` en cada refresco de
   caché - ninguna de esas dos vistas muestra este campo. Medido, no supuesto: la suite de
   integración completa (`pytest -q tests/integration`, ~80 tests) pasó de 1733 s (~28,9 min) con el
   campo sin condición a 1181 s (~19,7 min) tras el fix de abajo - una caída real del 32%.
   `run_triple_barrier_backtest` simula barra a barra en Python (no vectorizado como el resto de esta
   función) dos veces por señal "comprar" replayeada (fijo y trailing) más el benchmark de entradas
   aleatorias - unas 3x el coste del resto de `compute_core_signals` junto. Nuevo parámetro
   `include_triple_barrier_backtest: bool = False`: solo `TickerAnalysisService.analyze()` (lo único
   que el encargo pedía conectar aquí) lo activa; los otros dos llamadores se quedan sin este campo
   (`None`), exactamente como antes de este bloque.

   Los ~19,7 min restantes siguen por encima del tiempo histórico de esta suite antes de todo el
   Bloque 2 - atribuible en su mayor parte a que `multi_timeframe` (punto 5) sí quedó sin condición
   para los tres llamadores, `premium_watchlist_service.py` incluido, que antes nunca lo calculaba.
   Decisión explícita de no meterle también un flag de activación: su coste es pandas vectorizado
   (un resample semanal + una segunda pasada de SMA/RSI/ADX/DMI/MACD), no un bucle Python barra a
   barra como el del backtest - el mismo orden de magnitud que el resto de `compute_core_signals` ya
   hacía, no una categoría de coste nueva - y `portfolio_risk_service.py` ya lo necesitaba de todos
   modos en el caso real (con contexto de cartera), así que para ese llamador esto es, en la práctica,
   una sola pasada donde antes había dos. Queda como coste aceptado, no como pendiente.
7. **`portfolio_construction_service.py` (D12) tenía cero llamadores en `app/api`.** Nuevo endpoint
   `GET /api/v1/portfolios/{id}/construction`, orquestado en el propio endpoint (no un servicio
   nuevo): pesos por posición vía `PortfolioService.get_portfolio_summary` (ya calcula
   `market_value_base`/`quantity` correctamente en multi-moneda), sector vía
   `market_universe.sector_of`, retornos diarios vía un único `get_bulk_ohlcv` batched (nunca una
   llamada de red por ticker), y el stop vigente de cada posición vía su `trade_plan` abierto (lectura
   de BD, no de mercado). Una posición sin plan de trade abierto o sin cotización en vivo queda fuera
   de `aggregate_risk` y listada explícitamente en `tickers_without_trade_plan` - su riesgo
   genuinamente no se conoce, nunca se asume cero. Panel nuevo en el dashboard
   (`PortfolioConstructionPanel.jsx`): riesgo agregado si saltaran todos los stops, volatilidad de
   cartera vs. objetivo, pares correlacionados (>0,8) y concentración por sector con aviso visual
   cuando supera el 30%.
8. **`resample_ohlcv` comparaba contra el borde de calendario, no contra si de verdad había pasado
   el periodo.** `df.index[-1] < agg.index[-1]` compara la fecha del último dato crudo contra la
   etiqueta del periodo (el viernes calendario de una semana, el último día calendario de un mes) -
   un festivo en viernes (más común en Europa) o un fin de mes en fin de semana (~5/12 meses) hace
   que el último dato real caiga corto de ese borde *aunque el periodo ya haya terminado de verdad*,
   descartando una semana/mes ya completo. La pregunta real - ¿ya pasó este periodo, o seguimos
   dentro de él? - necesita saber "hoy", igual que `closed_bars`: nuevo parámetro `now` (mismo
   patrón, mismo valor por defecto), comparando la fecha de `now` contra el borde derecho del
   periodo directamente, no la fecha del último dato crudo.
9. **`closed_bars`'s `CLOSED_BAR_CUTOFF_UTC` (21:30 UTC, cierre de EE.UU.) se aplicaba a cualquier
   ticker, europeos incluidos** (cierre real 15:30-16:30 UTC) - una barra europea ya asentada se
   descartaba 5-6 horas de más cada tarde. `technical_analysis.py` se mantiene deliberadamente
   agnóstico del universo de mercado (funciones puras, sin imports de `services/`) - en vez de
   importar `market_universe` ahí, `closed_bars`/`resample_ohlcv` ganan un parámetro `cutoff`
   opcional (`None` = el valor por defecto de antes), y es el llamador que sí conoce el ticker
   (`market_universe.closed_bar_cutoff_for_ticker`, nuevo, factoriza la misma detección de región que
   ya usaba `benchmark_for_ticker`) quien decide cuál pasar. Enhebrado a través de
   `multi_timeframe.analyze_multi_timeframe`/`_read_timeframe`, `compute_core_signals` (que ahora
   acepta un `ticker` opcional para esto) y el propio `ta.closed_bars` de
   `portfolio_risk_service.assess_position_risk`.

**Criterios de aceptación verificados en este bloque**: test de `label_triple_barrier` con hueco de
apertura (stop y target) que reproduce el ejemplo exacto del encargo (-38% real vs. -5% sin
consciencia de hueco); test de drawdown con base 1,0 correcta (corrigiendo mi propio test anterior);
test del multiplicador de bloqueo +2R activándose en el trailing simulado; `grep
"analyze_multi_timeframe\|closed_bars" app/services/ticker_analysis_service.py` no vacío; `grep
portfolio_construction app/api` no vacío (el nuevo endpoint); `run_triple_barrier_backtest` con
llamador real (`compute_core_signals`); tests de `resample_ohlcv` con festivo en viernes y fin de mes
en fin de semana; tests de `closed_bars`/`market_universe` con cutoff específico por región; test de
`include_triple_barrier_backtest` sin activar por defecto (y de "Analizar activo" activándolo de
verdad, de punta a punta, contra la API real). 606 tests unitarios en verde, 80 tests de integración
en verde (19,7 min), `ruff check app tests` limpio.

## 15. Segunda auditoría independiente — Bloque 3, punto 1: universo dinámico punto-en-el-tiempo (agosto 2026)

El resto del Bloque 3 (separación de setups, percentil transversal, limpieza de `_approval_score`,
diversificación, logging de descartados, estadística por setup vía `backtest_engine`) queda pendiente
- ver la nota al final de esta sección sobre por qué se corta aquí, no a media implementación de algo
a medio probar.

**Lo que sí se construyó, probado de punta a punta:**

- **Tabla nueva `universe_memberships`** (`UniverseMembershipORM`, migración `697be1f02648`): una fila
  por `(region, ticker, as_of_date)`, con `source` (`"live"` | `"curated_fallback"`). Cada refresco
  **añade** un snapshot fechado, nunca sobrescribe uno anterior - así se acumula historia
  punto-en-el-tiempo real a partir de hoy, en vez de que cada refresco borre la posibilidad de
  reconstruir qué había en el universo hace meses. `UniverseMembershipRepositoryPort` +
  `UniverseMembershipRepository` siguen exactamente el mismo patrón puerto/adaptador que
  `TradePlanRepositoryPort`/`PositionSignalSnapshotRepositoryPort`.
- **`dynamic_universe_service.py`** (nuevo módulo - la excepción justificada de "no nuevos módulos":
  es la pieza sustantiva que el encargo pide para el Bloque 3, no algo no pedido):
  - `fetch_live_constituents(region)`: S&P 500 + S&P 400 (EE.UU., de sus páginas públicas de
    Wikipedia) o STOXX Europe 600 (Europa) - verificado en vivo contra la red real antes de escribir
    ninguna línea de parseo, no supuesto. STOXX 600 da un ticker desnudo + país, no un ticker listo
    para Yahoo Finance - `YAHOO_SUFFIX_BY_COUNTRY` lo resuelve al mismo sufijo que
    `market_universe.EUROPEAN_EXCHANGE_SUFFIXES` ya reconoce en el lado de lectura; un país sin
    mapeo se descarta con log, nunca se adivina.
  - `apply_liquidity_filter`: el filtro duro del encargo (volumen en $ de 20 sesiones ≥$20M,
    precio≥$5, capitalización≥$1.000M). Un ticker que ni siquiera se puede descargar (deslistado
    desde la foto de Wikipedia, un símbolo que Yahoo Finance no reconoce) falla el filtro igual que
    uno genuinamente ilíquido - no es una excepción especial, es exactamente el mismo "no es
    negociable a esta escala".
  - `refresh_universe_membership`: orquesta fetch → filtro → snapshot persistido. Si el fetch en vivo
    falla, o si sobrevive vacío tras el filtro (un fallo silencioso distinto pero igual de real), cae
    al universo curado de `market_universe.py` - **con log explícito**, nunca en silencio - y lo
    persiste igual (como snapshot `source="curated_fallback"`) para que `latest_as_of_date` no quede
    huérfano.
  - `scripts/refresh_universe_membership.py`: script mensual independiente, no un código de camino
    en vivo. Descargar y filtrar por liquidez ~1.000-1.500 tickers combinados es mucho más pesado que
    cualquier otra cosa en los caminos calientes de este proyecto - "no llamadas de red por ticker en
    los caminos calientes" (CLAUDE.md) aplica aquí también aunque el coste sea cómputo tanto como
    red: nadie debería pagar este coste por casualidad en una petición en vivo. `is_refresh_due`
    reutiliza la misma idea de TTL que `durable_cache.py` ya usa en todo el proyecto (aquí, 30 días).
  - Limitación honesta, no maquillada: esto **no puede reconstruir retroactivamente** quién estaba en
    cada índice hace años - solo un feed de pago con historial de constituyentes podría. Cada
    snapshot que esto guarda es "el universo tal como estaba el día que corrió esto", así que un
    estudio de factores que use un snapshot temprano para evaluar precios mucho más antiguos sigue
    cargando algo de sesgo de supervivencia para esas fechas antiguas - simplemente deja de
    **acumularse** hacia adelante, y el sesgo se reduce a cero para cualquier fecha de muestra en o
    después de que esto se desplegara.
  - Nueva dependencia: `lxml` (requirements.txt) - `pandas.read_html` la necesita; `bs4` sola (ya
    presente, transitiva de otra dependencia) no basta sin `lxml`/`html5lib` de todos modos.

**Decisión explícita de alcance, con la razón medida, no supuesta**: el universo dinámico **no** se
conectó todavía a `market_screener_service.get_universe_snapshot` (la base compartida de
`premium_watchlist_service.py`, rotación sectorial, movers, amplitud de tendencia). Ese snapshot hoy
descarga OHLCV de ~170 tickers curados en cualquier cache-miss - un camino **en vivo**, no un script
mensual. Conectarlo al universo dinámico (~1.000-1.500 tickers combinados) multiplicaría ese fetch
por 6-9x en cualquier petición que caiga en un cache-miss - exactamente el tipo de regresión de
latencia que `PortfolioRiskService`'s propio incidente en producción ya advierte evitar, y que el
propio Bloque 2 acaba de medir de verdad (29 min → 19,7 min) antes de decidir un gateo, no antes. No
tengo una medición real de cuánto tardaría ese cache-miss a 1.000+ tickers - y no voy a conectarlo sin
medirlo primero, mismo estándar que el resto de este documento exige para cualquier cambio de peso o
factor. El consumidor real, ya seguro de conectar sin ese riesgo, es `factor_ablation_study.py`
(Bloque 5): ese script ya es un proceso lento, offline, sin usuario esperando una respuesta en vivo -
es exactamente donde el encargo dice que el sesgo de supervivencia importa ("es la causa raíz de lo
que mide el estudio de ablación"). El resto del Bloque 3 (setups/percentil/limpieza/diversificación/
logging) puede construirse igualmente sobre el snapshot curado existente sin depender de esta
decisión - queda para continuar.

**Actualización (Tercera auditoría, Bloque F-1, agosto 2026)**: conectado. La objeción de coste de
arriba (multiplicar el fetch de OHLCV por 6-9x) resultó estar mal enfocada - el fetch en sí es una
sola llamada por lotes, cueste 170 o 1.000 tickers; el coste real era calcular indicadores completos
sobre 1.000. Resuelto con un cribado barato de precio/volumen (sin llamada de red por ticker) que
recorta a `CHEAP_SCREEN_KEEP_TOP_N` **antes** de calcular ningún indicador - ver §21.

**Por qué se corta aquí**: los 6 puntos restantes del Bloque 3 son, cada uno, del orden de un día de
trabajo por derecho propio (separación de 4 tipos de setup con su propio scoring, un percentil
transversal multi-factor que necesita datos que hoy no existen en `TickerSnapshot` -algunos
requieren enhebrar series temporales nuevas a través de todo el universo, no solo un valor puntual-,
limpieza validada de `_approval_score`, diversificación reutilizando `portfolio_construction_service`,
logging + UI de descartados, y estadística por setup vía `backtest_engine` sobre el universo
completo - esta última, en sí misma, de una escala parecida a la del propio estudio de ablación).
Completarlos a medias o con atajos que inventen datos que no existen contradiría exactamente el
estándar que este documento exige en cada sección anterior. Se documentan aquí como el trabajo
pendiente explícito del Bloque 3, no como "hecho a medias" sin decirlo.

**Tests**: `test_dynamic_universe_service.py` (17 tests - parseo de S&P 500/400 y STOXX 600 contra
HTML de referencia, no la red real; filtro de liquidez con los 4 casos límite - precio, volumen,
capitalización, sin datos; orquestación con fetch simulado en vivo y con fallback). Migración
verificada con `alembic heads`/`alembic history` y con `Base.metadata.create_all` contra SQLite (las
migraciones de este proyecto son Postgres-only por diseño - ver `tests/integration/conftest.py`, que
ya evita alembic por completo). 623 tests unitarios en verde, `ruff check app tests` limpio.

## 16. Segunda auditoría independiente — Bloque 3, puntos 2-6: setups, percentil, limpieza, diversificación (agosto 2026)

Sigue pendiente el punto 7 (estadística por setup vía `backtest_engine` sobre el universo completo) -
ver la nota al cierre de esta sección.

1. **Separación de setups.** `watchlist_service._short_term_reasons` hacía OR de tres reglas
   distintas en un solo cajón. Ahora son cuatro detectores independientes
   (`oversold_bounce`/`breakout_volume`/`trend_continuation`/`pullback_to_support`), cada uno
   produciendo su propio `WatchlistItem` - un ticker que cumple dos a la vez aparece dos veces, una
   por setup, nunca mezclado. `pullback_to_support` es la única regla genuinamente nueva (las otras
   tres ya existían, solo separadas): precio a 0-4% sobre una SMA50 que a su vez está por encima de
   la SMA200 (tendencia intermedia confirmada), con RSI > 40 (para no solaparse con
   `oversold_bounce`) - aproximado desde las medias móviles que cada snapshot ya trae, en vez de una
   segunda pasada de niveles de soporte/resistencia sobre todo el universo.
2. **Percentil transversal, específico de setup, reemplazando RS Rating como criterio de orden para
   los tiers de corto plazo.** Nuevos campos en `TickerSnapshot` (calculados una vez, en
   `market_screener_service._build_raw`, junto al resto de indicadores - sin pasada extra sobre el
   universo): `atr_ratio_50d` (ATR14 actual vs. su propia media de 50 sesiones - contracción/
   expansión), `atr_multiple_sma21` (`atr_multiple_from_sma` con `sma_window=21`, distinto del
   campo ya existente a 50 sesiones), `range_position_20d` (`rolling_position_in_range` a 20
   sesiones - ya existía como función, no como campo del snapshot), `mansfield_rs_4w` (ventana de
   20 sesiones, no las 200 del campo ya existente) y `relative_volume_trend` (relative_volume de
   hoy menos el de hace 5 sesiones). `watchlist_service.setup_percentile_scores` calcula, para
   cada uno de esos seis campos, el percentil (0-100, empates con rango promedio vía
   `pandas.Series.rank` - una implementación ingenua de "el primero en la lista gana el rango más
   bajo" sesgaría el resultado por orden de entrada, no por valor real, y así fue como se descubrió
   en el primer intento) dentro de **todo** el universo del día, no solo de quienes ya cumplen la
   regla de disparo del setup - la regla decide quién califica, el percentil decide el orden entre
   calificados. Ninguna dirección/peso por campo está validada por ablación todavía (a diferencia de
   los factores de `recommendation_engine.py`), así que es un promedio sin ponderar, transparente -
   la única inversión que se aplica es la que pide el encargo explícitamente: el retorno a 5 días
   cuenta al revés para `oversold_bounce` (una caída más profunda prepara un rebote mayor). RS
   Rating se queda como criterio de orden solo para el tier mensual.
3. **Limpieza de `_approval_score`** (`premium_watchlist_service.py`): fuera `BACKTEST_EDGE_BONUS`
   y `KELLY_SETUP_BONUS` (sin sustituto), fuera el rechazo `backtest_contradicts` (corría sobre el
   backtest legacy - reintroducir solo con `backtest_engine` y ≥30 operaciones/bucket, todavía por
   hacer), y `MONTE_CARLO_EDGE_WEIGHT` (`probability_target_before_stop × 2`, casi circular - la
   propia simulación Monte Carlo se deriva de la misma recomendación que este score ya usa) se
   reemplaza por `SETUP_PERCENTILE_BONUS_WEIGHT`, alimentado por el percentil del punto 2 - solo
   presente para el tier diario (setups de corto plazo); semanal/mensual se quedan sin ese empujón,
   no con uno inventado. El bonus de sector fuerte y la penalización de entrada extendida se
   mantienen sin cambios (baratos, direccionalmente sólidos, mismo carácter que el nuevo bonus).
4. **Log + exposición de candidatos descartados.** `MAX_CANDIDATES_PER_TIER=15` truncaba en
   silencio. Nuevo `TierDiscardStats` (`prefilter_matches`/`analyzed`/`approved` por tier), devuelto
   junto a los candidatos por `build_premium_watchlist` y cacheado junto a ellos
   (`PremiumWatchlistService.get_premium_watchlist_with_stats`, para no recomputar dos veces por
   petición). Expuesto en `GET /api/v1/market/watchlist/premium` (`discard_stats`) y en el panel
   "Premium" del frontend ("X de Y candidatos analizados en detalle").
5. **Diversificación (dedupe entre tiers, penalización por correlación, límite por sector)**: **no
   implementada en este bloque** - decisión explícita de alcance, no un olvido. Requiere
   `portfolio_construction_service.compute_correlation_matrix` sobre los propios candidatos de la
   lista (no solo la cartera existente, su uso actual) y una regla de desempate entre tiers que
   decida en cuál de sus horizontes calificados debe aparecer un ticker que califica en más de uno -
   ninguna de las dos cosas existe todavía en la forma que esto necesita, y el tiempo disponible en
   este bloque se agotó en los puntos 1-4 y 6. Candidato natural para la siguiente sesión sobre este
   mismo bloque.

**Por qué el punto 7 (estadística histórica por setup vía `backtest_engine` sobre el universo
completo) queda fuera de este bloque**: es, en sí mismo, una operación de la escala del propio
estudio de ablación (correr `backtest_engine` sobre cientos de tickers x 10 años x 4 tipos de setup),
no una función más para añadir a una tarde de trabajo - la ejecución real del estudio de ablación (§12)
ya documentó cuánto tarda ese tipo de cómputo. Se hará como parte del Bloque 5 (que de todos modos
re-ejecuta el estudio de ablación), no duplicado aquí.

**Tests**: `test_watchlist_service.py` ampliado (21 tests: los cuatro setups por separado, un ticker
que dispara dos a la vez produce dos items, `setup`/`percentile_score` presentes solo en items de
corto plazo, inversión del retorno a 5 días específica de `oversold_bounce`, empates con rango
promedio); `test_market_screener_service.py` nuevo (8 tests - primero que existe para este archivo:
contracción/expansión de ATR, posición en rango a 20 sesiones en máximos/mínimos, Mansfield RS a 4
semanas distinto del de 200 sesiones, tendencia de volumen relativo subiendo/bajando,
`atr_multiple_sma21` distinto del campo a 50 sesiones); `test_premium_watchlist_service.py`
reescrito (22 tests - bonus del percentil de setup, `TierDiscardStats`, el nuevo
`get_premium_watchlist_with_stats`); test de integración nuevo confirmando `discard_stats` y `setup`
de punta a punta contra la API real. 643 tests unitarios en verde, `ruff check app tests` limpio.

## 17. Segunda auditoría independiente — Bloque 4: decir la verdad en la interfaz (agosto 2026)

**Hallazgo real, encontrado al conectar esto por primera vez - no un ajuste de mi código, un dato que
esta misma auditoría existe para exponer**: `minervini_range_position` (el criterio de precio 25%+
sobre el mínimo anual y dentro del 25% del máximo) es, según el comentario propio de
`recommendation_engine.py` (líneas 236-248), *"el factor más robustamente validado de todo el
checklist... significativo (p<0.01) y con el signo correcto TANTO a 3 meses (+1,04pp) COMO a 6 meses
(+2,67pp)"* - una referencia al estudio de ablación **original** (pre-Fase 5). El estudio v2
reescrito (barrera triple, demeaned, Fase 5 - el que ahora lee este mismo bloque) mide, para ese
mismo factor: **signo contrario** a 63 sesiones (`mean_difference_pct=-0,112`, no significativo tras
corrección BH) y **signo contrario** a 126 sesiones (`mean_difference_pct=-0,51`, significativo en
crudo pero no tras corrección BH). El propio comentario del motor que justifica el peso de este
factor está citando un estudio que la Fase 5 ya reemplazó, y nadie lo había verificado hasta conectar
esto. No se ha tocado el peso - eso es exactamente lo que este documento (y CLAUDE.md) prohíben sin
decisión explícita del propietario - pero el comentario en `recommendation_engine.py` necesita una
revisión honesta antes de la próxima vez que alguien confíe en él, y **esto se deja anotado aquí para
que el propietario lo vea**, no corregido en silencio en este mismo commit. De los 15 factores con
clave de ablación, entre 7 (a 126 sesiones) y 11 (a 21 sesiones) miden signo contrario a su peso
actual - ver el propio endpoint/vista para el detalle completo por horizonte.

1. **"Rendimiento del sistema" ahora muestra peso vs. efecto medido por factor** - nuevo
   `ablation_report_service.py`: lee directamente los CSV que `scripts/factor_ablation_study.py` ya
   guardó (`docs/factor_ablation_report_v2_h{5,21,63,126}.csv`), nunca recalcula nada. Nuevo endpoint
   `GET /api/v1/system/factor-ablation?horizon_days=N`. Nueva sección en `SystemPerformanceView.jsx`
   con selector de horizonte y fila marcada ⚠️ cuando `directionally_consistent = False`.
2. **Aviso en la ficha del ticker cuando el veredicto se apoya en factores de signo contradicho** -
   `ablation_report_service.triggered_factors_with_contradicted_sign` cruza los factores
   *disparados* de la recomendación actual (por su etiqueta legible, vía un mapeo mantenido a mano
   `FACTOR_LABEL_TO_ABLATION_KEY` - un factor sin entrada ahí simplemente nunca se marca, nunca se
   asume consistente) contra el conjunto de claves con signo contrario al horizonte de referencia
   (21 sesiones, el horizonte real de esta cartera - igual que `TRIPLE_BARRIER_HORIZON_DAYS`). Nuevo
   campo `sign_contradicted_factors` en `CoreTickerSignals`/`TickerAnalysis`, mostrado como aviso en
   `RecommendationCard.jsx` y marcado factor por factor en la propia lista - nunca cambia la
   puntuación, solo la revela.
3. **Backtest etiquetado por lo que realmente es**: `BacktestCard.jsx` (el walk-forward legacy) ahora
   dice explícitamente que ignora el stop/objetivo y no descuenta costes. Nuevo
   `TripleBarrierBacktestCard.jsx` muestra el backtest de barrera triple ya conectado en el Bloque 2
   (`triple_barrier_backtest`) - expectativa/factor de beneficio/drawdown netos de costes, MAE/MFE
   brutos a propósito, las cuatro estrategias (fija/trailing real/comprar-y-mantener/aleatoria) una
   junto a la otra.

**Tests**: `test_ablation_report_service.py` (9 tests - parseo, archivo ausente, archivo corrupto,
mapeo de factores disparados a claves de ablación, un factor sin mapeo nunca se marca); tests nuevos
en `test_ticker_analysis_service.py`/`test_ticker_analysis_api.py` (el segundo contra los CSV reales
del repositorio, no una fábrica de datos) confirmando que `sign_contradicted_factors` es siempre un
subconjunto de los factores realmente disparados. 654 tests unitarios en verde, `ruff check app tests`
limpio.

## 18. Segunda auditoría independiente — Bloque 5: el script listo para la partición temporal y el
universo dinámico (agosto 2026) — **todavía no ejecutado contra datos reales**

Este bloque es, deliberadamente, solo la mitad "construir y probar" del encargo. La otra mitad -
correr el estudio completo (5-10 tickers x 10 años x varios horizontes, del orden de la misma
duración que el estudio v2 de la Fase 5) y traer una recomendación de recalibración con evidencia
dentro/fuera de muestra al lado - es un paso deliberadamente separado, no ejecutado todavía: es un
proceso largo, y decidir *cuándo* correrlo (y sobre qué universo - curado o dinámico) es una decisión
visible para el propietario, no algo que se lanza en silencio dentro de una sesión de código.

**Lo que sí se construyó y se probó de punta a punta, contra `scripts/factor_ablation_study.py`:**

- **Partición temporal calibrar/validar** (`split_samples_by_date`, `TEMPORAL_SPLIT_CUTOFF =
  2023-01-01`): reutiliza `backtest_engine.split_by_date` (ya existía desde la Fase 5) sobre las
  fechas de los propios `FactorSample`. Calibrar es estrictamente antes del corte, validar es en o
  después - nunca mezclados. Crítico: el *demeaning* transversal ocurre **después** de partir, no
  antes (`build_reports` desmedia solo sobre el subconjunto que se le pase) - desmediar contra la
  media de todo el periodo antes de partir habría filtrado información de validación hacia calibrar,
  exactamente el tipo de fuga que esta partición existe para evitar.
- **Segmentación por tipo de setup** (`segment_by_setup_type`, `SETUP_TRIGGER_KEYS`): las mismas
  cuatro heurísticas de `watchlist_service.py` (Bloque 3) - `oversold_bounce`, `breakout_volume`,
  `trend_continuation`, `pullback_to_support` - reimplementadas dentro de `compute_triggers_at` sobre
  series de indicadores en un índice histórico arbitrario (no sobre un `TickerSnapshot`, que
  `watchlist_service` construye solo para el presente vivo), reutilizando las mismas constantes con
  nombre (`wl.PULLBACK_MAX_DISTANCE_ABOVE_SMA50`, `wl.PULLBACK_MIN_RSI`) para no derivar en silencio
  de los umbrales que sí están en producción. No son mutuamente excluyentes (una muestra puede
  coincidir con varios setups o ninguno) y se excluyen explícitamente de los informes por factor
  (`factor_names` filtra cualquier clave `setup_*`) - son membresías de segmento, no factores
  puntuados por `recommendation_engine.py`.
- **Universo punto-en-el-tiempo opcional** (`resolve_universe_tickers`, flag `--use-dynamic-universe`):
  lee `universe_memberships` (Bloque 3, §15) por región vía `dynamic_universe_service.read_dynamic_universe`,
  con fallback explícito y logueado al diccionario curado por región cuando esa región todavía no
  tiene snapshot - nunca falla en silencio devolviendo un universo vacío para media convocatoria.
  Comportamiento por defecto (`--use-dynamic-universe` no pasado) sin cambios: sigue siendo el
  diccionario curado, igual que antes de este bloque.
- **CLI (`__main__`)**: `--use-dynamic-universe` y `--temporal-split` son opt-in explícitos - correr
  el script sin ellos produce exactamente los mismos archivos que producía antes de este bloque
  (`{prefix}_h{N}.csv`, `{prefix}_h{N}_regime_{name}.csv`), más los nuevos
  `{prefix}_h{N}_setup_{name}.csv` (siempre, son baratos - reutilizan las muestras ya recolectadas,
  sin recolección adicional). Con `--temporal-split` se añaden además
  `{prefix}_h{N}_calibrate.csv`/`{prefix}_h{N}_validate.csv` y sus propios desgloses por régimen/setup
  (`{prefix}_h{N}_calibrate_regime_{name}.csv`, etc.) - nunca un archivo que mezcle ambos periodos.

**Por qué no se ejecutó todavía**: correr esto de verdad, con `--use-dynamic-universe` real, primero
necesitaba que el universo dinámico existiera de verdad para ambas regiones - lo cual expuso (ver el
fix de `dynamic_universe_service.py` en el commit `6b9bd72`, el mismo día) un bug real de producción
(duplicado "SHEL.L" en la página de STOXX 600) que habría hecho fallar cualquier intento de correr el
estudio con `--use-dynamic-universe europe`. Con ese bug corregido y verificado en vivo (Europa guarda
242 tickers, `source=live`), el script está listo para correrse - pero el propio corrido (varias horas,
resultado con datos reales que hay que revisar antes de recomendar nada) es la siguiente decisión, no
parte de este commit.

**Tests**: `test_factor_ablation_study.py` (13 tests, nuevo - primer archivo de test para este script):
las cuatro heurísticas de setup contra series sintéticas donde el resultado es obvio a mano (una
caída pronunciada seguida de un día verde para `oversold_bounce`, una tendencia limpia para
`trend_continuation`, etc., validadas también contra las funciones reales de `technical_analysis.py`,
no contra un cálculo propio); `segment_by_setup_type` (agrupación, solape, entrada vacía);
`split_samples_by_date` (corte por defecto y personalizado, el propio día del corte cae en validar,
no en calibrar); `resolve_universe_tickers` (comportamiento curado por defecto, snapshot dinámico
cuando existe, fallback por región cuando no). 669 tests unitarios en verde, `ruff check app tests
scripts` limpio.

## 19. Panel "Tendencia": corto/mediano plazo y proyección de cruce (agosto 2026)

Pedido directo del propietario, fuera de los 5 bloques de la segunda auditoría: el panel "Tendencia"
(`market_screener_service.get_trend_breadth`/`get_trend_detail`, `TrendBreadthPanel.jsx`) usaba
`ma_cross` (SMA50/SMA200) para sus grupos "Golden cross"/"Death cross" - una señal de meses, demasiado
lenta para una cartera gestionada a días/semanas (CLAUDE.md) - y no ofrecía ninguna proyección de cruce
próximo, solo el cruce ya confirmado.

**Conectar, no construir de nuevo**: `technical_analysis.detect_imminent_cross` (D2, §8) ya existe y ya
está en producción en `multi_timeframe.py`/`exit_engine.py`/`ticker_analysis_service.py`, con el par
corto `FAST_MA_PERIOD`(21)/SMA50 ya siendo el que usan para gestión de posición a corto plazo. El panel
"Tendencia" simplemente no lo leía todavía - se conecta aquí, no se reimplementa una segunda vez.

1. **`ma_cross_short`** (`TickerSnapshot`, nuevo campo): `detect_recent_cross` sobre el mismo par
   SMA21/SMA50, en vez de `ma_cross` (SMA50/SMA200) - solo para este panel. `get_movers.jsx`/`MarketMovers`
   sigue usando el par largo a propósito, sin cambios - son componentes distintos con preguntas
   distintas ("¿qué cruzó hoy, de cualquier plazo?" vs. "¿qué tendencia de corto plazo importa para
   revisar la cartera esta semana?").
2. **`imminent_cross_short_term`** (nuevo campo): reutiliza `detect_imminent_cross` sobre el mismo par
   corto - nuevo grupo "Próximo cruce de medias esperado" en el panel, ordenado por sesiones estimadas
   (el más próximo primero), con el mismo componente `ImminentCrossBadge.jsx` que ya usa la ficha de un
   solo ticker (misma redacción, mismo umbral de R², una sola fuente de verdad para el texto).
3. **"Activos recomendados más precisos"**: los grupos alcistas (`uptrend`, `golden_cross`, `stage2`,
   `minervini_pass`, `strong_trend`) ahora empujan al final de su propia lista (sin ocultarlo) a
   cualquier nombre ya parabólico (`atr_multiple > 4` - el mismo umbral que `recommendation_engine.py`
   ya usa para `atr_parabolic`), en vez de dejar que un RS Rating alto por sí solo tape que el nombre ya
   está sobreextendido. `downtrend` y el resto de grupos bajistas/neutros no cambian - la
   sobreextensión no es la pregunta relevante ahí.

**Bug real encontrado al conectar esto, no al escribirlo**: un test de integración contra los datos
fake y deterministas de `tests/integration/conftest.py` hizo que `detect_imminent_cross` proyectara
`bars_until=0` - una proyección cruda entre 0 y 0,5 sesiones redondea a 0 con `round()`, y "cruce
esperado en ~0 sesiones" se lee como "ya pasó", exactamente el caso que `current_gap == 0` ya trata por
separado unas líneas antes. Ningún test existente (los suyos propios, ni los de `exit_engine.py`/
`multi_timeframe.py`/`ticker_analysis_service.py` que ya lo consumen) usaba una pendiente lo bastante
pronunciada para tocar este caso - una fracción de bar es un resultado legítimo del ajuste lineal,
simplemente nunca se había redondeado a exactamente 0 antes en ningún dato de prueba. Corregido con
`max(1, round(bars_until))`: nunca reporta menos de 1 sesión. No cambia ningún umbral (`IMMINENT_CROSS_MIN_R2`,
`IMMINENT_CROSS_HORIZON`) ni ninguna decisión de `exit_engine.py` - solo el valor mínimo reportable de
`bars_until`, que ya era, en espíritu, "al menos 1" en cada caso que los tests existentes cubrían.

**Tests**: `test_technical_analysis.py` (nuevo `test_detect_imminent_cross_never_reports_zero_bars_until`,
serie sintética con pendiente limpia que produce `bars_until` crudo ≈0,29); `test_market_screener_service.py`
(10 tests nuevos - `ma_cross_short`/`imminent_cross_short_term` en `_build_raw` con una reversión
declive→rally construida a mano y verificada contra las funciones reales antes de escribir el test;
`get_trend_breadth`/`get_trend_detail` con snapshots sintéticos para los conteos de cruce inminente, el
orden "más próximo primero", y que la desprioritización por sobreextensión afecta a `uptrend` pero no a
`downtrend`; reconstrucción de `imminent_cross_short_term` desde el caché durable, incluyendo el caso
"snapshot cacheado antes de que este campo existiera"); dos tests de integración nuevos en
`test_market_api.py` contra los endpoints reales `/market/trend` y `/market/trend/detail`. 680 tests
unitarios en verde, `ruff check app tests` limpio, `npm run lint`/`npm run build` limpios en el frontend.

## 20. Tercera auditoría independiente — Bloque A: regresiones y bugs con dinero real (agosto 2026)

Una tercera auditoría, ejecutando código real y simulando escenarios (no solo leyendo), encontró
regresiones introducidas por los propios arreglos de las rondas anteriores, además de bugs nuevos. Ninguno
toca los pesos de `recommendation_engine.py` - no hay bump de `ENGINE_VERSION` en este bloque.

1. **Un `current_stop` corrupto (de antes del fix del Chandelier) se quedaba en `EXIT_NOW` para
   siempre.** El fix de la ronda anterior (acotar `high` a `plan.entry_date`, descartar candidatos por
   encima del precio) solo protege candidatos *nuevos* - `update_trailing_stop`'s `max(current_stop,
   candidate)` seguía confiando ciegamente en cualquier `current_stop` ya persistido, así que un plan
   con un stop imposible (por encima del precio) de antes del fix nunca se recuperaba: cada evaluación
   volvía a dar `EXIT_NOW` con el mismo stop imposible, sin importar cuántos candidatos válidos
   llegaran después. `update_trailing_stop` ahora recibe `price` y descarta un `current_stop >= price`
   como corrupto (nunca como "ya protegido") antes de comparar - autocura en la siguiente evaluación,
   sin migración de datos, porque `portfolio_risk_service.py` ya pasaba `price=exit_price` en esa
   llamada. Decisión explícita de **no** acotar también el ATR del Chandelier a la entrada (lo que la
   propia auditoría sugería como "menor, del mismo bloque"): medido contra el decaimiento exponencial
   real de un EWM(1/14), un ATR calculado sobre el histórico completo converge al mismo valor que uno
   acotado a la entrada en cuanto han pasado ~60-90 sesiones desde la compra (cualquier pico de
   volatilidad anterior a esa ventana ya decayó por construcción) - y acotarlo de verdad *rompería* el
   caso de una posición joven (menos de 14 barras desde la entrada, el mínimo de `ATR(14)`), devolviendo
   `None` donde hoy da un stop real. El diagnóstico de "un pico previo a la compra sigue ensanchando el
   trail" no se sostiene contra cómo decae un EWM - no se tocó.
2. **Semanal sin confirmar (`unknown`) se convertía en "alineación bajista total".** El fallback de
   `combine_timeframes` para un semanal sin suficiente historia (`timeframe_bias == "unknown"`, menos de
   ~3,85 años de barras semanales) pasaba directamente el sesgo *diario* como si fuera una alineación de
   dos temporalidades - `"bearish_aligned", -1.0` con solo un diario bajista, exactamente el literal que
   `exit_engine.py` compara para disparar `EXIT_NOW` ("Alineación bajista total: la temporalidad semanal
   y la diaria coinciden en tendencia bajista") - una afirmación falsa cuando la semanal es
   desconocida, no confirmadamente bajista. Afecta a cualquier ticker con ~1,15-3,85 años de historia
   (OPVs recientes, ETFs jóvenes). Corregido: la rama `unknown` siempre devuelve `"transitioning", 0.0`,
   en ambas direcciones - "alineado" afirma que dos temporalidades coinciden, y con la semanal sin
   confirmar solo hay una opinión, por fuerte que sea. La dirección alcista (antes también devolvía
   `"bullish_aligned"` en este caso) se corrigió igual, aunque el daño práctico de un falso positivo
   alcista es menor que el de un falso `EXIT_NOW`.
3. **`ensure_trade_plan` cerraba el plan vivo antes de saber si podía crear el sustituto.** Si el
   `entry_date` de la recompra no coincidía con el histórico de `ohlcv` disponible (un lote DCA'd más
   antiguo que `HISTORY_YEARS`), el plan existente se cerraba igualmente y luego la función devolvía
   `None` por `as_of_entry.empty` - sin plan, `portfolio_risk_service.py` se salta todo el motor de
   salida (sin stop, sin trailing, sin `exit_urgency`) para esa posición, y el problema nunca se
   recuperaba solo (la siguiente llamada encuentra el plan ya cerrado, sigue sin poder reconstruirlo,
   y así indefinidamente). Movido el `repo.close()` a después de la comprobación `as_of_entry.empty` -
   un plan que no se puede sustituir de verdad ya no se destruye.
4. **`reverse=True` invertía también el indicador de "sin puntuación", premiando la ausencia de datos.**
   `watchlist_service._sort_key` y `market_screener_service.apply_filters` devolvían `(valor is None,
   valor)` y ordenaban con `sorted(..., reverse=True)` - pero `reverse=True` invierte *todo* el tuple,
   incluido el booleano, así que un ticker sin `rs_rating`/`percentile_score` (200-252 barras: pasa el
   mínimo del screener pero no alcanza lo que pide RS Rating) encabezaba la lista en vez de ir al final.
   Corregido reescribiendo ambas funciones como claves estrictamente ascendentes (el "sin dato" ya
   codificado como el valor más alto de la tupla, el signo de la magnitud invertido cuando hace falta
   orden descendente) - nunca más combinadas con `reverse=True`.
5. **El pipeline premium analizaba el mismo ticker hasta 3 veces.** Un candidato que dispara varios
   setups (deliberado, `watchlist_service.py`) generaba una fila por setup en `build_watchlist`, y
   `build_premium_watchlist` no deduplicaba antes de cortar a `MAX_CANDIDATES_PER_TIER` - así que
   consumía varias plazas y corría la lectura más cara de la app (GARCH+Markov+Monte
   Carlo+walk-forward+Kelly) varias veces sobre datos idénticos. Nuevo `_dedupe_by_ticker`: conserva la
   primera aparición de cada ticker (la de mayor puntuación, ya que `items` llega ordenado) y guarda los
   demás setups que también disparó como `also_matched_setups` - mostrados como etiqueta secundaria
   ("también: ruptura con volumen"), no como filas nuevas. De paso, `prefilter_matches` pasa a contar
   tickers únicos, no pares (ticker, setup) - inflaba el denominador ~1,5-2x - y `analyzed` en
   `TierDiscardStats` ahora solo cuenta candidatos que de verdad produjeron una señal utilizable, no
   todos los que entraron al bucle (antes "15 analizados" podía significar que 5 habían fallado en
   silencio por OHLCV ausente o una excepción).
6. **El filtro de liquidez del universo dinámico comparaba precio/volumen sin convertir divisa.**
   `dynamic_universe_service.apply_liquidity_filter` comparaba el precio/volumen en divisa local del
   ticker contra umbrales en USD - GBp (peniques, LSE) hacía el filtro ~127x más laxo, SEK/NOK/DKK ~10x,
   EUR/CHF ~10%. Un valor sueco con 20M SEK/día (~$1,9M reales) pasaba el "$20M" de sobra. Corregido
   usando `MarketDataService.get_fx_rate` (que ya normaliza GBp/GBX a una tasa GBP real, dividida entre
   100) antes de comparar contra `MIN_PRICE`/`MIN_DOLLAR_VOLUME_20D` - con caché por divisa dentro de la
   misma llamada para no repetir la consulta FX por cada ticker que comparte moneda. La capitalización
   de mercado se deja sin convertir a propósito: si `info.market_cap` de yfinance ya viene
   normalizado a USD para un listado extranjero no se verificó contra la API real, y adivinar mal en
   cualquier dirección corrompería en silencio el único número que hoy funciona. También se añadió un
   guard de recencia (`MAX_STALE_DAYS=10`): un ticker cuya última barra disponible sea más vieja que eso
   (deslistado/parado desde antes del inicio de la ventana de 45 días) ya no pasa con precios rancios
   solo por tener `>= 20` barras en algún punto de esa ventana.
7. **Bugs menores, cada uno con su fix y su test**:
   - `sector_rotation_service.assess_sector_rotation`: `if overlap > best_overlap` resolvía cualquier
     empate a favor de la primera fase del diccionario ("recuperación temprana" - que además tiene 4
     sectores en su set frente a los 3 de las demás, un sesgo compuesto) - ahora un empate genuino entre
     fases no elige ninguna (`cycle_phase=None`), en vez de fingir una lectura que el dato no respalda.
     Y `laggards = ranked[-top_n:]` podía repetir sectores ya listados como líderes cuando había menos
     de `2*top_n` sectores con rango - el corte de laggards ahora nunca empieza antes de `top_n`.
   - `signal_performance_service._deduplicate_latest_per_ticker_and_day`: la clave era `(ticker, día)`,
     pero `PositionSignalSnapshot` es multi-cartera - dos carteras con el mismo ticker el mismo día
     colapsaban en una sola fila, perdiendo la observación de una de ellas. Clave ahora
     `(portfolio_id, ticker, día)` vía `getattr(..., None)` (que da `None` para `RecommendationSnapshot`,
     que no tiene `portfolio_id` y nunca lo necesitó).
   - `find_false_negatives` deduplicaba *antes* de filtrar por `signal == "hold"` - un `hold` a las 9:00
     seguido de un `watch` a las 17:00 el mismo día hacía que el dedupe se quedara con el `watch` (más
     reciente), que el filtro descartaba después, perdiendo el `hold` real que la función existe para
     encontrar. Ahora filtra primero, deduplica después.
   - **Causa raíz de la duplicación, no solo el síntoma**: `PositionSignalSnapshotRepository.save()` no
     tenía ningún guard de escritura - cada evaluación fresca (incluyendo un "Actualizar ahora" manual,
     que salta el caché a propósito) insertaba otra fila para el mismo (cartera, ticker, día). Ahora
     borra cualquier fila existente para ese (portfolio_id, ticker, día calendario) antes de insertar -
     mismo patrón delete-then-insert que `UniverseMembershipRepository.save_snapshot` ya usa. Sin test
     antes de este bloque (ningún repositorio de este proyecto con acceso a BD real lo tenía) - nuevo
     `tests/integration/test_position_signal_snapshot_repository.py`.
   - `market_screener_service.get_sector_performance`: era el único método pesado de esta clase sin
     caché durable - cada redespliegue de Render pagaba la descarga síncrona de 11 ETFs en la primera
     llamada a `/sectors`, `/sectors/rotation` o el `sector_rs_rank` del watchlist. Ahora usa
     `durable_cache` exactamente igual que `get_universe_snapshot` (los campos de `SectorPerformance` son
     todos tipos planos, así que `asdict`/`**kwargs` hace de (de)serializador sin código nuevo).
   - `HISTORY_DAYS=400` (~261-265 sesiones tras festivos europeos) dejaba un margen de solo ~10 sesiones
     sobre las >252 que exige `rs_raw_score` - un ETF europeo con festivos extra o un hueco de datos
     caía por debajo en silencio y desaparecía del ranking sectorial. Nueva constante
     `SECTOR_HISTORY_DAYS=500` (solo para `get_sector_performance`, `HISTORY_DAYS` general sin cambios),
     con logging explícito de qué sectores quedan excluidos y por qué.
   - **Verificado y descartado**: la afirmación de que `pd.bdate_range(end=<fin de semana>, periods=25)`
     devuelve 24 elementos (y por tanto 6 tests fallarían en fin de semana) no se reprodujo contra la
     versión de pandas real de este proyecto (3.0.5) - `bdate_range` calcula hacia atrás desde el último
     día hábil antes de `end` y siempre devuelve exactamente `periods` elementos, caiga `end` en fin de
     semana o no. No se tocó ese test.
   - Pendiente, explícitamente no abordado en este bloque: `market_screener_service.py`'s ranking de
     industrias mezcla métricas incompatibles (ETF ponderado por capitalización vs. media equiponderada
     de 2-5 nombres curados) en una sola lista ordenada - es una decisión de diseño de UI (marcar el
     método visualmente, o separar los rankings), no un bug de datos incorrectos, y queda para una
     pasada de UI dedicada.

**Tests**: ~40 tests nuevos/actualizados a través de `test_trade_manager.py`, `test_multi_timeframe.py`,
`test_trade_plan_service.py`, `test_watchlist_service.py`, `test_market_screener_service.py`,
`test_premium_watchlist_service.py`, `test_dynamic_universe_service.py`, `test_sector_rotation_service.py`,
`test_signal_performance_service.py`, más el `test_position_signal_snapshot_repository.py` nuevo y dos
tests de integración nuevos en `test_market_api.py` para la caché durable de sectores. Suite completa
(unit+integración) en verde, `ruff check app tests scripts` limpio, `npm run lint`/`npm run build` limpios.

## 21. Tercera auditoría independiente — Bloque F: universo dinámico, setups semanales y percentil real (agosto 2026)

Ninguno de los diez puntos de este bloque toca los pesos de `recommendation_engine.py` - no hay bump de
`ENGINE_VERSION`. Todo el trabajo es en `watchlist_service.py`, `premium_watchlist_service.py`,
`market_screener_service.py`, `dynamic_universe_service.py`, `factor_ablation_study.py` y el par
`ticker_analysis_service.py`/`market_data_service.py` (fecha de earnings).

1. **Universo dinámico conectado al camino en vivo, con un cribado barato antes del cálculo caro
   (F-1)**: cierra el punto que §15 dejaba abierto. `market_screener_service.get_universe_snapshot` lee
   ahora `UniverseMembershipRepository` cuando hay una `db` disponible y sustituye por completo la lista
   curada de `market_universe.py`; si el universo leído supera `CHEAP_SCREEN_KEEP_TOP_N=400`,
   `dynamic_universe_service.apply_cheap_price_volume_screen` (precio/volumen del propio OHLCV ya
   descargado, cero llamadas de red por ticker) lo recorta antes de que `_build_raw` calcule un solo
   indicador. Efecto práctico hoy: nulo (solo existe un snapshot real en BD, ~900 US + ~240 Europa,
   ambos ya por debajo del umbral) - el cambio es estructuralmente correcto para cuando el universo
   dinámico crezca, no una mejora medible todavía.
   - El mismo punto-en-el-tiempo alcanza también al estudio de ablación:
     `filter_samples_by_point_in_time_membership` descarta una muestra histórica si el ticker no era
     miembro genuino del universo en la fecha de esa muestra (usando el snapshot más cercano anterior, o
     el snapshot más antiguo disponible como fallback honesto para fechas previas a cualquier
     snapshot) - reemplaza la comparación anterior contra el snapshot *de hoy*, que no reducía sesgo de
     supervivencia en absoluto. Mismo efecto práctico nulo hoy por la misma razón (un solo snapshot real
     en BD todavía), correcto de cara al futuro.
2. **Nivel semanal reescrito con setups propios, en vez de reglas ad hoc (F-2)**: `_medium_term_reasons`
   (un conjunto de reglas sueltas, sin percentil, sin arquitectura de setup) se sustituyó por
   `MEDIUM_TERM_SETUPS` (`FAST_GOLDEN_CROSS`, `FAST_CROSS_IMMINENT`, `STAGE2_LEADER`) - exactamente el
   mismo patrón que `SHORT_TERM_SETUPS` ya usa en el nivel diario: un `WatchlistItem` por setup que
   dispara, con su propio percentil (ver punto 4). El horizonte de Monte Carlo del nivel semanal en el
   premium watchlist pasa de "3m" a "1m" - "1m" es el horizonte de holding real de un setup semanal
   (días a pocas semanas), no 3 meses.
3. **RS Rating del nivel diario sustituido por un percentil de verdad (F-3)**: `rs_rating` es un cálculo
   de 52 semanas (`technical_analysis.rs_raw_score`/rating relativo del universo) - una medida
   *semanal/mensual* de fondo, no una lectura del horizonte de holding real del nivel diario (días a
   pocas semanas). Nuevo `watchlist_service.percentile_rank_by_ticker(snapshots, field)` (percentil
   transversal genérico, público); el nivel diario del premium watchlist ahora usa el percentil de
   `mansfield_rs_4w` (fuerza relativa a 4 semanas) en vez del `rs_rating` crudo. El nivel semanal sigue
   usando `rs_rating` sin cambios - es la temporalidad para la que ese cálculo sí tiene sentido.
4. **Cada setup puntúa con sus propios campos y signos, no un composite de 7 campos compartido (F-4)**:
   antes, `setup_percentile_scores` usaba el mismo composite de 7 campos para los 4 setups diarios (con
   un único caso especial: invertir `change_1w` para `oversold_bounce`) - así que `breakout_volume`,
   `trend_continuation` y `pullback_to_support` puntuaban de forma casi idéntica pese a ser tesis
   distintas. Nuevo `SETUP_PERCENTILE_FIELDS: dict[str, dict[str, bool]]` - un conjunto de campos y
   signos (mayor-es-mejor / invertido) por cada uno de los 7 setups (4 diarios + 3 semanales),
   documentado inline por qué cada campo entra o se invierte para ese setup concreto.
5. **Corrección de un docstring falso (F-5)**: el docstring de `watchlist_service.py` citaba una
   evidencia de ablación que en realidad mide algo distinto (factor-level, no setup-level) - corregido
   para no reclamar una validación que no existe todavía, con puntero explícito al punto 6.
6. **Medición real por tipo de setup, no solo por factor (F-6)**: el estudio de ablación ya medía
   *factores* (¿este campo por sí solo predice el retorno futuro?) pero nunca *setups* (¿esta
   combinación de reglas, tal y como dispara en producción, gana dinero de verdad?). `FactorSample` ahora
   guarda `exit_reason`/`bars_held`/`mae_pct`/`mfe_pct`/`risk_pct` directamente de
   `backtest_engine.TripleBarrierLabel` (antes se descartaban); `compute_setup_outcome_stats` agrega win
   rate (`exit_reason == "target"`), expectancy en R (`media(retorno_adelante / risk_pct)`), duración
   mediana en barras y MAE p80 - por setup, por horizonte. Expuesto vía
   `ablation_report_service.load_setup_outcome_stats`/`setup_outcome_by_name` (lee
   `factor_ablation_report_v3_h{N}_setup_outcomes.csv` - prefijo `v3` nuevo, **no** sobrescribe los CSV
   `v2` que §17 y la UI ya citan) y adjuntado a cada `WatchlistItemResponse`/`PremiumWatchlistItemResponse`
   como `setup_outcome_stats` cuando existe medición para ese setup+horizonte.
7. **Evidencia real, ejecutándose (F-7)**: `factor_ablation_study.py --horizons 5 10 21 63 126 --regions
   us europe --use-dynamic-universe --temporal-split --out-prefix docs/factor_ablation_report_v3` -
   lanzado contra el universo dinámico real (~900 US + ~240 Europa a 2026-08-21). **Todavía en curso al
   cierre de este bloque** - es un proceso de horas contra ~1.140 tickers × 2 regiones × 5 horizontes.
   Sus CSV de salida (incluidos los `*_setup_outcomes.csv` del punto 6) se verifican y se commitean por
   separado en cuanto termine - no se reclama aquí ninguna conclusión de esa corrida todavía, solo que
   el mecanismo que la produce está construido, probado (`test_factor_ablation_study.py`) y en marcha.
8. **Diversificación por sector y correlación, no solo por puntuación (F-8)**:
   `premium_watchlist_service._select_diversified` recorre la lista de candidatos ya ordenada por
   puntuación y descarta (no sustituye-y-reintenta) cualquiera que llevaría a un sector por encima de
   `MAX_PER_SECTOR=3`, o cuyos retornos de 60 días correlacionan `>= MAX_CANDIDATE_CORRELATION=0.7` con
   un nombre ya aprobado (reutilizando `portfolio_construction_service.compute_correlation_matrix`, sin
   reimplementar nada). Decisión explícita: no se añade una penalización de puntuación adicional al lado
   del bonus de sector fuerte ya existente - el propio tope estructural de sector ya es la fuerza
   contraria que pedía el encargo; apilar una segunda penalización blanda habría sido redundante.
9. **Filtro de liquidez en vivo (no solo en el batch mensual) + exclusión por earnings (F-9)**: la misma
   corrección de divisa del punto 6 de §20 (`dynamic_universe_service.usd_price_and_dollar_volume`) se
   extrajo como función compartida y ahora se aplica también dentro del bucle por ticker de
   `get_universe_snapshot` (con la tasa FX cacheada por divisa para toda la llamada, no por ticker - sin
   violar la regla de "sin llamadas de red por ticker"). Además, nuevo puerto
   `MarketDataProvider.get_next_earnings_date` (implementado en `YFinanceProvider` vía
   `yf.Ticker(ticker).calendar`), usado **solo** en los dos caminos ya establecidos de "profundizar en un
   único ticker" (`TickerAnalysis.analyze()` y el bucle acotado ≤15/nivel de `premium_watchlist_service`)
   - nunca en el snapshot del universo completo, para no violar la regla de llamadas de red por ticker en
   caminos calientes. El nivel diario excluye candidatos con earnings dentro de
   `SIGN_CHECK_HORIZON_DAYS=21` (el horizonte de holding real reutilizado, no una constante nueva); los
   niveles semanal/mensual conservan el candidato pero lo marcan visualmente.
   - **Regresión encontrada y corregida durante este mismo punto**: el nuevo filtro de liquidez en vivo
     rompió ~20 tests de integración porque `FakeMarketDataProvider._random_walk` (fixture de tests)
     generaba un volumen (500-5.000 acciones/día) y un suelo de precio (`max(1.0, ...)`) irrealmente
     bajos para simular un universo curado de grandes capitalizaciones - corregido a 500k-5M
     acciones/día y un suelo de $20, con comentario explícito de por qué esto es un arreglo de fidelidad
     del fixture (nunca se pretendió simular penny stocks) y no el patrón prohibido de "arreglar un test
     haciéndolo más permisivo" (que es sobre no debilitar aserciones de *regla de decisión*, no sobre
     datos de prueba poco realistas).
10. **Huecos del frontend cerrados (F-10)**: `percentile_score` visible en la tarjeta de watchlist,
    `premium_score` (no la puntuación interna de recomendación) como insignia del premium watchlist,
    claves de React corregidas para incluir `setup` (evitaba colisiones al mostrar el mismo ticker en
    varios setups), tarjeta de estadística de setup (punto 6) visible en ambas listas, insignia de
    earnings, aviso de descartes cuando 0 candidatos sobreviven, y retirada de la insignia del backtest
    legacy walk-forward del premium watchlist (redundante con el backtest de triple-barrera, ver también
    Bloque H).

**Tests**: ~55 tests nuevos/actualizados en `test_watchlist_service.py`, `test_premium_watchlist_service.py`,
`test_factor_ablation_study.py`, `test_ablation_report_service.py`, `test_dynamic_universe_service.py`, más
tests de integración nuevos en `test_market_api.py`. Suite completa (unit+integración): 870 tests en verde,
`ruff check app tests` limpio.

## 22. Tercera auditoría independiente — Bloque G: mapa de relaciones entre empresas (agosto 2026)

Función nueva explícitamente pedida por el encargo (única excepción a "no escribas módulos nuevos" de
esta ronda, junto con el propio Bloque F): `relationship_map_service.py` +
`GET /api/v1/market/tickers/{ticker}/relationships` + una tarjeta en `TickerAnalysisPanel` (pestaña
"Relaciones"). No hay ningún proveedor gratuito y fiable de cadena de suministro estructurada (FactSet
Supply Chain y similares son de pago), así que el mapa se construye en tres capas, de la más objetiva a
la más especulativa - **nunca mezcladas como si fueran la misma clase de evidencia**, cada una etiquetada
visualmente por fiabilidad en la interfaz:

1. **Estadística (la más fiable, y gratis)**: correlación de retornos a 60/250 sesiones, beta relativa,
   correlación con desfase (lead-lag, k∈[-5,+5] sesiones - qué activo se mueve primero), co-movimiento en
   días extremos (±2σ) y divergencia actual frente a la correlación histórica. Todo calculado sobre el
   OHLCV que `MarketScreenerService.get_universe_snapshot` ya tiene en memoria (nuevo getter público
   `get_cached_ohlcv()`) - cero llamadas de red nuevas por par.
   - **Bug real encontrado durante el desarrollo, no solo durante la revisión**: una ventana de 60
     sesiones con varianza cero hace que `pandas.Series.corr()` devuelva NaN, y `nan is not None` es
     `True` en Python - un guard `corr_60d is not None` dejaba pasar un NaN silencioso a la aritmética de
     `is_diverging`. Corregido con `pd.notna()` explícito en el origen: `corr_60d`/`corr_250d` son `None`
     (nunca un NaN crudo) cuando la correlación no está definida.
2. **Pares de sector/industria (contexto, gratis, ya en el repo)**: mismos `market_universe.Industry`
   del ticker analizado, con su RS Rating y estado técnico actual. Sin cuadrante RRG todavía - Bloque E
   (RRG) quedó fuera del alcance de esta pasada; si se construye después, es la extensión natural aquí.
3. **Relaciones declaradas en documentos SEC (la más especulativa, EE.UU. solamente)**: búsqueda de texto
   completo en EDGAR (gratuita, sin clave) de qué otras empresas mencionan a la analizada en su propio
   10-K/10-Q reciente - la señal más literal de relación comercial disponible sin pagar un feed. Cacheado
   ≥30 días vía `durable_cache` (una relación de un 10-K cambia una vez al año, no en cada request) y
   llamado una sola vez por análisis, igual que `get_ticker_info` - nunca desde un camino caliente.
   `disclosed_available=False` explícito (nunca una lista vacía silenciosa) para Europa o cuando EDGAR
   falla/tarda - degrada a las capas 1 y 2 sin romper la pantalla.
   - **Alcance explícito, no construido en esta pasada**: extraer la sección de concentración de clientes
     del propio 10-K de la empresa analizada (el otro sentido de la relación que pedía el encargo)
     necesitaría resolver el CIK del ticker y parsear el documento completo, no solo la búsqueda de texto
     - más frágil de lo que esta pasada puede verificar con confianza sin pruebas contra la API real.
     Queda señalado aquí como pendiente, no construido a medias.
- **Región no siempre conocida por el llamador**: a diferencia de cualquier otro endpoint de `market/`,
  este se alcanza desde una búsqueda de texto libre ("Analizar activo") que puede no saber a qué región
  pertenece el ticker. `region` no tiene valor por defecto fijo en este endpoint concreto - cuando el
  llamador lo omite, se resuelve con `market_universe.region_of(ticker)` (el mismo "mejor esfuerzo" ya
  usado para este caso exacto en `technical_analysis.closed_bars`/`benchmark_for_ticker`), no con un "us"
  fijo que buscaría un ticker europeo en el universo/caché equivocado.
- El resultado termina siempre en candidatos accionables: cada ticker relacionado en las tres capas es un
  botón que lanza su propio análisis completo (reutiliza `search()` del propio panel), no solo una
  etiqueta - "el mapa debe terminar en candidatos accionables, no en un diagrama bonito".

**Tests**: 22 tests unitarios nuevos en `test_relationship_map_service.py` (estadística, sector/industria,
EDGAR con `requests.get` mockeado - nunca red real) + 5 tests de integración nuevos en `test_market_api.py`
para el endpoint (tres capas para un ticker de EE.UU. con EDGAR mockeado, capa 3 no disponible en Europa,
resolución automática de región sin el parámetro, ticker desconocido con capas vacías en vez de error).
Suite completa (unit+integración): 870 tests en verde, `ruff check app tests` y `npm run lint`/
`npm run build` limpios.

**Bug real encontrado verificando el despliegue en producción, no en los tests**: nada más desplegar,
`GET /tickers/AAPL/relationships` devolvía `"statistical": []` - AAPL, con toda seguridad, tiene
correlaciones reales con el resto del universo tecnológico. Causa raíz en
`market_screener_service.get_universe_snapshot`: sus dos atajos de caché (TTL en proceso, y el durable
que sobrevive a un redeploy) devolvían el snapshot sin pasar nunca por la descarga de OHLCV, así que
`_ohlcv_cache` se quedaba vacío en cualquier worker que no hubiera hecho el recálculo completo desde que
arrancó - `get_cached_ohlcv()` (la fuente de `compute_statistical_relations`) devolvía `{}` en silencio.
`get_proximity_matches` ya tenía este mismo hueco, documentado en su día como "silencioso pero aceptable"
- aceptable para un screener que de por sí puede devolver `[]` con pocas consecuencias, no para una capa
que se anuncia explícitamente como "la más fiable" de una función nueva. Arreglado con
`_ensure_ohlcv_cache_warm`: tras cualquiera de los dos atajos de caché, si `_ohlcv_cache` está frío,
descarga el OHLCV de los tickers del propio snapshot (la misma llamada por lotes de siempre, sin repetir
el cálculo de indicadores - Bloque F-1 ya midió que esa descarga cuesta igual para 170 o 1.000 tickers)
antes de devolver. Verificado end-to-end contra el backend real en Render tras el fix: la misma llamada a
AAPL ahora completa en ~104s (primera vez, universo + búsqueda EDGAR en vivo) con datos reales en las tres
capas. `sector_peers: []` para AAPL en ese mismo chequeo, en cambio, **no** era un bug -
`market_universe.py` no tiene a AAPL en ningún `Industry.tickers` (solo en `TICKER_CAP_TIER`), así que la
lista vacía es la respuesta honesta, no un fallo. 6 tests nuevos en `test_market_screener_service.py`
reproducen ambos atajos de caché contra un `_ohlcv_cache` frío.

## 23. Tercera auditoría independiente — Bloque H: contradicciones de interfaz (agosto 2026)

Bloque puramente de frontend - ningún archivo de `app/` cambia, así que no hay tests de `pytest` nuevos
ni bump de `ENGINE_VERSION`; verificado con `npm run lint` y `npm run build` limpios. La regla general
aplicada a los siete puntos: cuando dos lecturas del sistema genuinamente responden preguntas distintas
(comprar es un checklist de puntuación, vender/gestionar una posición es el motor de salida - ver §8), la
contradicción **se declara explícitamente en la interfaz**, nunca se oculta ni se fuerza a que las dos
lecturas coincidan sin evidencia nueva que lo justifique.

1. **Cruce bajista inminente con confianza suficiente, bajo un veredicto COMPRAR**: el encargo original
   apuntaba a un veto (B-1.3) para resolver esto - fuera de alcance esta pasada (Bloque B no se tocó).
   `ImminentCrossBadge` ya declaraba que la proyección "no afecta a la puntuación de compra"; ahora,
   además, cuando el cruce es bajista **y** ya tiene confianza suficiente para que el motor de salida
   actuase (`clearsActionBar`) **y** el veredicto es "comprar", añade una nota explícita bajo el propio
   badge señalando que son preguntas distintas y que, con la posición todavía sin abrir, vale la pena
   esperar a que se resuelva - sin inventar un veto que no existe todavía.
2. **Dos backtests a distinto horizonte, apilados sin jerarquía**: `TripleBarrierBacktestCard.jsx`
   (Segunda auditoría, Bloque 2/4) ya documentaba en su propio código que ambos se muestran a propósito,
   "cada uno etiquetado por lo que realmente es" - retirar uno habría revertido esa decisión ya deliberada
   sin evidencia nueva. El problema real era de **orden**: el legacy (walk-forward, sin stop/objetivo
   reales) aparecía primero, y su propio texto ya decía "ver más abajo el backtest de triple-barrera... para
   la lectura honesta" - invertido respecto a lo que el texto prometía. Corregido intercambiando el orden
   en `TickerAnalysisPanel.jsx` (triple-barrera primero, con `<h3>` marcado "método principal"; walk-forward
   después, marcado "secundario") y actualizando "ver más abajo" → "ver arriba" en `BacktestCard.jsx`.
3. **Monte Carlo con P(stop) > P(objetivo), sin aviso, mientras Kelly sigue sugiriendo tamaño**:
   relacionado con el punto 4 pero no siempre el mismo caso - `ticker_analysis_service.py` deriva
   `win_probability` de Kelly de estas mismas dos probabilidades de Monte Carlo
   (`win_probability_from_barriers`), pero con un ratio riesgo/beneficio suficientemente grande, Kelly
   puede seguir siendo positivo pese a P(stop) > P(objetivo) - el ratio compensa la asimetría. Dos avisos
   distintos en `PositionSizingCard.jsx`, cada uno con su propio texto: Kelly negativo/nulo (punto 4,
   menciona la asimetría de Monte Carlo cuando también está presente) y Monte Carlo desfavorable con Kelly
   todavía positivo (el ratio riesgo/beneficio compensa, pero vale la pena mirar ambos números) - sin
   forzar que ninguno de los dos cambie.
4. **Kelly negativo bajo un veredicto COMPRAR**: `kelly_criterion.recommend_position_size` ya limitaba
   `recommended_position_pct` a un mínimo de 0% y ya explicaba en su `rationale` que "la ventaja esperada
   no compensa el riesgo" cuando el Kelly completo es negativo o nulo - pero nada conectaba visualmente
   ese 0% con el veredicto COMPRAR mostrado justo arriba. Resuelto por el mismo aviso del punto 3.
5. **Banner de "signo contrario" binario, siempre igual de alarmante sea 1 de 13 factores o 9 de 13**:
   `RecommendationCard.jsx` ahora calcula la fracción real de factores activos con signo contrario
   (`mismatchedTriggered.length / triggered.length`) y muestra el conteo explícito ("N de M factores...")
   en vez de un banner de sí/no. Con mayoría (`>= 50%`) usa el tono de aviso serio ya existente; en
   minoría, un tono neutro/atenuado nuevo (`--minor`). El umbral del 50% es una decisión de presentación
   sobre evidencia ya medida (cuántos factores discrepan), no un peso nuevo de `recommendation_engine.py`
   - no requiere estudio de ablación.
6. **`SectorForecastCard`: badge "sin señal estadística clara" con números impresos de todas formas**:
   `forecast_5d_return`/`prob_bullish_21d` son proyecciones reales del ajuste de Markov (nunca
   fabricadas), pero mostrarlas junto al badge que ya dice "no hay estructura distinguible del azar" las
   presenta como igual de fiables que una proyección genuina. Ahora se ocultan por completo cuando
   `has_statistical_structure` es falso, sustituidas por una nota explícita - `current_state_label` (una
   descripción, no una proyección direccional) se sigue mostrando siempre.
7. **`SectorStrength` ordena por retorno absoluto del periodo, ignorando `rs_rank`**: la propia etiqueta
   de la vista es "fuerza relativa", pero el orden de las barras solo miraba el retorno crudo del botón de
   periodo seleccionado (1D/1S/1M/3M/6M/1A) - un sector +5% en un mercado +8% es un rezagado en RS aunque
   su barra sea positiva, y podía aparecer por delante de un líder real. Ahora ordena por `rs_rank` (el
   mismo percentil de periodos combinados que ya decide "sector fuerte/rezagado" en otras vistas de la
   app, p. ej. `Watchlist.jsx`), con el retorno del periodo elegido de vuelta como criterio de desempate
   solo cuando ninguno de los dos sectores comparados tiene `rs_rank` (menos de 252 sesiones de
   historia). El valor de cada barra sigue siendo el retorno del periodo elegido - solo el orden cambia;
   el `rs_rank` de cada sector se añadió al tooltip para que la razón del orden sea visible, no implícita.

## 24. Cuarta auditoría independiente — resolución de recomendaciones (agosto 2026, `ENGINE_VERSION` → v5)

Un análisis independiente (publicado como artefacto, "Cuarta Auditoría Independiente") propuso 13
hallazgos priorizados en 5 áreas. Por instrucción explícita del propietario se omitió la sección A
(seguridad y operación - API sin autenticación, sin CI/CD, sin monitorización) de esta pasada; el resto
se resolvió, incluyendo retomar los Bloques B/C/D/E del encargo original que habían quedado
explícitamente fuera de alcance de la Tercera auditoría (decisión también explícita del propietario,
entre "dejarlo pendiente", "cerrarlo formalmente" y "retomarlo ahora").

1. **Bloque B (B-1.3): veto del par rápido EMA21/55**. El encargo original apuntaba a este veto como la
   solución "correcta" a la contradicción de Bloque H-1 (cruce bajista inminente bajo un veredicto
   COMPRAR) - no construido en su momento por estar fuera de alcance de esa pasada. Nuevo
   `technical_analysis.detect_fast_pair_bearish_veto(close)`: un cruce bajista confirmado (o proyectado,
   con R² ≥ 0,6 - el mismo umbral que `exit_engine.py` ya usa para su propio par rápido SMA20/50, reutilizado
   en vez de inventado) en un par EMA21/55 deliberadamente distinto de cualquier otro que el sistema ya
   siga (SMA20/50/200 del checklist, SMA50/200 de `ma_cross`, SMA21/50 de `multi_timeframe.py`). Cuando
   dispara, `build_recommendation` degrada "comprar" a "esperar" (nunca a "evitar" - el checklist puede
   seguir siendo genuinamente alcista, "esperar" es la lectura honesta de "todavía no, no de "esto es
   malo"") y expone `veto_reason` como campo propio de `Recommendation`, separado de `factors` (nunca
   entra en la puntuación) - mostrado en `RecommendationCard.jsx` como aviso explícito, distinto de los
   badges de cruce inminente SMA50/200 y SMA21/50 que ya existían (ambos pares siguen sin veto propio,
   por diseño - B-1.3 pidió específicamente este par nuevo). Cableado en los tres puntos donde
   `build_recommendation` se llama de verdad: `compute_core_signals` y `_confirmed_recommendation`
   (`ticker_analysis_service.py`) y `replay_recommendation_at` (`walk_forward_backtest.py`, para que el
   backtest simule el sistema que de verdad corre en producción, no una versión sin el veto). `ENGINE_VERSION`
   → v5 (ningún peso existente cambió, pero un "comprar" puede ahora volver "esperar" por una razón que
   la puntuación nunca llevaba). Es un chequeo exclusivamente del lado de compra, evaluado dentro de la
   propia decisión de comprar - nunca importa ni es importado por `exit_engine.py` (§8 sigue intacto:
   comprar y vender siguen siendo preguntas distintas). Sin ejecutar todavía el estudio de ablación
   específicamente contra este veto (la evidencia de que suprime más falsos positivos de los que cuesta
   en oportunidades reales queda pendiente de una corrida dedicada) - construido porque el propio
   encargo lo pidió explícitamente por nombre, no por intuición propia; documentado aquí como una
   limitación honesta, no ocultada. 21 tests nuevos (5 en `test_technical_analysis.py` para la detección
   en sí, 5 en `test_recommendation_engine.py` para la integración del veto, 1 en
   `test_walk_forward_backtest.py` para el cableado del replay).

2. **Bloques C/D: batería de escenarios dorados**. Nuevo `tests/unit/test_golden_scenarios.py` - series
   de precio sintéticas con forma de patrón técnico reconocible (ruptura de Fase 2 con volumen, tendencia
   bajista confirmada, lateral sin dirección clara, divergencia bajista de OBV) corridas de punta a punta
   por el pipeline real (`technical_analysis.py` → `recommendation_engine.build_recommendation`, incluido
   el veto nuevo de B-1.3), no contra una sola función aislada como hace el resto de `tests/unit/` a
   propósito. Incluye el escenario dorado más directamente relevante para el punto 1: la misma serie
   (tendencia alcista genuina, empezando a decaer con suavidad) puntúa idéntico con o sin el veto
   aplicado - la prueba más clara de que el veto cambia el veredicto sin tocar la puntuación del
   checklist. `exit_engine.py` (lado de venta) ya tenía esta clase de cobertura de escenario completo en
   su propio archivo de 47 tests - esta batería es el equivalente del lado de compra que nunca existió.

3. **Bloque E: reconstrucción del cuadrante RRG**. Nuevo `sector_rrg_service.py` (módulo nuevo,
   explícitamente permitido) - Relative Rotation Graph: RS-Ratio (fuerza relativa normalizada frente al
   benchmark, z-score móvil de 100 sesiones centrado en 100) cruzado con RS-Momentum (el ritmo de cambio
   de ese RS-Ratio, mismo tipo de normalización) en los cuatro cuadrantes clásicos (leading/weakening/
   lagging/improving). Complementa, no sustituye, a `sector_rotation_service.py` (que solo mira qué
   sector lidera hoy contra el patrón de ciclo económico) - añade el eje de momentum que ese servicio
   nunca tuvo, distinguiendo un líder que sigue acelerando de uno que ya está perdiendo fuelle. Nota de
   honestidad explícita en el propio módulo: la normalización exacta que StockCharts/JdK usa en su
   producto comercial no es pública - esto reproduce el comportamiento cualitativo del RRG con una
   normalización estándar de implementaciones abiertas, no pretende ser un clon numérico exacto. Nuevo
   endpoint `GET /api/v1/market/sectors/rrg` y tarjeta `SectorRrgCard.jsx`/`SectorRrgChart.jsx` (gráfico de
   dispersión con la cola de trayectoria de cada sector, coloreado por cuadrante - no por sector, con
   hasta 11 sectores distinguir por color de identidad obligaría a más matices de los que un vistazo puede
   separar). 15 tests unitarios (`test_sector_rrg_service.py`) + 3 de la capa de caché en
   `market_screener_service.py` (mismo patrón de `get_sector_forecast`: fetch propio de OHLCV, no
   reutiliza `_ohlcv_cache`) + 1 de integración.

4. **DEUDA-1: calibración del Chandelier Exit, medida por primera vez**. Nuevo
   `scripts/chandelier_calibration_study.py` (sibling script, no una extensión de
   `factor_ablation_study.py`: ese script fija `vol_regime=None` en cada muestra por coste - un refit de
   GARCH por ticker a escala de universo completo sería prohibitivo - así que estructuralmente nunca
   ejercita los multiplicadores por régimen; este script paga ese coste a propósito, sobre una muestra
   pequeña deliberada de 60 tickers, el mismo compromiso que el estudio de ablación original tomó a
   ~217 tickers antes de que el universo creciera). Requirió un refactor puro y verificado de
   `backtest_engine.py` (`find_triple_barrier_entries`, extraído de `run_triple_barrier_backtest` sin
   cambiar su comportamiento - 35/35 tests existentes en verde) para poder agrupar etiquetas de muchos
   tickers antes de agregar una sola vez, en vez de un `TradingMetrics` ya promediado por ticker.
   **Resultado real de la corrida** (`docs/chandelier_calibration_report.csv`, 60 tickers, horizonte 21
   días): desplazar los cuatro multiplicadores +0,5 mejora win_rate (0,436 vs. 0,408), expectancy (+0,23%
   vs. +0,14%), profit_factor (1,10 vs. 1,06) y reduce el drawdown (-0,89% vs. -0,93%) frente a los
   valores actuales; desplazarlos -0,5 empeora los cuatro. El multiplicador de profit-lock (1,75/2,0/2,25)
   no mostró ninguna diferencia - probablemente porque las operaciones de esta muestra rara vez alcanzan
   +2R antes de resolverse por otra vía. **No se ha tocado ningún valor en `trade_manager.py`** - esto es
   evidencia real y direccional, sobre una muestra pequeña y una sola corrida, no autorización para
   recalibrar; queda documentado para que el propietario decida, con el mismo estándar que cualquier otro
   peso de este sistema.

5. **DEUDA-3: método de cálculo visible en el ranking de industrias**. Nuevo campo
   `IndustryPerformance.performance_method` ("etf" | "basket_average") - refleja lo que realmente pasó
   para esa fila (un ETF *configurado* sin datos también cae a la media de la cesta curada), no solo si
   `industry.etf` está poblado. `IndustryCards.jsx` ahora etiqueta cada fila "· XLK" (vía ETF real) o
   "· cesta curada" (promedio equiponderado) en vez de mostrar el símbolo del ETF incluso cuando los
   números en realidad vinieron del promedio.

6. **TEST-1: cobertura de `market_screener_service.py`**. 9 tests nuevos cubriendo las tres rutas de
   caché (fría / en proceso / durable) de `get_industry_performance` y `get_sector_rrg` - exactamente la
   clase de ruta que el bug real de `_ohlcv_cache` (commit `7388ddc`, ver §22) demostró que ningún test
   existente ejercitaba.

7. **FE-1: `setup_label` como origen único de verdad**. El backend ya mantenía su propio
   `watchlist_service.SETUP_LABELS` en español para los textos de `reasons` - completamente
   desconectado del `SETUP_LABELS` que el frontend mantenía por su cuenta en `format.js` (que ya se
   había desincronizado una vez, ver §21 punto 10). En vez de solo añadir un test que detecte la próxima
   desincronización, se eliminó la clase de bug: `WatchlistItem`/`PremiumWatchlistItem`/
   `StatisticalRelation`/`SectorPeer` ganan una propiedad `setup_label` (y `PremiumWatchlistItem` un
   `also_matched_setup_labels`) calculada desde el único diccionario del backend; expuesta en las cuatro
   respuestas de API correspondientes. El frontend ahora prefiere `item.setup_label` y solo cae al mapa
   local como red de seguridad (`item.setup_label ?? SETUP_LABELS[item.setup] ?? item.setup`), nunca
   como fuente de verdad. Incluye un test que falla en cuanto un setup nuevo no tenga entrada en
   `SETUP_LABELS` - la comprobación real que sustituye la necesidad de un test de contrato aparte.

8. **FE-2: riesgo agregado visible en "Acciones requeridas hoy"**. `TodayActionsPanel` solo miraba
   señales por posición; ahora recibe `construction` (ya obtenido para `PortfolioConstructionPanel`,
   Segunda auditoría Bloque 2) y, cuando `aggregate_risk.exceeds_limit` es verdadero, añade una entrada
   de "Cartera completa" al principio de la lista con el mismo tono `exit_now` - el límite del 6% deja de
   ser algo que solo se ve si el usuario baja hasta el panel secundario.

**Tests**: ~50 tests nuevos/actualizados en total a través de `test_technical_analysis.py`,
`test_recommendation_engine.py`, `test_walk_forward_backtest.py`, `test_golden_scenarios.py` (nuevo),
`test_sector_rrg_service.py` (nuevo), `test_chandelier_calibration_study.py` (nuevo),
`test_market_screener_service.py`, `test_watchlist_service.py`, `test_premium_watchlist_service.py`,
`test_relationship_map_service.py`, `test_backtest_engine.py`, más 2 tests de integración nuevos en
`test_market_api.py`. `ruff check app tests scripts` limpio, `npm run lint`/`npm run build` limpios.

## 25. Reconstrucción de niveles/triggers (septiembre 2026) — Fases 1-10 completas del lado de este reconstructor

Encargo explícito del propietario: sustituir el checklist ponderado de 26 factores (secciones 1-24
arriba) por un sistema de niveles/triggers que responda exactamente 4 preguntas - qué hacer hoy con lo
que ya se tiene, qué está a punto de disparar una entrada, si una entrada concreta es buena, y si el
sistema está funcionando - en vez de una puntuación agregada opaca. Nota de procedencia: el texto
literal del encargo (20 partes) salió de contexto durante la compactación de la sesión que lo ejecutó;
el propietario dio luz verde explícita para continuar sobre la mejor reconstrucción posible de ese
encargo en vez de repetirlo, documentando cada decisión de diseño en el propio código/commits para
poder corregirla después. Lo que sigue es el resumen de auditoría; el razonamiento completo de cada
decisión vive en el docstring del módulo/función correspondiente, no solo aquí.

**Fase 1 — borrado de todo lo sin evidencia cruzada.** Retirados por completo (código y tests, no solo
desconectados): Monte Carlo, cadena de Markov, GARCH(1,1), criterio de Kelly, exponente de Hurst/ADF,
badge de entry-timing, rotación sectorial/RRG, informe de ablación peso-vs-signo, watchlist "Premium"
(embudo de 3 niveles) y sus sugerencias de coste de oportunidad, contexto macro (curva de tipos, paro,
IPC vía FRED), Fear & Greed compuesto, proxy de liquidez en dólares, capa de menciones SEC EDGAR del
mapa de relaciones, y - dentro del propio checklist de recomendación - los factores de fundamentales
(crecimiento/margen/apalancamiento), tenedores institucionales y consenso de analistas. El único
superviviente real de GARCH es el bucket de volatilidad del Chandelier Exit, ahora alimentado por un
percentil de ATR/precio (`technical_analysis.atr_percentile`/`volatility_regime_from_atr_percentile`)
en vez de un ajuste de modelo por ticker. `ENGINE_VERSION` → v7 (fundamentales fuera del checklist).

**Fase 2 — arquitectura de precálculo.** Seis tablas nuevas (`job_runs`, `ticker_daily_states`,
`ticker_intraday_states`, `position_daily_states`, `trigger_events`, `daily_briefs`) y dos cron jobs:
`daily_close.py` (Job A, recorre el universo evaluando el gate por ticker y cada cartera evaluando el
exit engine sobre sus posiciones abiertas, absorbiendo el cómputo en vivo que antes hacía
`market_screener_service.get_universe_snapshot` para este propósito) y `intraday_refresh.py` (Job B,
relectura barata de cotizaciones en vivo contra el `entry_trigger_price` ya calculado al cierre, sin
recalcular el gate completo). `render.yaml` declara ambos cron jobs (22:00 UTC L-V el primero, cada 30
min de 08:00 a 21:00 UTC el segundo) - cableados, no activados: crear los Cron Jobs reales en Render es
un recurso facturado aparte, decisión del propietario.

**Fase 3 — el motor nuevo.** `trade_geometry.py` (la mitad "a qué precio": `compute_entry_trigger` -
un nivel de ruptura o rebote en soporte concreto y vigilable - y `compute_stop_and_target`, movido aquí
sin cambios desde `recommendation_engine.py`) y `levels_engine.py` (la mitad "es esta entrada buena":
`evaluate_gate`, un AND duro de 6 condiciones transparentes - tendencia/Fase 2, sin extensión
parabólica, sin sobrecompra extrema fuera de tendencia fuerte, sin divergencia bajista de OBV, sin veto
del par rápido EMA21/55, relación beneficio:riesgo ≥ 1.5 - cada una visible en `GateResult.conditions`,
nunca colapsadas a un sí/no. RS Rating y Minervini 8/8 deliberadamente NO son gates duros: un setup
genuinamente bueno en un nombre que todavía no se ha ganado un RS Rating alto (una ruptura reciente, una
Fase 2 temprana) es exactamente el tipo de entrada que esta reconstrucción debe seguir detectando, no
excluir por construcción - ambos quedan visibles como contexto en el estado precalculado del ticker sin
condicionar el trigger. `GATE_VERSION = "2026-09-levels-v1"`, misma disciplina de versionado que
`ENGINE_VERSION`. Este reparto de condiciones es un juicio de primer trazo, no medido todavía - la Fase
8 reorienta `factor_ablation_study.py` a resultados de triggers precisamente para poder revisarlo con
evidencia en vez de intuición.

**Fase 4 — el cutover en vivo.**

1. **Retirada de `walk_forward_backtest.py`**: su único consumidor real que sobrevivía
   (`backtest_engine.find_triple_barrier_entries`, vía `replay_recommendation_at`) pasa a
   `levels_engine.replay_gate_at` - mismo contrato de replay point-in-time, mismas dos simplificaciones
   ya documentadas (sin RS Rating point-in-time, sin soporte/resistencia point-in-time), pero
   reconstruyendo el gate en vez del checklist retirado. `_permutation_test` (rutina estadística
   genérica) se mueve sin cambios a `scripts/factor_ablation_study.py`, su único otro consumidor. El
   backtest de triple-barrera valida ahora las entradas que el gate propondría - coherente con que la
   propia "Recomendación" en vivo pasa a ser el gate en el mismo paso.

2. **`ticker_analysis_service.py`/`portfolio_risk_service.py` repuntados al gate**: `CoreTickerSignals.
   recommendation` (`Recommendation`) se convierte en `.gate` (`GateResult`); `confirmed_recommendation`
   en `confirmed_gate`. `recommendation_engine.py` NO se borra - `scripts/factor_ablation_study.py`
   sigue midiendo su checklist antiguo hasta que la Fase 8 lo reoriente a triggers (decisión de
   secuenciación deliberada) - pero nada en el camino en vivo ("Analizar activo", riesgo de cartera) lo
   llama ya.

3. **Cambio de comportamiento deliberado en `portfolio_risk_service.assess_position_risk`**: `signal`
   (EXIT_WARNING/ADD_CANDIDATE/WATCH/HOLD) ya no puede convertirse en `EXIT_WARNING` solo por el lado de
   compra. El "evitar" del checklist antiguo (score ≤ AVOID_THRESHOLD, varios factores bajistas a la
   vez) acumulaba suficiente peso para leerse como una advertencia real; el gate es un booleano, y una
   entrada *fallida* sobre una posición ya abierta no es la misma afirmación - ruido ordinario (RSI
   pegado alto fuera de una tendencia fuerte, una extensión de ATR pasajera) también la hace fallar, en
   nombres sin nada realmente mal. Confundir "no es una compra fresca hoy" con "deberías preocuparte por
   esta posición" es exactamente la conflación entrada/salida que `exit_engine.py` ya evitaba a otro
   nivel (docs/quant_methodology.md §8) - esta función deja de cometer el mismo error un nivel más
   arriba. `EXIT_WARNING` viene ahora exclusivamente de la escalada propia e independiente del exit
   engine (EXIT_NOW/REDUCE), nunca de `gate.passes` siendo falso.

4. **`score` se redefine, no se elimina**: de la suma de puntos del checklist antiguo (rango abierto,
   con signo) a cuántas de las 6 condiciones del gate se cumplen (0-6) - la tabla `PositionSignalSnapshotORM`
   y `PositionRisk.score` siguen siendo un `int` sin migración de esquema; ningún lector real de ese
   campo (auditado con grep antes del cambio) hacía nada más que pasarlo a través, así que redefinir su
   significado no rompe nada más allá de lo cosmético. El snapshot persistido de "Analizar activo"
   (`RecommendationSnapshotORM`) recibe el mismo tratamiento: `verdict` se remapea a "comprar"/"esperar"
   (el gate no tiene un tercer estado peor que "no aprobado" - "evitar" simplemente deja de escribirse
   nunca más), y cada condición del gate se convierte en una fila de `factors` con el mismo esquema
   `{label, points, triggered}` que ya tenía esa columna JSON - sin migración. Los marcadores de versión
   de las tablas del lado de salida (`TradePlanORM.engine_version`, `PositionSignalSnapshotORM.
   engine_version`, `PositionDailyState.engine_version`) pasan de `recommendation_engine.ENGINE_VERSION`
   a `levels_engine.GATE_VERSION` - el mismo "qué generación del motor en vivo produjo esto", apuntando
   ahora al módulo que de verdad está vivo.

**Fase 5 (en curso) — endpoints de solo lectura contra las tablas de la Fase 2.**

1. **`GET /market/radar`**: "qué está a punto de disparar una entrada" (Parte 0, pregunta 2) - una
   lectura pura sobre `ticker_daily_states`, nunca un escaneo de universo en vivo (ese cómputo ya lo
   paga `daily_close.py` una vez por noche). Incluye un ticker si su gate pasa **o** si tiene un
   `entry_trigger_price` activo - un ticker sobreextendido acercándose a una ruptura sigue siendo
   relevante para el radar aunque el gate lo rechace hoy; se excluye solo cuando no pasa ninguna de las
   dos condiciones. `RadarResponse.computed_at` es `None` cuando `daily_close.py` todavía no ha corrido
   para esa región - distinto de "corrió y no hay candidatos".

2. **`GET /portfolios/{id}/today`**: "qué hago hoy con lo que ya tengo" (Parte 0, pregunta 1) - lectura
   pura sobre `position_daily_states`/`daily_briefs`, deliberadamente aditiva y no un reemplazo de
   `GET /portfolios/{id}/risk`: esta es la lectura rápida con la que abre el dashboard "Hoy"; la ficha de
   posición sigue abriendo contra la lectura en vivo, más rica (`signals`/`multi_timeframe`/
   `scaled_exit`, ninguno de los cuales carga `PositionDailyState`). Mismo criterio que el radar para
   "no ha corrido todavía" - `brief: None`, `positions: []`, nunca un 404.

3. **Consumidores mínimos en el frontend, no todavía las 4 vistas de la Fase 6**: una pestaña "Radar"
   nueva en Mercado (`RadarView.jsx`) junto a "A revisar" (`Watchlist.jsx`, que se queda tal cual -
   consolidarlas o retirarla es trabajo pendiente), y un banner pequeño en "Mi Cartera"
   (`DailyBriefBanner.jsx`) con el titular del brief y los conteos a nivel de universo
   (`new_entry_triggers`/`new_gate_passes`) - lo único que `/today` aporta que la lectura en vivo no
   podría calcular por sí sola. La urgencia por posición sigue siendo trabajo de `TodayActionsPanel`
   (`riskByTicker`, en vivo) - el banner no la duplica. La Fase 6 real (Hoy/Radar/Activo/Sistema como
   4 vistas de primer nivel, reemplazando la navegación actual) sigue pendiente.

**Fase 8 — medición basada en `TriggerEvent`, la nueva lectura principal de "¿está
funcionando el sistema?" (Parte 0, pregunta 4).**

`trigger_performance_service.py`: mismo patrón arquitectónico que `signal_performance_service.py`
(funciones de agregación puras + una única función de orquestación que trae histórico de precios en un
solo batch), pero midiendo `TriggerEvent` (el registro append-only que `daily_close.py`/
`intraday_refresh.py` ya escriben desde la Fase 2) en vez de un veredicto/señal de checklist. Mide dos
tipos de evento por separado - `gate_passed` (el gate pasó de fallar a aprobar) y `entry_triggered` (el
precio del disparador de entrada se cruzó de verdad) - cada uno una hipótesis real sobre si el gate
nuevo tiene valor predictivo, no una asumida. `exit_urgency_changed` (a nivel de posición) queda fuera a
propósito - es el motor de salida reaccionando sobre una posición ya abierta, no una señal de entrada
nueva; mezclar las dos tasas de acierto no respondería ninguna de las dos preguntas con honestidad
(mismo razonamiento que `BEARISH_SIGNAL_LABELS` ya codifica en `signal_performance_service.py`).
`GET /system/signal-performance` gana un campo `trigger_outcomes` nuevo, sin tocar los tres que ya
tenía - los datos de antes de la Fase 4 siguen siendo correctos bajo la lectura antigua. La vista
"Rendimiento del sistema" muestra esta tabla primero, como la lectura principal actual, con las tablas
de veredicto/señal debajo, marcadas explícitamente como históricas.

`factor_ablation_study.py` (el estudio offline, distinto de la lectura en vivo de arriba) también se
reorienta: `compute_triggers_at` ahora llama a `levels_engine.replay_gate_at` - la misma función de
replay que usa el backtest en vivo, no una segunda aproximación hecha a mano - y expone cinco de sus
seis condiciones como factores propios (`gate_trend_or_stage2`, `gate_not_parabolic`,
`gate_not_overbought_outside_strong_trend`, `gate_no_obv_bearish_divergence`, `gate_no_fast_pair_veto`),
más el compuesto `gate_passes`. Tres factores del checklist antiguo se retiraron, no se dejaron al lado
de su equivalente nuevo: `atr_parabolic`, `rsi_overbought_outside_strong_trend` y `obv_bearish` son cada
uno la negación lógica exacta de un `gate_*` - mantener ambos le daría a la regresión multivariante dos
columnas perfectamente colineales (una matriz de diseño singular), un defecto real, no solo redundancia.
La sexta condición del gate (beneficio:riesgo >= 1.5) se deja fuera a propósito: sin soporte/resistencia
point-in-time, `compute_stop_and_target` siempre cae al stop/objetivo fijo (ATR/2:1), así que esa
condición sería una constante `True` en cada muestra - una columna sin varianza es colineal con la
propia constante de la regresión, lo que corrompería el coeficiente de *todos* los demás factores, no
solo el suyo. Esa condición se mide correctamente, contra soporte/resistencia real, en
`trigger_performance_service.py` en su lugar. `golden_cross`/`death_cross`/`rsi_oversold_bounce`/
`minervini_range_position`/`trend_down`/`stage4` se quedan sin tocar - son señales informativas que el
sistema en vivo todavía muestra aunque ninguna condicione el gate.

### 25.1 Primera ejecución real del estudio reorientado (2026-09-12) - resultado, no una acción

`python scripts/factor_ablation_study.py --temporal-split` corrido de verdad contra las ~216 acciones del
universo curado (US + Europa), 10 años de histórico, horizontes 5/10/21 sesiones, split calibrar
(< 2023-01-01) / validar (>= esa fecha). Salida en `docs/factor_ablation_report_v4_gate_*.csv` (54
archivos, sin comitear - mismo criterio que los `factor_ablation_report_v3_*.csv` previos: son artefactos
de investigación locales, no código; la evidencia que importa queda registrada aquí, en prosa).

**Hallazgo principal - honesto, no cómodo**: `gate_trend_or_stage2` (y por extensión `gate_passes`, que
lo incluye) mide un efecto **negativo** y con frecuencia significativo (BH-ajustado) sobre el retorno
demeaneado a 5 y 10 sesiones - lo contrario del signo que su diseño asume:

| horizonte | conjunto | `gate_passes` diff. | p (BH) | `gate_trend_or_stage2` diff. | p (BH) |
|---|---|---|---|---|---|
| 5d | pooled | -0.128pp | 0.00055 (sig.) | -0.148pp | 0.0003 (sig.) |
| 5d | calibrar | -0.146pp | 0.0006 (sig.) | -0.174pp | 0.0003 (sig.) |
| 5d | validar | -0.102pp | 0.048 (no sig. al 1%) | -0.112pp | 0.017 (no sig. al 1%) |
| 10d | pooled | -0.206pp | 0.0006 (sig.) | -0.315pp | 0.0003 (sig.) |
| 10d | calibrar | -0.152pp | 0.037 (no sig. al 1%) | -0.363pp | 0.0003 (sig.) |
| 10d | validar | -0.285pp | 0.0026 (sig.) | -0.246pp | 0.005 (sig.) |
| 21d | pooled | -0.153pp | 0.214 (no sig.) | -0.169pp | 0.181 (no sig.) |

El efecto no es un accidente de una sola muestra: aparece en calibrar **y** en validar (fuera de
muestra) a 5 y 10 sesiones, con el mismo signo. Se debilita y deja de ser significativo a 21 sesiones -
el horizonte de holding real de esta cartera. Segmentado por régimen (h10): el efecto es fuerte en
`vix_calm`/`market_above_sma200` (la mayoría de las muestras) y se invierte de signo, no significativo,
en `vix_stress` (muestra pequeña, n≈2000). No es un hallazgo aislado de `trend_or_stage2`: **todo** el
grupo de factores "confirmación alcista" del checklist retirado (`trend_up`, `stage2`,
`adx_strong_trend`) mide el mismo signo negativo a 5-10 sesiones, con sus opuestos (`trend_down`,
`stage4`, `death_cross`) midiendo el signo positivo espejo - consistente con el efecto de **reversión de
muy corto plazo** ampliamente documentado en la literatura (Lehmann 1990, Jegadeesh 1990), distinto del
momentum de 6-12 meses que este mismo proyecto ya cita (Jegadeesh & Titman 1993) y que opera en un
horizonte mucho más largo. Comprar una tendencia ya confirmada parece, en esta muestra, comprar
justo antes de una pausa de corto plazo relativa al resto del universo - no evidencia de que la
tendencia en sí sea mala, sino de que 5-10 sesiones es demasiado poco tiempo para que se exprese.

**El único gate condition que sí se valida limpiamente**: `gate_not_parabolic` mide el signo correcto
(positivo - evitar extensión ayuda) y es significativo BH-ajustado a 5d y 10d pooled (+0.236pp/+0.445pp,
p<0.001), aunque pierde significación en el split validar aislado (muestra más pequeña). Las otras tres
condiciones (`gate_not_overbought_outside_strong_trend`, `gate_no_obv_bearish_divergence`,
`gate_no_fast_pair_veto`) no muestran nada concluyente en ningún sentido - probablemente infra-medidas:
la condición de sobrecompra en particular solo tiene 30-152 muestras "activadas" (RSI≥80 fuera de
tendencia fuerte) frente a decenas de miles del lado contrario en cada horizonte.

**Qué NO se hace con esto todavía**: ningún cambio al gate en vivo. Es exactamente la situación que la
sección "Cómo usar los resultados" de este mismo script anticipa - una señal real, consistente en
calibrar y validar, pero que pide una decisión deliberada del propietario (¿acortar el horizonte de
"gate_trend_or_stage2" a algo que no choque con la reversión de corto plazo? ¿mantenerlo, dado que 21
sesiones - el horizonte real - no muestra el mismo efecto? ¿investigar si el muestreo por stride del
estudio interactúa mal con la reversión?), no una acción automática de este reconstructor a partir de un
único estudio, por bien corrido que esté. Queda registrado aquí para esa decisión, no aplicado.

**Setup outcome stats (h21, contexto)**: expectancy positiva y consistente en los cuatro setups de
`watchlist_service.py` (`oversold_bounce` +0.114R, `breakout_volume` +0.126R, `trend_continuation`
+0.085R, `pullback_to_support` +0.133R sobre esa muestra) pese a win rates bajos (12-15%) - la asimetría
esperada de un stop ajustado con objetivo 2:1, no una señal de que el setup "falle" la mayoría de las
veces en el sentido coloquial.

**Fase 8 completa** en el sentido en que este reconstructor puede cerrarla: script reorientado,
ejecutado de verdad contra el universo real, resultado documentado arriba. Lo que queda es una decisión
del propietario (qué hacer, si algo, con el hallazgo de reversión de corto plazo), no trabajo de
ingeniería pendiente.

### 25.2 Capa Gemini de narrativa (Fase 7) - cableada, no activada

`LLMNarrator` (`app/domain/interfaces/llm_narrator.py`): puerto que traduce un `GateResult` ya calculado
a una explicación corta en lenguaje llano - nunca una segunda opinión, nunca una puntuación, nunca una
decisión propia. Deliberadamente tipado solo con primitivos (`str`/`bool`/`list[tuple[str, bool]]`), no
con `GateResult` directamente - el servicio que ya importa `levels_engine` (`ticker_analysis_service.
_explain_gate`) traduce antes de llamar al puerto, así el puerto no depende de la forma interna de otro
módulo de `services/`. `GeminiNarrator` (`app/infrastructure/llm/gemini_narrator.py`) es el adaptador
real (`google-genai`), estructuralmente incapaz de influir el gate: solo recibe hechos ya decididos, y
cualquier fallo (sin clave, límite de tasa, timeout, respuesta vacía) se traga en `None` - una narrativa
es contexto opcional, nunca una dependencia que pueda bloquear "Analizar activo".

**Cableada, no activada** - mismo criterio que los Cron Jobs de la Fase 2: `GEMINI_API_KEY` no tiene
valor por defecto (`.env.example` la deja vacía, `render.yaml` la declara `sync: false` sin valor) -
mientras no se configure, `explain_gate` devuelve `None` de inmediato, sin ningún intento de red, y
ningún dato del propietario (tickers, gate, cartera) sale nunca de esta app hacia Google. Activarla es
una decisión de coste y privacidad del propietario, confirmada explícitamente antes de construir esto
(no algo que este reconstructor decidiera por su cuenta, a diferencia del resto de la Fase 7).
`TickerAnalysisResponse.llm_narrative` (`GET /market/tickers/{ticker}/analysis`) es el único consumidor
por ahora - el único sitio donde "explicar una entrada concreta en lenguaje natural" tiene sentido
acotado; el Radar (muchos tickers a la vez) queda fuera a propósito, no es una vista deep-dive.
`GateNarrative.jsx` la muestra en el frontend, justo debajo de `RecommendationCard` - visualmente
distinta (fondo/borde propios) y con su propio descargo ("no es una segunda opinión"), sin renderizar
nada mientras `GEMINI_API_KEY` no esté configurada. **Fase 7 completa** en el sentido en que este
reconstructor puede cerrarla - activarla es la decisión pendiente del propietario, no trabajo de
ingeniería.

### 25.3 Frontend de 4 vistas (Fase 6) - Hoy/Radar/Activo/Sistema

Reescritura completa de la navegación, no una adición: "Mi Cartera / Rendimiento del sistema / Mercado
(8 secciones)" se convierte en 4 vistas de primer nivel, una por cada pregunta de la Parte 0 del encargo
- `Sidebar.jsx` y `App.jsx` reescritos, sin dejar la navegación antigua como alternativa.

- **Hoy** = el antiguo "Mi Cartera" (mismo contenido - `DailyBriefBanner`/`TodayActionsPanel` primero,
  luego el dashboard completo), solo renombrado.
- **Radar** = agrupa todas las herramientas de cribado a nivel de universo, no solo `RadarView.jsx`:
  `A revisar` (Watchlist), `Screener`, `Movers`, `Sectores`, `Tendencia`, `Soportes/Resistencias` se
  mueven aquí desde el antiguo "Mercado" - son la misma familia de herramientas que el Radar, no una
  degradación. Cada una mantiene su propio selector de región.
- **Activo** = solo el deep-dive de un ticker: `Analizar activo` y `Contexto` (contexto macro, relevante
  al analizar una entrada concreta) - nada de universo aquí, a propósito.
- **Sistema** = el antiguo "Rendimiento del sistema", sin cambios de contenido.

`sectionNav.js` (antes `marketSections.js`) exporta `RADAR_SECTIONS`/`ACTIVO_SECTIONS` en vez de una
lista única; `SectionedView.jsx` (antes `MarketView.jsx`) queda genérico - recibe qué mapa de
componentes usar por prop - y los dos mapas (`RADAR_COMPONENT_BY_SECTION`/`ACTIVO_COMPONENT_BY_SECTION`)
viven en `sectionComponents.js`, un módulo no-componente aparte (regla de Vite/react-refresh: un
archivo que exporta un componente no puede exportar también constantes). Verificado en vivo, no solo por
build: servidor + frontend reales lanzados con Playwright, las 4 vistas navegadas una por una, sin
errores de consola ni de página.

### 25.4 Escenarios dorados del gate + presupuesto de latencia (Fase 9)

`test_golden_gate_scenarios.py`: misma metodología que `test_golden_scenarios.py` (series de precios
con forma de manual, respuesta verificable a simple vista, corridas de punta a punta) aplicada al gate
que de verdad decide ahora (`levels_engine.evaluate_gate`), no al checklist retirado - archivo separado,
no una reescritura del original (ese sigue documentando correctamente el comportamiento del checklist
retirado). Siete escenarios: ruptura limpia de Fase 2 con volumen (aprueba las 6 condiciones), tendencia
bajista confirmada y lateralidad (fallan por tendencia), veto del par rápido y divergencia bajista de
OBV (cada uno aislado - falla solo esa condición, con tendencia todavía aprobada), extensión parabólica
(aislada igual), y un pico de sobrecompra dentro de un mercado lateral (no aislado - un impulso corto lo
bastante fuerte para disparar el RSI también dispara el ADX y el múltiplo de ATR en la misma ventana de
14 barras, así que las tres condiciones fallan juntas en un mismo gráfico real, documentado como tal en
vez de forzar una fixture artificial). Limitación conocida y documentada: ningún escenario aquí prueba
la sexta condición (beneficio:riesgo) fallando - los seis usan `nearest_support=nearest_resistance=None`
(sin escaneo de niveles), así que esa condición siempre cae al objetivo fijo 2:1 y sale `True`; una
fixture real con una resistencia cercana que ofrezca entre 1.0x y 1.5x el riesgo no se construyó en esta
pasada.

`test_latency_budgets.py`: no una prueba de corrección, sino una red contra la otra forma en que un
camino caliente se vuelve lento - un bucle Python pesado por barra u O(n²) introducido por accidente en
cómputo puro, algo que ninguna prueba de "¿es correcta la respuesta?" detectaría nunca. Mide
`build_ticker_daily_state` (la función que el cron nocturno paga una vez por ticker) sobre 50 tickers
sintéticos de ~2520 barras (10 años) cada uno, con un presupuesto deliberadamente generoso (0.5s/ticker,
~40x el tiempo real medido en hardware ordinario, ~12ms/ticker) - pensado para atrapar una regresión real
(una función que se volvió 10x-100x más lenta), no para perseguir una cifra exacta que haría esta prueba
inestable entre distinta hardware de CI. Relevante ahora concretamente porque la Fase 10 planea crecer
el universo de ~217 a ~400 tickers sobre la misma ventana de cron.

### 25.5 Retirada de watchlist_service.py (resto de la Fase 5)

Retirado por completo (código y tests, no solo desconectado): `app/services/watchlist_service.py`
(445 líneas - los cuatro detectores de setup de corto plazo, los tres de plazo medio, el scoring por
percentil cruzado, `WatchlistItem`, `build_watchlist`), el endpoint `GET /market/watchlist` y sus
esquemas (`WatchlistItemResponse`/`WatchlistResponse`), y la pestaña "A revisar" del frontend
(`Watchlist.jsx`) - `GET /market/radar` cubre la misma pregunta ("qué merece la pena mirar") con
evidencia real detrás, no con una regla barata sin medir.

Dos dependencias reales encontradas y resueltas, no solo referencias en comentarios:

1. **`relationship_map_service.py`** llamaba a `build_watchlist(universe_snapshot)` de verdad (dos
   veces) para anotar cada ticker relacionado/par de sector con su setup y percentil del día -
   `StatisticalRelation`/`SectorPeer` pierden esos campos (`setup`/`percentile_score`/`setup_label`)
   en vez de duplicar la lógica de detección de setups aquí solo para mantener vivas dos etiquetas.
   Decisión distinta de la planeada originalmente (migrar `SETUP_LABELS` a un sitio compartido) -
   la investigación mostró que la dependencia real era la salida de `build_watchlist`, no solo las
   etiquetas, así que quitar la anotación por completo es la simplificación más proporcionada,
   coherente con el resto de esta reconstrucción (retirar señales sin invalidar en vez de migrarlas).
2. **`scripts/factor_ablation_study.py`** importaba `PULLBACK_MAX_DISTANCE_ABOVE_SMA50`/`PULLBACK_MIN_RSI`
   de `watchlist_service.py` para su propio detector de `setup_pullback_to_support` (ya duplicado a
   mano contra series crudas, nunca llamaba a `build_watchlist`). Las dos constantes se copian al
   script en vez de dejarlas como dependencia de un módulo que ya no existe - `segment_by_setup_type`/
   `SetupOutcomeStats` se quedan intactos: miden si estos cuatro patrones tienen edge como pregunta de
   investigación propia, independiente de si alguna vista en vivo los sigue mostrando hoy.

### 25.6 opportunity_cost.py en pequeño, contra el Radar

Reemplaza en espíritu al `opportunity_cost.py` original (retirado en la Fase 1 junto con la watchlist
"Premium" de 3 niveles que alimentaba) - deliberadamente pequeño: una comparación booleana, no una
puntuación. `find_opportunity_cost_notes(held_states, radar_candidates)` recorre cada posición cuyo
propio gate no aprueba hoy y busca, en el mismo sector curado (`market_universe.sector_of`), candidatos
del Radar cuyo gate sí aprueba - sin inventar una segunda opinión sobre si la posición debería venderse
(eso sigue siendo trabajo exclusivo de `exit_engine.py`, ver §8). Ordenado por RS Rating del candidato
(sin datos al final) - no una puntuación nueva, el mismo campo que el resto del proyecto ya trata como
criterio de "qué líder es más fuerte".

`GET /portfolios/{id}/today` gana un campo `opportunity_cost` - combina el Radar de ambas regiones
(mismo razonamiento "una cartera personal no se limita a un mercado" que ya usa `/risk`) contra el
estado de gate más reciente de cada posición (`TickerDailyStateRepository.latest_for_ticker`).
`OpportunityCostPanel.jsx` lo muestra en "Hoy", debajo de `TodayActionsPanel` - renderiza nada si no hay
nada que señalar.

### 25.7 Fase 10 - universo dinámico: cableado, no activado en producción

Investigado antes de tocar nada: `market_screener_service.get_universe_snapshot` **ya** detecta y
prefiere el universo dinámico (`dynamic_universe_service.read_dynamic_universe`) sobre el diccionario
curado de ~217 tickers de `market_universe.py` en el momento en que exista un snapshot en la tabla
`universe_memberships` para una región (Tercera auditoría, Bloque F-1 - ver el comentario de ese propio
método) - "activar" esta fase no es un cambio de código, es puramente un asunto de datos:
`scripts/refresh_universe_membership.py` es el único sitio que escribe esa tabla.

`render.yaml` gana un tercer Cron Job (`quantumalpha-refresh-universe-membership`, 1º de cada mes a las
05:00 UTC) - mismo criterio "cableado, no activado" que los otros dos: crear el recurso de verdad en
Render es una decisión de coste del propietario, esto solo lo deja listo para revisar antes de esa
decisión. El propio script se salta una región que no lleva 30 días sin refrescar, así que una ejecución
ocasionalmente tardía o perdida no es grave.

Verificado en local (no contra producción - este reconstructor no tiene credenciales de la base de datos
de Render, ni las necesita: el script acepta cualquier `DATABASE_URL`, incluida una base de dev local)
que el pipeline completo funciona de verdad, no solo en teoría - descarga real de los índices S&P 500/
400 y STOXX Europe 600 desde Wikipedia, filtro de liquidez real vía yfinance, y persistencia en
`universe_memberships`: **896 tickers guardados para EE.UU.**, **196 para Europa** (de partida, 189 -
ver el bug real encontrado y corregido justo abajo).

**Bug real encontrado y corregido, no solo una hipótesis**: la primera corrida en vivo falló 113/~300
búsquedas europeas. 18 de esas eran un problema de formato reproducible y ya arreglado -
`parse_stoxx600_constituents` pegaba el sufijo de Yahoo Finance directo sobre el ticker crudo de
Wikipedia, que trae la clase de acción con un espacio ("VOLV B", bolsas nórdicas) o un punto ("BT.A",
Londres) - Yahoo Finance exige un guion en los dos casos ("VOLV-B.ST", "BT-A.L"), exactamente la misma
normalización que `parse_us_constituents` ya aplicaba para "BRK.B" -> "BRK-B" y que este `parse_stoxx600_constituents`
simplemente no tenía. Confirmado con `yfinance` en vivo antes de tocar el código (`VOLV-B.ST` responde,
`VOLV B.ST` no) y con un test nuevo (`test_parse_stoxx600_constituents_normalizes_space_and_dot_share_classes_to_a_dash`).
Una segunda corrida tras el arreglo bajó los fallos a 96 y subió el total guardado a 196.

Los ~96 fallos restantes son un problema distinto y más difícil, fuera de alcance de esta pasada: una
mezcla de tickers realmente deslistados/fusionados en el mundo real desde que se escribió la página de
Wikipedia, y casos donde el ticker que Wikipedia lista no es el que Yahoo Finance realmente usa para esa
empresa (ej. Ferrari aparece como "FERR" ahí, pero cotiza en Yahoo como "RACE.MI") - verificarlos uno por
uno no es una normalización mecánica como la de arriba, es trabajo de investigación por empresa.
`refresh_universe_membership.py` ya está diseñado para este caso exacto: registra y salta cualquier
ticker que Yahoo Finance no pueda cotizar, sin romper la corrida ni inventar un precio.

**Pendiente**: decisión del propietario sobre si/cuándo activar el Cron Job de verdad en Render (crear
el recurso, con su coste asociado) - o ejecutar el script a mano una vez contra la base de datos de
producción para probar el universo dinámico sin comprometerse todavía a la recurrencia mensual. Ninguna
de las dos cosas es trabajo de ingeniería que falte por hacer. Opcional, no bloqueante: investigar los
~96 tickers europeos restantes uno por uno si se quiere un universo europeo más completo que 196/~300.

**Tests**: 15 nuevos en `test_trade_geometry.py`, 12 en `test_levels_engine.py`, 17 en
`test_precompute_repositories.py`, 21 en `test_daily_close.py` + 5 de integración, 11 en
`test_intraday_refresh.py` + 4 de integración, 5 en `test_levels_engine_replay.py`, 6 en
`test_radar_api.py`, 5 en `test_portfolio_today_api.py`, 8 en `test_trigger_performance_service.py` + 2
de integración en `test_system_api.py`, 2 nuevos en `test_factor_ablation_study.py` (los factores del
gate y el retiro de sus tres equivalentes del checklist), 4 en `test_gemini_narrator.py`, 4 más en
`test_ticker_analysis_service.py` (el cableado de `_explain_gate`) + 1 de integración en
`test_ticker_analysis_api.py` (`llm_narrative: null` sin clave configurada), 7 en
`test_golden_gate_scenarios.py`, 1 en `test_latency_budgets.py`, más las actualizaciones de
`test_portfolio_risk_service.py`, `test_ticker_analysis_api.py` y `test_portfolios_api.py` para el nuevo
contrato del gate. Retirados con `watchlist_service.py`: `test_watchlist_service.py` completo (34 tests)
y 4 en `test_relationship_map_service.py` que probaban la anotación de setup/percentil ya eliminada (38
en total, 713 → 675 unitarios). 8 nuevos en `test_opportunity_cost.py`, 3 de integración nuevos en
`test_portfolio_today_api.py` (675 → 683 unitarios). 1 nuevo en `test_dynamic_universe_service.py`
(normalización de espacio/punto a guion en tickers STOXX 600 - 683 → 684 unitarios).

## 26. Quinta auditoría: verificación contra el texto literal del encargo (septiembre 2026)

La sección 25 se escribió, como su propia nota de procedencia explica, después de que el texto
literal de las 20 partes del encargo saliera de contexto - una reconstrucción de memoria, con el
visto bueno explícito del propietario para proceder así en vez de repetir el pegado completo. En
una sesión posterior el propietario sí volvió a pegar el texto literal completo, con una instrucción
adicional explícita: seguir el plan tal cual, fase por fase, sin volver a consultar nada. Esta
sección documenta esa verificación - qué de la sección 25 resultó ser exactamente lo que el encargo
pedía (o una desviación deliberada y ya razonada, que se deja tal cual), y qué resultó ser una
aproximación real que necesitaba corrección. Ninguno de los hallazgos de abajo implica que la
sección 25 fuera descuidada - la mayoría de sus decisiones de diseño (el reparto de condiciones del
gate, el alcance reducido de la capa Gemini, el campo único de percentil sectorial) resultaron
coincidir con el espíritu del encargo real o estar ya justificadas con un razonamiento que esta
auditoría no tenía motivo para deshacer. Lo que sigue son las excepciones reales, encontradas con
grep y lectura directa antes de tocar nada, nunca asumidas.

### 26.1 Fase 1 (cierre real), incluido un error propio corregido antes de comitear

Un grep sistemático confirmó que la mayoría del borrado de la Parte 2 ya estaba hecho en la sesión
anterior; lo que quedaba de verdad: `analysis_tools.py` (Gann/estacionalidad/analogos históricos) y
toda su cablería en `ticker_analysis_service.py`/`PriceChart.jsx`/`TickerAnalysisPanel.jsx`,
retirados por completo (no solo desconectados); el rendimiento por sector/industria como
endpoint/vista independiente (`get_sector_performance`/`get_industry_performance`,
`/market/sectors`, `/market/industries`, `SectorsView.jsx` y sus componentes) - Parte 2.5 lo
sustituye por un único campo de percentil, no por nada de lo retirado aquí; y
`portfolio_construction_service.final_position_size`, cuyo parámetro seguía llamándose `kelly_size`
pese a que Kelly ya no existía - renombrado a `risk_based_size` para que el grep de aceptación de la
Parte 17 (`monte_carlo|markov|kelly|garch|hurst|walk_forward|rrg`) diera cero fuera de comentarios
históricos.

Error real cometido y corregido en la misma pasada, documentado aquí por disciplina, no para
ocultarlo: se borró `opportunity_cost.py` completo pensando que era la versión vieja (el embudo de 3
niveles) sin darse cuenta de que la sesión anterior ya lo había reescrito en pequeño, contra el
Radar (sección 25.6) - exactamente la versión correcta. Detectado revisando `git log` del archivo
antes de comitear (mostraba una reescritura posterior a su primer retiro), restaurado íntegro
(servicio, test, cableado en `GET /today`, schema, test de integración, panel de frontend y su CSS)
sin pérdida de funcionalidad.

### 26.2 `recommendation_engine.py`: el checklist retirado de verdad, no solo de la práctica

La sección 25.4 documenta correctamente que nada en el camino en vivo llamaba ya a
`build_recommendation` - pero el propio archivo, con las clases del checklist todavía definidas,
seguía ahí (313 líneas). Un grep confirmó que ni un job, ni un endpoint, ni `scripts/
factor_ablation_study.py` (que mide sus propios valores como constantes literales, nunca importando
las clases retiradas) tenían ninguna razón real para que siguiera existiendo. Retirado: `build_recommendation`,
`Recommendation`, `RecommendationFactor`, `BUY_THRESHOLD`, `AVOID_THRESHOLD`; `test_recommendation_engine.py`
(sus 5 tests de `compute_stop_and_target` eran duplicados exactos de `test_trade_geometry.py`) y
`test_golden_scenarios.py` (escenarios end-to-end del checklist retirado - `test_golden_gate_scenarios.py`
ya cubre el mismo terreno contra `evaluate_gate`, el que de verdad decide). El archivo queda en 48
líneas: solo re-exporta `compute_stop_and_target`/`StopAndTarget` de `trade_geometry.py` y define
`ENGINE_VERSION = "2026-09-v6-levels"` - el valor exacto que la Parte 6/17 del encargo pide, no el
`"2026-09-audit-v7"` que tenía (esa cadena era la numeración incremental correcta del checklist
mientras seguía vivo, per su propio comentario de versión - simplemente nunca se hizo el último
salto a la cadena final una vez el checklist quedó totalmente sustituido).

### 26.3 `app/core/trading_params.py`: el módulo único de parámetros no existía, y varios valores en producción seguían siendo los de ANTES de esta reconstrucción

La Parte 19 del encargo pide un módulo único con cada parámetro configurable y su valor exacto - no
existía. Al crearlo y cablearlo donde de verdad importa, un grep reveló que el Chandelier Exit en
`trade_manager.py` seguía en los valores canónicos de Chuck LeBeau sin recalibrar (ventana de 22
barras, multiplicadores 2.5/3.0/3.25/3.5 por régimen, profit lock 2.0×@+2R) - los mismos números de
ANTES de que esta reconstrucción empezara, no una aproximación de la Parte 3.2, que pide
explícitamente 10 barras, 1.75/2.0/2.25/2.5, profit lock 1.5×@+1.5R (más ajustado a un holding de
2-10 sesiones, no al swing de varias semanas para el que el original estaba pensado).
`HIGH_CORRELATION_THRESHOLD` en `portfolio_construction_service.py` estaba en 0.8, no en el 0.7 que
pide la Parte 19. La escalera de salida (`trade_manager.compute_scaled_exit_plan`) tampoco
implementaba la Parte 8 completa: el break-even en +1R no incluía el coste de ida y vuelta; +2R
dejaba el stop en `None` ("que gobierne el Chandelier") en vez de subirlo al nivel de +1R; no existía
la excepción de posición pequeña (< $150 al abrir, sin escalado); y no existía el cierre por tiempo
del último tercio (`LAST_TRANCHE_TIME_STOP_BARS`). Los tres primeros son value-level: valores
correctos pero desactualizados, no un diseño equivocado; el cuarto y la excepción de posición pequeña
eran comportamiento genuinamente ausente. Cada test de `compute_trailing_stop`/`label_triple_barrier`
que fijaba un multiplicador a mano se recalculó a mano contra los nuevos valores - ningún assert se
relajó para que pasara.

### 26.4 El "par rápido" no estaba unificado: `multi_timeframe.py`/`market_screener_service.py`/`exit_engine.py` seguían en SMA21/SMA50

La Parte 3.2 nombra explícitamente el síntoma: "3 definiciones competidoras (EMA21/55, SMA21/50,
SMA20)". Un grep confirmó las tres siguen existiendo en el código de la sección 25:
`technical_analysis.detect_fast_pair_bearish_veto` calculaba correctamente EMA21/EMA55 (el veto
bajista del gate), pero `multi_timeframe.py` (el "fast pair" que `ma_cross_20_50`/
`cross_quality_20_50`/`price_vs_sma20`/`price_vs_sma50` en realidad describen) lo hacía con
`ta.sma(close, FAST_MA_PERIOD)` contra una SMA50 real, y `market_screener_service.py`'s panel
"Tendencia" repetía el mismo cálculo SMA por su cuenta; las reglas duras de `exit_engine.py`
(pérdida del par rápido, con o sin volumen) leían una TERCERA computación SMA independiente hecha en
`portfolio_risk_service.py`. Corregido introduciendo `SLOW_MA_PERIOD=55` junto a `FAST_MA_PERIOD=21`
en `multi_timeframe.py` y recalculando esas tres piezas contra EMA21/EMA55 reales - deliberadamente
SIN tocar el cruce dorado/de la muerte SMA50/SMA200 (un concepto estándar y separado que la Parte 3.2
nunca pidió cambiar) ni `atr_multiple`/`CoreTickerSignals` (ver 26.6). El impacto en tests fue menor
de lo esperado: la mayoría de los escenarios end-to-end de `test_multi_timeframe.py` usan series de
tendencia larga e inequívoca, insensibles a SMA vs EMA - solo 1 test de `test_market_screener_service.py`
(una serie de reversión afinada a mano para un cruce SMA exacto) y ~15 de `test_exit_engine.py`
(renombrado de parámetro, textos "SMA"→"EMA", y 2 tests cuya premisa - "semanal alcista suprime la
ruptura de la pierna lenta" - dejó de aplicar, porque la Parte 9 quita ese calificador a propósito
para el disparador de EMA55) necesitaron cambios.

Aprovechando el mismo disparador dividido en dos exactos que pide la Parte 9 (antes uno solo, sobre
SMA50, con y sin volumen): "pérdida de EMA21 confirmada con volumen relativo ≥1.3 + semanal no
alcista" (antes ≥1.5 sobre SMA50) y "2 cierres consecutivos bajo EMA55" (nuevo, sin el calificador
semanal - la Parte 9 lo lista como disparador propio, sin condición extra). El disparador de death
cross del par rápido quedó corregido gratis, sin tocar `exit_engine.py`: ya leía
`cross_quality_20_50`/`price_vs_sma20`/`price_vs_sma50`, que ahora son EMA21/55 reales.

### 26.5 `trade_geometry.py`: el diseño real de la Parte 7 nunca se construyó

La propia sección 25.3 documenta con honestidad que `compute_stop_and_target` se "movió sin cambios"
desde `recommendation_engine.py` - y ese es exactamente el problema: la Parte 7 del encargo describe
una cascada de stop por tipo de entrada (ruptura, rebote en soporte, retroceso a EMA21, continuación
sobre EMA55) con un techo duro de 2.0 ATR, un techo de riesgo adaptativo por percentil de ATR, un
objetivo neto de costes de transacción, y un tamaño de posición con tres límites - nada de eso existía;
lo que había era un stop de ATR fijo (`ATR_STOP_MULTIPLE=2.5`) y un objetivo 2:1 simple, la misma
aproximación que el checklist retirado ya usaba. Construido de cero como `compute_entry_geometry` +
`TradeGeometry`/`EntryType`, verificado contra los dos ejemplos numéricos exactos que la Parte 15/20
del encargo dan (techo de 3.0% de riesgo para un nombre calmado de 1.2% ATR con un stop natural de
4% → rechazado; techo de 7.0% para un semiconductor de 3.5% ATR con un stop natural de 6% →
aceptado). Un bug real de ordenamiento se encontró y corrigió antes de publicar el diseño: la primera
versión comprobaba el techo de riesgo adaptativo DESPUÉS de aplicar el techo duro de 2.0 ATR, lo que
enmascaraba silenciosamente el rechazo en el primer ejemplo de arriba (el stop ya recortado a 2.0 ATR
mostraba 2.4% de riesgo, por debajo del techo de 3.0%, cuando el stop natural real - 4% - sí debía
rechazarse). Corregido comprobando el techo contra la distancia natural, sin recortar, antes de
aplicar el techo duro - que entonces nunca puede volver a disparar el rechazo, porque solo reduce el
riesgo.

Separado deliberadamente en dos funciones antes de cablearlo en ningún sitio: `daily_close.py` puntúa
todo el universo curado una vez al día, en un paso estructuralmente independiente del bucle por
cartera (ese es solo para el riesgo de posiciones ya abiertas) - no hay ningún `capital_total` que
darle al gate de un ticker en ese punto. `compute_entry_geometry` (stop/objetivo/techo de riesgo, sin
capital) es lo que `evaluate_gate` puede llamar; `size_position` (acciones/valor de posición/% de
cartera) solo tiene sentido una vez se conoce el capital de una cartera específica, y sigue sin
llamarse desde ningún sitio de producción - el Radar renderizado para una cartera, o
`trade_plan_service.py` al abrir una posición de verdad, son los candidatos naturales, todavía
pendientes. `evaluate_gate` sí quedó conectado a `compute_entry_geometry`: acepta `ema21`/`ema55`
opcionales y, cuando `ticker_analysis_service.compute_core_signals` se los da (ya calcula el mismo
EMA21/55 unificado en 26.4), `GateResult.entry_geometry` lleva la geometría real - verificado
end-to-end contra la respuesta real de `/market/tickers/{ticker}/analysis`. `daily_close.py` sigue
sin pasarle `ema21`/`ema55` a propósito: `TickerDailyState` no tiene columnas para persistir
`entry_geometry`, así que calcularlo ahí sería trabajo tirado hasta que exista esa migración de
esquema, que queda fuera de esta pasada.

### 26.6 Extensión parabólica de `exit_engine.py`: recalibrada a EMA21 sin tocar el campo compartido

La Parte 9 recalibra este disparador a "3 ATR sobre EMA21" (antes "4 ATR sobre SMA50") - pero
`atr_multiple` (`technical_analysis.atr_multiple_from_sma`) es un campo ampliamente compartido:
también lo lee la condición "sin extensión parabólica" del propio gate (`levels_engine.evaluate_gate`)
y `market_screener_service.py`'s snapshots, ninguno de los cuales la Parte 9 pide tocar. Cambiar su
base habría alterado esos dos consumidores como efecto secundario, no como decisión. Resuelto con
una función hermana, `atr_multiple_from_ema` (misma lógica, `ema()` en vez de `sma()`), y un
parámetro propio en `exit_engine.evaluate_exit` (`atr_multiple_from_ema21`, calculado en
`portfolio_risk_service.py` con las series que ya tiene en scope para el resto de disparadores
EMA21/55) - el gate y el screener siguen leyendo la versión SMA50 sin cambios.

### 26.7 Qué queda abierto, honestamente

~~`scripts/daily_close.py` no persiste `entry_geometry`/`size_position`... `size_position` no se
llama desde ningún camino de producción todavía.~~ Resuelto en la sección 26.12 - dejado tachado,
no borrado, para que quede constancia de que esto sí fue una deuda real y no una afirmación vacía
de "casi listo".

Sigue abierto, honestamente: `GET /market/screener` sigue calculando en vivo por request
(`MarketScreenerService.get_universe_snapshot`) en vez de leer `ticker_daily_state` como describen
la Parte 4/10.2 - un endpoint nuevo filtrable por SQL más un rediseño de frontend no trivial
(presets guardables, columnas ordenables, exportación CSV), deliberadamente fuera del alcance de
una pasada incremental. `trade_plan_service.py` tampoco llama a `size_position` ni lo llamará -
una posición reconstruida ya tiene una cantidad real y fija (ver 26.12), dimensionarla no es una
pregunta coherente.

### 26.8 Test explícito de degradación total sin `GEMINI_API_KEY` (criterio de aceptación de la Parte 17)

La Parte 12/17 exige un test que arranque la app sin `GEMINI_API_KEY` y verifique que los endpoints
responden - existía la condición (ningún test de integración configura una clave real; `.env.example`
la deja comentada) pero no un test propio, explícito, que lo afirme como su único propósito. Toda la
suite de integración ya pasaba "sin clave" por accidente de configuración, no por diseño verificado -
`test_gemini_degradation.py` lo hace explícito: confirma primero la premisa
(`get_settings().gemini_api_key is None`) y luego siete endpoints, incluidos los dos que de verdad
ejecutan `GeminiNarrator.explain_gate` bajo el capó (`compute_core_signals`, compartido por "Analizar
activo" y el riesgo de cartera) - todos responden con 200, `llm_narrative: null` donde aplica, sin
ningún intento de red. Si alguna vez se configura un valor por defecto para la clave en el entorno de
tests, este archivo es el primero en dejar de probar el camino "sin clave" en absoluto, no solo dejar
de fallar en silencio.

### 26.9 "Qué pasó con lo que no compraste" (Parte 13) - taken vs. no taken sobre `entry_triggered`

El esquema literal `trigger_history` de la Parte 4.3 lleva un booleano `taken` que la
`TriggerEvent` real de la sección 25.2 nunca tuvo - no hay ninguna columna que el propietario
marque al abrir de verdad una posición. Resuelto sin añadir esa columna (nada más la habría
escrito jamás): `taken` se deriva en `trigger_performance_service.py` a partir del historial real
de `Transaction` - una compra real de ese ticker dentro de `TAKEN_WINDOW_DAYS` (10 días naturales,
un margen holgado sobre el horizonte primario de 5 sesiones) después del `entry_triggered`, mirando
las transacciones de **todas** las carteras, no una en particular. Deliberadamente acotado a
`entry_triggered` únicamente - `gate_passed` es un estado más temprano y menos específico que el
propietario no "actúa" directamente, así que "tomado" no es una pregunta coherente para hacerle.

`compute_trigger_outcomes` acepta `buy_dates_by_ticker` opcional (`None` reproduce exactamente el
comportamiento anterior, una sola fila combinada por tipo de evento/horizonte - nada se rompe para
quien no lo pase); cuando se da, cada fila `entry_triggered` se divide en `taken=True`/`taken=False`,
medidas cada una contra su propio subconjunto de eventos, sin tocar la fila combinada
(`taken=None`) que sigue cubriendo ambos. `GET /system/signal-performance` construye ese
diccionario a partir de `PortfolioRepository.list_all()`/`get_transactions` - una lectura completa,
no optimizada, pero razonable para un puñado de carteras con unos cientos de transacciones cada
una, y no es un camino caliente de las reglas de decisión (la prohibición de cómputo en vivo de la
Parte 12/17 es sobre el propio gate, no sobre este endpoint de medición).

Como todo lo demás de la Fase 8 (sección 25/25.1), esto empieza vacío y solo puede responder hacia
adelante - no hay atajo honesto para comparar retroactivamente lo que se tomó de lo que no, dado
que la propia tabla de eventos empezó a acumularse recién en la Fase 2.

### 26.10 Tesis autogenerada al abrir posición (Parte 5.4)

`trade_plan_service.py` persistía siempre `RECONSTRUCTED_THESIS`, una constante fija sin ningún
hecho real del setup - la Parte 5.4 pide una tesis autogenerada a partir de lo ya calculado
(tendencia, base del stop, objetivo), persistida al abrir la posición. El propio módulo explica por
qué nunca captura el plan de forma síncrona en el momento de la compra (mantener el endpoint de
transacciones rápido, sin coste de red/suite cuantitativa - ver su docstring) - lo que sí puede
hacer, sin ese coste, es generar una tesis *factual* (no el razonamiento subjetivo del propietario,
que de verdad es irrecuperable) a partir de los mismos datos que `reconstruct_stop_and_target` ya
calcula: `generate_thesis` la construye (tendencia vía el mismo `classify_trend(sma20,50,200)` que
el propio gate usa para decidir entradas, distancia del stop, objetivo y su base) y
`ensure_trade_plan` la persiste seguida del disclaimer honesto de reconstrucción - todo plan por
este camino se construye después del hecho, así que ese disclaimer nunca queda obsoleto ni engañoso
al mantenerlo.

### 26.11 El gráfico dibuja EMA21/EMA55 y el semáforo semanal/diario se monta en la ficha

Dos huecos de la Parte 11 confirmados con grep, no supuestos: el gráfico de precio solo dibujaba
SMA50/SMA200 - EMA21/EMA55, el par del que de verdad dependen el gate y el motor de salida desde la
sección 26.4, nunca fue visible; y `MultiTimeframeSemaphore.jsx` existía y se usaba en las
posiciones de cartera, pero `multi_timeframe` en `TickerAnalysisResponse` - calculado en el backend
desde siempre - nunca se renderizaba en la ficha del activo.

- `PricePoint` (dominio, schema, `ticker_analysis_service.py`) gana `ema21`/`ema55`, calculados con
  el mismo `mtf.FAST_MA_PERIOD`/`SLOW_MA_PERIOD` que todo lo demás desde la sección 26.4 - no una
  sexta definición. `PriceChart.jsx` los dibuja justo después del precio, antes de las SMA - son las
  medias que de verdad deciden algo aquí, el resto es contexto.
- `MultiTimeframeSemaphore` se monta en la sección "Gate de entrada" de `TickerAnalysisPanel.jsx`,
  sin ningún cambio de backend - el dato ya estaba en la respuesta, solo nadie lo leía.
- Aprovechado para corregir dos textos que seguían nombrando Gann/estacionalidad/análogos
  históricos, retirados por completo en el cierre de la Fase 1 (sección 26.1) y nunca actualizados
  en la UI hasta ahora.

**Tests**: 5 nuevos en `test_trade_manager.py` reescritos + 4 nuevos (ladder completo con costes,
techo de posición pequeña, cierre por tiempo del último tercio); 3 en `test_backtest_engine.py`
recalculados contra los nuevos multiplicadores del Chandelier; 1 nuevo en `test_market_screener_service.py`
recalculado contra el cruce EMA real; ~15 en `test_exit_engine.py` renombrados/recalculados; 44 en
`test_trade_geometry.py` (39 para `compute_entry_geometry`/`size_position`/`compute_trade_geometry`,
5 para el split); 2 nuevos en `test_levels_engine.py` (`entry_geometry` presente/ausente según
`ema21`/`ema55`); 3 nuevos en `test_technical_analysis.py` (`atr_multiple_from_ema`, incluida la
comparación de reactividad contra la versión SMA); 8 nuevos en `test_gemini_degradation.py`; 7
nuevos en `test_trigger_performance_service.py` (`taken`/`_was_taken`) + 1 de integración nueva en
`test_system_api.py` end-to-end contra portafolio/transacción reales; 3 nuevos en
`test_trade_plan_service.py` (`generate_thesis` con y sin stop/objetivo disponibles, y
`ensure_trade_plan` persistiendo la tesis real + el disclaimer); integración de `test_ticker_analysis_api.py`
actualizada para verificar `ema21`/`ema55` en `price_history`. Suite completa verde en cada commit
(`pytest -q`, unit + integración, ejecutada en domingo sin fallos - la propia Parte 17 exige verde
cualquier día de la semana).

### 26.12 `daily_close.py` persiste `entry_geometry` y `GET /market/radar` lo dimensiona por cartera (Parte 7, cierre)

La sección 26.7 dejaba dos deudas explícitas, ambas con la misma raíz: `size_position` no se podía
conectar en ningún sitio real sin antes resolver la persistencia de `entry_geometry`, porque sus
dos candidatos naturales (la propia `trade_geometry.py` los nombraba) resultan, al mirarlos de
cerca, el mismo problema. `trade_plan_service.py` "al abrir una posición" no aplica: para cuando
`ensure_trade_plan` corre, la compra ya ocurrió con una cantidad real y fija - dimensionar algo que
ya no se puede cambiar no es una sugerencia, es ruido. "El Radar renderizado para una cartera" sí
aplica, pero `GET /portfolios/{id}/today` (donde el Radar ya se cruza con una cartera concreta, vía
`opportunity_cost`) es, por diseño propio (su docstring lo dice explícitamente), una lectura pura
sobre tablas precomputadas - nunca recomputa nada en vivo. Calcular `entry_geometry` ahí exigiría
recalcular ATR/soportes/resistencias/EMA21/55 por cada candidato del Radar en cada request,
exactamente la llamada en caliente por ticker que CLAUDE.md prohíbe. La única salida honesta era
la migración de esquema que la sección 26.7 había pospuesto a propósito.

**Persistencia (`daily_close.py`, `TickerDailyState`, migración `d3f7a2b8c1e4`)**: `build_ticker_daily_state`
ahora calcula `ema21`/`ema55` con `ta.ema(close, mtf.FAST_MA_PERIOD/SLOW_MA_PERIOD)` sobre el mismo
`close` que ya tenía en memoria para ATR/niveles/OBV - sin llamada de red nueva, mismo criterio que
el resto de esta función - y se los pasa a `evaluate_gate`, que ya sabía construir `entry_geometry`
cuando recibe ambos (sección 26.5) pero nunca los había recibido desde este job.
`TickerDailyState.entry_geometry: dict | None` (columna `JSON` nullable, mismo patrón que
`gate_conditions`) persiste el resultado vía `trade_geometry.geometry_to_dict` - siempre la mitad
*sin dimensionar* (`compute_entry_geometry`, nunca `size_position`): un ticker del universo no
pertenece a ninguna cartera, no hay capital que darle todavía. `geometry_to_dict`/`geometry_from_dict`
(nuevas en `trade_geometry.py`) son el par de serialización - `entry_type` (el único campo no
JSON-safe, un `EntryType`) se guarda como su `.value` y se reconstruye con `EntryType(...)`; todo lo
demás ya era JSON-safe. `None` para cada fila calculada antes de esta migración o cuando
`evaluate_gate` no recibió `ema21`/`ema55` reales - nunca una aproximación silenciosa.

**Lectura y dimensionado (`GET /market/radar`)**: `RadarItemResponse.entry_geometry` expone la
geometría sin dimensionar tal cual está persistida - ya es información nueva y útil por sí sola
("a qué precio exacto dispararía esta entrada, con qué stop, con qué riesgo:beneficio neto") para
cada candidato del Radar, no solo el `stop_and_target` más simple que ya existía. El parámetro
opcional `portfolio_id` es la conexión de `size_position` que la sección 26.7 dejaba pendiente:
cuando se da, el endpoint lee `total_portfolio_value` de esa cartera (mismo campo, mismo patrón de
lectura en vivo puntual que `GET /portfolios/{id}/construction` ya usa para la suya - una llamada
de red por *request*, no por ticker) y llama a `size_position` sobre cada geometría viable ya en
memoria - una función pura, sin coste añadido por candidato. Omitir `portfolio_id` mantiene el
endpoint exactamente como era: cero red, pura lectura de `ticker_daily_states`. Una geometría con
`viable=False` se deja tal cual (`size_position` ya es un no-op sobre una geometría rechazada, ver
sección 26.5) - dimensionar un setup que nunca estuvo sobre la mesa fabricaría un número sin
sentido.

**Tests**: 2 nuevos en `test_trade_geometry.py` (`geometry_to_dict`/`geometry_from_dict`, round-trip
con `entry_type` presente y `None`); 1 nuevo en `test_daily_close.py`
(`build_ticker_daily_state` persiste `entry_geometry` viable, nunca dimensionado); 2 nuevos en
`test_precompute_repositories.py` (round-trip por la columna JSON real, y `None` por defecto); 5
nuevos en `test_radar_api.py` (geometría ausente antes de la migración, expuesta sin dimensionar
sin `portfolio_id`, dimensionada correctamente con `portfolio_id` contra el capital real de una
cartera con una posición real, una geometría no viable que `portfolio_id` deja intacta, y 404 con
un `portfolio_id` inexistente). Suite completa verde (`pytest -q`, unit + integración).

### 26.13 `apply_portfolio_limits`: el techo agregado (6%) y de sector (30%) que `size_position` no podía ver

Una auditoría independiente (agente de exploración, dedicado a buscar exactamente este patrón:
comentarios que admiten algo pendiente sin la justificación explícita que sí acompaña a cada deuda
ya conocida de este documento) encontró una más, real, no re-litigada: `size_position` (26.12) fija
una posición en `RISK_PER_TRADE_PCT` del capital y un techo plano `MAX_POSITION_PCT` -
deliberadamente ciega, por su propio diseño, a cualquier *otra* posición abierta. Su propio
docstring ya lo decía: *"The 6% aggregate-risk-across-all-positions cap from the brief is
deliberately NOT enforced here - that needs every other open position's own risk, which is
`portfolio_construction_service.final_position_size`'s job, one layer up from a single ticker's own
geometry."* Esa función - y `trade_manager.max_shares_for_position_risk`, su otra mitad - existían,
probadas (`test_portfolio_construction_service.py`, `test_trade_manager.py`), sin un solo llamador
real en `app/api`. A diferencia de cada "primera pasada, sin calibrar todavía" de este documento
(todas con un porqué explícito y un estudio de ablación al que apuntan), esta frase no tenía ningún
"queda pendiente a propósito" adjunto - simplemente daba la integración por hecha cuando no lo
estaba.

`apply_portfolio_limits` (nuevo, `portfolio_construction_service.py`) es esa capa que faltaba,
aplicada *después* de `size_position`, nunca en su lugar: sobre una geometría ya dimensionada,
calcula cuánto del 6% de riesgo agregado ya está comprometido por el resto de posiciones abiertas
de la cartera (`compute_aggregate_risk` sobre sus `trade_plan`s reales, el mismo cálculo que
`GET /portfolios/{id}/construction` ya hace) y cuánto del 30% de un sector ya ocupa esa cartera
(`compute_sector_concentration` sobre los pesos actuales) - y estrecha el tamaño sugerido a lo que
de verdad queda de margen en cada uno, vía `final_position_size` (el `min()` de las tres
restricciones) y la nueva `sector_limit_shares` (el remanente de un sector concreto, `0.0` en vez
de negativo una vez que ya está en o por encima del techo, para que ese `min()` lo excluya
correctamente en vez de dejar pasar un número sin sentido). Una posición que el estrechamiento deja
por debajo de `MIN_POSITION_USD` se rechaza igual que `size_position` ya rechaza la suya - nunca se
persiste un tamaño tan pequeño que los costes de transacción se lo coman entero.

`GET /market/radar?portfolio_id=` es el único llamador real: para cada candidato ya dimensionado,
una lectura acotada por posición abierta (el mismo puñado de `trade_plan`s que `/construction` ya
lee, nunca uno por candidato del Radar) más `sector_of` del propio candidato - toda la aritmética
sobre datos ya en memoria, cero llamada de red nueva. El peso/riesgo aquí se mide contra el mismo
`capital_total` (`total_portfolio_value`) que `size_position` ya usa - deliberadamente no el
`total_market_value` que `/construction` usa para el suyo (una pregunta distinta: "de lo que ya
invertí, cuánto hay en este sector" contra "de mi capital total, cuánto cabría aquí") - para que el
6% y el 30% de esta narrowing sean comparables entre sí sin mezclar dos bases distintas.

**Tests**: 3 nuevos en `test_portfolio_construction_service.py` para `sector_limit_shares`
(remanente real, cero una vez agotado el techo, remanente completo para un sector no ocupado
todavía, bucket "Desconocido" para un ticker fuera del universo curado) + 4 nuevos para
`apply_portfolio_limits` (no-op sobre una geometría no viable, no-op sobre una geometría que
`size_position` nunca llegó a dimensionar, estrechamiento a la más ajustada de riesgo agregado y
sector con los campos de stop/objetivo/tipo de entrada intactos, rechazo cuando el tamaño ya
estrechado cae por debajo de `MIN_POSITION_USD`); 1 nuevo en `test_radar_api.py` de integración
end-to-end (una cartera real con dos posiciones - una en el mismo sector que el candidato, otra en
uno distinto solo para diluir capital - confirma que el techo de sector, no el de riesgo por
posición, es lo que efectivamente estrecha el tamaño sugerido). Suite completa verde
(`pytest -q`, unit + integración), ruff limpio.

**Complemento de interfaz (mismo día)**: `entry_geometry` llevaba conectado en el backend desde la
sección 26.5/26.12 (`GateResultResponse.entry_geometry` en "Analizar activo", el `/risk` de cartera
y ahora `GET /market/radar`) sin que ningún componente de React lo dibujara jamás -
`RecommendationCard.jsx` solo conocía el `stop_and_target` más simple, el mismo hueco de "el
backend ya calcula el dato real, nadie lo lee" que la sección 26.11 encontró para el semáforo
multi-temporalidad. `EntryGeometryBlock` (nuevo, mismo archivo) sustituye ese bloque cuando
`gate.entry_geometry` existe - stop con su base en prosa (qué peldaño de la cascada, si el techo
duro de ATR lo recortó), objetivo con su base, beneficio:riesgo neto de costes, y el tamaño sugerido
cuando `GET /market/radar?portfolio_id=` lo trae dimensionado. `viable=False` se muestra
explícitamente con su motivo (nunca se oculta como si la geometría no existiera) - el gate booleano
puede aprobar por su chequeo más simple de beneficio:riesgo mientras el diseño real, neto de
costes, lo rechaza, y esa discrepancia es información real, no ruido. `stop_and_target` se mantiene
como resguardo para cualquier lectura que nunca calculó `entry_geometry` (p. ej. el replay de
backtest, que no tiene EMA21/55 punto-en-el-tiempo) - `RecommendationCard` es el único componente
que renderiza ninguno de los dos, así que este único cambio cubre tanto "Analizar activo" como el
Radar. `npm run lint`/`npm run build` limpios; sin infraestructura de tests de componentes en este
frontend (no hay Vitest/Jest configurado), verificado por revisión de código - el flujo de props ya
existía, tipado y probado en el backend, para ambos casos.

### 26.14 Dos huecos más del mismo patrón: `taken` colisionaba claves de React, `confirmed_gate` nunca se mostraba

Tercera pasada del mismo agente de exploración (backend calcula un campo real, ningún componente
lo lee) sobre los esquemas de respuesta restantes. Dos hallazgos, ambos genuinos:

**`TriggerOutcomeResponse.taken` (Parte 13, sección 26.9) nunca llegaba a la interfaz - y peor, su
ausencia producía filas duplicadas con la misma clave de React.** `compute_trigger_outcomes` emite,
para `entry_triggered`, tres filas por horizonte (`taken=None` combinada, `taken=True`,
`taken=False`) - exactamente la pregunta que Parte 13 pedía poder responder. `SystemPerformanceView.jsx`
las recibía las tres, pero su `OutcomeTable` las mapeaba con la misma etiqueta ("Entrada disparada")
y la misma `key={`${o.label}-${o.horizon_days}`}` para las tres - una colisión de claves de React
real, no solo un dato que faltaba: sin la columna nueva, un usuario con historial de compras reales
vería tres filas visualmente idénticas para el mismo horizonte, sin ninguna forma de saber cuál era
cuál. Arreglado con una columna "¿Comprado?" (`showTaken`, activada solo en la tabla de
gate/disparador - las de veredicto/señal nunca tienen `taken`) y una clave de fila que ahora incluye
`taken`, eliminando la colisión de raíz.

**`TickerAnalysisResponse.confirmed_gate` (Fase 4) nunca se comparaba contra `gate` en la
interfaz.** Solo se calcula cuando `is_intraday_snapshot` es verdadero (la sesión de hoy sigue en
curso) - la misma re-evaluación del gate, pero sobre `technical_analysis.closed_bars` en vez del
marco en vivo, exactamente la pregunta "¿esto seguiría aprobando con el último cierre confirmado,
o es un artefacto de una barra que todavía no cierra?" que la propia insignia "● Sesión en curso"
ya advertía sin poder responder. `TickerAnalysisPanel.jsx` ya mostraba esa insignia con una
`title` genérica; ahora, cuando `confirmed_gate` existe, compara `gate.passes` contra
`confirmed_gate.passes` - una nota discreta (`ticker-analysis__section-hint`) cuando coinciden, un
aviso más visible (`.banner--warning`, nueva variante ámbar - mismo tono que la propia insignia
intradía, agregada a `App.css`) cuando no, con el resultado exacto de cada lectura en el texto.

**Tests**: sin infraestructura de tests de componentes en este frontend (mismo estado que la
sección 26.12/complemento anterior) - verificado por revisión de código y por `npm run lint`/
`npm run build` limpios; el `taken`/`confirmed_gate` que consumen ambos cambios ya estaba tipado y
probado en el backend desde sus propias secciones (26.9, Fase 4).

**Tercer hallazgo de la misma pasada, resuelto igual**: `PortfolioConstructionResponse.risk_contributions`
(la contribución de *cada* posición al riesgo real de la cartera, no solo `suggested_to_trim` - el
subconjunto top-N que solo aparece cuando la volatilidad ya supera el objetivo) se calculaba y se
enviaba, pero `PortfolioConstructionPanel.jsx` nunca lo dibujaba - la información de "esta posición
pesa poco en capital pero mucho en riesgo real" quedaba invisible hasta que la volatilidad agregada
ya había superado el objetivo, en vez de estar disponible de forma proactiva. Nuevo bloque
"Contribución al riesgo por posición" (mismo estilo de barra que "Concentración por sector", misma
clase CSS reutilizada sin duplicar reglas) - siempre visible cuando hay posiciones con retorno
suficiente para calcularla, marcando con el mismo aviso visual las que ya aparecen en
`suggested_to_trim`. ~~**Deliberadamente no resuelto en la misma pasada**: `correlation_matrix`...
queda como hueco conocido, no como omisión silenciosa.~~ Resuelto en la sección 26.16 (revisión
propia, mismo día) - tachado, no borrado, mismo criterio que la sección 26.7.

### 26.15 `GET /market/radar?portfolio_id=` era real y probado, pero inalcanzable desde la interfaz

Revisión propia del cierre de la sección 26.13: el parámetro `portfolio_id` existía, probado
end-to-end contra el backend, pero `frontend/src/api.js`'s `getRadar` nunca lo aceptaba - ninguna
pantalla del producto podía pedirlo jamás, solo Swagger o una llamada manual. El mismo patrón de
"el dato/la ruta ya está lista, nadie la usa" que las secciones 26.11/complemento-26.12/26.14 ya
encontraron para lecturas, ahora en un parámetro.

`App.jsx` ya llevaba un `selectedId` (la cartera activa) a nivel de toda la app, usado hasta ahora
solo por la pestaña "Hoy" - reutilizado aquí, no duplicado: el selector de cartera
(`PortfolioSelect`) ahora también aparece en la pestaña "Radar", y `SectionedView` (el envoltorio
genérico que monta cualquier sub-sección de "Radar"/"Activo") gana un prop `portfolioId` más que
reenvía sin condición - inofensivo para las sub-secciones que no lo usan (el screener, movers,
tendencia, soportes/resistencias, "Analizar activo"), real para `RadarView.jsx`, que ahora lo pasa
a `api.getRadar` como `portfolio_id`. Sin cartera seleccionada (`portfolios.length === 0`, la app
recién instalada), el parámetro simplemente no se envía - `toQueryString` ya omite valores
`null`/`undefined` - y el Radar se comporta exactamente igual que antes de esta sección.

**Tests**: sin infraestructura de tests de componentes en este frontend - verificado por revisión
de código y `npm run lint`/`npm run build` limpios; el propio parámetro ya estaba probado
end-to-end en `test_radar_api.py` desde la sección 26.13.

### 26.16 `correlation_matrix`: revisión propia del hueco que la sección 26.14 dejó deliberadamente abierto

Antes de cerrar esta ronda de auditoría se revisó una vez más el propio criterio usado para
diferir `correlation_matrix` (26.14): la razón dada - "una matriz completa es una decisión de
diseño propia (disposición, mapa de color, qué hacer con carteras grandes)" - se sostiene para un
mapa de calor elaborado, pero no para una tabla plana con las celdas coloreadas por magnitud, que
es un patrón estándar y ya tiene precedente directo en el propio panel (`construction-sector__bar-fill`
usa `style` en línea con un valor calculado exactamente de la misma forma). Revisado el criterio,
la pieza restante no era una decisión de diseño pendiente sino, otra vez, el mismo patrón de esta
sección: un dato real (`PortfolioConstructionResponse.correlation_matrix`, ya calculado, ya
enviado) sin nada que lo dibujara.

Tabla N×N en `PortfolioConstructionPanel.jsx`, mostrada solo con más de un ticker (con uno solo la
matriz es un 1×1 trivial): fondo rojo (`var(--series-critical)`) para correlación positiva, verde
(`var(--series-good)`) para negativa, opacidad proporcional a `|valor|` vía `color-mix` - mismo
patrón CSS que la insignia `.construction-risk-card--over` y los indicadores `delta-up`/`delta-down`
ya usan en el resto de la app, ninguna paleta nueva inventada. La diagonal (siempre 1.00) se
muestra sin colorear. Envuelta en `table-scroll` (ya usado en `SupportResistancePanel.jsx`/
`SystemPerformanceView.jsx`) para que una cartera con muchas posiciones se desplace horizontalmente
en vez de romper el layout - la única concesión real a "qué hacer con carteras grandes" que la nota
original de 26.14 mencionaba, y resulta que ya había una solución genérica hecha para eso en el
propio proyecto.

**Tests**: sin infraestructura de tests de componentes en este frontend - verificado por revisión
de código (incluida la coincidencia exacta entre las claves de `correlation_matrix` y las de
`sector_concentrations`/`risk_contributions`, ya cubiertas por tests de integración en
`test_portfolios_api.py`) y `npm run lint`/`npm run build` limpios.

## 27. Sexta auditoría: verificación contra el texto literal completo de las 20 partes (septiembre 2026)

Las auditorías previas (§25, §26) se hicieron contra fragmentos y contra el propio código, sin el
texto literal completo del encargo original en contexto - las secciones 25/26 lo dicen explícitamente
("escrito después de que el texto literal ya había salido de contexto"). El propietario volvió a
pegar las 20 partes completas en esta sesión. Esta auditoría las compara, parte por parte, contra lo
ya construido - y encuentra un panorama más divergente de lo que las auditorías previas asumían.

### 27.1 Lo que coincide fielmente con el texto literal

- **Parte 19 (parámetros)**: `trading_params.py` coincide casi exactamente con la lista literal -
  `RISK_CEILING_ATR_MULTIPLE=2.5`, `RISK_CEILING_MIN_PCT/MAX_PCT=0.020/0.070`,
  `CHANDELIER_WINDOW=10`, los multiplicadores 1.75/2.0/2.25/2.5, `CHANDELIER_PROFIT_LOCK_R=1.5`,
  `SCALE_OUT_1R/2R_FRACTION=0.33`, `MIN_POSITION_FOR_SCALING=150`, `LAST_TRANCHE_TIME_STOP_BARS=15`,
  `MIN_POSITION_USD=80`, `TRANSACTION_COST_PCT=0.001`.
- **Parte 7 (geometría) y Parte 8 (escalera)**: la cascada de stop, el techo de riesgo adaptativo,
  el sizing con sus tres límites y la escalera +1R/+2R/resto con break-even neto de costes coinciden
  con los ejemplos numéricos literales de la Parte 15/20.
- **Parte 2 (borrado)**: `monte_carlo_simulation.py`, `markov_chain_model.py`, `kelly_criterion.py`,
  `volatility_model.py` (GARCH), `statistical_structure.py` (Hurst/ADF), `analysis_tools.py`
  (Gann/estacionalidad/análogos), `walk_forward_backtest.py`, `sector_rotation_service.py`,
  `sector_rrg_service.py`, `premium_watchlist_service.py`, `entry_timing.py`,
  `ablation_report_service.py`, `macro_data_service.py`, `fred_client.py` - todos borrados. Solo
  quedan menciones históricas en comentarios explicando por qué se retiraron, nada de código vivo.
  `opportunity_cost.py` sigue vivo con 71 líneas, coherente con "se reescribe en ~30 líneas".
- **Parte 13 (parcial)**: existe `trigger_event`/`trigger_performance_service` con el campo `taken`,
  funcionalmente equivalente a `trigger_history`, aunque con otro nombre y un esquema más simple.

### 27.2 Divergencias reales encontradas (verificadas contra el código, no supuestas)

1. **El gate no es el gate literal.** El actual (`levels_engine.evaluate_gate`) tiene 6 condiciones
   (tendencia/Fase2, no parabólico, no sobrecompra RSI≥80, no divergencia OBV, no veto par rápido,
   R/R≥1.5). El literal (Parte 6.2) pide 5 criterios eliminatorios distintos: `liquidity_ok`
   (volumen-dólar 20d≥20M USD y precio≥5 USD), `data_quality_ok` (≥250 barras), `weekly_not_stage4`
   (unknown tampoco pasa), `no_fast_bearish_cross` (confirmado o proyectado con R²≥0.6),
   `no_event_risk` (vía Gemini/earnings). Solo el del par rápido coincide conceptualmente. El
   R/R≥1.5 en el literal es un criterio de *viabilidad del disparador* (Parte 5.2/7.4), no del gate
   de elegibilidad (Parte 6.2) - son dos conceptos que el diseño literal separa y esta reconstrucción
   había fusionado en un único gate de 6 condiciones.
2. **`MIN_BARS_REQUIRED = 60`**, no 250 como pide la Parte 3.2 - **resuelto en esta misma sección**
   (27.3).
3. **No existe el sistema de grados A/B/C** (Parte 5.3) en ningún sitio. No existe `LevelKind`/
   `LevelState` como enums (Parte 5.1) - el `PriceLevel` actual (`technical_analysis.py`) solo tiene
   `price/kind/strength/distance_pct`, sin `distance_atr`, `bars_in_state`, `slope_pct_20d`, ni los 6
   estados (FAR/APPROACHING/TESTING/BREAKING/BROKEN_CONFIRMED/LOST_CONFIRMED). Además,
   `support_resistance_levels` filtra los pivotes por el lado del precio actual en el momento de
   detectarlos (`p > current_price` para resistencia, `p < current_price` para soporte) -
   exactamente el sesgo que la Parte 5.1 describe explícitamente como bug a corregir ("guarda los
   pivotes sin filtrar por el precio actual y deja que el estado del nivel diga de qué lado estás").
4. **`GET /market/screener` sigue calculando en vivo** (`market_screener_service.py`, ~800 líneas)
   en vez de leer `ticker_daily_state` con filtros SQL - ya identificado como pendiente en auditorías
   previas (§26.7), y el texto literal (Parte 10.2) confirma que es exactamente lo que pide.
5. **Gemini tiene 1 uso** (`GeminiNarrator.explain_gate`), no los 5 de la Parte 12.3 (G1 catalizadores
   fechados, G2 explicación de movimientos anómalos, G3 triaje de noticias, G4 revisión adversarial,
   G5 journal al cerrar posición).
6. **`ticker_daily_state` tiene muchos menos campos** que la Parte 4.3 (faltan `weekly_ma30`,
   `weekly_stage/bias`, `daily_bias`, `rs_percentile_20d`, `sector_rs_percentile`,
   `relative_volume(_trend)`, `di_bias`, `levels`/`triggers` JSONB ricos, `eligible`/`eligibility`
   JSONB, `data_quality`).
7. **Weinstein semanal usa una proxy diaria** (`classify_stage` con SMA150 diaria como sustituto de
   la MA30 semanal), no la MA30 semanal real que pide la Parte 5.5 ("la de Weinstein de verdad").
8. **~8.460 líneas de servicios**, no <6.500 (criterio de aceptación literal, Parte 17) - sin
   verificar todavía cuánto de ese exceso es lógica real del sistema nuevo (que la Parte 17 no
   contemplaba con este nivel de detalle) frente a algo recortable.

### 27.3 Primer cierre: `MIN_BARS_REQUIRED` a 250

Cambiado en los dos sitios donde vivía duplicado (`market_screener_service.py`,
`ticker_analysis_service.py`) de 60 a 250, tal como pide la Parte 3.2 literalmente ("con 60 barras
no hay SMA200 ni rango anual: el análisis sale degenerado... por debajo de 250, marca el activo como
«datos insuficientes»"). Doce tests unitarios construían series sintéticas de 60-220 barras
asumiendo que ese rango ya era "suficiente" - ninguno probaba el propio umbral (eso lo cubren tests
dedicados aparte), así que se extendieron con relleno plano antes del patrón relevante (contracción
de rango, tendencia de volumen, cruce de medias, rebote en soporte) sin cambiar la intención de cada
uno. La excepción es `test_52_week_range_fields_never_fabricated_with_only_60_bars`
(`test_ticker_analysis_service.py`): su propósito - verificar que sin las 252 barras completas los
campos de rango de 52 semanas nunca se fabrican - ya no se podía probar con 60 barras (ahora
"insuficiente" de entrada), así que pasó a usar 251 barras (por encima del nuevo mínimo, por debajo
de las 252 que ese cálculo concreto exige) y se renombró en consecuencia.

**Tests**: suite completa verde (830 passed, unit+integración) tras el ajuste, sin regresiones.
ruff limpio.

### 27.4 Qué queda pendiente de esta auditoría

El resto de 27.2 (grados A/B/C, `Level`/`LevelKind`/`LevelState` completos con el sesgo de
filtrado corregido, el gate de 5 criterios eliminatorios separado de la viabilidad del disparador,
Weinstein semanal real, `ticker_daily_state` enriquecido, Gemini G1-G5, el screener por SQL) es un
cambio de núcleo que toca prácticamente todo el pipeline de decisión (14 archivos consumen
`PriceLevel` directamente). Se aborda en sub-pasos separados, cada uno con su propio commit y suite
verde, siguiendo la propia Parte 16 ("cada fase termina con la suite en verde y un commit propio").

### 27.5 `Level`/`LevelKind`/`LevelState`: el motor de niveles real (Parte 5.1), añadido junto al existente

Primer sub-paso del núcleo pendiente (27.4). `PriceLevel`/`support_resistance_levels`
(`technical_analysis.py`) se dejan intactos - siguen siendo lo que 14 consumidores ya usan hoy;
migrarlos es un paso posterior. `Level`/`LevelKind`/`LevelState`/`detect_levels` son el diseño real
de la Parte 5.1, añadidos al lado, todavía sin ningún llamador en producción.

- `LevelKind` (12 valores: EMA21/55, SMA50/200, MA30 semanal, pivote de soporte/resistencia, rango
  alto/bajo de 20 días, máximo de 52 semanas, máximo/mínimo del día anterior) y `LevelState` (6
  valores: FAR/APPROACHING/TESTING por distancia en ATR, más BREAKING/BROKEN_CONFIRMED/
  LOST_CONFIRMED por transición de lado) tal como los describe el texto literalmente.
- `_level_state_and_duration` es el corazón: mide primero la *racha actual* (`run_length`) del lado
  en el que está el precio, contando desde la última barra hacia atrás. Si la racha cubre toda la
  ventana, no ha habido cruce reciente y el estado es puro de distancia. Si no, hubo un cruce hace
  `run_length` barras, y se confirma con 1 cierre con volumen relativo >= 1,2 (`run_length == 1`) o
  con 2 cierres consecutivos sin exigir volumen (`run_length >= 2`) - sin ninguna de las dos,
  `BREAKING`, no confirmado. La primera versión de esta función comparaba solo la última barra
  contra la penúltima para decidir "¿hubo cruce?", lo que fallaba exactamente en el caso de "2
  cierres consecutivos confirman la ruptura": con ambas últimas barras ya al nuevo lado, esa
  comparación nunca veía ningún cruce en absoluto. Se detectó con los propios tests de esta sección
  (dos fallos reales, no cosméticos) antes de comitear - corregido con la racha completa, no un par
  de barras.
- **El sesgo de filtrado que la Parte 5.1 señala explícitamente queda corregido en `detect_levels`**:
  los pivotes se detectan con `_fractal_pivots` (sin el `if p > current_price`/`if p < current_price`
  que `support_resistance_levels` sigue aplicando) - un soporte roto sigue apareciendo en la lista,
  con `side="above"` y el estado que le corresponda, en vez de desaparecer porque el precio ya lo
  dejó atrás. Verificado con un test dedicado
  (`test_detect_levels_pivots_are_not_filtered_by_which_side_of_price_they_sit_on`).
- `strength` solo se rellena para pivotes (nº de toques históricos); `slope_pct_20d` solo para
  medias - los niveles estáticos (rango, 52 semanas, día anterior) no tienen ninguno de los dos,
  honestamente `None`.
- `weekly_close=None` omite `WEEKLY_MA30` en vez de fabricarla desde una proxy diaria - la MA30
  semanal real (Parte 5.5, todavía pendiente de conectar a `classify_stage`) sigue siendo trabajo
  de un sub-paso posterior; este módulo ya sabe construir el nivel una vez que alguien le pase el
  cierre semanal (`resample_ohlcv`, ya existente).

**Tests**: 20 nuevos en `test_technical_analysis.py` - los 6 estados y sus condiciones de frontera
(FAR/APPROACHING/TESTING por umbral de ATR, BREAKING sin confirmar, BROKEN_CONFIRMED tanto por
volumen como por 2 cierres, LOST_CONFIRMED simétrico, el caso "1 cierre sin volumen ni segundo
cierre se queda en BREAKING"), `bars_in_state` contando solo la racha de la categoría actual, y 9
tests de `detect_levels` (lista vacía con poco histórico, cada `LevelKind` de media presente,
rango/52 semanas/día anterior presentes, `WEEKLY_MA30` ausente/presente según se dé el cierre
semanal, el propio test del sesgo de filtrado corregido, y que `strength`/`slope_pct_20d` solo se
rellenan donde corresponde). Suite completa verde (715 unit, 133 integración), ruff limpio - sin
tocar ningún consumidor existente todavía.

### 27.6 El gate real: los 5 criterios eliminatorios de la Parte 6.2, no los 6 de la aproximación anterior

El cambio de mayor alcance de esta auditoría. `levels_engine.evaluate_gate` se reescribe con los 5
criterios eliminatorios literales - `liquidity_ok`, `data_quality_ok`, `weekly_not_stage4`,
`no_fast_bearish_cross`, `no_event_risk` (`Eligibility`, nuevo dataclass) - sustituyendo por
completo los 6 de la reconstrucción anterior (tendencia/Fase2, no parabólico, no sobrecompra, no
divergencia OBV, no veto del par rápido, R:R≥1.5), que eran una aproximación razonada, nunca lo que
pedía el texto. `GATE_VERSION` sube a `"2026-09-levels-v2"`.

**Qué se retira, y por qué no es una pérdida silenciosa** (ver el propio docstring de
`levels_engine.py`, ahora reescrito con el razonamiento completo):
- "Tendencia alcista o Fase 2" ya no es un criterio del gate - la dirección de la tendencia la
  exige la propia cascada de la geometría (`trade_geometry._stop_cascade`: los peldaños de
  retroceso a EMA21/continuación sobre EMA55 solo aplican en `TrendState.UPTREND`), no un criterio
  de elegibilidad aparte. Un ticker en tendencia lateral o incluso bajista puede, en teoría, pasar
  el gate ahora - simplemente no producirá un disparador viable casi nunca, sin necesidad de un
  segundo candado.
- "No parabólico"/"no sobrecompra extrema" no tienen respaldo literal como criterios de
  *elegibilidad* - la extensión parabólica ya vive en `exit_engine.py` (REDUCE, Parte 9, sobre
  posiciones abiertas) desde la sección 26.6.
- "Sin divergencia bajista de volumen (OBV)" nunca apareció en ninguna de las 20 partes como
  criterio de entrada - pura invención de la reconstrucción anterior. `technical_analysis.
  obv_divergence` queda sin llamador en el gate (sigue usándose donde antes, como campo
  informativo en `CoreTickerSignals`) - marcado para revisión de código muerto en un sub-paso
  posterior, no borrado en el mismo commit que reescribe el gate.
- El R:R mínimo (antes criterio nº6 del gate, sobre `compute_stop_and_target`) se separa hacia la
  *viabilidad del disparador* (Parte 5.2/7.4) - ya vivía ahí, en
  `trade_geometry.compute_entry_geometry`'s propio chequeo de `MIN_RISK_REWARD_NET`, sin
  duplicarse en el gate. Un ticker puede aprobar el gate de elegibilidad sin tener hoy una entrada
  geométricamente viable - son preguntas distintas en el texto literal, fusionadas por error en la
  reconstrucción anterior.

**Los 5 criterios nuevos, con sus piezas ya existentes reutilizadas, no reconstruidas desde cero:**
- `liquidity_ok`: nueva `dynamic_universe_service.passes_liquidity_floor` reutiliza
  `MIN_DOLLAR_VOLUME_20D`/`MIN_PRICE`, las mismas constantes que ya filtran el universo dinámico
  mensual (Job C) - evaluado aquí a diario por ticker. En divisa nativa, sin conversión a USD, en
  los dos sitios que lo calculan bajo demanda (`ticker_analysis_service.py`,
  `scripts/daily_close.py`) - una simplificación deliberada y documentada, distinta de cómo
  `market_screener_service.py` sí convierte a USD para su propio filtro de universo.
- `data_quality_ok`: en la práctica, casi siempre `True` - el propio llamador ya garantiza
  `MIN_BARS_REQUIRED` (250, sección 27.3) antes de construir cualquier lectura. Se mantiene como
  criterio explícito y persistido, no implícito, tal como pide la Parte 6.2 ("todos los criterios
  se persisten individualmente").
- `weekly_not_stage4`: lee `multi_timeframe.py`'s Stage *semanal* real (MA30 sobre barras
  semanales genuinas vía `resample_ohlcv`) - un hallazgo grato de esta pasada: `multi_timeframe.py`
  YA calculaba esto correctamente desde antes (Parte 5.5 ya estaba resuelta ahí); el hueco real era
  que el gate seguía leyendo `classify_stage` sobre una SMA150 *diaria* (la proxy documentada para
  cuando no hay barras semanales, aplicada aquí por costumbre, no por necesidad). `unknown`
  (menos de ~60 semanas, `MIN_WEEKLY_BARS_FOR_STAGE`) tampoco pasa, literal.
- `no_fast_bearish_cross`: sin cambios - ya reutilizaba `detect_fast_pair_bearish_veto`
  correctamente desde antes de esta auditoría.
- `no_event_risk`: nuevo `_no_event_risk(next_earnings_date, as_of)`, ventana de 14 días naturales
  (aproximación a "10 sesiones", mismo criterio que `TAKEN_WINDOW_DAYS` ya usa en otro sitio sin
  calendario de mercado exacto disponible). `None` (sin fecha de resultados conocida) cuenta como
  *sin* riesgo, no como "no se pudo comprobar" - una decisión propia, documentada: la literal "si
  no se puede comprobar, cuenta como no cumplido" se interpreta sobre un fallo real de la propia
  comprobación (la llamada de red falla, o no hay proveedor), no sobre el resultado normal de "hoy
  no hay nada programado", que es el caso la mayoría de los días del año para la mayoría de los
  tickers - la lectura estrictamente literal bloquearía el gate casi siempre, incluso en
  producción. `get_next_earnings_date` ya existía en el `MarketDataProvider` desde antes; solo
  faltaba llegar hasta el gate.

**Dónde vive cada pieza tras el cambio** - los 3 llamadores reales de `evaluate_gate`
(`ticker_analysis_service.compute_core_signals`/`_confirmed_gate`, `scripts/daily_close.py`) y
`replay_gate_at` (el replay punto-en-el-tiempo del backtest):
- "Analizar activo"/`/risk`: `liquidity_ok` sobre precio/volumen nativos (sin FX);
  `next_earnings_date` reutiliza la llamada que `TickerAnalysisService.analyze()` ya hacía para
  mostrar "días para resultados" en la ficha - antes nunca llegaba al gate.
- `daily_close.py`: mismo `liquidity_ok` nativo; `next_earnings_date` es una llamada de red nueva
  *por ticker del universo*, aceptada explícitamente aquí (no en ningún endpoint) porque este job
  corre una vez por noche, fuera del camino de una petición - la distinción exacta que la propia
  regla de CLAUDE.md ("no llamadas de red por ticker en los caminos calientes") traza entre un job
  y un endpoint.
- `replay_gate_at` (backtest histórico): ni el volumen-dólar en USD ni un calendario de resultados
  históricos están disponibles barato por barra - `liquidity_ok=True`/`next_earnings_date=None`
  siempre, documentado como limitación aceptada (mismo patrón que ya usaba para
  `nearest_support`/`nearest_resistance`). `weekly_not_stage4` usa la proxy diaria (SMA150) en vez
  del semanal real, por el mismo motivo de coste (remuestrear a semanal en cada barra de un
  backtest sería prohibitivo) - una real, reconocida excepción a "el gate real, no una
  aproximación", documentada en el propio docstring de la función, no oculta.

**`scripts/factor_ablation_study.py`** se actualiza en consecuencia: de los 5 criterios, solo 2
varían genuinamente dentro de un replay histórico (`gate_weekly_not_stage4`,
`gate_no_fast_bearish_cross`) - los otros 3 son constantes `True` en `replay_gate_at` por las
razones de arriba, así que exponerlos como factores del estudio de ablación produciría columnas de
varianza cero, colineales con el propio intercepto de la regresión (el mismo problema que el R:R
del gate viejo ya tenía, y por el que nunca se expuso). `CURRENT_POINTS` y el desempaquetado de
`compute_triggers_at` se actualizan a los 2 nombres nuevos.

**Tests**: `test_levels_engine.py` reescrito por completo (18 tests: cada uno de los 5 criterios en
aislamiento, sus combinaciones, `Eligibility.passes`/`failing`, y que un R:R pobre ya no falla el
gate por sí solo); `test_golden_gate_scenarios.py` reescrito por completo (10 escenarios: pase
limpio, Fase 4 semanal confirmada contra un fixture verificado directamente contra
`multi_timeframe.py` antes de escribirlo, semanal desconocida por poco historial, veto del par
rápido, liquidez insuficiente, riesgo de evento dentro/fuera de ventana, y que una tendencia bajista
diaria ya no descalifica por sí sola); `test_levels_engine_replay.py` (1 test actualizado: OBV ya
no cambia el resultado del replay); `test_daily_close.py`/`test_ticker_analysis_service.py`
(versión del gate, forma de `GateResult`); `test_portfolio_risk_service.py` (3 fixtures extendidos
a 700 barras para que el Stage semanal sea conocible, no solo el umbral de 250 de
`MIN_BARS_REQUIRED`); `test_factor_ablation_study.py` (9 tests actualizados a los 2 factores
reales del replay). Suite completa verde (855 passed, unit+integración), ruff limpio.

**Un bug real encontrado por los propios tests, no cosmético**: la primera versión de
`_level_state_and_duration` (sección 27.5) comparaba solo la última barra contra la penúltima para
decidir si hubo un cruce de lado - fallaba exactamente en "2 cierres consecutivos confirman la
ruptura" (con ambas últimas barras ya al nuevo lado, esa comparación nunca detectaba ningún cruce
en absoluto). Corregido con la racha completa (`run_length`) antes de comitear nada de esta
sección - un recordatorio de por qué esta auditoría escribe los tests antes de dar por buena la
lógica, no después.

### 27.7 Fuerza relativa recalibrada a 20 sesiones, `sector_rs_percentile` (Parte 2.5), y los grados A/B/C (Parte 5.3) - construidos, todavía sin conectar

Tres piezas del mismo sub-paso, todas nuevas o recalibradas, ninguna conectada todavía a los
consumidores reales (mismo patrón que la sección 27.5 con `Level`/`LevelKind`/`LevelState` -
construir y probar primero, conectar en un sub-paso aparte).

**`rs_raw_score` recalibrado (Parte 3.2, otro punto de esa tabla que se había marcado por error
como resuelto en el mapeo inicial de esta auditoría)**: de un compuesto ponderado 63/126/189/252
sesiones (estilo IBD, momentum de 3-12 meses - "a 5 días no aplica", literal) a un simple retorno a
20 sesiones. El propio `_percentile_rank` transversal que ya existía no cambia - solo el valor
crudo que alimenta. `RS_RAW_SCORE_WINDOW = 20` como constante nombrada, ya no un `dict` de cuatro
pesos.

**`sector_rs_percentile` (Parte 2.5)**: "un campo por ticker" que sustituye por completo los cinco
servicios/cuatro vistas de sectores retirados en las auditorías previas (§25/26) - hasta ahora
nunca implementado, ni siquiera parcialmente (`SECTOR_ETFS`/`EUROPE_SECTOR_ETFS` existían en
`market_universe.py` sin ningún consumidor). `_sector_rs_percentiles` en `market_screener_service.py`:
retorno a 20 sesiones del ETF de cada sector, percentilado entre los sectores de la región - los
ETFs se descargan en el mismo lote bulk que el universo ya pedía (nunca una llamada de red nueva
por ticker, ~11 tickers más en una petición que ya existía). Nuevo campo en `TickerSnapshot`,
`None` hasta que el ETF de ese sector tenga suficiente historial.

**Grados A/B/C (Parte 5.3)**: `Grade`/`GradeResult`/`compute_grade`/`apply_portfolio_grade_modifiers`
en `levels_engine.py`, junto al gate. "No inventes una probabilidad; no la tienes" - geometría
pura, nunca una probabilidad de éxito. Grado base por distancia al nivel del disparador (en ATR,
un número distinto de `risk_atr`, la distancia al *stop*), riesgo en ATR, R:R neto y volumen
relativo (los cuatro ya calculados por `trade_geometry`/`technical_analysis` - `compute_grade` no
recalcula nada, solo los compara contra los umbrales literales), más semanal alcista/bajista de
`multi_timeframe.py`. Los modificadores de fuerza relativa/SMA200/sector se aplican después,
literalmente: como máximo un escalón de subida por el conjunto de razones de subida (no uno por
cada razón), y un techo de B en cuanto cualquier razón de bajada aplica. Los dos modificadores que
necesitan una cartera específica (correlación con una posición abierta, tope de posiciones) viven
en una función separada, `apply_portfolio_grade_modifiers` - mismo motivo de la separación
`compute_entry_geometry`/`size_position` ya usa: un disparador del universo no pertenece a ninguna
cartera en particular. `grade=None` (nunca se emite el disparador) cuando la geometría de origen no
es viable - no hay nada que gradar.

**Deliberadamente no tocado en este sub-paso**: la ventana de Mansfield RS por defecto sigue en 200
sesiones (Parte 3.2 también la pide en 20) - ya existe un campo separado `mansfield_rs_4w`
(window=20, Segunda auditoría Bloque 3) sin consumidor todavía, así que la pieza de datos ya existe;
cambiar el *default* global tiene más riesgo (Mansfield RS se muestra como serie/gráfico, no solo
como número puntual para un ranking, así que una ventana de 20 sesiones cambiaría su naturaleza
visual, no solo su calibración) - queda como hueco conocido, documentado, no una omisión silenciosa.

**Tests**: 3 nuevos para `rs_raw_score` (positivo en una subida sostenida, `None` con historial
genuinamente insuficiente para 20 sesiones, y que de verdad usa una ventana de 20 no la vieja de
63+); 3 nuevos para `_sector_rs_percentiles` (ranking correcto entre sectores, sectores sin
historial suficiente excluidos, vacío sin ningún ETF con historial); 16 nuevos para el sistema de
grados (A/B/C limpios, `None` sin geometría viable, cada modificador de fuerza relativa/SMA200/
sector en aislamiento, el tope de un solo escalón con dos razones de subida a la vez, y los tres
casos de `apply_portfolio_grade_modifiers` - correlación alta, cartera llena, no-op sin grado).
Suite completa verde, ruff limpio.

### 27.8 Los grados A/B/C conectados a "Analizar activo" - dejan de estar inertes

Sub-paso siguiente a 27.7, autorizado explícitamente por el propietario ("Vale, continúa") tras
proponer esta conexión como el paso más visible: sin ella, el ejemplo central de la Parte 0
("NVDA - Grado A") seguía sin poder ocurrir de verdad en ningún camino en vivo - el sistema de
grados existía, tenía sus 16 tests, pero ningún endpoint lo llamaba.

**Cableado**: `ticker_analysis_service.compute_core_signals` calcula `grade` justo después de
`gate = evaluate_gate(...)`, y solo cuando hay algo que gradar de verdad -
`gate.entry_trigger is not None and gate.entry_geometry is not None and gate.entry_geometry.viable`
- exactamente el mismo criterio que `compute_grade` ya documentaba como su propio precondición
("no hay nada que gradar" si la geometría de origen no es viable). El sesgo semanal alcista/bajista
se lee de `mtf.timeframe_bias(multi_timeframe.weekly)`, ya calculado unas líneas antes para el
propio `multi_timeframe` de la respuesta - ninguna llamada nueva.

`sector_rs_percentile` necesitaba un segundo dato que "Analizar activo" no leía todavía: el
`TickerSnapshot` completo del universo (antes solo se leía `.rs_rating` vía un método
`_rs_rating_for`, ahora renombrado a `_universe_snapshot_for` y devolviendo el snapshot entero) -
mismo lookup, un campo más aprovechado, cero llamadas nuevas. `compute_core_signals` gana un
parámetro `sector_rs_percentile: int | None = None` explícitamente documentado como "ya
precomputado por el screener, no se recalcula aquí" (mismo patrón que `next_earnings_date`, que
`analyze()` ya pasaba desde antes).

`CoreTickerSignals`/`TickerAnalysis` (dominio) ganan un campo `grade: GradeResult | None` junto a
`gate`. En el límite de la API, `GradeResponse` (nuevo, en `schemas/quant_analysis.py`, junto a
`GateResultResponse`) - `grade: str | None` + `reasons: list[str]`, deliberadamente sin los
modificadores de cartera (`apply_portfolio_grade_modifiers` sigue sin consumidor: correlación con
posición abierta y tope de posiciones solo tienen sentido dentro de una cartera concreta, no en una
lectura de "Analizar activo" que no está scoped a ninguna). `TickerAnalysisResponse.grade` expone
esto en `GET /api/v1/market/tickers/{ticker}/analysis`.

**Qué no cambia todavía**: `portfolio_risk_service.py` reutiliza `compute_core_signals` pero no le
pasa `sector_rs_percentile` (sigue con el default `None`) ni persiste `grade` en ninguna tabla -
`scripts/daily_close.py`/`TickerDailyState` no lo calculan, así que el Radar y "Hoy" (que leen
estado precomputado, no recalculan) todavía no muestran grado. Ambos quedan como sub-pasos
explícitamente pendientes, no como omisiones descubiertas después.

**Tests**: sin tests unitarios nuevos (el cálculo del grado en sí ya tiene sus 16 tests en
`test_levels_engine.py` desde 27.7; este sub-paso es cableado, no lógica nueva) - se extendió
`test_ticker_analysis_returns_full_payload` (integración) para comprobar que `body["grade"]` viaja
en la respuesta real y, cuando no es `None`, tiene la forma esperada (`grade` en
`{"A", "B", "C"}`, `reasons` una lista). Suite completa verde (742 unitarios + integración), ruff
limpio.

### 27.9 `sector_rs_percentile` y el grado también llegan a `portfolio_risk_service.py`

Mismo sub-paso que 27.8, un consumidor más: `assess_position_risk`/`get_portfolio_positions_risk`/
`PortfolioRiskService.get_positions_risk` reutilizan `compute_core_signals` (documentado desde
hace tiempo como "una señal, un solo sitio donde se calcula"), pero hasta ahora solo le pasaban
`rs_rating` del `universe_snapshot` - `sector_rs_percentile` se quedaba en su default `None`, así
que una posición abierta nunca podía recibir el modificador de sector del grado aunque estuviera
en un ticker del universo dinámico con el dato disponible.

**Cableado**: mismo patrón que `rs_by_ticker = {s.ticker: s.rs_rating for s in universe_snapshot}`,
una línea más: `sector_rs_by_ticker = {s.ticker: s.sector_rs_percentile for s in universe_snapshot}`,
en los tres sitios que ya construían el primer diccionario (`get_portfolio_positions_risk`,
`PortfolioRiskService.get_positions_risk`). `assess_position_risk` gana el parámetro
`sector_rs_percentile: int | None = None` y lo reenvía a `compute_core_signals` sin tocarlo -
la única lógica que lo consume sigue siendo `compute_grade`'s modificador de sector, ya probado en
27.7. `CoreSignalsResponse` (usado por `PositionRiskResponse` en `schemas/market.py`, el DTO real
de `GET /portfolios/{id}/risk`) gana su propio `grade: GradeResponse | None` - `_core_signals_to_response`
en `api/v1/endpoints/portfolios.py` no necesitó cambios: ya construye la respuesta vía
`CoreSignalsResponse(**asdict(signals))`, así que el nuevo campo de `CoreTickerSignals` viaja solo.

**Qué sigue sin cambiar**: `daily_close.py`/`TickerDailyState` (el precompute nocturno que alimenta
el Radar y "Hoy") no calcula ni persiste `grade` todavía - una posición abierta ya lo tiene vía
`/risk` (cómputo en caliente, aceptado para ese endpoint desde su propio diseño), pero un candidato
nuevo del Radar todavía no. Queda como el próximo sub-paso natural, no una omisión de este.

**Tests**: 2 nuevos en `test_portfolio_risk_service.py` - `sector_rs_percentile` llega intacto de
`assess_position_risk` a `compute_core_signals` (monkeypatch capturando el kwarg, sin depender de
que el grado termine siendo A/B/C de verdad - eso ya lo prueba 27.7), y `get_portfolio_positions_risk`
lo busca en el `universe_snapshot` correcto por ticker (incluido el caso `None` para un ticker sin
el dato). Suite completa verde, ruff limpio.

### 27.10 El grado llega al Radar - persistido por `daily_close.py`, migración `4f7f279777aa`

Cierra el círculo de 27.8/27.9: las tres superficies que comparten `compute_core_signals`/el gate
("Analizar activo", `/risk` de cartera, y ahora el Radar) exponen el grado A/B/C. El Radar es
distinto de las otras dos - nunca calcula nada en caliente (regla no negociable de CLAUDE.md, ver
`ticker_daily_state.py`), así que el grado tiene que persistirse en `ticker_daily_states` por el
job nocturno, exactamente el mismo patrón que `entry_geometry` (migración `d3f7a2b8c1e4`, Parte 7)
ya estableció.

**Migración `4f7f279777aa`** (`down_revision=d3f7a2b8c1e4`): columna `grade` JSON, nullable, en
`ticker_daily_states` - `TickerDailyStateORM.grade`, `TickerDailyState.grade: dict | None`.
Deliberadamente sin una función `grade_from_dict` gemela de `geometry_from_dict`: nada relee un
grado para recalcularlo o dimensionarlo en el momento de la lectura (a diferencia de
`entry_geometry`, que `GET /market/radar?portfolio_id=` sí reconstruye para pasarlo por
`size_position`) - es un valor terminal, de solo mostrar, así que un dict plano
`{"grade": "A"|"B"|"C"|None, "reasons": [...]}` basta, mismo criterio que ya usa `gate_conditions`.

**`daily_close.py` (`build_ticker_daily_state`)**: calcula el grado con la misma precondición
exacta que "Analizar activo" - `gate.entry_trigger is not None and gate.entry_geometry is not None
and gate.entry_geometry.viable` (dos disparadores distintos que deben coincidir a la vez:
`entry_trigger`, el más simple, basado en soporte/resistencia; `entry_geometry`, el real de la
cascada de stop de Parte 7 - un ticker puede tener geometría viable sin sentarse cerca de un
soporte/resistencia clásico, y viceversa; solo cuando ambos coinciden hay algo que gradar). Todos
los insumos ya estaban en memoria en este job: `snapshot.relative_volume`/`sma200`/`rs_rating`/
`sector_rs_percentile` (ya calculados por el screener para este ticker) y
`mtf.timeframe_bias(multi_timeframe.weekly)` (el semanal real, ya leído unas líneas antes para
`weekly_stage`) - cero llamadas nuevas.

**`GET /market/radar`**: `RadarItemResponse.grade: GradeResponse | None`, leído tal cual desde
`TickerDailyState.grade` (`_grade_dict_to_response`, mismo patrón que `_geometry_dict_to_response`)
- nunca resized ni modificado por `?portfolio_id=` (eso solo afecta `entry_geometry`/`size_position`;
los modificadores de cartera del grado, `apply_portfolio_grade_modifiers`, siguen sin consumidor -
ver 27.7).

**Tests**: 2 nuevos en `test_daily_close.py` (grado presente con geometría viable + trigger real
usando el mismo fixture de pullback-a-soporte que `test_portfolio_risk_service.py` ya usa para
"add candidate"; `None` cuando una subida monótona sin soporte/resistencia cercano nunca produce un
`entry_trigger` aunque la geometría EMA sí sea viable - el caso que demuestra por qué la
precondición necesita *ambos* disparadores, no solo uno); 2 nuevos en `test_radar_api.py` (`grade`
viaja tal cual desde una fila sembrada; `None` en una fila pre-migración). Suite completa verde,
ruff limpio (fuera de `alembic/versions/`, que nunca ha estado bajo `ruff check app tests` - mismo
estilo que toda migración autogenerada existente).

### 27.11 `Level`/`LevelKind`/`LevelState` conectado a "Analizar activo"/`/risk` de cartera

Primer consumidor real del motor de niveles construido en 27.5 (`technical_analysis.detect_levels`,
"añadido, todavía sin ningún llamador en producción"). `compute_core_signals` lo llama junto a
`support_resistance_levels` (el sistema simple, sin tocar - 14+ consumidores siguen leyéndolo sin
cambios), así que `CoreTickerSignals`/`TickerAnalysis` ganan un campo `levels: list[ta.Level]`
puramente aditivo.

**Cierra de paso el hueco documentado en 27.5**: `WEEKLY_MA30` estaba ausente porque nadie le pasaba
un `weekly_close` real a `detect_levels`. `compute_core_signals` ya construye `daily_df` para
`multi_timeframe.analyze_multi_timeframe` (que resamplea semanalmente por dentro, pero no expone la
serie cruda) - un segundo `ta.resample_ohlcv(daily_df, mtf.WEEKLY_RULE)` (mismo `daily_df` ya en
memoria, coste de CPU trivial, cero llamadas de red nuevas) obtiene el cierre semanal real que
faltaba. `WEEKLY_MA30` aparece ahora en la lista siempre que haya al menos 2 semanas cerradas.

**API**: `LevelResponse` nuevo en `schemas/common.py` (junto a `PriceLevelResponse`, mismo motivo de
colocación: que `quant_analysis.py` y `market.py` lo puedan usar sin importarse entre sí).
`CoreSignalsResponse.levels`/`TickerAnalysisResponse.levels` - ninguno de los dos endpoints
(`_core_signals_to_response` en `portfolios.py`, `_to_response` en `ticker_analysis.py`) necesitó
cambios propios: ambos ya construyen la respuesta vía `asdict(...)` genérico, así que el campo nuevo
del dataclass de dominio viaja solo, mismo mecanismo que ya benefició a `grade` en 27.8.

**Deliberadamente fuera de este sub-paso**: el Radar/`daily_close.py` no persisten `levels` todavía
(a diferencia de `entry_geometry`/`grade`) - la lista completa de niveles es más pesada que un
`grade` de dos campos, y el Radar no tiene todavía una vista de UI que la use; conectar los 14
consumidores existentes de `support_resistance`/`nearest_support`/`nearest_resistance` al sistema
nuevo tampoco - ambos quedan como trabajo futuro explícito, no como omisión.

**Tests**: 1 nuevo en `test_ticker_analysis_service.py` (`levels` no vacío, un `LevelKind.EMA21`
presente y sin racha reciente de cruce en una subida sostenida, `support_resistance` sigue
poblado sin cambios); `test_ticker_analysis_returns_full_payload` (integración) extendido para
comprobar `body["levels"]` en la respuesta real de la API, incluido `"weekly_ma30"` entre los
`kind` presentes (los 10 años de histórico falso de AAPL superan de sobra el mínimo semanal).

## 28. Biblioteca de setups del Radar (septiembre 2026, en curso)

Encargo nuevo del propietario, independiente de las 20 Partes originales: el Radar
(`GET /market/radar`) responde hoy "qué está a punto de dar entrada" con una lista plana -
ticker, precio, RS, tendencia, etapa - sin decir *por qué* ese ticker merece atención ni en qué
punto de su formación está. El encargo pide una biblioteca de detectores de patrones técnicos
con nombre propio (transición de etapa 1→2 de Weinstein, VCP, rupturas, retrocesos, cruces de
medias, canales, patrones clásicos) que conviertan esa lista en setups identificados, con su
gatillo, su geometría y su estadística histórica medida.

**Por qué esto no contradice "menos factores, no más"** (la regla fundacional de toda la
reconstrucción de 2026-09, ver §1/§6.6): la distinción está en el propio módulo -
`app/services/setups/__init__.py` documenta la regla completa, resumida aquí porque es el
estándar contra el que se audita cada fase siguiente: ningún setup suma puntos a otro (el mejor
gana, los demás son contexto - `setups/arbitration.py`, fase posterior); los modificadores de
contexto suben como máximo un escalón de grado en total, nunca más; ningún setup entra a
producción sin su propio detector, tests y medición histórica (replay con triple barrera); un
setup medido peor que la entrada aleatoria en dos regímenes se retira por completo, misma
disciplina que ya retiró el filtro de Faber y la cadena de Markov. Los patrones clásicos
(taza-con-asa, H-C-H, doble suelo, triángulos) reciben trato asimétrico: se detectan porque el
propietario los quiere ver, no por evidencia sólida de que operarlos sea rentable (Lo, Mamaysky
y Wang 2000 - informativos, no necesariamente rentables, sin costes de transacción en su
prueba) - solo taza-con-asa/doble-suelo/triángulo-ascendente disparan por sí solos, el resto
(H-C-H incluido) siempre acompaña.

**Decisión de esquema, tomada antes de escribir el primer detector**: `ticker_daily_states.setups`
será una columna JSON nueva (mismo patrón exacto que `entry_geometry`/`grade`, §27.8-§27.10) -
no una tabla `setup_matches` propia, que multiplicaría filas (~7 familias × universo × sesión)
para datos que el Radar siempre lee en bloque y nunca filtra por SQL. `setup_performance` sí será
una tabla nueva de verdad (dominio/ORM/repo/migración propios) porque su grano es distinto -
agregado por `(setup_name, grado, régimen)`, no por ticker/día.

### 28.1 Fase 1: tipos y registro, sin detectores todavía

Primer sub-paso, puramente de andamiaje - ningún detector real todavía, siguiendo la misma
disciplina de la reconstrucción de que cada fase termine con tests en verde y su propio commit.

- `app/services/setups/types.py`: `SetupFamily`/`SetupStage`/`SetupConfidence` (los tres enums
  literales del encargo) y `SetupMatch` (dataclass frozen con los campos exactos que pide el
  encargo - `family, name, label_es, stage, bars_in_stage, timeframe, trigger_price,
  trigger_condition, invalidation_price, invalidation_condition, evidence, narrative_es,
  confidence`).
- `app/services/setups/context.py`: `SetupContext` (frozen, **`kw_only=True`** a propósito - más
  de una docena de campos, una llamada posicional sería ilegible) con todo lo que un detector
  puede necesitar ya calculado: OHLCV diario y semanal (remuestreado una sola vez por quien
  construye el contexto, nunca por el propio detector), `atr_series`/`atr14`, ema21/55,
  sma20/50/150/200, `levels: list[ta.Level]` (el motor de niveles de Parte 5.1/§27.5, ya con
  estado y duración), `multi_timeframe`, `trend`, `weekly_stage`, `relative_volume` y los
  percentiles de RS/sector. Ningún campo se calcula aquí - el módulo entero es un contenedor.
- `app/services/setups/registry.py`: `SETUP_DETECTORS: list[SetupDetector]` (vacío a propósito -
  cada familia se añade explícitamente en su propia fase, nunca antes de tener detector+tests+
  medición) y `detect_all(ctx) -> list[SetupMatch]`, con aislamiento por detector vía
  `try/except` - un detector con un borde no cubierto no debe vaciar el resultado de los demás
  del mismo ticker, la misma disciplina que `portfolio_risk_service._safe_assess_position_risk`
  ya aplica un nivel más arriba (por ticker, no por detector).
- `app/services/setups/__init__.py`: la nota de estándar completa de la biblioteca (regla
  anti-suma, tope de un escalón, regla de admisión, tratamiento asimétrico de patrones clásicos
  con la cita de Lo-Mamaysky-Wang) como docstring del paquete - es el único `__init__.py` del
  proyecto con contenido real (los de `infrastructure/llm`/`infrastructure/market_data` están
  vacíos a propósito), justificado porque el propio encargo pide este texto exactamente en este
  archivo, como referencia obligada antes de escribir cualquier detector nuevo.

**Presupuesto de rendimiento** (documentado, no medido todavía - se mide de verdad en la Fase 7
vía una columna JSON nueva `job_runs.detail`): 40 ms/ticker para todos los detectores juntos. El
riesgo real no está en los detectores en sí, sino en que la Fase 2 (conectar `SetupContext` a
`daily_close.py`) evite que cualquier detector recalcule `resample_ohlcv`/`detect_levels` por su
cuenta - si eso se cumple, el coste real ya está pagado por el gate/la geometría existentes antes
de que el primer detector corra.

**Tests**: `test_setups_registry.py` - `detect_all` con cero detectores registrados devuelve
`[]`; recolecta coincidencias de varios detectores en orden; aísla un detector que lanza
excepción sin perder las coincidencias de los demás (mismo patrón que
`test_get_portfolio_positions_risk_isolates_a_ticker_whose_compute_raises`); nunca ordena ni
deduplica (eso es trabajo de `arbitration.py`, fase posterior). Suite completa verde, ruff
limpio.

**Plan de fases 2-11** (arquitectura completa ya diseñada y presentada al propietario, pendiente
de ejecutar paso a paso): wiring a `daily_close.py` sin duplicar cómputo (Fase 2); familias
baratas - transición de etapa, cruce rápido, retroceso (Fase 3); ruptura, canal (con una
primitiva nueva compartida `technical_analysis.linear_regression_fit`, reutilizada también por
triángulos) (Fase 4); VCP y modificadores de contexto (Fase 5); arbitraje - el único sitio donde
viven las reglas "nadie suma"/"máximo un escalón" (Fase 6); persistencia - migraciones, columna
`setups` (Fase 7); `GET /market/radar` extendido - orden lexicográfico, agrupación por sector,
cortes duros (Fase 8); replay histórico de setups reutilizando `label_triple_barrier`/
`compute_trading_metrics` tal cual (Fase 9); `taken` derivado contra transacciones reales, nunca
persistido (Fase 10, misma decisión que `trigger_performance_service.py` ya tomó); frontend
(Fase 11).

**Correcciones ya incorporadas al plan original tras revisión propia, antes de implementar**:
tolerancia de "contracción decreciente" del VCP bajada de 15% a 5-8% (con 15%, una secuencia
20%→22%→18% pasaba como "decreciente"); umbral de secado de volumen del retroceso bajado de 0,9×
a 0,70-0,75× (0,9× apenas filtra nada - la mayoría de sesiones caen ahí por varianza natural);
"RS girando" simplificado a solo el cruce de Mansfield RS sobre su MA10 (se descarta, de momento,
detectar también "mínimo más alto" sobre la propia línea de RS - complejidad real por beneficio
marginal); el cruce rápido "proyectado" no toca el umbral interno de `detect_imminent_cross`
(`IMMINENT_CROSS_MIN_R2=0.5`, usado por otros consumidores) - en su lugar post-filtra el
`r_squared` ya devuelto para exigir 0,6 solo en este setup; `RADAR_DROP_FORMING_BELOW_GRADE` se
descarta tal cual estaba especificado - un setup en `FORMING` no tiene `entry_trigger`/geometría
todavía, así que no puede tener un grado A/B/C real que cortar (`compute_grade` exige ambos); la
propia ordenación por `SetupStage` ya los deja al final, sin necesidad de un grado especulativo
sobre una entrada que aún no existe.
Suite completa verde, ruff limpio.
