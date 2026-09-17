"""La biblioteca de setups del Radar (`GET /market/radar`): detectores de
patrones técnicos con nombre propio - transición de etapa 1→2 de Weinstein,
VCP, rupturas, retrocesos, cruces de medias, canales, patrones clásicos -
que responden a "¿qué está a punto de dar entrada?" con algo más que un
ticker y un precio: un setup identificado, su gatillo exacto, su geometría
y su estadística histórica medida. Ver `app/services/levels_engine.py` para
el gate/grado que esta biblioteca complementa, nunca sustituye.

Toda la reconstrucción de 2026-09 se hizo bajo "menos factores, no más" -
retirar el checklist ponderado de 26 factores por un gate de reglas
booleanas. Una biblioteca de detectores podría parecer ir en contra. No lo
es, y la distinción es la regla de diseño de este paquete:

- Lo que envenena un sistema son N factores colineales sumados en una
  puntuación: todos miden lo mismo desde ángulos distintos, ninguno se
  puede auditar por separado, y el resultado es un número que nadie puede
  verificar.
- Lo que funciona es una biblioteca de setups mutuamente distinguibles,
  cada uno con un nombre, condiciones que se cumplen o no, un gatillo
  propio, y su propia tasa de acierto medida de forma independiente.

Por eso **ningún setup suma puntos a otro** (`setups/arbitration.py`): un
ticker que cumple tres setups no obtiene "tres veces más nota" - obtiene el
mejor de los tres, y los otros dos se muestran como contexto. Los
modificadores de contexto (`setups/context.py`) son la única excepción, y
está acotada: entre todos suben como máximo un escalón de grado, nunca más,
sea cual sea el número de modificadores que apliquen a la vez. Sin este
tope, la biblioteca degenera en el score de 26 factores por la puerta de
atrás - es la regla más fácil de romper sin querer y la más importante de
vigilar en cada revisión.

Regla de admisión, no negociable: un setup solo llega a producción con su
detector, sus tests unitarios y su medición histórica (`setup_replay.py`,
replay punto-en-el-tiempo con triple barrera y costes reales, mismo patrón
que `levels_engine.replay_gate_at`/`backtest_engine.find_triple_barrier_entries`).
Un setup sin muestra suficiente se muestra etiquetado como tal
(`SetupConfidence.THIN`) y se ordena el último. Un setup medido peor que la
entrada aleatoria en dos regímenes de mercado distintos se retira por
completo - no se rebaja de peso ni se esconde - la misma disciplina "medir
antes de confiar" que ya retiró el filtro de régimen de Faber y la cadena
de Markov (ver `docs/quant_methodology.md`).

**Los patrones clásicos reciben un tratamiento deliberadamente asimétrico.**
Taza con asa, hombro-cabeza-hombro, doble suelo, triángulos - se detectan
porque el propietario los usa y quiere verlos, no porque exista evidencia
sólida de que operarlos sea rentable. La referencia académica más citada,
Lo, Mamaysky y Wang (2000), *Foundations of Technical Analysis* (Journal of
Finance), encontró mediante detección automática por regresión kernel que
varios de estos patrones - hombro-cabeza-hombro entre ellos - alteran la
distribución de retornos de forma estadísticamente significativa, con más
fuerza en Nasdaq que en NYSE/AMEX. Los propios autores advierten, sin
embargo, que solo probaron que el patrón es *informativo*, no que operarlo
sea *rentable*: no incluyeron costes de transacción, y su prueba es, en sus
palabras, débil respecto a la eficacia real del análisis técnico.

Por eso el tratamiento es asimétrico. Los setups con reglas cuantificadas y
verificables por construcción - transición de etapa, VCP, caja, ruptura de
nivel, retroceso - generan gatillo (`trigger_price` no `None`). Los
patrones clásicos acompañan como contexto; solo taza-con-asa, doble-suelo y
triángulo-ascendente disparan por sí solos (`classic_patterns.py`, Fase 4),
el resto (hombro-cabeza-hombro incluido) nunca lo hace. Y todos, sin
excepción, se someten al mismo replay histórico con triple barrera y costes
que el resto del sistema - no hay un patrón "de segunda" exento de medirse.

No inventes probabilidades: los grados son geometría (`levels_engine.py`);
las cifras de un setup son historia medida, no una promesa sobre el próximo
trade. No uses Gemini en ningún detector - `narrative_es` es plantilla
determinista, nunca una llamada a `GeminiNarrator`."""
