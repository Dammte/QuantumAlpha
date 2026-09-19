"""Auditoria del Radar, bloque E4: "el primario lleva... dos o tres frases
de tesis que resuman qué hace ese setup interesante hoy... generada con
Gemini a partir de los números ya calculados... si la llamada falla, se
muestra un resumen plantilla determinista. El LLM redacta, no decide."

Este módulo es esa plantilla determinista - el fallback que SIEMPRE
funciona, sin red, sin Gemini, y la única pieza que existe hasta que el
bloque 12 conecte la llamada real. Puro (sin FastAPI/Pydantic) - toma
primitivos ya calculados por el resto del Radar, nunca recomputa nada
(mismo principio que `trade_plan_service.generate_thesis` ya aplica para la
tesis de apertura de una posición real - "el LLM redacta, no decide" es la
misma regla, un nivel más abajo)."""


def generate_deterministic_thesis(
    ticker: str,
    setup_narrative: str | None,
    sector: str | None,
    rs_rating: int | None,
    entry_price: float | None,
    stop_price: float | None,
    stop_basis: str | None,
    target_price: float | None,
    risk_reward_net: float | None,
    score_total: float | None,
) -> str:
    """Tres frases, en el mismo orden que la ficha ampliada las presenta:
    (1) la narrativa del setup ya generada por su propio detector -
    reutilizada tal cual, nunca reescrita, porque ya es una frase
    determinista y factual (Parte 15: "el LLM redacta, no decide" - aquí ni
    siquiera hay LLM); (2) la geometría (entrada/stop/objetivo/R:R); (3) el
    contexto de fuerza relativa y el score compuesto, para que el número
    del bloque 12.1 ("por qué está aquí") tenga una frase que lo explique
    incluso sin Gemini."""
    sentences: list[str] = []

    if setup_narrative:
        sentences.append(setup_narrative)

    if entry_price is not None and stop_price is not None:
        stop_desc = stop_basis if stop_basis else f"stop en {stop_price:.2f}"
        geometry_sentence = f"Entrada en {entry_price:.2f}, {stop_desc}"
        if target_price is not None:
            geometry_sentence += f", objetivo en {target_price:.2f}"
        if risk_reward_net is not None:
            geometry_sentence += f" (R:R {risk_reward_net:.1f}:1)"
        sentences.append(geometry_sentence + ".")

    context_parts = []
    if rs_rating is not None:
        context_parts.append(f"RS {rs_rating}")
    if sector is not None:
        context_parts.append(f"sector {sector}")
    if score_total is not None:
        context_parts.append(f"score {score_total:.0f}/100")
    if context_parts:
        sentences.append(f"{ticker}: " + ", ".join(context_parts) + ".")

    if not sentences:
        return f"{ticker} no tiene suficiente información calculada todavía para una tesis."
    return " ".join(sentences)
