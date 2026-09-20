"""Reconstruction (2026-09), Fase 7: concrete `LLMNarrator` adapter over
Google's Gemini API - see `app.domain.interfaces.llm_narrator` for the full
"why a port, why it can never score or decide" reasoning this class exists
under.

**Cableado, no activado** (mismo criterio que los Cron Jobs de la Fase 2 en
`render.yaml`): `settings.gemini_api_key` es `None` por defecto
(`.env.example` la deja vacía a propósito) - sin clave, `explain_gate`
devuelve `None` de inmediato, sin ningún intento de red, y ningún dato del
propietario (tickers, gate, cartera) sale nunca de esta app hacia Google.
Activar esta capa - poner una clave real en `.env` - es una decisión de
coste y privacidad del propietario, no algo que este reconstructor decida
por su cuenta (ver la conversación que sustenta esta decisión).

Cualquier fallo de la llamada en sí (clave inválida, límite de tasa,
timeout, una respuesta vacía o mal formada) también se traga en `None` -
igual que `MarketDataProvider.get_latest_quote` documenta para un fallo de
cotización: una narrativa es contexto opcional sobre una decisión que el
gate ya tomó, nunca una dependencia que pueda bloquear o tumbar la lectura
principal de "Analizar activo"."""

import json
import logging

from google import genai

from app.domain.interfaces.llm_narrator import LLMNarrator

logger = logging.getLogger(__name__)

# Modelo rápido/económico a propósito - esta capa solo redacta una
# explicación de hechos ya calculados, no necesita razonamiento de frontera.
_MODEL = "gemini-2.5-flash"

# Auditoria del Radar, bloque E4/12, literal: "gemini-2.5-flash-lite, salida
# JSON estructurada" - un modelo distinto (y más ligero) del de
# `explain_gate` de arriba, elegido explícitamente para esta tarea concreta
# (redactar 2-3 frases sobre un puñado de hechos ya calculados, la tarea más
# barata de las dos que este archivo cubre).
_RADAR_PRIMARY_MODEL = "gemini-2.5-flash-lite"

_SYSTEM_INSTRUCTION = (
    "Eres un asistente que redacta, en español y en 2-4 frases, una explicación en lenguaje llano de por "
    "qué un gate de entrada técnico aprobó o no aprobó, y qué significan en la práctica su disparador de "
    "entrada y su stop/objetivo - a partir exclusivamente de los hechos que se te dan. No inventes ningún "
    "dato, indicador o cifra que no se te haya proporcionado. No emitas una recomendación de compra/venta "
    "propia, ni un veredicto distinto del que ya se te da (el gate ya está decidido, tu trabajo es "
    "explicarlo, no volver a juzgarlo). No es asesoramiento financiero."
)

_RADAR_PRIMARY_SYSTEM_INSTRUCTION = (
    "Eres un asistente que redacta, en español y en 2-3 frases, la tesis de por qué un candidato del "
    "Radar de trading ya marcado como principal a entrar resulta interesante hoy - a partir "
    "EXCLUSIVAMENTE de los hechos ya calculados que se te dan (narrativa del setup, sector, fuerza "
    "relativa, entrada, stop y su anclaje, objetivo, beneficio:riesgo neto, score). No inventes ningún "
    "dato, indicador o cifra que no se te haya proporcionado. No analices el valor por tu cuenta, no "
    "emitas una recomendación de compra/venta propia, ni cambies que este candidato es el principal - "
    "esa decisión ya se tomó por su score antes de que tú intervengas; tu único trabajo es redactarla en "
    "prosa clara, no volver a juzgarla. No es asesoramiento financiero. Responde únicamente con el JSON "
    "pedido, sin texto adicional."
)

_RADAR_PRIMARY_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"thesis": {"type": "string"}},
    "required": ["thesis"],
}


class GeminiNarrator(LLMNarrator):
    def __init__(self, api_key: str | None) -> None:
        self._client = genai.Client(api_key=api_key) if api_key else None

    def explain_gate(
        self,
        ticker: str,
        gate_passes: bool,
        conditions: list[tuple[str, bool]],
        trend_label: str,
        stage_label: str | None,
        entry_trigger_summary: str | None,
        stop_and_target_summary: str | None,
    ) -> str | None:
        if self._client is None:
            return None

        conditions_text = "\n".join(
            f"- {label}: {'cumple' if passed else 'no cumple'}" for label, passed in conditions
        )
        prompt = (
            f"Ticker: {ticker}\n"
            f"Tendencia: {trend_label}\n"
            f"Fase de Weinstein: {stage_label or 'sin datos suficientes'}\n"
            f"Gate: {'APROBADO' if gate_passes else 'NO APROBADO'}\n"
            f"Condiciones evaluadas:\n{conditions_text}\n"
            f"Disparador de entrada: {entry_trigger_summary or 'ninguno activo ahora mismo'}\n"
            f"Stop y objetivo: {stop_and_target_summary or 'no calculados'}\n"
        )
        try:
            response = self._client.models.generate_content(
                model=_MODEL,
                contents=prompt,
                config={"system_instruction": _SYSTEM_INSTRUCTION},
            )
            text = response.text
            return text.strip() if text else None
        except Exception:
            # Never lets a Gemini-side failure (rate limit, timeout, invalid
            # key, malformed response) reach the caller - see this module's
            # own docstring.
            logger.warning("GeminiNarrator.explain_gate failed for %s", ticker, exc_info=True)
            return None

    def explain_radar_primary(
        self,
        ticker: str,
        setup_narrative: str | None,
        sector: str | None,
        rs_rating: int | None,
        entry_price: float | None,
        stop_price: float | None,
        stop_basis: str | None,
        target_price: float | None,
        risk_reward_net: float | None,
        score_total: float,
    ) -> str | None:
        if self._client is None:
            return None

        stop_text = "no calculado"
        if stop_price is not None:
            stop_text = f"{stop_price:.2f}" + (f" ({stop_basis})" if stop_basis else "")
        prompt = (
            f"Ticker: {ticker}\n"
            f"Setup: {setup_narrative or 'sin narrativa de setup específica'}\n"
            f"Sector: {sector or 'sin sector conocido'}\n"
            f"RS Rating: {rs_rating if rs_rating is not None else 'sin dato'}\n"
            f"Entrada: {entry_price if entry_price is not None else 'no calculada'}\n"
            f"Stop: {stop_text}\n"
            f"Objetivo: {target_price if target_price is not None else 'no calculado'}\n"
            f"Beneficio:riesgo neto: {risk_reward_net if risk_reward_net is not None else 'sin dato'}\n"
            f"Score compuesto: {score_total:.0f}/100\n"
        )
        try:
            response = self._client.models.generate_content(
                model=_RADAR_PRIMARY_MODEL,
                contents=prompt,
                config={
                    "system_instruction": _RADAR_PRIMARY_SYSTEM_INSTRUCTION,
                    "response_mime_type": "application/json",
                    "response_schema": _RADAR_PRIMARY_RESPONSE_SCHEMA,
                },
            )
            text = response.text
            if not text:
                return None
            thesis = json.loads(text).get("thesis")
            return thesis.strip() if thesis else None
        except Exception:
            # Same "never lets an LLM-side failure reach the caller" rule as
            # explain_gate above - the caller (market.py) always has the
            # deterministic thesis (setups/thesis.py) ready to use instead.
            logger.warning("GeminiNarrator.explain_radar_primary failed for %s", ticker, exc_info=True)
            return None
