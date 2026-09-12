"""Reconstruction (2026-09), Fase 7: GeminiNarrator - see its own docstring
for the "cableado, no activado" reasoning. These tests never make a real
network call: the no-key path returns before touching the SDK at all, and
the configured-path tests monkeypatch `self._client.models.generate_content`
directly rather than hitting Google's API."""

from app.infrastructure.llm.gemini_narrator import GeminiNarrator

_CONDITIONS = [("Tendencia alcista o Fase 2 de Weinstein", True), ("Sin extensión parabólica", False)]


def test_explain_gate_returns_none_without_an_api_key():
    narrator = GeminiNarrator(api_key=None)

    result = narrator.explain_gate(
        ticker="AAPL",
        gate_passes=False,
        conditions=_CONDITIONS,
        trend_label="uptrend",
        stage_label="stage2",
        entry_trigger_summary=None,
        stop_and_target_summary=None,
    )

    assert result is None
    assert narrator._client is None  # never even constructs a genai.Client


class _FakeResponse:
    def __init__(self, text: str | None) -> None:
        self.text = text


class _FakeModels:
    def __init__(self, response: _FakeResponse | None = None, error: Exception | None = None) -> None:
        self._response = response
        self._error = error
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._response


class _FakeClient:
    def __init__(self, models: _FakeModels) -> None:
        self.models = models


def test_explain_gate_returns_stripped_text_on_success():
    narrator = GeminiNarrator(api_key="fake-key-for-test")
    fake_models = _FakeModels(response=_FakeResponse("  Explicación en lenguaje llano.  "))
    narrator._client = _FakeClient(fake_models)

    result = narrator.explain_gate(
        ticker="AAPL",
        gate_passes=True,
        conditions=_CONDITIONS,
        trend_label="uptrend",
        stage_label="stage2",
        entry_trigger_summary="ruptura en 125.40 (ya disparado)",
        stop_and_target_summary="stop en 110.00, objetivo en 140.00",
    )

    assert result == "Explicación en lenguaje llano."
    assert len(fake_models.calls) == 1
    assert fake_models.calls[0]["model"]
    assert "AAPL" in fake_models.calls[0]["contents"]


def test_explain_gate_returns_none_when_the_response_has_no_text():
    narrator = GeminiNarrator(api_key="fake-key-for-test")
    narrator._client = _FakeClient(_FakeModels(response=_FakeResponse(None)))

    result = narrator.explain_gate(
        ticker="AAPL", gate_passes=True, conditions=_CONDITIONS, trend_label="uptrend", stage_label=None,
        entry_trigger_summary=None, stop_and_target_summary=None,
    )

    assert result is None


def test_explain_gate_swallows_any_api_error_into_none():
    narrator = GeminiNarrator(api_key="fake-key-for-test")
    narrator._client = _FakeClient(_FakeModels(error=RuntimeError("rate limited")))

    result = narrator.explain_gate(
        ticker="AAPL", gate_passes=True, conditions=_CONDITIONS, trend_label="uptrend", stage_label=None,
        entry_trigger_summary=None, stop_and_target_summary=None,
    )

    assert result is None
