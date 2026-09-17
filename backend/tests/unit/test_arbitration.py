"""Parte 0/6 (biblioteca de setups del Radar, quant_methodology.md §28):
"ningún setup suma puntos a otro" - el único módulo que compara/elige entre
varios `SetupMatch` de un mismo ticker."""

from app.services.setups import arbitration
from app.services.setups.types import SetupConfidence, SetupFamily, SetupMatch, SetupStage


def _match(name: str, stage: SetupStage, family: SetupFamily = SetupFamily.STAGE_TRANSITION) -> SetupMatch:
    return SetupMatch(
        family=family, name=name, label_es=name, stage=stage, bars_in_stage=1, timeframe="daily",
        trigger_price=None, trigger_condition="", invalidation_price=None, invalidation_condition="",
        evidence={}, narrative_es="", confidence=SetupConfidence.UNVALIDATED,
    )


def test_select_best_returns_none_for_an_empty_list():
    assert arbitration.select_best([]) is None


def test_select_best_returns_the_only_match():
    match = _match("a", SetupStage.FORMING)
    assert arbitration.select_best([match]) is match


def test_select_best_prefers_triggered_over_ready_over_forming():
    forming = _match("forming", SetupStage.FORMING)
    ready = _match("ready", SetupStage.READY)
    triggered = _match("triggered", SetupStage.TRIGGERED)

    assert arbitration.select_best([forming, ready, triggered]) is triggered
    assert arbitration.select_best([triggered, forming, ready]) is triggered
    assert arbitration.select_best([ready, forming]) is ready


def test_select_best_prefers_any_non_failed_stage_over_failed():
    failed = _match("failed", SetupStage.FAILED)
    forming = _match("forming", SetupStage.FORMING)

    assert arbitration.select_best([failed, forming]) is forming


def test_select_best_breaks_ties_by_the_order_matches_arrived_in():
    # Sin un criterio explícito del encargo para comparar familias distintas
    # en la misma etapa, el desempate es "el primero que llegó" - estable y
    # documentado, no un orden no determinista.
    first = _match("primero", SetupStage.READY, family=SetupFamily.MA_CROSS)
    second = _match("segundo", SetupStage.READY, family=SetupFamily.BREAKOUT)

    assert arbitration.select_best([first, second]) is first
    assert arbitration.select_best([second, first]) is second


def test_order_by_rank_puts_the_best_match_first_without_dropping_the_rest():
    forming = _match("forming", SetupStage.FORMING)
    triggered = _match("triggered", SetupStage.TRIGGERED)
    ready = _match("ready", SetupStage.READY)

    ordered = arbitration.order_by_rank([forming, ready, triggered])

    assert ordered[0] is triggered
    assert {m.name for m in ordered} == {"forming", "ready", "triggered"}
    assert len(ordered) == 3


def test_order_by_rank_returns_a_single_match_unchanged():
    match = _match("a", SetupStage.READY)
    assert arbitration.order_by_rank([match]) == [match]


def test_order_by_rank_returns_empty_list_unchanged():
    assert arbitration.order_by_rank([]) == []
