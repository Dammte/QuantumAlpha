"""Auditoria del Radar, bloque 9: suggest_rotations - pura comparación, sin
DB/red, sector lookups monkeypatched igual que test_opportunity_cost.py, para
que esta suite nunca dependa de la composición real del universo curado."""

from datetime import UTC, date, datetime

from app.domain.models.ticker_daily_state import TickerDailyState
from app.services import portfolio_rotation_service as prs


def _state(
    ticker: str,
    gate_passes: bool = True,
    grade: str | None = "A",
    rs_rating: int | None = 90,
) -> TickerDailyState:
    return TickerDailyState(
        id=None,
        region="us",
        ticker=ticker,
        trade_date=date(2026, 9, 19),
        computed_at=datetime(2026, 9, 19, tzinfo=UTC),
        price=100.0,
        currency="USD",
        trend="uptrend",
        stage="stage2",
        rs_rating=rs_rating,
        adx14=None,
        atr_multiple=None,
        rsi14=None,
        gate_passes=gate_passes,
        gate_conditions=[],
        gate_version="v1",
        entry_trigger_type=None,
        entry_trigger_price=None,
        entry_already_triggered=False,
        stop_loss=None,
        take_profit=None,
        take_profit_method=None,
        risk_reward=None,
        grade={"grade": grade, "reasons": [], "distance_atr": None} if grade is not None else None,
    )


_SECTORS = {
    "WEAKCO": "Tecnología", "STRONGCO": "Tecnología", "TECHALT": "Tecnología",
    "BCO": "Tecnología", "ACO_LOW_RS": "Tecnología", "ACO_HIGH_RS": "Tecnología",
    "ALREADYHELD": "Tecnología", "ONLY_ALT": "Tecnología", "WEAK1": "Tecnología", "WEAK2": "Tecnología",
    "REDUCECO": "Salud", "HEALTHALT": "Salud",
}
_SECTORS.update({f"CAPWEAK{i}": f"Sector{i}" for i in range(5)})
_SECTORS.update({f"CAPALT{i}": f"Sector{i}" for i in range(5)})


def _patch_sectors(monkeypatch) -> None:
    monkeypatch.setattr(prs, "sector_of", lambda ticker: _SECTORS.get(ticker))


def test_no_suggestions_when_the_portfolio_has_room(monkeypatch):
    _patch_sectors(monkeypatch)
    held = [_state("WEAKCO")]
    candidates = [_state("STRONGCO")]
    result = prs.suggest_rotations(
        held, {"WEAKCO": "exit_now"}, candidates, open_positions_count=5, max_open_positions=10
    )
    assert result == []


def test_no_suggestions_when_nothing_held_is_weak(monkeypatch):
    _patch_sectors(monkeypatch)
    held = [_state("HEALTHYCO")]
    candidates = [_state("STRONGCO")]
    result = prs.suggest_rotations(
        held, {"HEALTHYCO": "hold"}, candidates, open_positions_count=10, max_open_positions=10
    )
    assert result == []


def test_a_ticker_missing_from_urgency_map_is_never_assumed_weak(monkeypatch):
    # No se evaluó todavía por el motor de salida - nunca elegible por defecto.
    _patch_sectors(monkeypatch)
    held = [_state("UNKNOWNCO")]
    candidates = [_state("STRONGCO")]
    result = prs.suggest_rotations(held, {}, candidates, open_positions_count=10, max_open_positions=10)
    assert result == []


def test_suggests_a_swap_for_an_exit_now_position_with_a_same_sector_alternative(monkeypatch):
    _patch_sectors(monkeypatch)
    held = [_state("WEAKCO")]
    candidates = [_state("STRONGCO", grade="A", rs_rating=95)]
    result = prs.suggest_rotations(
        held, {"WEAKCO": "exit_now"}, candidates, open_positions_count=10, max_open_positions=10
    )
    assert len(result) == 1
    suggestion = result[0]
    assert suggestion.sell_ticker == "WEAKCO"
    assert suggestion.buy_ticker == "STRONGCO"
    assert "salida inmediata" in suggestion.sell_reason
    assert "grado A" in suggestion.buy_reason
    assert suggestion.sector == "Tecnología"


def test_never_suggests_a_candidate_from_a_different_sector(monkeypatch):
    _patch_sectors(monkeypatch)
    held = [_state("REDUCECO")]  # Salud
    candidates = [_state("STRONGCO", grade="A")]  # Tecnología
    result = prs.suggest_rotations(
        held, {"REDUCECO": "exit_now"}, candidates, open_positions_count=10, max_open_positions=10
    )
    assert result == []


def test_never_suggests_a_candidate_whose_gate_does_not_pass(monkeypatch):
    _patch_sectors(monkeypatch)
    held = [_state("WEAKCO")]
    candidates = [_state("STRONGCO", gate_passes=False, grade="A")]
    result = prs.suggest_rotations(
        held, {"WEAKCO": "exit_now"}, candidates, open_positions_count=10, max_open_positions=10
    )
    assert result == []


def test_never_suggests_a_candidate_with_no_grade(monkeypatch):
    _patch_sectors(monkeypatch)
    held = [_state("WEAKCO")]
    candidates = [_state("STRONGCO", grade=None)]
    result = prs.suggest_rotations(
        held, {"WEAKCO": "exit_now"}, candidates, open_positions_count=10, max_open_positions=10
    )
    assert result == []


def test_never_suggests_a_ticker_already_held(monkeypatch):
    _patch_sectors(monkeypatch)
    held = [_state("WEAKCO"), _state("ALREADYHELD", grade="A")]
    candidates = [_state("ALREADYHELD", grade="A")]
    result = prs.suggest_rotations(
        held, {"WEAKCO": "exit_now"}, candidates, open_positions_count=10, max_open_positions=10
    )
    assert result == []


def test_picks_the_best_grade_then_the_highest_rs_rating_as_tiebreak(monkeypatch):
    _patch_sectors(monkeypatch)
    held = [_state("WEAKCO")]
    candidates = [
        _state("BCO", grade="B", rs_rating=99),
        _state("ACO_LOW_RS", grade="A", rs_rating=60),
        _state("ACO_HIGH_RS", grade="A", rs_rating=95),
    ]
    result = prs.suggest_rotations(
        held, {"WEAKCO": "exit_now"}, candidates, open_positions_count=10, max_open_positions=10
    )
    assert len(result) == 1
    assert result[0].buy_ticker == "ACO_HIGH_RS"


def test_exit_now_positions_are_prioritized_over_reduce_positions(monkeypatch):
    _patch_sectors(monkeypatch)
    held = [_state("REDUCECO"), _state("WEAKCO")]  # WEAKCO plays "exit_now" here
    candidates = [
        _state("HEALTHALT", grade="A"),  # Salud, matches REDUCECO
        _state("STRONGCO", grade="A"),  # Tecnología, matches WEAKCO
    ]
    result = prs.suggest_rotations(
        held,
        {"REDUCECO": "reduce", "WEAKCO": "exit_now"},
        candidates,
        open_positions_count=10,
        max_open_positions=10,
        max_suggestions=1,
    )
    assert len(result) == 1
    assert result[0].sell_ticker == "WEAKCO"


def test_hard_cap_on_suggestions_per_call(monkeypatch):
    _patch_sectors(monkeypatch)
    held = [_state(f"CAPWEAK{i}") for i in range(5)]
    candidates = [_state(f"CAPALT{i}", grade="A") for i in range(5)]
    urgency = {f"CAPWEAK{i}": "exit_now" for i in range(5)}
    result = prs.suggest_rotations(
        held, urgency, candidates, open_positions_count=10, max_open_positions=10, max_suggestions=2
    )
    assert len(result) == 2


def test_never_reuses_the_same_candidate_for_two_suggestions(monkeypatch):
    _patch_sectors(monkeypatch)
    held = [_state("WEAK1"), _state("WEAK2")]
    candidates = [_state("ONLY_ALT", grade="A")]
    result = prs.suggest_rotations(
        held,
        {"WEAK1": "exit_now", "WEAK2": "exit_now"},
        candidates,
        open_positions_count=10,
        max_open_positions=10,
    )
    assert len(result) == 1


def test_a_held_ticker_with_no_sector_mapping_is_skipped(monkeypatch):
    monkeypatch.setattr(prs, "sector_of", lambda ticker: None)
    held = [_state("WEAKCO")]
    candidates = [_state("STRONGCO", grade="A")]
    result = prs.suggest_rotations(
        held, {"WEAKCO": "exit_now"}, candidates, open_positions_count=10, max_open_positions=10
    )
    assert result == []
