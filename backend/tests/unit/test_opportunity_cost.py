"""Reconstruction (2026-09), Fase 5 (resto): find_opportunity_cost_notes -
pure comparison, no DB/network, sector lookups monkeypatched so this suite
never depends on the real curated universe's exact membership."""

from datetime import UTC, date, datetime

import app.services.opportunity_cost as oc
from app.domain.models.ticker_daily_state import TickerDailyState


def _state(ticker: str, gate_passes: bool, rs_rating: int | None = None) -> TickerDailyState:
    return TickerDailyState(
        id=None,
        region="us",
        ticker=ticker,
        trade_date=date.today(),
        computed_at=datetime.now(UTC),
        price=100.0,
        currency="USD",
        trend="uptrend" if gate_passes else "sideways",
        stage="stage2" if gate_passes else None,
        rs_rating=rs_rating,
        adx14=None,
        atr_multiple=None,
        rsi14=None,
        gate_passes=gate_passes,
        gate_conditions=[],
        gate_version="2026-09-levels-v1",
        entry_trigger_type=None,
        entry_trigger_price=None,
        entry_already_triggered=False,
        stop_loss=None,
        take_profit=None,
        take_profit_method=None,
        risk_reward=None,
    )


_SECTORS = {"AAPL": "Tecnología", "MSFT": "Tecnología", "GOOGL": "Tecnología", "JPM": "Financiero"}


def _patch_sectors(monkeypatch) -> None:
    monkeypatch.setattr(oc, "sector_of", lambda ticker: _SECTORS.get(ticker))


def test_no_notes_when_the_holding_already_passes_its_own_gate(monkeypatch):
    _patch_sectors(monkeypatch)
    held = [_state("AAPL", gate_passes=True)]
    radar = [_state("MSFT", gate_passes=True, rs_rating=95)]

    assert oc.find_opportunity_cost_notes(held, radar) == []


def test_flags_a_same_sector_alternative_whose_gate_passes(monkeypatch):
    _patch_sectors(monkeypatch)
    held = [_state("AAPL", gate_passes=False)]
    radar = [_state("MSFT", gate_passes=True, rs_rating=95)]

    notes = oc.find_opportunity_cost_notes(held, radar)

    assert len(notes) == 1
    assert notes[0].held_ticker == "AAPL"
    assert notes[0].alternative_ticker == "MSFT"
    assert notes[0].alternative_rs_rating == 95
    assert notes[0].sector == "Tecnología"


def test_ignores_a_different_sector_alternative(monkeypatch):
    _patch_sectors(monkeypatch)
    held = [_state("AAPL", gate_passes=False)]
    radar = [_state("JPM", gate_passes=True, rs_rating=95)]

    assert oc.find_opportunity_cost_notes(held, radar) == []


def test_ignores_a_same_sector_candidate_whose_own_gate_does_not_pass(monkeypatch):
    _patch_sectors(monkeypatch)
    held = [_state("AAPL", gate_passes=False)]
    radar = [_state("MSFT", gate_passes=False, rs_rating=95)]

    assert oc.find_opportunity_cost_notes(held, radar) == []


def test_never_flags_the_held_ticker_as_its_own_alternative(monkeypatch):
    _patch_sectors(monkeypatch)
    held = [_state("AAPL", gate_passes=False)]
    radar = [_state("AAPL", gate_passes=True, rs_rating=95)]

    assert oc.find_opportunity_cost_notes(held, radar) == []


def test_skips_a_holding_with_no_curated_sector(monkeypatch):
    _patch_sectors(monkeypatch)
    held = [_state("UNKNOWN_TICKER", gate_passes=False)]
    radar = [_state("MSFT", gate_passes=True, rs_rating=95)]

    assert oc.find_opportunity_cost_notes(held, radar) == []


def test_sorts_multiple_alternatives_by_rs_rating_descending_missing_last(monkeypatch):
    _patch_sectors(monkeypatch)
    held = [_state("AAPL", gate_passes=False)]
    radar = [
        _state("MSFT", gate_passes=True, rs_rating=60),
        _state("GOOGL", gate_passes=True, rs_rating=None),
        _state("JPM", gate_passes=True, rs_rating=99),  # different sector - excluded regardless of rank
    ]

    notes = oc.find_opportunity_cost_notes(held, radar)

    assert [n.alternative_ticker for n in notes] == ["MSFT", "GOOGL"]


def test_multiple_holdings_each_get_their_own_notes(monkeypatch):
    monkeypatch.setattr(
        oc,
        "sector_of",
        lambda ticker: {
            "AAPL": "Tecnología", "MSFT": "Tecnología", "JPM": "Financiero", "BAC": "Financiero",
        }.get(ticker),
    )
    held = [_state("AAPL", gate_passes=False), _state("JPM", gate_passes=False)]
    radar = [
        _state("MSFT", gate_passes=True, rs_rating=90),
        _state("BAC", gate_passes=True, rs_rating=70),
    ]

    notes = oc.find_opportunity_cost_notes(held, radar)

    by_held = {n.held_ticker: n.alternative_ticker for n in notes}
    assert by_held == {"AAPL": "MSFT", "JPM": "BAC"}
