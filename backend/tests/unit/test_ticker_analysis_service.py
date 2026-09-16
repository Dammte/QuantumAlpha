import numpy as np
import pandas as pd

from app.domain.interfaces.llm_narrator import LLMNarrator
from app.services import ticker_analysis_service as tas
from app.services import trade_geometry as tg
from app.services.levels_engine import Eligibility, GateCondition, GateResult

_SeriesQuintet = tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]


def _series(close: np.ndarray, wiggle: float = 1.0) -> _SeriesQuintet:
    # A real DatetimeIndex, ending well before "today" - compute_core_signals
    # now derives weekly bars (multi_timeframe.analyze_multi_timeframe) off
    # this index via technical_analysis.closed_bars/resample_ohlcv, which
    # both need real dates, not a bare positional RangeIndex. Fixed in the
    # past (not ending "today") keeps these tests' bars all settled, same as
    # the plain int index they used before (is_intraday_snapshot was always
    # False for that shape of index too).
    index = pd.bdate_range(end=pd.Timestamp("2024-01-01"), periods=len(close))
    close_s = pd.Series(close, index=index)
    return close_s, close_s + wiggle, close_s - wiggle, pd.Series([1_000_000.0] * len(close), index=index), close_s


def test_none_when_not_enough_bars():
    close, high, low, volume, open_ = _series(np.array([100.0] * 10))
    assert tas.compute_core_signals(close, high, low, volume, open_, None, rs_rating=None) is None


def test_returns_a_fully_populated_result_for_an_uptrend():
    close, high, low, volume, open_ = _series(100 + np.arange(260) * 0.4)
    signals = tas.compute_core_signals(close, high, low, volume, open_, None, rs_rating=85)

    assert signals is not None
    assert signals.trend == tas.ta.TrendState.UPTREND
    assert signals.rs_rating == 85
    assert signals.gate is not None


def test_rs_rating_passed_through_unchanged():
    close, high, low, volume, open_ = _series(100 + np.arange(260) * 0.4)
    signals = tas.compute_core_signals(close, high, low, volume, open_, None, rs_rating=42)
    assert signals.rs_rating == 42


def test_mansfield_rs_is_none_without_a_benchmark():
    close, high, low, volume, open_ = _series(100 + np.arange(260) * 0.4)
    signals = tas.compute_core_signals(close, high, low, volume, open_, None, rs_rating=None)
    assert signals.mansfield_rs is None


def test_mansfield_rs_is_computed_when_a_benchmark_is_given():
    close, high, low, volume, open_ = _series(100 + np.arange(260) * 0.4)
    benchmark = pd.Series(100 + np.arange(260) * 0.1, index=close.index)  # mansfield_rs inner-joins by index
    signals = tas.compute_core_signals(close, high, low, volume, open_, benchmark, rs_rating=None)
    assert signals.mansfield_rs is not None


def test_multi_timeframe_is_always_populated():
    # Segunda auditoría, Bloque 2: before this, ticker_analysis_service.py
    # never referenced analyze_multi_timeframe/closed_bars at all - every
    # signal was read off the live/possibly-still-forming last bar with no
    # weekly/confirmed counterpart exposed anywhere.
    close, high, low, volume, open_ = _series(100 + np.arange(260) * 0.4)
    signals = tas.compute_core_signals(close, high, low, volume, open_, None, rs_rating=None)
    assert signals is not None
    assert signals.multi_timeframe is not None
    assert signals.multi_timeframe.daily is not None


def test_confirmed_gate_is_none_when_the_last_bar_is_already_settled():
    # These fixtures end well in the past (see _series) - is_intraday_snapshot
    # is False, so there is nothing to separate the live read from.
    close, high, low, volume, open_ = _series(100 + np.arange(260) * 0.4)
    signals = tas.compute_core_signals(close, high, low, volume, open_, None, rs_rating=None)
    assert signals is not None
    assert signals.is_intraday_snapshot is False
    assert signals.confirmed_gate is None


def test_confirmed_gate_is_populated_when_the_last_bar_is_still_forming():
    # A series whose last bar is dated *today* - the live read is real but not
    # repaint-proof, so confirmed_gate must be filled in from
    # technical_analysis.closed_bars instead of silently staying None.
    n = 260
    # Calendar days (not bdate_range) so the last bar lands on "today"
    # regardless of which weekday the test happens to run on.
    index = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq="D")
    close_s = pd.Series(100 + np.arange(n) * 0.4, index=index)
    high, low = close_s + 1.0, close_s - 1.0
    volume = pd.Series([1_000_000.0] * n, index=index)
    signals = tas.compute_core_signals(close_s, high, low, volume, close_s, None, rs_rating=None)
    assert signals is not None
    assert signals.is_intraday_snapshot is True
    assert signals.confirmed_gate is not None


def test_triple_barrier_backtest_is_populated_with_enough_history():
    # Segunda auditoría, Bloque 2: run_triple_barrier_backtest must actually
    # have a caller - before this, it existed, was tested in isolation, and
    # was never wired into "Analizar activo" at all. Needs more history than
    # the other fixtures here (WARMUP_BARS=260 plus the horizon) to produce a
    # real (non-None) result, at the fixed 21-day horizon (never the Monte
    # Carlo preset - see TRIPLE_BARRIER_HORIZON_DAYS).
    close, high, low, volume, open_ = _series(100 + np.arange(400) * 0.4)
    signals = tas.compute_core_signals(
        close, high, low, volume, open_, None, rs_rating=None, include_triple_barrier_backtest=True
    )
    assert signals is not None
    assert signals.triple_barrier_backtest is not None
    assert signals.triple_barrier_backtest.horizon_days == tas.TRIPLE_BARRIER_HORIZON_DAYS


def test_triple_barrier_backtest_is_skipped_by_default():
    # Measured at ~3x this function's own cost - portfolio_risk_service.py
    # (which runs this per held position on every cache refresh) must not
    # pay for a field that view doesn't show. Only TickerAnalysisService.
    # analyze() opts in.
    close, high, low, volume, open_ = _series(100 + np.arange(400) * 0.4)
    signals = tas.compute_core_signals(close, high, low, volume, open_, None, rs_rating=None)
    assert signals is not None
    assert signals.triple_barrier_backtest is None


def test_52_week_range_fields_never_fabricated_with_only_251_bars():
    # D11: rolling_extreme_price/distance_to_rolling_extreme require the full
    # 252-bar window by default - a ticker with 251 bars (one short of that,
    # but comfortably past MIN_BARS_REQUIRED=250 since Parte 3.2 - the rest
    # of the analysis genuinely runs) must not have its "52-week range"
    # answered with whatever 251 days happen to be available (CLAUDE.md: "no
    # inventes datos"). The old checklist's own range-confirmed factor that
    # used to be asserted here was retired with the rest of the weighted
    # checklist (2026-09, Fase 4) - the gate has no 52-week-range condition
    # to fire in the first place, so there's nothing left to assert about it
    # beyond these two fields themselves staying honestly None.
    close, high, low, volume, open_ = _series(100 + np.arange(251) * 0.4)
    signals = tas.compute_core_signals(close, high, low, volume, open_, None, rs_rating=None)
    assert signals is not None
    assert signals.dist_52w_high is None
    assert signals.dist_52w_low is None


# --- Fase 7: _explain_gate wiring (LLMNarrator) ---------------------------


class _FakeNarrator(LLMNarrator):
    """Records exactly what TickerAnalysisService hands it - never talks to
    a real LLM. See LLMNarrator's own docstring for why this port is
    primitive-typed rather than taking a GateResult directly."""

    def __init__(self, response: str | None = "una explicación") -> None:
        self.response = response
        self.calls: list[dict] = []

    def explain_gate(self, **kwargs) -> str | None:
        self.calls.append(kwargs)
        return self.response


def _gate(
    passes: bool = True,
    entry_trigger: tg.EntryTrigger | None = None,
    stop_and_target: tg.StopAndTarget | None = None,
) -> GateResult:
    return GateResult(
        passes=passes,
        conditions=[GateCondition(label="Semanal no en Fase 4 de Weinstein", passed=passes)],
        entry_trigger=entry_trigger,
        stop_and_target=stop_and_target,
        eligibility=Eligibility(
            liquidity_ok=True,
            data_quality_ok=True,
            weekly_not_stage4=passes,
            no_fast_bearish_cross=True,
            no_event_risk=True,
        ),
    )


def test_explain_gate_is_none_without_a_narrator_configured():
    service = tas.TickerAnalysisService(market_data=None, screener=None, narrator=None)
    result = service._explain_gate("AAPL", _gate(), tas.ta.TrendState.UPTREND, tas.ta.Stage.STAGE_2)
    assert result is None


def test_explain_gate_passes_primitive_facts_through_to_the_narrator():
    narrator = _FakeNarrator()
    service = tas.TickerAnalysisService(market_data=None, screener=None, narrator=narrator)
    gate = _gate(
        passes=True,
        entry_trigger=tg.EntryTrigger(trigger_type="breakout", trigger_price=125.4, already_triggered=True),
        stop_and_target=tg.StopAndTarget(
            stop_loss=110.0, take_profit=140.0, take_profit_method="objetivo 2:1 sobre el riesgo", risk_reward=2.0
        ),
    )

    result = service._explain_gate("AAPL", gate, tas.ta.TrendState.UPTREND, tas.ta.Stage.STAGE_2)

    assert result == "una explicación"
    assert len(narrator.calls) == 1
    call = narrator.calls[0]
    assert call["ticker"] == "AAPL"
    assert call["gate_passes"] is True
    assert call["conditions"] == [("Semanal no en Fase 4 de Weinstein", True)]
    assert call["trend_label"] == "uptrend"
    assert call["stage_label"] == "stage2"
    assert call["entry_trigger_summary"] == "ruptura en 125.40 (ya disparado)"
    assert call["stop_and_target_summary"] == (
        "stop en 110.00, objetivo en 140.00 (objetivo 2:1 sobre el riesgo), relación beneficio:riesgo 2.0:1"
    )


def test_explain_gate_handles_no_trigger_and_no_stop_target():
    narrator = _FakeNarrator()
    service = tas.TickerAnalysisService(market_data=None, screener=None, narrator=narrator)

    service._explain_gate("AAPL", _gate(passes=False), tas.ta.TrendState.SIDEWAYS, None)

    call = narrator.calls[0]
    assert call["stage_label"] is None
    assert call["entry_trigger_summary"] is None
    assert call["stop_and_target_summary"] is None


def test_explain_gate_returns_none_when_the_narrator_itself_returns_none():
    narrator = _FakeNarrator(response=None)
    service = tas.TickerAnalysisService(market_data=None, screener=None, narrator=narrator)
    result = service._explain_gate("AAPL", _gate(), tas.ta.TrendState.UPTREND, tas.ta.Stage.STAGE_2)
    assert result is None
