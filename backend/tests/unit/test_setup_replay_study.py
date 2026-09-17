"""Parte 10 (biblioteca de setups del Radar, quant_methodology.md §28):
`scripts/setup_replay_study.py`. Mismo criterio que `test_factor_ablation_study.py`
- se prueba la pieza pura (`_stats_to_rows`), no la orquestación de nivel
superior que solo pega piezas ya probadas por separado
(`replay_setups_for_ticker`/`aggregate_setup_performance` en
`test_setup_replay.py`, `resolve_universe_tickers` en
`test_factor_ablation_study.py`) y que además descarga datos reales."""

import scripts.setup_replay_study as srs
from app.services.setup_replay import SetupPerformanceStats
from app.services.setups.types import SetupConfidence


def test_stats_to_rows_maps_every_field():
    stats = [
        SetupPerformanceStats(
            setup_name="vcp_3_contracciones", family="vcp", grade="A", market_regime=None,
            n_observations=35, trigger_rate=0.6, win_rate=0.55, expectancy_r=0.42,
            median_bars_held=6.0, mae_p80_pct=-0.03, failure_rate_3d=0.1,
            confidence=SetupConfidence.MEASURED,
        )
    ]
    rows = srs._stats_to_rows(stats)

    assert len(rows) == 1
    row = rows[0]
    assert row.id is None
    assert row.setup_name == "vcp_3_contracciones"
    assert row.family == "vcp"
    assert row.grade == "A"
    assert row.market_regime is None
    assert row.n_observations == 35
    assert row.trigger_rate == 0.6
    assert row.win_rate == 0.55
    assert row.expectancy_r == 0.42
    assert row.median_bars_held == 6.0
    assert row.mae_p80_pct == -0.03
    assert row.failure_rate_3d == 0.1
    assert row.confidence == "measured"  # .value, no el enum - así se persiste
    assert row.computed_at is not None


def test_stats_to_rows_preserves_none_fields_for_a_never_triggered_setup():
    stats = [
        SetupPerformanceStats(
            setup_name="caja_de_darvas", family="breakout", grade=None, market_regime=None,
            n_observations=5, trigger_rate=0.0, win_rate=None, expectancy_r=None,
            median_bars_held=None, mae_p80_pct=None, failure_rate_3d=None,
            confidence=SetupConfidence.THIN,
        )
    ]
    rows = srs._stats_to_rows(stats)

    assert rows[0].win_rate is None
    assert rows[0].expectancy_r is None
    assert rows[0].confidence == "thin"


def test_stats_to_rows_empty_input_returns_empty_list():
    assert srs._stats_to_rows([]) == []
