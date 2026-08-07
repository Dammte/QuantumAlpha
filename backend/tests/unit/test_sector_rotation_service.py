from app.domain.models.ticker_snapshot import SectorPerformance
from app.services.sector_rotation_service import assess_sector_rotation

ALL_SECTORS = [
    "Tecnología",
    "Financiero",
    "Salud",
    "Energía",
    "Industrial",
    "Consumo discrecional",
    "Consumo defensivo",
    "Utilities",
    "Materiales",
    "Comunicación",
    "Inmobiliario",
]


def _performance(rank_by_sector: dict[str, int]) -> list[SectorPerformance]:
    return [
        SectorPerformance(
            sector=sector,
            etf=f"{sector[:3].upper()}",
            change_1d=0.01,
            change_1w=0.01,
            change_1m=0.01,
            change_3m=0.01,
            change_6m=0.01,
            change_1y=0.01,
            rs_rank=rank_by_sector.get(sector),
        )
        for sector in ALL_SECTORS
    ]


def test_none_when_no_ranks_available():
    performance = _performance({})
    assert assess_sector_rotation(performance) is None


def test_detects_early_cycle_recovery_leadership():
    ranks = {
        "Financiero": 99,
        "Inmobiliario": 90,
        "Consumo discrecional": 85,
        "Utilities": 10,
        "Consumo defensivo": 5,
        "Salud": 15,
    }
    summary = assess_sector_rotation(_performance(ranks), top_n=3)

    assert summary is not None
    assert summary.leaders == ["Financiero", "Inmobiliario", "Consumo discrecional"]
    assert summary.cycle_phase == "recuperación temprana"
    assert summary.cycle_confidence == 1.0
    assert summary.defensive_leadership is False
    assert summary.warning is None


def test_detects_recession_defensive_leadership_and_warns():
    ranks = {
        "Consumo defensivo": 99,
        "Utilities": 95,
        "Salud": 90,
        "Tecnología": 10,
        "Consumo discrecional": 5,
        "Financiero": 8,
    }
    summary = assess_sector_rotation(_performance(ranks), top_n=3)

    assert summary is not None
    assert set(summary.leaders) == {"Consumo defensivo", "Utilities", "Salud"}
    assert summary.cycle_phase == "contracción / recesión"
    assert summary.defensive_leadership is True
    assert summary.warning is not None
    assert "defensivos" in summary.warning


def test_laggards_are_weakest_first():
    ranks = {sector: rank for rank, sector in enumerate(ALL_SECTORS, start=1)}
    summary = assess_sector_rotation(_performance(ranks), top_n=3)

    assert summary is not None
    # ALL_SECTORS[0] ("Tecnología") got rank 1 - the single weakest.
    assert summary.laggards[0] == "Tecnología"
    assert summary.laggards == ["Tecnología", "Financiero", "Salud"]


def test_only_sectors_with_a_rank_are_considered():
    ranks = {"Energía": 80, "Materiales": 70}
    summary = assess_sector_rotation(_performance(ranks), top_n=3)

    assert summary is not None
    assert set(summary.leaders) == {"Energía", "Materiales"}
