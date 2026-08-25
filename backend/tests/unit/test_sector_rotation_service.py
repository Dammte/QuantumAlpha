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


# --- Tercera auditoría, Bloque A-7 ------------------------------------------


def test_leaders_and_laggards_never_overlap_with_a_thin_ranked_universe():
    # Only 4 sectors have a rank at all (top_n=3) - a plain ranked[-3:] would
    # re-include 2 of the 3 leaders as "laggards" too. The laggard slice
    # must shrink instead of double-counting.
    ranks = {"Tecnología": 90, "Financiero": 80, "Salud": 70, "Energía": 60}
    summary = assess_sector_rotation(_performance(ranks), top_n=3)

    assert summary is not None
    assert summary.leaders == ["Tecnología", "Financiero", "Salud"]
    assert summary.laggards == ["Energía"]
    assert not (set(summary.leaders) & set(summary.laggards))


def test_cycle_phase_is_none_on_a_genuine_three_way_tie_not_insertion_order():
    # Tecnología (expansión media), Energía (expansión tardía) and
    # Inmobiliario (recuperación temprana) each contribute exactly one
    # sector to a *different* phase - a genuine 3-way tie at overlap=1. The
    # old `>` comparison always resolved this to "recuperación temprana"
    # (first in CYCLE_PHASE_LEADERS, and the only phase with 4 sectors in
    # its set instead of 3 - a second, compounding bias) regardless of which
    # phase the data actually favors.
    ranks = {
        "Tecnología": 90, "Energía": 80, "Inmobiliario": 70,
        "Comunicación": 10, "Materiales": 8, "Financiero": 5,
    }
    summary = assess_sector_rotation(_performance(ranks), top_n=3)

    assert summary is not None
    assert set(summary.leaders) == {"Tecnología", "Energía", "Inmobiliario"}
    assert summary.cycle_phase is None
    assert summary.cycle_confidence == 0.0
    assert summary.cycle_description is None
