from app.services import market_universe as mu


def test_us_and_europe_universes_have_no_overlapping_tickers():
    us = set(mu.universe_tickers("us"))
    europe = set(mu.universe_tickers("europe"))
    assert us & europe == set()
    assert len(us) > 100  # sanity: the original universe wasn't accidentally gutted
    assert len(europe) > 30  # sanity: Europe is a real, non-trivial universe


def test_every_european_ticker_has_a_cap_tier():
    europe = mu.universe_tickers("europe")
    missing = [t for t in europe if t not in mu.EUROPE_TICKER_CAP_TIER]
    assert missing == []


def test_no_orphaned_cap_tier_or_currency_entries():
    """Catches copy-paste leftovers: an entry in TICKER_CAP_TIER/TICKER_CURRENCY
    for a ticker that was since removed from the actual curated industries."""
    europe = set(mu.universe_tickers("europe"))
    us = set(mu.universe_tickers("us"))
    orphaned_cap_tier = [t for t in mu.EUROPE_TICKER_CAP_TIER if t not in europe]
    orphaned_currency = [t for t in mu.TICKER_CURRENCY if t not in europe and t not in us]
    assert orphaned_cap_tier == []
    assert orphaned_currency == []


def test_us_and_europe_cover_the_same_11_sectors():
    us_sectors = {i.sector for i in mu.INDUSTRIES}
    europe_sectors = {i.sector for i in mu.EUROPE_INDUSTRIES}
    assert us_sectors == europe_sectors
    assert len(us_sectors) == 11


def test_europe_sector_etfs_cover_every_sector():
    europe_sectors = {i.sector for i in mu.EUROPE_INDUSTRIES}
    assert set(mu.EUROPE_SECTOR_ETFS) == europe_sectors


def test_currency_of_defaults_to_usd():
    assert mu.currency_of("AAPL") == "USD"
    assert mu.currency_of("SOME_UNKNOWN_TICKER") == "USD"


def test_currency_of_known_european_tickers():
    assert mu.currency_of("SAP.DE") == "EUR"
    assert mu.currency_of("NESN.SW") == "CHF"
    assert mu.currency_of("AZN.L") == "GBp"  # pence, not pounds - LSE convention


def test_cap_tier_of_checks_both_regions():
    assert mu.cap_tier_of("AAPL") == "mega"
    assert mu.cap_tier_of("SAP.DE") == "mega"
    assert mu.cap_tier_of("SOME_UNKNOWN_TICKER") == "large"  # safe default


def test_sector_of_checks_both_regions():
    assert mu.sector_of("NVDA") == "Tecnología"
    assert mu.sector_of("SAP.DE") == "Tecnología"
    assert mu.sector_of("SOME_UNKNOWN_TICKER") is None


def test_benchmark_for_ticker_us_default():
    assert mu.benchmark_for_ticker("AAPL") == "^GSPC"
    assert mu.benchmark_for_ticker("SOME_UNKNOWN_TICKER") == "^GSPC"


def test_benchmark_for_ticker_curated_european():
    assert mu.benchmark_for_ticker("SAP.DE") == "^STOXX"


def test_benchmark_for_ticker_recognizes_european_suffix_even_if_uncurated():
    assert mu.benchmark_for_ticker("SOMECOMPANY.PA") == "^STOXX"
    assert mu.benchmark_for_ticker("SOMECOMPANY.MI") == "^STOXX"


def test_benchmark_for_region():
    assert mu.benchmark_for_region("us") == "^GSPC"
    assert mu.benchmark_for_region("europe") == "^STOXX"
    assert mu.benchmark_for_region("unknown_region") == "^GSPC"  # falls back to default


def test_region_config_falls_back_to_default_for_unknown_region():
    assert mu.region_config("nonsense").key == mu.DEFAULT_REGION


def test_industries_by_sector_is_region_scoped():
    us_grouped = mu.industries_by_sector("us")
    europe_grouped = mu.industries_by_sector("europe")
    assert "Tecnología" in us_grouped
    assert "Tecnología" in europe_grouped
    # Industry names genuinely differ between regions (different constituents)
    us_names = {i.name for i in us_grouped["Tecnología"]}
    europe_names = {i.name for i in europe_grouped["Tecnología"]}
    assert us_names != europe_names
