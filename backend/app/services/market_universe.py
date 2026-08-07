"""A curated, hardcoded ticker universe for the market-intelligence features
(screener, movers, sector/industry strength, trend breadth, watchlist).

There is no live index-membership API wired up (no S&P 500 constituents feed), so —
same trade-off as most retail screener tools — this is a hand-picked sample of
liquid, well-known names per industry. Sectors are derived from industries (each
industry declares its parent sector), so a stock's sector classification and its
more specific industry/subsector both come from one source of truth.

Market-cap tiers are hand-tagged (mega/large/mid/small), not fetched live: a real
per-ticker market-cap lookup via yfinance's `.info` is slow and rate-limited across
~180 tickers, and since this universe is already hand-curated, a static, coarse
tier is good enough to tell "the mega-cap leading a subsector" apart from
"the small-cap also showing up there" - which is the whole point of tagging it.
Tiers are approximate and will drift out of date; treat them as directional.

Two independent regions are curated - "us" (the original universe, still the
default everywhere for backward compatibility) and "europe" (added later,
covering the UK/Germany/France/Netherlands/Spain/Switzerland/Italy blue chips).
They are deliberately NOT blended into one cross-sectional ranking: RS Rating and
sector rotation are both benchmark-relative (S&P 500 for the US, STOXX Europe 600
for Europe) and currency-relative, so comparing a European stock's percentile rank
against US peers under a US benchmark would conflate two different questions.
Every European ticker was individually confirmed to resolve on yfinance (name,
sector, currency) before being added here - see the region's own comment block
for the exchange-suffix convention used (.PA/.DE/.AS/.MC/.MI/.SW/.L).
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Industry:
    name: str
    sector: str
    etf: str | None  # representative ETF proxy for the industry, if a liquid one exists
    tickers: tuple[str, ...]


CAP_TIERS = ("mega", "large", "mid", "small")

# ticker -> approximate market-cap tier (mega ≥ $200B, large $10-200B, mid $2-10B, small < $2B)
TICKER_CAP_TIER: dict[str, str] = {
    # Mega caps
    "AAPL": "mega", "MSFT": "mega", "NVDA": "mega", "GOOGL": "mega", "AMZN": "mega",
    "META": "mega", "AVGO": "mega", "TSLA": "mega", "ORCL": "mega", "LLY": "mega",
    "JNJ": "mega", "XOM": "mega", "CVX": "mega", "WMT": "mega",
    "COST": "mega", "UNH": "mega", "GE": "mega", "ASML": "mega",
    # Large caps
    "AMD": "large", "QCOM": "large", "TXN": "large", "AMAT": "large", "MU": "large",
    "CRM": "large", "ADBE": "large", "NOW": "large", "INTU": "large", "PANW": "large",
    "CRWD": "large", "FTNT": "large", "JPM": "large", "BAC": "large", "WFC": "large",
    "C": "large", "GS": "large", "MS": "large", "V": "large", "MA": "large",
    "PYPL": "large", "FI": "large", "PGR": "large", "CB": "large", "TRV": "large",
    "MRK": "large", "PFE": "large", "ABBV": "large", "BMY": "large", "VRTX": "large",
    "REGN": "large", "GILD": "large", "AMGN": "large", "ABT": "large", "TMO": "large",
    "DHR": "large", "ISRG": "large", "MDT": "large", "SYK": "large", "CVS": "large",
    "CI": "large", "HUM": "large", "ELV": "large", "COP": "large", "EOG": "large",
    "SLB": "large", "MPC": "large", "PSX": "large", "VLO": "large", "RTX": "large",
    "LMT": "large", "BA": "large", "GD": "large", "NOC": "large", "UNP": "large",
    "UPS": "large", "FDX": "large", "CAT": "large", "DE": "large", "HON": "large",
    "ETN": "large", "HD": "large", "LOW": "large", "TJX": "large", "F": "large",
    "MCD": "large", "SBUX": "large", "CMG": "large", "YUM": "large", "BKNG": "large",
    "ABNB": "large", "PEP": "large", "PG": "large", "MDLZ": "large", "PM": "large",
    "NEE": "large", "SO": "large", "DUK": "large", "LIN": "large", "APD": "large",
    "SHW": "large", "NFLX": "large", "DIS": "large", "CMCSA": "large", "TMUS": "large",
    "VZ": "large", "T": "large", "PLD": "large", "AMT": "large", "EQIX": "large",
    "SPG": "large", "O": "large", "PSA": "large", "TGT": "large", "ORLY": "large",
    "AZO": "large",
    # Mid caps
    "WDAY": "mid", "ZS": "mid", "OKTA": "mid", "CYBR": "mid", "FITB": "mid",
    "MTB": "mid", "RF": "mid", "CFG": "mid", "HBAN": "mid", "GPN": "mid",
    "FIS": "mid", "MET": "mid", "ALL": "mid", "BIIB": "mid", "MRNA": "mid",
    "HAL": "mid", "BKR": "mid", "ENPH": "mid", "FSLR": "mid", "HII": "mid",
    "CSX": "mid", "NSC": "mid", "DAL": "mid", "UAL": "mid", "AEP": "mid",
    "D": "mid", "ECL": "mid", "FCX": "mid", "NEM": "mid", "NUE": "mid",
    "WBD": "mid", "KEY": "small",
    # Small caps (relative to this universe - still liquid, just smaller)
    "SEDG": "small",
}


def cap_tier_of(ticker: str) -> str:
    # Checked as one combined lookup (not region-parametrized): tickers are
    # unique strings across both universes (a bare "AAPL" never collides with
    # a suffixed "SAP.DE"), so there's no ambiguity in looking up either dict.
    if ticker in TICKER_CAP_TIER:
        return TICKER_CAP_TIER[ticker]
    if ticker in EUROPE_TICKER_CAP_TIER:
        return EUROPE_TICKER_CAP_TIER[ticker]
    return "large"  # unseen tickers default to "large" (safe middle)


INDUSTRIES: tuple[Industry, ...] = (
    # --- Tecnología ---
    Industry(
        "Semiconductores", "Tecnología", "SOXX", ("NVDA", "AVGO", "AMD", "QCOM", "TXN", "AMAT", "MU", "ASML")
    ),
    Industry("Software empresarial", "Tecnología", "IGV", ("MSFT", "ORCL", "CRM", "ADBE", "NOW", "INTU", "WDAY")),
    Industry("Ciberseguridad", "Tecnología", "CIBR", ("PANW", "CRWD", "FTNT", "ZS", "OKTA", "CYBR")),
    # --- Financiero ---
    Industry("Grandes bancos", "Financiero", "KBE", ("JPM", "BAC", "WFC", "C", "GS", "MS")),
    Industry("Bancos regionales", "Financiero", "KRE", ("FITB", "MTB", "RF", "CFG", "HBAN", "KEY")),
    Industry("Pagos y fintech", "Financiero", None, ("V", "MA", "PYPL", "FI", "GPN", "FIS")),
    Industry("Seguros", "Financiero", "KIE", ("PGR", "CB", "TRV", "ALL", "MET")),
    # --- Salud ---
    Industry("Farmacéuticas", "Salud", "PPH", ("LLY", "JNJ", "MRK", "PFE", "ABBV", "BMY")),
    Industry("Biotecnología", "Salud", "XBI", ("VRTX", "REGN", "GILD", "AMGN", "BIIB", "MRNA")),
    Industry("Equipos médicos", "Salud", "IHI", ("ABT", "TMO", "DHR", "ISRG", "MDT", "SYK")),
    Industry("Seguros de salud", "Salud", "IHF", ("UNH", "CVS", "CI", "HUM", "ELV")),
    # --- Energía ---
    Industry("Petroleras integradas", "Energía", "IEO", ("XOM", "CVX", "COP", "EOG")),
    Industry("Servicios petroleros", "Energía", "OIH", ("SLB", "HAL", "BKR")),
    Industry("Refino", "Energía", "CRAK", ("MPC", "PSX", "VLO")),
    Industry("Energías renovables", "Energía", "TAN", ("ENPH", "FSLR", "SEDG")),
    # --- Industrial ---
    Industry("Aeroespacial y defensa", "Industrial", "ITA", ("RTX", "LMT", "BA", "GD", "NOC", "HII")),
    Industry("Transporte y logística", "Industrial", "IYT", ("UNP", "UPS", "CSX", "NSC", "FDX")),
    Industry("Maquinaria industrial", "Industrial", None, ("CAT", "DE", "HON", "GE", "ETN")),
    # --- Consumo discrecional ---
    Industry(
        "Comercio minorista", "Consumo discrecional", "XRT", ("AMZN", "HD", "LOW", "TJX", "TGT", "ORLY", "AZO")
    ),
    Industry("Automóviles", "Consumo discrecional", "CARZ", ("TSLA", "F")),
    Industry("Restaurantes", "Consumo discrecional", None, ("MCD", "SBUX", "CMG", "YUM")),
    Industry("Viajes y aerolíneas", "Consumo discrecional", "JETS", ("BKNG", "ABNB", "DAL", "UAL")),
    # --- Consumo defensivo ---
    Industry("Retail defensivo", "Consumo defensivo", None, ("WMT", "COST")),
    Industry("Alimentación y bebidas", "Consumo defensivo", None, ("KO", "PEP", "PG", "MDLZ", "PM")),
    # --- Utilities ---
    Industry("Eléctricas", "Utilities", "XLU", ("NEE", "SO", "DUK", "AEP", "D")),
    # --- Materiales ---
    Industry("Química", "Materiales", None, ("LIN", "APD", "SHW", "ECL")),
    Industry("Minería y metales", "Materiales", "XME", ("FCX", "NEM", "NUE")),
    # --- Comunicación ---
    Industry("Medios y streaming", "Comunicación", None, ("NFLX", "DIS", "CMCSA", "WBD")),
    Industry("Redes sociales e internet", "Comunicación", None, ("META", "GOOGL")),
    Industry("Telecomunicaciones", "Comunicación", None, ("TMUS", "VZ", "T")),
    # --- Inmobiliario ---
    Industry("REITs", "Inmobiliario", "XLRE", ("PLD", "AMT", "EQIX", "SPG", "O", "PSA")),
)

SECTOR_ETFS: dict[str, str] = {
    "Tecnología": "XLK",
    "Financiero": "XLF",
    "Salud": "XLV",
    "Energía": "XLE",
    "Industrial": "XLI",
    "Consumo discrecional": "XLY",
    "Consumo defensivo": "XLP",
    "Utilities": "XLU",
    "Materiales": "XLB",
    "Comunicación": "XLC",
    "Inmobiliario": "XLRE",
}

BENCHMARK_TICKER = "^GSPC"


# ============================================================================
# Europe: a second, independent curated universe (see module docstring for why
# it isn't blended with the US one). Exchange suffixes follow Yahoo Finance's
# convention: .PA Paris, .DE Xetra/Frankfurt, .AS Amsterdam, .MC Madrid,
# .MI Milan, .SW Zurich/SIX, .L London. Every ticker below was individually
# confirmed via a live yfinance call (name, sector, currency, price history)
# before being added - see the sector-rotation ETF picks (the iShares STOXX
# Europe 600 sub-industry range) confirmed the same way.
# ============================================================================

EUROPE_TICKER_CAP_TIER: dict[str, str] = {
    # Mega caps (roughly >$150B)
    "MC.PA": "mega", "RMS.PA": "mega", "NESN.SW": "mega", "NOVN.SW": "mega",
    "SAP.DE": "mega", "SHEL.L": "mega", "AZN.L": "mega", "HSBA.L": "mega", "ULVR.L": "mega",
    # Large caps
    "OR.PA": "large", "AIR.PA": "large", "SU.PA": "large", "SAN.PA": "large", "BNP.PA": "large",
    "TTE.PA": "large", "DG.PA": "large", "AI.PA": "large", "SIE.DE": "large", "ALV.DE": "large",
    "MBG.DE": "large", "BMW.DE": "large", "VOW3.DE": "large", "DTE.DE": "large", "MUV2.DE": "large",
    "DBK.DE": "large", "IFX.DE": "large", "ADS.DE": "large", "ITX.MC": "large", "IBE.MC": "large",
    "SAN.MC": "large", "REP.MC": "large", "UBSG.SW": "large", "ZURN.SW": "large", "ABBN.SW": "large",
    "ADYEN.AS": "large", "HEIA.AS": "large", "ENI.MI": "large", "RACE.MI": "large",
    "GSK.L": "large", "DGE.L": "large", "RIO.L": "large", "BATS.L": "large", "BP.L": "large",
    "GLEN.L": "large", "AAL.L": "large", "VOD.L": "large", "NG.L": "large", "LSEG.L": "large",
    "PRU.L": "large", "LGEN.L": "large", "ENEL.MI": "large", "EOAN.DE": "large", "STM": "large",
    # Mid caps
    "BAS.DE": "mid", "STLAM.MI": "mid", "FER.MC": "mid", "AENA.MC": "mid",
    "ENGI.PA": "mid", "ORA.PA": "mid", "PUB.PA": "mid", "REL.L": "mid", "WKL.AS": "mid",
    "DSY.PA": "mid", "ASM.AS": "mid", "PRX.AS": "mid", "VNA.DE": "mid", "URW.PA": "mid",
    "SGRO.L": "mid", "G.MI": "mid", "BARC.L": "mid",
}


# ticker -> currency, for every non-USD ticker in either universe (yfinance's own
# `.info["currency"]`, confirmed live - not inferred from the exchange suffix,
# since e.g. London Stock Exchange quotes most stocks in pence (GBp), not pounds).
TICKER_CURRENCY: dict[str, str] = {
    # EUR
    **dict.fromkeys(
        (
            "MC.PA", "OR.PA", "AIR.PA", "SU.PA", "SAN.PA", "BNP.PA", "TTE.PA", "DG.PA", "RMS.PA", "AI.PA",
            "ENGI.PA", "ORA.PA", "PUB.PA", "URW.PA", "DSY.PA",
            "SAP.DE", "SIE.DE", "ALV.DE", "MBG.DE", "BMW.DE", "VOW3.DE", "DTE.DE", "BAS.DE", "MUV2.DE",
            "DBK.DE", "IFX.DE", "ADS.DE", "EOAN.DE", "VNA.DE",
            "ITX.MC", "IBE.MC", "SAN.MC", "REP.MC", "FER.MC", "AENA.MC",
            "ADYEN.AS", "HEIA.AS", "PRX.AS", "WKL.AS", "ASM.AS",
            "ENI.MI", "RACE.MI", "STLAM.MI", "G.MI", "ENEL.MI",
        ),
        "EUR",
    ),
    # CHF
    **dict.fromkeys(("NESN.SW", "NOVN.SW", "UBSG.SW", "ZURN.SW", "ABBN.SW"), "CHF"),
    # GBp (pence, not pounds - London Stock Exchange convention)
    **dict.fromkeys(
        (
            "AZN.L", "ULVR.L", "HSBA.L", "SHEL.L", "BP.L", "GSK.L", "DGE.L", "RIO.L", "BATS.L", "REL.L",
            "LSEG.L", "BARC.L", "NG.L", "VOD.L", "AAL.L", "GLEN.L", "SGRO.L", "PRU.L", "LGEN.L",
        ),
        "GBp",
    ),
    # STM (STMicroelectronics) trades in USD on the NYSE despite being a European
    # company - correctly omitted here, defaults to USD like everything else.
}


def currency_of(ticker: str) -> str:
    return TICKER_CURRENCY.get(ticker, "USD")


EUROPE_INDUSTRIES: tuple[Industry, ...] = (
    # --- Tecnología ---
    Industry(
        "Software y semiconductores", "Tecnología", "EXV3.DE", ("SAP.DE", "IFX.DE", "ASM.AS", "DSY.PA", "STM")
    ),
    # --- Financiero ---
    Industry(
        "Grandes bancos", "Financiero", "EXV1.DE",
        ("BNP.PA", "DBK.DE", "UBSG.SW", "HSBA.L", "BARC.L", "SAN.MC"),
    ),
    Industry("Seguros", "Financiero", "EXH5.DE", ("ALV.DE", "MUV2.DE", "ZURN.SW", "G.MI", "PRU.L", "LGEN.L")),
    Industry("Pagos y fintech", "Financiero", None, ("ADYEN.AS", "LSEG.L")),
    # --- Salud ---
    Industry("Farmacéuticas", "Salud", "EXV4.DE", ("SAN.PA", "NOVN.SW", "AZN.L", "GSK.L")),
    # --- Energía ---
    Industry("Petroleras integradas", "Energía", "EXH1.DE", ("TTE.PA", "REP.MC", "ENI.MI", "SHEL.L", "BP.L")),
    # --- Industrial ---
    Industry("Industrial y aeroespacial", "Industrial", "EXH4.DE", ("AIR.PA", "SU.PA", "SIE.DE", "ABBN.SW")),
    Industry("Infraestructuras y construcción", "Industrial", "EXV8.DE", ("DG.PA", "FER.MC", "AENA.MC")),
    # --- Consumo discrecional ---
    Industry("Lujo y bienes personales", "Consumo discrecional", "EXH7.DE", ("MC.PA", "RMS.PA", "ADS.DE")),
    Industry(
        "Automóviles", "Consumo discrecional", "EXV5.DE",
        ("MBG.DE", "BMW.DE", "VOW3.DE", "RACE.MI", "STLAM.MI"),
    ),
    Industry("Retail y e-commerce", "Consumo discrecional", "EXH8.DE", ("ITX.MC", "PRX.AS")),
    # --- Consumo defensivo ---
    Industry(
        "Alimentación y bebidas", "Consumo defensivo", "EXH3.DE",
        ("OR.PA", "NESN.SW", "HEIA.AS", "ULVR.L", "DGE.L", "BATS.L"),
    ),
    # --- Utilities ---
    Industry("Eléctricas", "Utilities", "EXH9.DE", ("IBE.MC", "ENEL.MI", "EOAN.DE", "ENGI.PA", "NG.L")),
    # --- Materiales ---
    Industry("Química", "Materiales", "EXV7.DE", ("BAS.DE", "AI.PA")),
    Industry("Minería y metales", "Materiales", "EXV6.DE", ("RIO.L", "GLEN.L", "AAL.L")),
    # --- Comunicación ---
    Industry("Telecomunicaciones", "Comunicación", "EXV2.DE", ("DTE.DE", "ORA.PA", "VOD.L")),
    Industry("Medios e información", "Comunicación", "EXH6.DE", ("PUB.PA", "REL.L", "WKL.AS")),
    # --- Inmobiliario ---
    Industry("REITs europeos", "Inmobiliario", "EXI5.DE", ("VNA.DE", "URW.PA", "SGRO.L")),
)

EUROPE_SECTOR_ETFS: dict[str, str] = {
    "Tecnología": "EXV3.DE",  # iShares STOXX Europe 600 Technology
    "Financiero": "EXV1.DE",  # iShares STOXX Europe 600 Banks (Europe's financial sector is bank-dominated)
    "Salud": "EXV4.DE",  # iShares STOXX Europe 600 Health Care
    "Energía": "EXH1.DE",  # iShares STOXX Europe 600 Oil & Gas
    "Industrial": "EXH4.DE",  # iShares STOXX Europe 600 Industrial Goods & Services
    "Consumo discrecional": "EXH8.DE",  # iShares STOXX Europe 600 Retail
    "Consumo defensivo": "EXH3.DE",  # iShares STOXX Europe 600 Food & Beverage
    "Utilities": "EXH9.DE",  # iShares STOXX Europe 600 Utilities
    "Materiales": "EXV6.DE",  # iShares STOXX Europe 600 Basic Resources
    "Comunicación": "EXV2.DE",  # iShares STOXX Europe 600 Telecommunications
    "Inmobiliario": "EXI5.DE",  # iShares STOXX Europe 600 Real Estate
}

EUROPE_BENCHMARK_TICKER = "^STOXX"  # STOXX Europe 600 - broad, includes UK/CH alongside the eurozone

# Recognized European exchange suffixes, used to guess the right benchmark/region
# for a ticker a user searches directly that isn't necessarily in the curated
# universe above (see `benchmark_for_ticker`).
EUROPEAN_EXCHANGE_SUFFIXES: tuple[str, ...] = (
    ".PA", ".DE", ".AS", ".MC", ".MI", ".SW", ".L", ".BR", ".VI", ".LS", ".ST", ".CO", ".HE", ".OL",
)


@dataclass(frozen=True, slots=True)
class RegionConfig:
    key: str
    label: str
    industries: tuple[Industry, ...]
    sector_etfs: dict[str, str]
    benchmark_ticker: str


DEFAULT_REGION = "us"

REGIONS: dict[str, RegionConfig] = {
    "us": RegionConfig("us", "Estados Unidos", INDUSTRIES, SECTOR_ETFS, BENCHMARK_TICKER),
    "europe": RegionConfig("europe", "Europa", EUROPE_INDUSTRIES, EUROPE_SECTOR_ETFS, EUROPE_BENCHMARK_TICKER),
}


def region_config(region: str) -> RegionConfig:
    return REGIONS.get(region, REGIONS[DEFAULT_REGION])


def benchmark_for_region(region: str) -> str:
    return region_config(region).benchmark_ticker


def benchmark_for_ticker(ticker: str) -> str:
    """Best-effort benchmark pick for a ticker searched directly (e.g. from
    "Analizar activo"), which may or may not be in the curated universe above:
    Europe's benchmark if it's a curated European ticker or carries a
    recognized European exchange suffix, the S&P 500 otherwise."""
    ticker = ticker.upper()
    if ticker in all_sector_tickers("europe"):
        return EUROPE_BENCHMARK_TICKER
    if any(ticker.endswith(suffix) for suffix in EUROPEAN_EXCHANGE_SUFFIXES):
        return EUROPE_BENCHMARK_TICKER
    return BENCHMARK_TICKER


# Broad indices tracked for market context / comparison (not part of the stock universe).
BENCHMARK_INDICES: dict[str, str] = {
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "Dow Jones": "^DJI",
    "Russell 2000": "^RUT",
    "STOXX 600": "^STOXX",
    "IBEX 35": "^IBEX",
}

VIX_TICKER = "^VIX"
VIX_3M_TICKER = "^VIX3M"

# Instruments used only for the market-context composite (Fear & Greed proxy, liquidity).
MACRO_TICKERS: dict[str, str] = {
    "spx": "^GSPC",
    "vix": "^VIX",
    "treasury_long": "TLT",
    "high_yield_bonds": "HYG",
    "investment_grade_bonds": "LQD",
    "dollar_index": "UUP",  # ETF proxy for DXY - more reliably available via yfinance
}


def sector_of(ticker: str) -> str | None:
    # Combined across both regions, same rationale as cap_tier_of.
    for industry in INDUSTRIES + EUROPE_INDUSTRIES:
        if ticker in industry.tickers:
            return industry.sector
    return None


def all_sector_tickers(region: str = DEFAULT_REGION) -> dict[str, str]:
    """ticker -> sector, for every stock in the given region's universe."""
    mapping: dict[str, str] = {}
    for industry in region_config(region).industries:
        for ticker in industry.tickers:
            mapping[ticker] = industry.sector
    return mapping


def all_industry_tickers(region: str = DEFAULT_REGION) -> dict[str, str]:
    """ticker -> industry name, for every stock in the given region's universe."""
    mapping: dict[str, str] = {}
    for industry in region_config(region).industries:
        for ticker in industry.tickers:
            mapping[ticker] = industry.name
    return mapping


def industries_by_sector(region: str = DEFAULT_REGION) -> dict[str, list[Industry]]:
    result: dict[str, list[Industry]] = {}
    for industry in region_config(region).industries:
        result.setdefault(industry.sector, []).append(industry)
    return result


def universe_tickers(region: str = DEFAULT_REGION) -> list[str]:
    return sorted(all_sector_tickers(region).keys())
