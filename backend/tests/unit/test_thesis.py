"""Auditoria del Radar, bloque E4: plantilla determinista de tesis - el
fallback que siempre funciona, sin Gemini."""

from app.services.setups import thesis


def test_generate_deterministic_thesis_includes_the_setup_narrative_verbatim():
    result = thesis.generate_deterministic_thesis(
        ticker="NVDA", setup_narrative="VCP de 3 contracciones, falta el disparo.",
        sector="Tecnología", rs_rating=90, entry_price=130.0, stop_price=122.0,
        stop_basis="bajo el soporte en 122.00", target_price=145.0, risk_reward_net=2.5,
        score_total=85.0,
    )
    assert "VCP de 3 contracciones, falta el disparo." in result


def test_generate_deterministic_thesis_includes_the_geometry():
    result = thesis.generate_deterministic_thesis(
        ticker="NVDA", setup_narrative=None, sector=None, rs_rating=None,
        entry_price=130.0, stop_price=122.0, stop_basis="bajo el soporte en 122.00",
        target_price=145.0, risk_reward_net=2.5, score_total=None,
    )
    assert "130.00" in result
    assert "bajo el soporte en 122.00" in result
    assert "145.00" in result
    assert "2.5:1" in result


def test_generate_deterministic_thesis_falls_back_to_stop_price_without_a_basis():
    result = thesis.generate_deterministic_thesis(
        ticker="NVDA", setup_narrative=None, sector=None, rs_rating=None,
        entry_price=130.0, stop_price=122.0, stop_basis=None,
        target_price=None, risk_reward_net=None, score_total=None,
    )
    assert "stop en 122.00" in result


def test_generate_deterministic_thesis_includes_context_when_available():
    result = thesis.generate_deterministic_thesis(
        ticker="NVDA", setup_narrative=None, sector="Tecnología", rs_rating=90,
        entry_price=None, stop_price=None, stop_basis=None,
        target_price=None, risk_reward_net=None, score_total=85.0,
    )
    assert "RS 90" in result
    assert "sector Tecnología" in result
    assert "score 85/100" in result


def test_generate_deterministic_thesis_with_nothing_at_all_says_so_honestly():
    result = thesis.generate_deterministic_thesis(
        ticker="NVDA", setup_narrative=None, sector=None, rs_rating=None,
        entry_price=None, stop_price=None, stop_basis=None,
        target_price=None, risk_reward_net=None, score_total=None,
    )
    assert "NVDA" in result
    assert "no tiene suficiente información" in result


def test_generate_deterministic_thesis_omits_geometry_sentence_without_a_stop():
    result = thesis.generate_deterministic_thesis(
        ticker="NVDA", setup_narrative="Narrativa.", sector=None, rs_rating=None,
        entry_price=130.0, stop_price=None, stop_basis=None,
        target_price=None, risk_reward_net=None, score_total=None,
    )
    assert "Entrada en" not in result
