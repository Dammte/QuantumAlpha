from types import SimpleNamespace

from app.services import ablation_report_service as ars

_HEADER = (
    "factor,current_points,n_triggered,n_not_triggered,mean_return_triggered_pct,"
    "mean_return_not_triggered_pct,mean_difference_pct,t_stat,p_value,permutation_p_value,"
    "permutation_p_value_bh,significant_at_1pct,significant_at_1pct_bh,directionally_consistent,"
    "mean_ic,ic_ir,n_ic_buckets,multivariate_coef_pct,multivariate_p_value"
)


def _row(factor, points, mean_diff, consistent, significant=True):
    return (
        f"{factor},{points},2304,20598,0.959,-0.107,{mean_diff},5.999,0.0,0.0002,0.001,"
        f"{significant},{significant},{consistent},0.02,0.1,50,0.9,0.01"
    )


def _write_report(tmp_path, horizon_days, rows):
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    path = docs / f"factor_ablation_report_v2_h{horizon_days}.csv"
    path.write_text("\n".join([_HEADER, *rows]) + "\n", encoding="utf-8")
    return docs


def test_load_ablation_report_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(ars, "DOCS_DIR", tmp_path / "docs")
    assert ars.load_ablation_report(21) == []


def test_load_ablation_report_parses_real_rows(tmp_path, monkeypatch):
    docs = _write_report(
        tmp_path, 21,
        [_row("trend_down", -3, 1.066, "False"), _row("trend_up", 2, 0.5, "True")],
    )
    monkeypatch.setattr(ars, "DOCS_DIR", docs)
    results = ars.load_ablation_report(21)
    assert len(results) == 2
    trend_down = next(r for r in results if r.factor == "trend_down")
    assert trend_down.current_points == -3
    assert trend_down.mean_difference_pct == 1.066
    assert trend_down.directionally_consistent is False


def test_load_ablation_report_malformed_file_returns_empty_not_raises(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "factor_ablation_report_v2_h21.csv").write_text("not,a,valid,header\ngarbage\n", encoding="utf-8")
    monkeypatch.setattr(ars, "DOCS_DIR", docs)
    assert ars.load_ablation_report(21) == []


def test_sign_mismatched_factor_keys_only_includes_inconsistent_ones(tmp_path, monkeypatch):
    docs = _write_report(
        tmp_path, 21,
        [_row("trend_down", -3, 1.066, "False"), _row("trend_up", 2, 0.5, "True")],
    )
    monkeypatch.setattr(ars, "DOCS_DIR", docs)
    assert ars.sign_mismatched_factor_keys(21) == {"trend_down"}


def test_sign_mismatched_factor_keys_empty_when_no_report_on_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ars, "DOCS_DIR", tmp_path / "docs")
    assert ars.sign_mismatched_factor_keys(21) == set()


def _factor(label, triggered):
    return SimpleNamespace(label=label, triggered=triggered, points=1)


def test_triggered_factors_with_contradicted_sign_flags_a_matching_triggered_factor(tmp_path, monkeypatch):
    docs = _write_report(tmp_path, 21, [_row("trend_down", -3, 1.066, "False")])
    monkeypatch.setattr(ars, "DOCS_DIR", docs)
    factors = [_factor("Tendencia bajista - evitar entradas largas", triggered=True)]
    assert ars.triggered_factors_with_contradicted_sign(factors, 21) == [
        "Tendencia bajista - evitar entradas largas"
    ]


def test_triggered_factors_with_contradicted_sign_ignores_a_non_triggered_factor(tmp_path, monkeypatch):
    docs = _write_report(tmp_path, 21, [_row("trend_down", -3, 1.066, "False")])
    monkeypatch.setattr(ars, "DOCS_DIR", docs)
    factors = [_factor("Tendencia bajista - evitar entradas largas", triggered=False)]
    assert ars.triggered_factors_with_contradicted_sign(factors, 21) == []


def test_triggered_factors_with_contradicted_sign_ignores_an_unmapped_label(tmp_path, monkeypatch):
    # A factor label with no entry in FACTOR_LABEL_TO_ABLATION_KEY must never
    # be flagged - absence of a mapping isn't evidence of a contradiction.
    docs = _write_report(tmp_path, 21, [_row("trend_down", -3, 1.066, "False")])
    monkeypatch.setattr(ars, "DOCS_DIR", docs)
    factors = [_factor("Un factor que no existe en el mapeo", triggered=True)]
    assert ars.triggered_factors_with_contradicted_sign(factors, 21) == []


def test_triggered_factors_with_contradicted_sign_empty_when_factor_is_consistent(tmp_path, monkeypatch):
    docs = _write_report(tmp_path, 21, [_row("trend_up", 2, 0.5, "True")])
    monkeypatch.setattr(ars, "DOCS_DIR", docs)
    factors = [_factor("Tendencia alcista (MA20 > MA50 > MA200)", triggered=True)]
    assert ars.triggered_factors_with_contradicted_sign(factors, 21) == []


# --- setup outcome stats (Tercera auditoría, Bloque F-6/F-7) ----------------

_OUTCOME_HEADER = "setup,n,win_rate,expectancy_r,median_bars_held,mae_p80_pct"


def _write_outcomes(tmp_path, horizon_days, rows):
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    path = docs / f"factor_ablation_report_v3_h{horizon_days}_setup_outcomes.csv"
    path.write_text("\n".join([_OUTCOME_HEADER, *rows]) + "\n", encoding="utf-8")
    return docs


def test_load_setup_outcome_stats_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(ars, "DOCS_DIR", tmp_path / "docs")
    assert ars.load_setup_outcome_stats(21) == []


def test_load_setup_outcome_stats_parses_real_rows(tmp_path, monkeypatch):
    docs = _write_outcomes(
        tmp_path, 21,
        ["oversold_bounce,412,0.54,0.42,6.0,2.1", "breakout_volume,201,0.48,0.15,8.0,3.4"],
    )
    monkeypatch.setattr(ars, "DOCS_DIR", docs)
    results = ars.load_setup_outcome_stats(21)
    assert len(results) == 2
    oversold = next(r for r in results if r.setup == "oversold_bounce")
    assert oversold.n == 412
    assert oversold.win_rate == 0.54
    assert oversold.expectancy_r == 0.42
    assert oversold.median_bars_held == 6.0
    assert oversold.mae_p80_pct == 2.1


def test_load_setup_outcome_stats_malformed_file_returns_empty_not_raises(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "factor_ablation_report_v3_h21_setup_outcomes.csv").write_text(
        "not,valid\ngarbage\n", encoding="utf-8"
    )
    monkeypatch.setattr(ars, "DOCS_DIR", docs)
    assert ars.load_setup_outcome_stats(21) == []


def test_setup_outcome_by_name_keys_by_setup(tmp_path, monkeypatch):
    docs = _write_outcomes(tmp_path, 21, ["oversold_bounce,412,0.54,0.42,6.0,2.1"])
    monkeypatch.setattr(ars, "DOCS_DIR", docs)
    by_name = ars.setup_outcome_by_name(21)
    assert set(by_name) == {"oversold_bounce"}
    assert by_name["oversold_bounce"].win_rate == 0.54


def test_setup_outcome_by_name_empty_when_no_report_on_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ars, "DOCS_DIR", tmp_path / "docs")
    assert ars.setup_outcome_by_name(21) == {}
