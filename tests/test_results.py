"""Checks that the numbers published in the README match the data.

Run with ``pytest tests`` or directly with ``python tests/test_results.py``.
The correlation tests read the README table and compare each row with a
correlation computed for the pair named in that row's label, so a number that
belongs to a different pair of variables (e.g. LCPD instead of league position)
makes the suite fail.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dea_bcc import compute_efficiency, load_data  # noqa: E402
from laliga_analysis import (  # noqa: E402
    ALPHA,
    EFFICIENT_THRESHOLD,
    season_summary,
    spearman,
)

CORR_TOL = 0.002
README = (REPO / "README.md").read_text(encoding="utf-8")
DF = load_data()

# Values published in the README.
PUBLISHED_MEAN_EFFICIENCY = {
    "2019/2020": 0.687,
    "2020/2021": 0.803,
    "2021/2022": 0.857,
    "2022/2023": 0.839,
    "2023/2024": 0.775,
    "2024/2025": 0.756,
}
PUBLISHED_PCT_EFFICIENT = {
    "2019/2020": 30.0,
    "2020/2021": 30.0,
    "2021/2022": 47.4,
    "2022/2023": 45.0,
    "2023/2024": 35.0,
    "2024/2025": 31.6,
}
# README row label -> (x column, y column, published rho)
PUBLISHED_CORRELATIONS = {
    "Efficiency vs league position": ("Efektywnosc", "Pozycja", -0.159),
    "Efficiency vs points": ("Efektywnosc", "Punkty", 0.161),
    "Efficiency vs LCPD limit": ("Efektywnosc", "LCPD", -0.309),
    "Efficiency vs squad value": ("Efektywnosc", "Wartosc_kadry", -0.226),
    "LCPD limit vs league position": ("LCPD", "Pozycja", -0.716),
    "Squad value vs league position": ("Wartosc_kadry", "Pozycja", -0.790),
}


def test_dataset_shape():
    assert len(DF) == 118
    assert DF["Klub"].nunique() == 27
    assert DF["Sezon"].nunique() == 6
    assert int((DF["Efektywnosc"] >= EFFICIENT_THRESHOLD).sum()) == 43


def test_dea_reproduces_csv():
    diff = (compute_efficiency(DF) - DF["Efektywnosc"]).abs()
    assert diff.max() < 1e-9


def test_season_means():
    means = season_summary(DF)["mean"]
    for season, published in PUBLISHED_MEAN_EFFICIENCY.items():
        assert abs(means[season] - published) <= CORR_TOL, season


def test_season_efficient_share():
    pct = season_summary(DF)["pct_efficient"]
    assert not pct.isna().any()
    for season, published in PUBLISHED_PCT_EFFICIENT.items():
        assert abs(pct[season] - published) <= 0.06, season


def test_three_headline_correlations():
    for x, y, published in [
        ("Efektywnosc", "LCPD", -0.309),
        ("Efektywnosc", "Pozycja", -0.159),
        ("Efektywnosc", "Punkty", 0.161),
    ]:
        rho, _ = spearman(DF, x, y)
        assert abs(rho - published) <= CORR_TOL, f"{x} vs {y}: {rho:.3f}"


def test_significance_claims():
    assert spearman(DF, "Efektywnosc", "LCPD")[1] < ALPHA
    assert spearman(DF, "Efektywnosc", "Pozycja")[1] >= ALPHA
    assert spearman(DF, "Efektywnosc", "Punkty")[1] >= ALPHA


def test_readme_correlation_table_matches_data():
    rows = dict(re.findall(r"^\|\s*([^|]+?)\s*\|\s*([+-]?\d\.\d{3})\s*\|", README, re.M))
    for label, (x, y, published) in PUBLISHED_CORRELATIONS.items():
        assert label in rows, f"README table is missing the row '{label}'"
        rho, _ = spearman(DF, x, y)
        assert abs(float(rows[label]) - rho) <= CORR_TOL, (
            f"README says {label}: {rows[label]}, data give {rho:.3f}")
        assert abs(rho - published) <= CORR_TOL


def test_readme_season_table_matches_data():
    pattern = (r"^\|\s*(\d{4}/\d{4})\s*\|\s*(\d+)\s*\|\s*(\d\.\d{3})\s*\|"
               r"\s*(\d+)\s*\|\s*([\d.]+)%\s*\|")
    rows = {m[0]: m[1:] for m in re.findall(pattern, README, re.M)}
    summary = season_summary(DF)
    assert set(rows) == set(summary.index), "README season table is incomplete"
    for season, (n, mean, n_eff, pct) in rows.items():
        assert int(n) == summary.loc[season, "count"]
        assert abs(float(mean) - summary.loc[season, "mean"]) <= CORR_TOL
        assert int(n_eff) == round(summary.loc[season, "pct_efficient"] / 100 * int(n))
        assert abs(float(pct) - summary.loc[season, "pct_efficient"]) <= 0.06


if __name__ == "__main__":
    failures = 0
    for name, func in list(globals().items()):
        if name.startswith("test_") and callable(func):
            try:
                func()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)
