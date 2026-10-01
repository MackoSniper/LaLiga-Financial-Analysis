"""Statistics and charts for the La Liga DEA efficiency results.

Reads ``wyniki_DEA.csv`` (efficiency scores produced by ``dea_bcc.py``), prints
per-season statistics and Spearman correlations, and writes the PNG charts next
to this script. Paths are resolved from ``__file__``, so the script can be run
from any working directory.

Usage:
    python laliga_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy import stats

from dea_bcc import DEFAULT_CSV, DEAError, load_data

OUTPUT_DIR = Path(__file__).resolve().parent
ALPHA = 0.05
EFFICIENT_THRESHOLD = 0.999  # scores >= this count as "fully efficient"
MIN_SEASONS_FOR_CLUB_CHART = 3

# (x column, y column) pairs reported in the correlation table.
CORRELATION_PAIRS = [
    ("Efektywnosc", "Pozycja"),
    ("Efektywnosc", "Punkty"),
    ("Efektywnosc", "LCPD"),
    ("Efektywnosc", "Wartosc_kadry"),
    ("LCPD", "Pozycja"),
    ("Wartosc_kadry", "Pozycja"),
]

LABELS = {
    "Efektywnosc": "DEA efficiency",
    "Pozycja": "League position",
    "Punkty": "Points",
    "LCPD": "LCPD spending limit",
    "Wartosc_kadry": "Squad value",
}


def spearman(df: pd.DataFrame, x: str, y: str) -> tuple[float, float]:
    """Spearman rank correlation between two columns of ``df``: (rho, p)."""
    result = stats.spearmanr(df[x], df[y])
    return float(result.statistic), float(result.pvalue)


def season_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-season mean, min, max, standard deviation and share of efficient clubs.

    The share of efficient clubs is the mean of a boolean condition computed
    within each season, so it is robust to seasons without any efficient club
    and to floating-point noise around 1.0.
    """
    grouped = df.groupby("Sezon")["Efektywnosc"]
    summary = grouped.agg(["count", "mean", "min", "max", "std"])
    summary["pct_efficient"] = (
        (df["Efektywnosc"] >= EFFICIENT_THRESHOLD).groupby(df["Sezon"]).mean() * 100
    )
    return summary


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman rho, p-value and significance flag for every reported pair."""
    rows = []
    for x, y in CORRELATION_PAIRS:
        rho, p = spearman(df, x, y)
        rows.append({"x": x, "y": y, "rho": rho, "p": p, "n": len(df),
                     "significant": p < ALPHA})
    return pd.DataFrame(rows)


def print_report(df: pd.DataFrame) -> None:
    """Print dataset info, season statistics and the correlation table."""
    print("=== DATASET ===")
    print(f"Observations: {len(df)}")
    print(f"Seasons: {', '.join(sorted(df['Sezon'].unique()))}")
    print(f"Unique clubs: {df['Klub'].nunique()}")
    print()

    summary = season_summary(df)
    print("=== DEA EFFICIENCY PER SEASON ===")
    print(summary.rename(columns={
        "count": "n", "mean": "Mean", "min": "Min", "max": "Max",
        "std": "Std", "pct_efficient": f"% efficient (>= {EFFICIENT_THRESHOLD})",
    }).round(3).to_string())
    print()

    print(f"=== SPEARMAN CORRELATIONS (n = {len(df)}, alpha = {ALPHA}) ===")
    for row in correlation_table(df).itertuples():
        verdict = "significant" if row.significant else "NOT significant"
        p_text = f"{row.p:.5f}" if row.p >= 1e-4 else f"{row.p:.1e}"
        print(f"{LABELS[row.x]} vs {LABELS[row.y]}: "
              f"rho = {row.rho:+.3f}, p = {p_text} ({verdict})")
    print()


def _stat_line(df: pd.DataFrame, x: str, y: str) -> str:
    """Correlation caption computed for exactly the pair (x, y)."""
    rho, p = spearman(df, x, y)
    verdict = ("statistically significant" if p < ALPHA
               else "NOT statistically significant")
    return f"Spearman ρ = {rho:+.3f}, p = {p:.3f}, n = {len(df)} — {verdict} (α = {ALPHA})"


def _season_scatter(ax, df, x, y, *, percent_x=False, log_x=False, invert_y=False):
    """Scatter of column ``x`` against ``y`` coloured by season."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    colors = plt.cm.tab10.colors
    for i, season in enumerate(sorted(df["Sezon"].unique())):
        subset = df[df["Sezon"] == season]
        ax.scatter(subset[x], subset[y], label=season, color=colors[i],
                   alpha=0.75, s=40)
    ax.set_xlabel(LABELS[x] + (" (log scale)" if log_x else ""))
    ax.set_ylabel(LABELS[y])
    if log_x:
        ax.set_xscale("log")
    if percent_x:
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    if invert_y:
        ax.invert_yaxis()
    ax.grid(alpha=0.25)


def _save(fig, name: str) -> None:
    import matplotlib.pyplot as plt

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / name, dpi=150)
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / name}")


def plot_efficiency_vs_outcome(df: pd.DataFrame, y: str, name: str, title: str,
                               invert_y: bool) -> None:
    """Scatter of DEA efficiency against a sporting outcome (points or position)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.2))
    _season_scatter(ax, df, "Efektywnosc", y, percent_x=True, invert_y=invert_y)
    ax.set_title(f"{title}\n{_stat_line(df, 'Efektywnosc', y)}", fontsize=10)
    ax.legend(title="Season", fontsize=8, title_fontsize=8)
    _save(fig, name)


def plot_resources_vs_position(df: pd.DataFrame) -> None:
    """Squad value and LCPD limit against league position (two panels)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for ax, x in zip(axes, ["Wartosc_kadry", "LCPD"]):
        _season_scatter(ax, df, x, "Pozycja", log_x=True)
        ax.set_title(f"{LABELS[x]} vs league position\n{_stat_line(df, x, 'Pozycja')}",
                     fontsize=9)
    axes[0].invert_yaxis()  # shared y axis: invert once so position 1 is on top
    axes[0].legend(title="Season", fontsize=8, title_fontsize=8)
    axes[1].set_ylabel("")
    _save(fig, "zasoby_vs_pozycja.png")


def plot_club_averages(df: pd.DataFrame) -> None:
    """Mean efficiency per club, for clubs with enough seasons."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    seasons_per_club = df.groupby("Klub")["Sezon"].transform("nunique")
    club_avg = (df[seasons_per_club >= MIN_SEASONS_FOR_CLUB_CHART]
                .groupby("Klub")["Efektywnosc"].mean().sort_values())

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["darkorange" if v >= 0.9 else "steelblue" for v in club_avg.values]
    bars = ax.barh(club_avg.index, club_avg.values, color=colors, edgecolor="white")
    for bar, val in zip(bars, club_avg.values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.2f}",
                va="center", fontsize=7.5)

    ax.axvline(x=0.9, color="darkorange", linestyle="--", linewidth=1)
    ax.legend(handles=[Patch(color="darkorange", label="Mean efficiency ≥ 0.90"),
                       Patch(color="steelblue", label="Mean efficiency < 0.90")],
              fontsize=8, loc="lower right")
    ax.set_xlabel("Mean DEA efficiency score")
    ax.set_title("Mean financial efficiency of La Liga clubs\n"
                 f"(clubs with at least {MIN_SEASONS_FOR_CLUB_CHART} seasons, 2019/2020–2024/2025)")
    ax.set_xlim(0, 1.12)
    _save(fig, "efektywnosc_klubow.png")


def plot_trend(df: pd.DataFrame) -> None:
    """Mean efficiency per season."""
    import matplotlib.pyplot as plt

    means = df.groupby("Sezon")["Efektywnosc"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(means.index, means.values, marker="o", color="steelblue", linewidth=2)
    for x, y in zip(means.index, means.values):
        ax.text(x, y + 0.01, f"{y:.3f}", ha="center", fontsize=8.5)
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("Season")
    ax.set_ylabel("Mean DEA efficiency")
    ax.set_title("Mean financial efficiency of La Liga clubs by season")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(alpha=0.25)
    _save(fig, "efektywnosc_trend.png")


def make_charts(df: pd.DataFrame) -> None:
    """Generate and save all charts."""
    plot_efficiency_vs_outcome(
        df, "Punkty", "efektywnosc_vs_punkty.png",
        "DEA efficiency vs points, La Liga 2019/2020–2024/2025", invert_y=False)
    plot_efficiency_vs_outcome(
        df, "Pozycja", "efektywnosc_vs_pozycja.png",
        "DEA efficiency vs final league position (1 = champion)", invert_y=True)
    plot_resources_vs_position(df)
    plot_club_averages(df)
    plot_trend(df)


def main() -> int:
    """Run the analysis; returns the process exit code."""
    try:
        df = load_data(DEFAULT_CSV)
    except DEAError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print_report(df)
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("ERROR: matplotlib is required to generate charts "
              "(pip install -r requirements.txt)", file=sys.stderr)
        return 1
    make_charts(df)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
