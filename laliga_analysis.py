import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

# ── 1. WCZYTANIE DANYCH ───────────────────────────────────────────────────────

df = pd.read_csv("wyniki_DEA.csv")

# Skrócone nazwy sezonów do wykresów
df["Sezon_short"] = df["Sezon"].str[:4]

print("=== PODSTAWOWE INFO ===")
print(f"Obserwacji: {len(df)}")
print(f"Sezony: {df['Sezon'].unique()}")
print(f"Liczba unikalnych klubów: {df['Klub'].nunique()}")
print()

# ── 2. STATYSTYKI EFEKTYWNOŚCI PER SEZON ─────────────────────────────────────

print("=== ŚREDNIA EFEKTYWNOŚĆ PER SEZON ===")
season_stats = (
    df.groupby("Sezon")["Efektywnosc"]
    .agg(["mean", "min", "max", "std"])
    .round(3)
)
season_stats.columns = ["Średnia", "Min", "Max", "Odch.std"]
print(season_stats)
print()

# Odsetek klubów efektywnych (score = 1.0) per sezon
efficient_pct = (
    df[df["Efektywnosc"] == 1.0]
    .groupby("Sezon")["Klub"]
    .count()
    .div(df.groupby("Sezon")["Klub"].count())
    .mul(100)
    .round(1)
)
print("=== % KLUBÓW EFEKTYWNYCH (DEA = 1.0) PER SEZON ===")
print(efficient_pct.to_string())
print()

# ── 3. KORELACJA SPEARMANA: EFEKTYWNOŚĆ vs PUNKTY ────────────────────────────

rho, p_value = stats.spearmanr(df["Efektywnosc"], df["Pozycja"])
print("=== KORELACJA SPEARMANA: EFEKTYWNOŚĆ vs PUNKTY ===")
print(f"rho = {rho:.3f}")
print(f"p-value = {p_value:.5f}")
print(f"n = {len(df)}")
print()

# ── 4. WYKRES 1: EFEKTYWNOŚĆ vs PUNKTY (scatter) ─────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))

# Różny kolor dla każdego sezonu
seasons = df["Sezon"].unique()
colors = plt.cm.tab10.colors

for i, season in enumerate(seasons):
    subset = df[df["Sezon"] == season]
    ax.scatter(
        subset["Efektywnosc"],
        subset["Punkty"],
        label=season,
        color=colors[i],
        alpha=0.75,
        s=50,
    )

# Linia trendu
slope, intercept, *_ = stats.linregress(df["Efektywnosc"], df["Punkty"])
x_range = [df["Efektywnosc"].min(), df["Efektywnosc"].max()]
y_range = [slope * x + intercept for x in x_range]
ax.plot(x_range, y_range, color="black", linewidth=1.2, linestyle="--", label="Trend")

ax.set_xlabel("Wynik efektywności DEA (BCC input-oriented)")
ax.set_ylabel("Punkty w sezonie")
ax.set_title(
    f"Efektywność finansowa a wyniki sportowe w La Liga (2019–2025)\n"
    f"Korelacja Spearmana: ρ = {rho:.3f}, p = {p_value:.4f}, n = {len(df)}"
)
ax.legend(title="Sezon", fontsize=8, title_fontsize=8)
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

plt.tight_layout()
plt.savefig("efektywnosc_vs_punkty.png", dpi=150)
plt.close()
print("Zapisano: efektywnosc_vs_punkty.png")

# ── 5. WYKRES 2: ŚREDNIA EFEKTYWNOŚĆ KLUBÓW (min. 3 sezony) ──────────────────

club_avg = (
    df.groupby("Klub")
    .filter(lambda x: len(x) >= 3)   # tylko kluby z co najmniej 3 sezonami
    .groupby("Klub")["Efektywnosc"]
    .mean()
    .sort_values(ascending=True)
)

fig, ax = plt.subplots(figsize=(9, 7))
bars = ax.barh(club_avg.index, club_avg.values, color="steelblue", edgecolor="white")

# Zaznacz kluby efektywne (średnia >= 0.9)
for bar, val in zip(bars, club_avg.values):
    color = "steelblue" if val < 0.9 else "darkorange"
    bar.set_color(color)
    ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", fontsize=7.5)

ax.set_xlabel("Średni wynik efektywności DEA")
ax.set_title("Średnia efektywność finansowa klubów La Liga\n(kluby z min. 3 sezonami, 2019–2025)")
ax.axvline(x=0.9, color="darkorange", linestyle="--", linewidth=1, label="Próg 0.90")
ax.legend(fontsize=8)
ax.set_xlim(0, 1.12)

plt.tight_layout()
plt.savefig("efektywnosc_klubow.png", dpi=150)
plt.close()
print("Zapisano: efektywnosc_klubow.png")

# ── 6. WYKRES 3: ŚREDNIA EFEKTYWNOŚĆ PER SEZON ───────────────────────────────

fig, ax = plt.subplots(figsize=(7, 4))
means = df.groupby("Sezon")["Efektywnosc"].mean()
ax.plot(means.index, means.values, marker="o", color="steelblue", linewidth=2)

for x, y in zip(means.index, means.values):
    ax.text(x, y + 0.005, f"{y:.3f}", ha="center", fontsize=8.5)

ax.set_ylim(0.5, 1.0)
ax.set_xlabel("Sezon")
ax.set_ylabel("Średnia efektywność DEA")
ax.set_title("Zmiana średniej efektywności finansowej La Liga w czasie")
ax.tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig("efektywnosc_trend.png", dpi=150)
plt.close()
print("Zapisano: efektywnosc_trend.png")

print()
print("Gotowe.")
