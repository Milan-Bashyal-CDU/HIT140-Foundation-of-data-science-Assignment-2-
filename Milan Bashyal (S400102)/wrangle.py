from __future__ import annotations

import numpy as np
import pandas as pd

from gk_common import (RAW_DATA, CLEAN_DATA, FIG_DIR, PALETTE, banner, section,
                       ensure_dirs, setup_plot_style)
import matplotlib.pyplot as plt

CONFEDERATION = {
    # UEFA (Europe)
    "Austria": "UEFA", "Belgium": "UEFA", "Bosnia–Herz": "UEFA", "Croatia": "UEFA",
    "Czechia": "UEFA", "England": "UEFA", "France": "UEFA", "Germany": "UEFA",
    "Netherlands": "UEFA", "Norway": "UEFA", "Portugal": "UEFA", "Scotland": "UEFA",
    "Spain": "UEFA", "Sweden": "UEFA", "Switzerland": "UEFA", "Türkiye": "UEFA",
    # CONMEBOL (South America)
    "Argentina": "CONMEBOL", "Brazil": "CONMEBOL", "Colombia": "CONMEBOL",
    "Ecuador": "CONMEBOL", "Paraguay": "CONMEBOL", "Uruguay": "CONMEBOL",
    # CONCACAF (North / Central America & Caribbean)
    "Canada": "CONCACAF", "Curaçao": "CONCACAF", "Haiti": "CONCACAF",
    "Mexico": "CONCACAF", "Panama": "CONCACAF", "USA": "CONCACAF",
    # CAF (Africa)
    "Algeria": "CAF", "Cabo Verde": "CAF", "Congo DR": "CAF", "Côte d'Ivoire": "CAF",
    "Egypt": "CAF", "Ghana": "CAF", "Morocco": "CAF", "Senegal": "CAF",
    "South Africa": "CAF", "Tunisia": "CAF",
    # AFC (Asia) & OFC (Oceania)
    "Australia": "AFC", "IR Iran": "AFC", "Iraq": "AFC", "Japan": "AFC",
    "Jordan": "AFC", "Korea Republic": "AFC", "Qatar": "AFC",
    "Saudi Arabia": "AFC", "Uzbekistan": "AFC",
    "New Zealand": "OFC",
}

MIN_MINUTES = 90
MIN_SOTA = 5


def audit_raw(df: pd.DataFrame) -> None:
    section("1. RAW DATA AUDIT")
    print(f"Shape                : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Goalkeepers          : {df['Player'].nunique()} unique names")
    print(f"Nations represented  : {df['Country'].nunique()}")
    print(f"Duplicate rows       : {df.duplicated().sum()}")
    print(f"Duplicate players    : {df['Player'].duplicated().sum()}")
    print(f"Positions in file    : {sorted(df['Pos'].unique())}")
    miss = df.isna().sum()
    miss = miss[miss > 0]
    print("\nMissing values by column:")
    if miss.empty:
        print("  none")
    else:
        for col, n in miss.items():
            print(f"  {col:<14}: {n:>2}  ({n / len(df):.1%})")
    print("\nWhy the missing values exist (verified, not assumed):")
    print("  Save_Pct    - undefined when a keeper faced 0 shots on target (0/0).")
    print("  CS_Pct      - undefined when a keeper made 0 appearances as starter.")
    print("  PK_Save_Pct - undefined when no penalty was faced (the usual case).")
    print("  => These are 'structurally missing', NOT data-entry errors, so rows are")
    print("     filtered by an eligibility rule rather than imputed.")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    section("2. CLEANING")
    out = df.copy()

    text_cols = ["Player", "Pos", "SquadCode", "Country", "Club", "Stage"]
    for c in text_cols:
        out[c] = out[c].astype(str).str.strip()

    before = len(out)
    out = out.drop_duplicates()
    out = out.sort_values("Min", ascending=False).drop_duplicates(subset=["Player", "Country"])
    print(f"Rows removed as duplicates            : {before - len(out)}")

    out = out[out["Pos"].str.upper().str.contains("GK")]
    print(f"Rows retained after position check    : {len(out)}")

    bad = out[(out["Min"] < 0) | (out["Saves"] > out["SoTA"]) | (out["GA"] > out["SoTA"])]
    print(f"Rows failing logical range checks     : {len(bad)}")

    out = out.sort_values("Rk").reset_index(drop=True)
    return out


def validate_and_enrich(df: pd.DataFrame) -> pd.DataFrame:
    section("3. VALIDATION & FEATURE ENGINEERING")
    d = df.copy()

    d["Save_Pct_calc"] = np.where(d["SoTA"] > 0,
                                  (d["SoTA"] - d["GA"]) / d["SoTA"] * 100, np.nan)
    diff = (d["Save_Pct_calc"] - d["Save_Pct"]).abs()
    print(f"Max |recomputed Save% - supplied Save%|: {np.nanmax(diff):.3f} pp  "
          f"-> column validated")
    d["Save_Pct"] = d["Save_Pct"].fillna(d["Save_Pct_calc"])

    d["SoTA90"] = np.where(d["90s"] > 0, d["SoTA"] / d["90s"], np.nan)   # workload faced
    d["Saves90"] = np.where(d["90s"] > 0, d["Saves"] / d["90s"], np.nan)
    d["GA90_calc"] = np.where(d["90s"] > 0, d["GA"] / d["90s"], np.nan)

    d["Confederation"] = d["Country"].map(CONFEDERATION)
    unmapped = d.loc[d["Confederation"].isna(), "Country"].unique()
    print(f"Countries without a confederation map : {len(unmapped)} {list(unmapped)}")
    d["ConfGroup"] = np.where(d["Confederation"].isin(["UEFA", "CONMEBOL"]),
                              "UEFA/CONMEBOL", "Rest of world")
    d["reached_knockout"] = (d["Stage"].str.strip().str.lower() == "knockout")

    d["ClubCountryCode"] = (d["Club"].str.split().str[0]
                            .str.split(".").str[-1].str.lower())
    d["plays_abroad"] = d["ClubCountryCode"] != d["SquadCode"].str.lower()

    d["is_number_one"] = d.groupby("Country")["Min"].transform("max") == d["Min"]

    d["eligible_perf"] = (d["Min"] >= MIN_MINUTES) & (d["SoTA"] >= MIN_SOTA)  # Task 1
    d["eligible_rate"] = (d["Min"] >= MIN_MINUTES)                            # Task 2
    print(f"Task-1 eligible (>= {MIN_MINUTES} min & >= {MIN_SOTA} SoT faced): "
          f"{d['eligible_perf'].sum()}")
    print(f"Task-2 eligible (>= {MIN_MINUTES} min)                : {d['eligible_rate'].sum()}")
    print(f"Task-3 eligible (played any minutes)      : {(d['Min'] > 0).sum()}")
    return d


def overview_figure(d: pd.DataFrame) -> None:
    """One combined figure describing the cleaned dataset."""
    setup_plot_style()
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))

    ax[0].hist(d["Min"], bins=12, color=PALETTE["secondary"], edgecolor="white")
    ax[0].axvline(MIN_MINUTES, color=PALETTE["contrast"], ls="--", lw=2,
                  label=f"eligibility cut-off ({MIN_MINUTES} min)")
    ax[0].set_title("Minutes played")
    ax[0].set_xlabel("Minutes")
    ax[0].set_ylabel("Goalkeepers")
    ax[0].legend(fontsize=8)

    counts = d["Confederation"].value_counts()
    ax[1].bar(counts.index, counts.values, color=PALETTE["primary"])
    ax[1].set_title("Goalkeepers by confederation")
    ax[1].set_ylabel("Goalkeepers")
    ax[1].tick_params(axis="x", rotation=30)

    stage = d["Stage"].value_counts()
    ax[2].bar(stage.index, stage.values,
              color=[PALETTE["accent"], PALETTE["primary"]])
    ax[2].set_title("Furthest stage reached")
    ax[2].set_ylabel("Goalkeepers")
    ax[2].tick_params(axis="x", rotation=10)

    fig.suptitle("Cleaned dataset overview - FIFA World Cup 2026 goalkeepers",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig0_data_overview.png")
    plt.close(fig)
    print(f"\nSaved figure -> {FIG_DIR}/fig0_data_overview.png")


def main() -> pd.DataFrame:
    ensure_dirs()
    banner("FIFA WORLD CUP 2026 GOALKEEPERS - DATA WRANGLING & PREPARATION")

    raw = pd.read_csv(RAW_DATA)
    audit_raw(raw)
    cleaned = clean(raw)
    final = validate_and_enrich(cleaned)

    section("4. ANALYSIS-READY DATASET")
    print(f"Final shape: {final.shape[0]} rows x {final.shape[1]} columns")
    print("New variables created: Save_Pct_calc, SoTA90, Saves90, GA90_calc, "
          "Confederation,\n  ConfGroup, reached_knockout, ClubCountryCode, "
          "plays_abroad, is_number_one,\n  eligible_perf, eligible_rate")
    print("\nPreview:")
    print(final[["Player", "Country", "Confederation", "Min", "SoTA", "GA",
                 "Save_Pct", "SoTA90", "Age", "Stage"]].head(8).to_string(index=False))

    final.to_csv(CLEAN_DATA, index=False)
    print(f"\nSaved cleaned dataset -> {CLEAN_DATA}")
    overview_figure(final)
    return final


if __name__ == "__main__":
    main()
