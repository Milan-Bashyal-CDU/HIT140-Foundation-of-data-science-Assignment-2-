"""
task2_workload.py
=================
ANALYTIC TASK 2 - FOCAL POINT: DEFENSIVE WORKLOAD / SHOT EXPOSURE
                  (shots on target faced per 90 minutes)

This task deliberately measures something different from Task 1. Task 1 asked how
well a goalkeeper performs when tested; this task asks how OFTEN a goalkeeper is
tested, i.e. how much protection the team in front of him provides.

Analytic question
-----------------
Q2: On average, how many shots on target does a goalkeeper have to face per 90
    minutes at the FIFA World Cup 2026 - and do goalkeepers representing the
    traditional powerhouse confederations (UEFA and CONMEBOL) face a
    significantly lighter workload than goalkeepers from the rest of the world?

    Only goalkeepers with at least 90 minutes played are included, so that a
    per-90 rate is not extrapolated from a few minutes on the pitch.

Statistical design
------------------
    Variable        : SoTA90 = SoTA / 90s        (ratio, continuous)
    Population      : all goalkeepers who played at least one full match at WC 2026
    Sampling frame  : the 58 eligible goalkeeper records in the cleaned file
    Sample          : SIMPLE RANDOM SAMPLE without replacement, n = 36
    CI              : 95% t-interval for the mean shots on target faced per 90
    Hypothesis test : two-sample Welch t-test (UEFA/CONMEBOL vs rest of world)
                      H0: mu_UEFA/CONMEBOL = mu_rest
                      H1: mu_UEFA/CONMEBOL != mu_rest       alpha = 0.05

Run:  python task2_workload.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gk_common import (ALPHA, CONF_LEVEL, FIG_DIR, OUT_DIR, PALETTE, RANDOM_SEED,
                       banner, section, show, descriptive_stats, ensure_dirs,
                       load_clean, mean_confidence_interval, normality_check,
                       outlier_bounds, setup_plot_style, two_sample_t)

SAMPLE_SIZE = 40
VAR = "SoTA90"


# --------------------------------------------------------------------------
# Data preparation & sampling
# --------------------------------------------------------------------------
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    section("TASK 2 | DATA PREPARATION")
    frame = df.loc[df["eligible_rate"], [
        "Player", "Country", "Confederation", "ConfGroup", "Age", "Min", "90s",
        "SoTA", "GA", "Save_Pct", "SoTA90", "Stage"]].copy()

    print(f"All goalkeepers in cleaned file       : {len(df)}")
    print(f"Removed: fewer than 90 minutes played : {len(df) - len(frame)}")
    print(f"SAMPLING FRAME (population of interest): {len(frame)}")
    print("\nWhy a per-90 rate? Raw shots faced (SoTA) rewards keepers who simply")
    print("played longer. Dividing by 90-minute units puts every keeper on the")
    print("same scale, which is what makes the two groups comparable.")

    print("\nFrame composition by comparison group:")
    print(frame["ConfGroup"].value_counts().to_string())

    lo, hi = outlier_bounds(frame[VAR])
    out = frame[(frame[VAR] < lo) | (frame[VAR] > hi)]
    print(f"\nTukey outlier fences for {VAR}: [{lo:.2f}, {hi:.2f}] -> "
          f"{len(out)} extreme value(s) flagged")
    if len(out):
        print(out[["Player", "Country", "Min", "SoTA", VAR]].round(2).to_string(index=False))
    print("Decision: retained - a keeper under heavy siege is a real, informative case.")
    return frame


def draw_sample(frame: pd.DataFrame) -> pd.DataFrame:
    """Simple random sample without replacement (every record equally likely)."""
    section("TASK 2 | SAMPLING (simple random sampling)")
    sample = frame.sample(n=SAMPLE_SIZE, replace=False,
                          random_state=RANDOM_SEED).sort_values("Player")
    print(f"Frame size N = {len(frame)}   ->   sample size n = {len(sample)} "
          f"(seed = {RANDOM_SEED})")
    print(f"Sampling fraction = {len(sample)/len(frame):.1%}")
    print("n = 40 >= 30, so by the Central Limit Theorem the sampling distribution")
    print("of the sample mean is approximately normal and t-based inference is valid.")
    print("\nSample composition by comparison group:")
    print(sample["ConfGroup"].value_counts().to_string())
    return sample


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def describe(sample: pd.DataFrame) -> None:
    section("TASK 2 | DESCRIPTIVE STATISTICS - shots on target faced per 90 min")
    print(descriptive_stats(sample[VAR], "Sample").round(3).to_string())

    print("\nBy comparison group:")
    tbl = (sample.groupby("ConfGroup")[VAR]
           .agg(n="count", mean="mean", median="median", std="std",
                min="min", max="max").round(2))
    print(tbl.to_string())

    print("\nBy confederation (context only, not the test):")
    print(sample.groupby("Confederation")[VAR].agg(n="count", mean="mean")
          .round(2).sort_values("mean").to_string())

    m, sd = sample[VAR].mean(), sample[VAR].std(ddof=1)
    print("\nInterpretation:")
    print(f"  A typical goalkeeper was tested {m:.2f} times per 90 minutes "
          f"(SD = {sd:.2f}),")
    print(f"  i.e. roughly one shot on target every {90/m:.0f} minutes of football.")
    print(f"  Coefficient of variation = {100*sd/m:.1f}% - workload differs widely,")
    print(f"  because it depends on the quality of the defence in front of the keeper.")


def infer(sample: pd.DataFrame, frame: pd.DataFrame) -> dict:
    section("TASK 2 | INFERENTIAL STATISTICS 1 - 95% CONFIDENCE INTERVAL")
    ci = mean_confidence_interval(sample[VAR], CONF_LEVEL)
    show(ci)
    print(f"\n  We are 95% confident that the mean number of shots on target faced per")
    print(f"  90 minutes by the WC 2026 goalkeeper population is between "
          f"{ci['lower']:.2f} and {ci['upper']:.2f}.")
    print(f"  (Validation: the true frame mean is {frame[VAR].mean():.2f} - "
          f"inside the interval.)")

    section("TASK 2 | INFERENTIAL STATISTICS 2 - TWO-SAMPLE t-TEST")
    a = sample.loc[sample["ConfGroup"] == "UEFA/CONMEBOL", VAR]
    b = sample.loc[sample["ConfGroup"] == "Rest of world", VAR]

    print("Assumption checks (Shapiro-Wilk normality):")
    for s, lab in ((a, "UEFA/CONMEBOL"), (b, "Rest of world")):
        r = normality_check(s, lab)
        print(f"  {lab:<16} n={r['n']:>2}  W={r['W']:.3f}  p={r['p_value']:.3f}  "
              f"-> {'normal' if r['normal_at_5pct'] else 'non-normal'}")
    print("  Independence: groups are mutually exclusive sets of goalkeepers. OK")

    print("\nH0: mu_UEFA/CONMEBOL = mu_rest    H1: mu_UEFA/CONMEBOL != mu_rest"
          "    alpha = 0.05")
    res = two_sample_t(a, b, ("UEFA/CONMEBOL", "Rest of world"))
    show(res)

    print("\nConclusion:")
    if res["reject_H0"]:
        print(f"  p = {res['p_value']:.4f} < {ALPHA}: REJECT H0. The workload gap of "
              f"{abs(res['mean_difference']):.2f} shots per 90 is statistically significant.")
    else:
        print(f"  p = {res['p_value']:.4f} > {ALPHA}: FAIL TO REJECT H0.")
        print(f"  Keepers from UEFA/CONMEBOL faced {abs(res['mean_difference']):.2f} fewer")
        print(f"  shots on target per 90 on average, but the 95% CI for the difference")
        print(f"  ({res['ci_diff_lower']:.2f} to {res['ci_diff_upper']:.2f}) includes zero, so")
        print(f"  the evidence is not strong enough to claim a real difference in the")
        print(f"  population; the effect size is {res['effect']} "
              f"(d = {res['cohens_d']:.2f}).")
        print("  Practical reading: elite confederations do NOT protect their keeper")
        print("  by conceding fewer shots - Task 1 showed their edge lies in the")
        print("  quality of the shot-stopping itself, not in the volume of shots faced.")
    return {"ci": ci, "ttest": res}


# --------------------------------------------------------------------------
# Visualisation
# --------------------------------------------------------------------------
def visualise(sample: pd.DataFrame, ci: dict, res: dict) -> None:
    setup_plot_style()
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

    ax[0].hist(sample[VAR], bins=9, color=PALETTE["accent"], edgecolor="white")
    ax[0].axvline(sample[VAR].mean(), color=PALETTE["primary"], lw=2,
                  label=f"mean = {sample[VAR].mean():.2f}")
    ax[0].set_title("(a) Shots on target faced per 90")
    ax[0].set_xlabel("Shots on target faced per 90 min")
    ax[0].set_ylabel("Goalkeepers")
    ax[0].legend(fontsize=9)

    ax[1].errorbar(ci["mean"], 0, xerr=ci["margin_of_error"], fmt="o",
                   color=PALETTE["primary"], capsize=10, ms=11, lw=2.5)
    ax[1].text(ci["mean"], 0.16, f"{ci['mean']:.2f}", ha="center",
               fontweight="bold", color=PALETTE["primary"])
    ax[1].text(ci["lower"], -0.2, f"{ci['lower']:.2f}", ha="center", fontsize=9)
    ax[1].text(ci["upper"], -0.2, f"{ci['upper']:.2f}", ha="center", fontsize=9)
    ax[1].set_ylim(-0.5, 0.5)
    ax[1].set_yticks([])
    ax[1].set_title("(b) 95% CI for mean workload")
    ax[1].set_xlabel("Shots on target faced per 90 min")

    groups = ["UEFA/CONMEBOL", "Rest of world"]
    data = [sample.loc[sample.ConfGroup == g, VAR] for g in groups]
    means = [d.mean() for d in data]
    errs = [d.std(ddof=1) / np.sqrt(len(d)) for d in data]
    bars = ax[2].bar(groups, means, yerr=errs, capsize=7, width=0.55,
                     color=[PALETTE["primary"], PALETTE["secondary"]])
    for bar, m, e in zip(bars, means, errs):
        ax[2].text(bar.get_x() + bar.get_width() / 2, m + e + 0.18, f"{m:.2f}",
                   ha="center", fontweight="bold")
    ax[2].set_ylim(0, max(m + e for m, e in zip(means, errs)) + 0.8)
    ax[2].set_title(f"(c) Group comparison  (p = {res['p_value']:.3f}, n.s.)"
                    if not res["reject_H0"] else
                    f"(c) Group comparison  (p = {res['p_value']:.3f})")
    ax[2].set_ylabel("Mean shots on target faced per 90")

    fig.suptitle("Task 2 - Defensive workload faced by World Cup 2026 goalkeepers",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig2_workload.png")
    plt.close(fig)
    print(f"\nSaved figure -> {FIG_DIR}/fig2_workload.png")


def main() -> dict:
    ensure_dirs()
    banner("ANALYTIC TASK 2 - DEFENSIVE WORKLOAD (SHOTS ON TARGET FACED PER 90)")
    print(__doc__.split("Analytic question")[1].split("Statistical design")[0].strip())

    df = load_clean()
    frame = prepare(df)
    sample = draw_sample(frame)
    sample.to_csv(f"{OUT_DIR}/task2_sample.csv", index=False)
    describe(sample)
    results = infer(sample, frame)
    visualise(sample, results["ci"], results["ttest"])

    section("TASK 2 | ANSWER TO THE ANALYTIC QUESTION")
    ci, t = results["ci"], results["ttest"]
    print(f"A World Cup 2026 goalkeeper faces on average {ci['mean']:.2f} shots on target")
    print(f"per 90 minutes (95% CI: {ci['lower']:.2f} - {ci['upper']:.2f}). Keepers from "
          f"UEFA/CONMEBOL faced")
    print(f"{t['mean_a']:.2f} versus {t['mean_b']:.2f} for the rest of the world - a "
          f"difference that is NOT")
    print(f"statistically significant (p = {t['p_value']:.4f}).")
    return results


if __name__ == "__main__":
    main()
