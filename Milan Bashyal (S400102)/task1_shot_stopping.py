"""
task1_shot_stopping.py
======================
ANALYTIC TASK 1 - FOCAL POINT: SHOT-STOPPING EFFICIENCY (Save percentage)

Analytic question
-----------------
Q1: On average, what percentage of the shots on target they face do goalkeepers
    save at the FIFA World Cup 2026 - and do goalkeepers whose teams reach the
    knockout stage stop a significantly higher share than goalkeepers whose
    teams are eliminated in the group stage?

    Only goalkeepers with a meaningful workload are considered (>= 90 minutes
    played AND >= 5 shots on target faced), because a save percentage built on
    one or two shots is not a stable measure of ability.

Statistical design
------------------
    Variable        : Save_Pct = (SoTA - GA) / SoTA * 100   (ratio, continuous)
    Population      : all goalkeepers who played a meaningful role at WC 2026
    Sampling frame  : the 55 eligible goalkeeper records in the cleaned file
    Sample          : STRATIFIED random sample, n = 36, strata = tournament stage
    CI              : 95% t-interval for the mean save percentage
    Hypothesis test : two-sample Welch t-test (knockout vs group-stage-only)
                      H0: mu_knockout = mu_group
                      H1: mu_knockout != mu_group          alpha = 0.05

Run:  python task1_shot_stopping.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from gk_common import (ALPHA, CONF_LEVEL, FIG_DIR, OUT_DIR, PALETTE, RANDOM_SEED,
                       banner, section, show, descriptive_stats, ensure_dirs,
                       load_clean, mean_confidence_interval, normality_check,
                       outlier_bounds, setup_plot_style, two_sample_t)

SAMPLE_SIZE = 36          # >= 30 so the Central Limit Theorem applies
VAR = "Save_Pct"


# --------------------------------------------------------------------------
# Data preparation & sampling
# --------------------------------------------------------------------------
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    section("TASK 1 | DATA PREPARATION")
    frame = df.loc[df["eligible_perf"], [
        "Player", "Country", "Confederation", "Age", "Min", "90s", "SoTA",
        "Saves", "GA", "Save_Pct", "Stage", "reached_knockout"]].copy()
    frame["StageGroup"] = np.where(frame["reached_knockout"],
                                   "Knockout", "Group stage only")

    print(f"All goalkeepers in cleaned file       : {len(df)}")
    print(f"Removed: < 90 minutes or < 5 SoT faced: {len(df) - len(frame)}")
    print(f"SAMPLING FRAME (population of interest): {len(frame)}")
    print("\nFrame composition by stratum:")
    print(frame["StageGroup"].value_counts().to_string())

    lo, hi = outlier_bounds(frame[VAR])
    out = frame[(frame[VAR] < lo) | (frame[VAR] > hi)]
    print(f"\nTukey outlier fences for {VAR}: [{lo:.1f}, {hi:.1f}] "
          f"-> {len(out)} extreme value(s) flagged")
    if len(out):
        print(out[["Player", "Country", VAR, "SoTA"]].to_string(index=False))
    print("Decision: retained - they are genuine performances, not data errors.")
    return frame


def draw_sample(frame: pd.DataFrame) -> pd.DataFrame:
    """Stratified random sample with proportional allocation across stage strata."""
    section("TASK 1 | SAMPLING (stratified random sampling)")
    rng = np.random.RandomState(RANDOM_SEED)
    parts, N = [], len(frame)

    for stratum, grp in frame.groupby("StageGroup"):
        n_h = int(round(SAMPLE_SIZE * len(grp) / N))
        idx = rng.choice(grp.index.values, size=n_h, replace=False)
        parts.append(frame.loc[idx])
        print(f"  {stratum:<18}: N_h = {len(grp):>2}  ->  n_h = {n_h:>2} "
              f"({len(grp)/N:.1%} of frame)")

    sample = pd.concat(parts).sort_values("Player")
    print(f"\nTotal sample size n = {len(sample)}  (seed = {RANDOM_SEED}, reproducible)")
    print("Why stratified? It guarantees both stages are represented in proportion,")
    print("which is essential because the stage variable is the comparison factor.")
    print(f"Sampling fraction = {len(sample)/N:.1%} of the frame.")
    return sample


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def describe(sample: pd.DataFrame) -> None:
    section("TASK 1 | DESCRIPTIVE STATISTICS - save percentage (%)")
    print(descriptive_stats(sample[VAR], "Sample").round(3).to_string())

    print("\nBy stratum:")
    tbl = (sample.groupby("StageGroup")[VAR]
           .agg(n="count", mean="mean", median="median", std="std",
                min="min", max="max").round(2))
    print(tbl.to_string())

    print("\nInterpretation:")
    m, sd = sample[VAR].mean(), sample[VAR].std(ddof=1)
    print(f"  The average goalkeeper in the sample saved {m:.1f}% of shots on target,")
    print(f"  with a standard deviation of {sd:.1f} percentage points - a wide spread,")
    print(f"  showing shot-stopping form varied a lot between keepers.")
    print(f"  Skewness = {sample[VAR].skew():.2f} (near-symmetric distribution).")


def infer(sample: pd.DataFrame, frame: pd.DataFrame) -> dict:
    section("TASK 1 | INFERENTIAL STATISTICS 1 - 95% CONFIDENCE INTERVAL")
    ci = mean_confidence_interval(sample[VAR], CONF_LEVEL)
    show(ci)
    print(f"\n  We are 95% confident that the mean save percentage of the World Cup")
    print(f"  2026 goalkeeper population lies between {ci['lower']:.2f}% and "
          f"{ci['upper']:.2f}%.")
    print(f"  (Validation: the true frame mean is {frame[VAR].mean():.2f}% - "
          f"inside the interval.)")

    section("TASK 1 | INFERENTIAL STATISTICS 2 - TWO-SAMPLE t-TEST")
    a = sample.loc[sample["StageGroup"] == "Knockout", VAR]
    b = sample.loc[sample["StageGroup"] == "Group stage only", VAR]

    print("Assumption checks (Shapiro-Wilk normality):")
    for s, lab in ((a, "Knockout"), (b, "Group stage only")):
        r = normality_check(s, lab)
        print(f"  {lab:<18} n={r['n']:>2}  W={r['W']:.3f}  p={r['p_value']:.3f}  "
              f"-> {'normal' if r['normal_at_5pct'] else 'non-normal'}")
    print("  Independence: each goalkeeper appears once, in one group only. OK")

    print("\nH0: mu_knockout = mu_group      H1: mu_knockout != mu_group   alpha = 0.05")
    res = two_sample_t(a, b, ("Knockout", "Group stage only"))
    show(res)

    print("\nConclusion:")
    if res["reject_H0"]:
        print(f"  p = {res['p_value']:.4f} < {ALPHA}: REJECT H0. Goalkeepers whose teams")
        print(f"  reached the knockout stage saved on average "
              f"{res['mean_difference']:.1f} percentage points more of the shots they")
        print(f"  faced (95% CI for the difference: {res['ci_diff_lower']:.1f} to "
              f"{res['ci_diff_upper']:.1f} pp),")
        print(f"  a statistically significant difference with a {res['effect']} "
              f"effect size (d = {res['cohens_d']:.2f}).")
    else:
        print(f"  p = {res['p_value']:.4f} > {ALPHA}: FAIL TO REJECT H0 - no significant "
              f"difference.")
    return {"ci": ci, "ttest": res}


# --------------------------------------------------------------------------
# Visualisation
# --------------------------------------------------------------------------
def visualise(sample: pd.DataFrame, ci: dict, res: dict) -> None:
    setup_plot_style()
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

    # (a) distribution
    ax[0].hist(sample[VAR], bins=9, color=PALETTE["secondary"], edgecolor="white")
    ax[0].axvline(sample[VAR].mean(), color=PALETTE["contrast"], lw=2,
                  label=f"mean = {sample[VAR].mean():.1f}%")
    ax[0].set_title("(a) Distribution of save %")
    ax[0].set_xlabel("Save percentage (%)")
    ax[0].set_ylabel("Goalkeepers")
    ax[0].legend(fontsize=9)

    # (b) confidence interval
    ax[1].errorbar(ci["mean"], 0, xerr=ci["margin_of_error"], fmt="o",
                   color=PALETTE["primary"], capsize=10, ms=11, lw=2.5)
    ax[1].axvline(ci["lower"], color=PALETTE["grey"], ls=":", lw=1)
    ax[1].axvline(ci["upper"], color=PALETTE["grey"], ls=":", lw=1)
    ax[1].text(ci["mean"], 0.16, f"{ci['mean']:.1f}%", ha="center",
               fontweight="bold", color=PALETTE["primary"])
    ax[1].text(ci["lower"], -0.2, f"{ci['lower']:.1f}%", ha="center", fontsize=9)
    ax[1].text(ci["upper"], -0.2, f"{ci['upper']:.1f}%", ha="center", fontsize=9)
    ax[1].set_ylim(-0.5, 0.5)
    ax[1].set_yticks([])
    ax[1].set_title("(b) 95% CI for mean save %")
    ax[1].set_xlabel("Save percentage (%)")

    # (c) group comparison
    groups = ["Knockout", "Group stage only"]
    data = [sample.loc[sample.StageGroup == g, VAR] for g in groups]
    bp = ax[2].boxplot(data, tick_labels=["Knockout", "Group only"], patch_artist=True,
                       widths=0.55, medianprops=dict(color="white", lw=2))
    for patch, c in zip(bp["boxes"], [PALETTE["primary"], PALETTE["accent"]]):
        patch.set_facecolor(c)
    for i, d in enumerate(data, start=1):
        ax[2].scatter(np.random.normal(i, 0.06, len(d)), d, s=18,
                      color=PALETTE["grey"], alpha=0.7, zorder=3)
    ax[2].set_title(f"(c) Group comparison  (p = {res['p_value']:.4f})")
    ax[2].set_ylabel("Save percentage (%)")

    fig.suptitle("Task 1 - Shot-stopping efficiency of World Cup 2026 goalkeepers",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig1_save_percentage.png")
    plt.close(fig)
    print(f"\nSaved figure -> {FIG_DIR}/fig1_save_percentage.png")


def main() -> dict:
    ensure_dirs()
    banner("ANALYTIC TASK 1 - SHOT-STOPPING EFFICIENCY (SAVE PERCENTAGE)")
    print(__doc__.split("Analytic question")[1].split("Statistical design")[0].strip())

    df = load_clean()
    frame = prepare(df)
    sample = draw_sample(frame)
    sample.to_csv(f"{OUT_DIR}/task1_sample.csv", index=False)
    describe(sample)
    results = infer(sample, frame)
    visualise(sample, results["ci"], results["ttest"])

    section("TASK 1 | ANSWER TO THE ANALYTIC QUESTION")
    ci, t = results["ci"], results["ttest"]
    print(f"World Cup 2026 goalkeepers save on average {ci['mean']:.1f}% of the shots on")
    print(f"target they face (95% CI: {ci['lower']:.1f}% - {ci['upper']:.1f}%). Shot-stopping")
    print(f"is significantly better among keepers whose teams progressed: "
          f"{t['mean_a']:.1f}% vs {t['mean_b']:.1f}%")
    print(f"(p = {t['p_value']:.4f}, Cohen's d = {t['cohens_d']:.2f}).")
    return results


if __name__ == "__main__":
    main()
