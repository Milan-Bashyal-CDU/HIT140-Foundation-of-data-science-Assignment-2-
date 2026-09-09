"""
task3_age_profile.py
====================
ANALYTIC TASK 3 - FOCAL POINT: THE AGE (CAREER-STAGE) PROFILE OF THE POSITION

Tasks 1 and 2 examined what happens on the pitch. This task examines WHO the
nations pick: goalkeeping is widely described as a position where experience
outweighs athletic peak, so the squad selections themselves should look old
compared with the rest of the squad.

Analytic question
-----------------
Q3: What is the average age of the goalkeepers who took the field at the FIFA
    World Cup 2026, and is that mean age significantly different from 27 years -
    the age commonly used as the peak-performance age of an outfield
    professional footballer and the typical mean age of a World Cup squad?

Statistical design
------------------
    Variable        : Age in completed years at the tournament (ratio, discrete)
    Population      : all goalkeepers who appeared at FIFA World Cup 2026
    Sampling frame  : the 62 goalkeeper records in the cleaned file
                      (no minutes filter: age does not depend on playing time)
    Sample          : SYSTEMATIC random sample, every k-th record, n = 31
    Benchmark mu0   : 27 years (peak age of an outfield footballer)
    CI              : 95% t-interval for the mean age
    Hypothesis test : one-sample t-test
                      H0: mu = 27      H1: mu != 27        alpha = 0.05

Run:  python task3_age_profile.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from gk_common import (ALPHA, CONF_LEVEL, FIG_DIR, OUT_DIR, PALETTE, RANDOM_SEED,
                       banner, section, show, descriptive_stats, ensure_dirs,
                       load_clean, mean_confidence_interval, normality_check,
                       one_sample_t, outlier_bounds, setup_plot_style)

SAMPLE_SIZE = 31          # >= 30 for the Central Limit Theorem
BENCHMARK_AGE = 27.0      # mu0: conventional peak age of an outfield footballer
VAR = "Age"


# --------------------------------------------------------------------------
# Data preparation & sampling
# --------------------------------------------------------------------------
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    section("TASK 3 | DATA PREPARATION")
    frame = df.loc[df["Min"] > 0, [
        "Player", "Country", "Confederation", "Club", "Age", "Born", "Min",
        "MP", "Save_Pct", "plays_abroad", "is_number_one", "Stage"]].copy()

    print(f"All goalkeepers in cleaned file        : {len(df)}")
    print(f"Removed: never took the field          : {len(df) - len(frame)}")
    print(f"SAMPLING FRAME (population of interest): {len(frame)}")
    print("\nNote on the frame: unlike Tasks 1 and 2 no minutes threshold is applied,")
    print("because age is a property of the player, not of his playing time. Every")
    print("goalkeeper who appeared is therefore in scope.")

    # cross-check age against year of birth (data-quality control)
    implied = 2026 - frame["Born"]
    mismatch = (implied - frame["Age"]).abs() > 1
    print(f"\nConsistency check Age vs (2026 - Born): {mismatch.sum()} mismatches "
          f"of more than 1 year")

    lo, hi = outlier_bounds(frame[VAR])
    out = frame[(frame[VAR] < lo) | (frame[VAR] > hi)]
    print(f"Tukey outlier fences for {VAR}: [{lo:.1f}, {hi:.1f}] -> "
          f"{len(out)} extreme value(s)")
    if len(out):
        print(out[["Player", "Country", "Age"]].to_string(index=False))
    return frame


def draw_sample(frame: pd.DataFrame) -> pd.DataFrame:
    """Systematic sampling: random start, then every k-th record of the frame."""
    section("TASK 3 | SAMPLING (systematic random sampling)")
    rng = np.random.RandomState(RANDOM_SEED)
    N = len(frame)
    k = N // SAMPLE_SIZE                     # sampling interval
    start = rng.randint(0, k)                # random start within the first interval
    idx = np.arange(start, N, k)[:SAMPLE_SIZE]
    sample = frame.iloc[idx].copy()

    print(f"Frame size N = {N}")
    print(f"Sampling interval k = N / n = {N} / {SAMPLE_SIZE} = {k}")
    print(f"Random start = record {start + 1}; then every {k}nd record is taken.")
    print(f"Sample size n = {len(sample)}   (seed = {RANDOM_SEED}, reproducible)")
    print("The frame is ordered alphabetically by player name, which is unrelated to")
    print("age, so systematic selection carries no periodicity bias here and behaves")
    print("like a simple random sample while being easier to audit.")
    return sample


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def describe(sample: pd.DataFrame) -> None:
    section("TASK 3 | DESCRIPTIVE STATISTICS - age (years)")
    print(descriptive_stats(sample[VAR], "Sample").round(3).to_string())

    print("\nAge bands:")
    bands = pd.cut(sample[VAR], bins=[21, 25, 29, 33, 41],
                   labels=["22-25", "26-29", "30-33", "34+"])
    tbl = bands.value_counts().sort_index().to_frame("goalkeepers")
    tbl["share"] = (tbl["goalkeepers"] / len(sample)).map("{:.1%}".format)
    print(tbl.to_string())

    print("\nContext (descriptive only):")
    print(sample.groupby("is_number_one")[VAR].agg(n="count", mean_age="mean")
          .round(2).rename(index={True: "Nation's No.1", False: "Understudy"})
          .to_string())

    m, sd = sample[VAR].mean(), sample[VAR].std(ddof=1)
    print("\nInterpretation:")
    print(f"  Mean age {m:.2f} years, median {sample[VAR].median():.1f}, "
          f"SD {sd:.2f} years.")
    print(f"  Range {sample[VAR].min():.0f}-{sample[VAR].max():.0f} years; "
          f"skewness {sample[VAR].skew():.2f}.")
    print(f"  {(sample[VAR] >= 30).mean():.0%} of the sampled goalkeepers were 30 or older.")


def infer(sample: pd.DataFrame, frame: pd.DataFrame) -> dict:
    section("TASK 3 | INFERENTIAL STATISTICS 1 - 95% CONFIDENCE INTERVAL")
    ci = mean_confidence_interval(sample[VAR], CONF_LEVEL)
    show(ci)
    print(f"\n  We are 95% confident that the mean age of the World Cup 2026")
    print(f"  goalkeeper population is between {ci['lower']:.2f} and "
          f"{ci['upper']:.2f} years.")
    print(f"  (Validation: the true frame mean is {frame[VAR].mean():.2f} years - "
          f"inside the interval.)")
    print(f"  Note the benchmark of {BENCHMARK_AGE:.0f} years lies well BELOW the whole")
    print(f"  interval, which already anticipates the outcome of the test below.")

    section("TASK 3 | INFERENTIAL STATISTICS 2 - ONE-SAMPLE t-TEST")
    r = normality_check(sample[VAR], "Age")
    print(f"Assumption check (Shapiro-Wilk): W = {r['W']:.3f}, p = {r['p_value']:.3f} "
          f"-> {'normality supported' if r['normal_at_5pct'] else 'departure from normality'}")
    print("Sample size n = 31 >= 30, so the CLT also supports the use of the t-test.")
    print("Independence: one record per goalkeeper, selected by a random-start scheme. OK")

    print(f"\nH0: mu = {BENCHMARK_AGE:.0f} years    H1: mu != {BENCHMARK_AGE:.0f} years"
          f"    alpha = {ALPHA}")
    res = one_sample_t(sample[VAR], BENCHMARK_AGE)
    show(res)

    print("\nConclusion:")
    if res["reject_H0"]:
        print(f"  p = {res['p_value']:.6f} < {ALPHA}: REJECT H0. World Cup 2026 goalkeepers")
        print(f"  are on average {res['mean'] - BENCHMARK_AGE:.1f} years older than the "
              f"{BENCHMARK_AGE:.0f}-year benchmark,")
        print(f"  a {res['effect']} effect (Cohen's d = {res['cohens_d']:.2f}).")
        print("  This supports the football view that goalkeeping rewards accumulated")
        print("  experience, reading of the game and command of the area far longer")
        print("  than outfield positions reward physical peak.")
    else:
        print(f"  p = {res['p_value']:.4f} > {ALPHA}: FAIL TO REJECT H0.")
    return {"ci": ci, "ttest": res}


# --------------------------------------------------------------------------
# Visualisation
# --------------------------------------------------------------------------
def visualise(sample: pd.DataFrame, ci: dict, res: dict) -> None:
    setup_plot_style()
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

    # (a) age distribution vs benchmark
    ax[0].hist(sample[VAR], bins=np.arange(21.5, 41.5, 2),
               color=PALETTE["primary"], edgecolor="white")
    ax[0].axvline(BENCHMARK_AGE, color=PALETTE["contrast"], ls="--", lw=2,
                  label=f"benchmark = {BENCHMARK_AGE:.0f} yrs")
    ax[0].axvline(sample[VAR].mean(), color=PALETTE["accent"], lw=2.5,
                  label=f"sample mean = {sample[VAR].mean():.1f} yrs")
    ax[0].set_title("(a) Age distribution")
    ax[0].set_xlabel("Age (years)")
    ax[0].set_ylabel("Goalkeepers")
    ax[0].legend(fontsize=8.5)

    # (b) CI vs benchmark
    ax[1].errorbar(ci["mean"], 0, xerr=ci["margin_of_error"], fmt="o",
                   color=PALETTE["primary"], capsize=10, ms=11, lw=2.5,
                   label="95% CI for mean age")
    ax[1].axvline(BENCHMARK_AGE, color=PALETTE["contrast"], ls="--", lw=2,
                  label=f"H0: mu = {BENCHMARK_AGE:.0f}")
    ax[1].text(ci["mean"], 0.18, f"{ci['mean']:.1f}", ha="center",
               fontweight="bold", color=PALETTE["primary"])
    ax[1].set_xlim(25.5, 34)
    ax[1].set_ylim(-0.5, 0.5)
    ax[1].set_yticks([])
    ax[1].set_title("(b) 95% CI excludes the benchmark")
    ax[1].set_xlabel("Age (years)")
    ax[1].legend(fontsize=8.5, loc="lower right")

    # (c) t-distribution with the observed statistic
    dfree = res["df"]
    xs = np.linspace(-8, 8, 500)
    ax[2].plot(xs, stats.t.pdf(xs, dfree), color=PALETTE["primary"], lw=2)
    crit = stats.t.ppf(1 - ALPHA / 2, dfree)
    ax[2].fill_between(xs, 0, stats.t.pdf(xs, dfree), where=np.abs(xs) >= crit,
                       color=PALETTE["contrast"], alpha=0.5,
                       label=f"rejection region (+/-{crit:.2f})")
    ax[2].axvline(res["t_stat"], color=PALETTE["accent"], lw=2.5,
                  label=f"observed t = {res['t_stat']:.2f}")
    ax[2].set_title(f"(c) One-sample t-test  (p < 0.001)")
    ax[2].set_xlabel(f"t (df = {dfree:.0f})")
    ax[2].set_ylabel("Density")
    ax[2].legend(fontsize=8.5)

    fig.suptitle("Task 3 - Age profile of World Cup 2026 goalkeepers",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig3_age_profile.png")
    plt.close(fig)
    print(f"\nSaved figure -> {FIG_DIR}/fig3_age_profile.png")


def main() -> dict:
    ensure_dirs()
    banner("ANALYTIC TASK 3 - AGE / CAREER-STAGE PROFILE OF GOALKEEPERS")
    print(__doc__.split("Analytic question")[1].split("Statistical design")[0].strip())

    df = load_clean()
    frame = prepare(df)
    sample = draw_sample(frame)
    sample.to_csv(f"{OUT_DIR}/task3_sample.csv", index=False)
    describe(sample)
    results = infer(sample, frame)
    visualise(sample, results["ci"], results["ttest"])

    section("TASK 3 | ANSWER TO THE ANALYTIC QUESTION")
    ci, t = results["ci"], results["ttest"]
    print(f"The mean age of a World Cup 2026 goalkeeper is estimated at "
          f"{ci['mean']:.1f} years")
    print(f"(95% CI: {ci['lower']:.1f} - {ci['upper']:.1f}), significantly older than the "
          f"{BENCHMARK_AGE:.0f}-year")
    print(f"benchmark for an outfield professional (t = {t['t_stat']:.2f}, "
          f"p = {t['p_value']:.6f}, d = {t['cohens_d']:.2f}).")
    return results


if __name__ == "__main__":
    main()
