"""
gk_common.py
============
Shared utilities for the FIFA World Cup 2026 goalkeeper analysis.

Contains the statistical and plotting helpers that are re-used by the three
analytic task scripts so that every task applies *identical* methodology:

    * descriptive_stats()      -> descriptive statistics table
    * mean_confidence_interval -> t-based confidence interval for a mean
    * one_sample_t()           -> one-sample t-test + Cohen's d
    * two_sample_t()           -> Welch two-sample t-test + Cohen's d + CI of difference
    * normality_check()        -> Shapiro-Wilk assumption check
    * PALETTE / setup_plot_style() -> consistent visual identity across figures

Author : Data Analytics Assignment - Group Project
Python : 3.10+
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")            # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Project-wide constants
# --------------------------------------------------------------------------
RAW_DATA = os.environ.get("GK_RAW", "keepers_2026.csv")
CLEAN_DATA = "output/keepers_clean.csv"
FIG_DIR = "figures"
OUT_DIR = "output"

RANDOM_SEED = 2026          # fixed seed -> every sample is reproducible
CONF_LEVEL = 0.95           # 95% confidence throughout
ALPHA = 0.05                # significance level throughout

# Visual identity (deep pitch green / gold / charcoal)
PALETTE = {
    "primary":   "#0B3D2E",
    "secondary": "#2E8B57",
    "accent":    "#E4B33C",
    "contrast":  "#B33A3A",
    "grey":      "#5A6068",
    "light":     "#EDF2EF",
}


def setup_plot_style() -> None:
    """Apply one consistent matplotlib style to every figure in the project."""
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.edgecolor": PALETTE["grey"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "legend.frameon": False,
    })


def ensure_dirs() -> None:
    """Create the output folders if they do not yet exist."""
    for d in (FIG_DIR, OUT_DIR):
        os.makedirs(d, exist_ok=True)


# --------------------------------------------------------------------------
# Descriptive statistics
# --------------------------------------------------------------------------
def descriptive_stats(x: pd.Series, label: str = "sample") -> pd.Series:
    """Return the full descriptive-statistics profile of a numeric series.

    Includes measures of central tendency, dispersion and shape, plus the
    standard error of the mean (required later for the confidence interval).
    """
    x = pd.Series(x).dropna().astype(float)
    n = len(x)
    desc = {
        "n": n,
        "mean": x.mean(),
        "median": x.median(),
        "mode": x.mode().iloc[0] if not x.mode().empty else np.nan,
        "std_dev": x.std(ddof=1),
        "variance": x.var(ddof=1),
        "min": x.min(),
        "Q1": x.quantile(0.25),
        "Q3": x.quantile(0.75),
        "IQR": x.quantile(0.75) - x.quantile(0.25),
        "max": x.max(),
        "range": x.max() - x.min(),
        "skewness": x.skew(),
        "kurtosis": x.kurt(),
        "std_error": x.std(ddof=1) / np.sqrt(n),
        "CV_pct": 100 * x.std(ddof=1) / x.mean() if x.mean() else np.nan,
    }
    return pd.Series(desc, name=label)


def outlier_bounds(x: pd.Series) -> tuple[float, float]:
    """Tukey 1.5 x IQR fences, used to flag (not automatically delete) outliers."""
    x = pd.Series(x).dropna().astype(float)
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


# --------------------------------------------------------------------------
# Inferential statistics
# --------------------------------------------------------------------------
def mean_confidence_interval(x: pd.Series, conf: float = CONF_LEVEL) -> dict:
    """t-distribution confidence interval for a population mean.

    The population standard deviation is unknown and is estimated from the
    sample, so the t-distribution (df = n - 1) is the correct sampling model.
    """
    x = pd.Series(x).dropna().astype(float)
    n = len(x)
    mean, se = x.mean(), x.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(1 - (1 - conf) / 2, df=n - 1)
    margin = t_crit * se
    return {
        "n": n, "mean": mean, "std_dev": x.std(ddof=1), "std_error": se,
        "df": n - 1, "conf_level": conf, "t_critical": t_crit,
        "margin_of_error": margin, "lower": mean - margin, "upper": mean + margin,
    }


def normality_check(x: pd.Series, label: str = "") -> dict:
    """Shapiro-Wilk test of normality (an assumption check for the t-test)."""
    x = pd.Series(x).dropna().astype(float)
    w, p = stats.shapiro(x)
    return {"group": label, "n": len(x), "W": w, "p_value": p,
            "normal_at_5pct": p > ALPHA}


def cohens_d_one(x: pd.Series, mu0: float) -> float:
    x = pd.Series(x).dropna().astype(float)
    return (x.mean() - mu0) / x.std(ddof=1)


def cohens_d_two(a: pd.Series, b: pd.Series) -> float:
    """Cohen's d with a pooled standard deviation (effect size)."""
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    return (a.mean() - b.mean()) / sp


def interpret_d(d: float) -> str:
    d = abs(d)
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


def one_sample_t(x: pd.Series, mu0: float, alternative: str = "two-sided") -> dict:
    """One-sample t-test of H0: mu = mu0."""
    x = pd.Series(x).dropna().astype(float)
    t_stat, p_val = stats.ttest_1samp(x, popmean=mu0, alternative=alternative)
    d = cohens_d_one(x, mu0)
    return {
        "test": "One-sample t-test", "mu0": mu0, "alternative": alternative,
        "n": len(x), "mean": x.mean(), "std_dev": x.std(ddof=1),
        "df": len(x) - 1, "t_stat": t_stat, "p_value": p_val,
        "reject_H0": p_val < ALPHA, "cohens_d": d, "effect": interpret_d(d),
    }


def two_sample_t(a: pd.Series, b: pd.Series, labels=("Group A", "Group B"),
                 alternative: str = "two-sided") -> dict:
    """Welch two-sample t-test of H0: mu_A = mu_B (does not assume equal variances).

    Levene's test is reported so the choice of Welch's version is evidence-based;
    Welch is used regardless because it is robust when variances are unequal and
    loses almost nothing when they are equal.
    """
    a = pd.Series(a).dropna().astype(float)
    b = pd.Series(b).dropna().astype(float)
    lev_stat, lev_p = stats.levene(a, b, center="median")
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False, alternative=alternative)

    n1, n2 = len(a), len(b)
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    se_diff = np.sqrt(v1 / n1 + v2 / n2)
    df_welch = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    t_crit = stats.t.ppf(1 - (1 - CONF_LEVEL) / 2, df_welch)
    diff = a.mean() - b.mean()
    d = cohens_d_two(a, b)

    return {
        "test": "Two-sample (Welch) t-test", "alternative": alternative,
        "group_a": labels[0], "group_b": labels[1],
        "n_a": n1, "n_b": n2, "mean_a": a.mean(), "mean_b": b.mean(),
        "sd_a": a.std(ddof=1), "sd_b": b.std(ddof=1),
        "levene_stat": lev_stat, "levene_p": lev_p,
        "equal_var_assumed": lev_p > ALPHA,
        "mean_difference": diff, "se_difference": se_diff, "df": df_welch,
        "t_stat": t_stat, "p_value": p_val, "reject_H0": p_val < ALPHA,
        "ci_diff_lower": diff - t_crit * se_diff,
        "ci_diff_upper": diff + t_crit * se_diff,
        "cohens_d": d, "effect": interpret_d(d),
    }


# --------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------
def banner(title: str, char: str = "=", width: int = 78) -> None:
    print("\n" + char * width)
    print(title.center(width))
    print(char * width)


def section(title: str, width: int = 78) -> None:
    print("\n" + "-" * width)
    print(title)
    print("-" * width)


def show(d: dict, decimals: int = 4) -> None:
    """Pretty-print a results dictionary."""
    for k, v in d.items():
        if isinstance(v, (int, np.integer)):
            print(f"  {k:<22}: {v}")
        elif isinstance(v, (float, np.floating)):
            print(f"  {k:<22}: {v:.{decimals}f}")
        else:
            print(f"  {k:<22}: {v}")


def load_clean() -> pd.DataFrame:
    """Load the cleaned dataset produced by wrangle.py (runs it if missing)."""
    if not os.path.exists(CLEAN_DATA):
        raise FileNotFoundError(
            f"{CLEAN_DATA} not found - run `python wrangle.py` first (or `python run_all.py`)."
        )
    return pd.read_csv(CLEAN_DATA)
