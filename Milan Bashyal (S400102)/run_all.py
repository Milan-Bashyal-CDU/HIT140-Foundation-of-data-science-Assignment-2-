"""
run_all.py
==========
Master script: runs the whole project end-to-end and writes a consolidated
results file.

Pipeline
--------
    wrangle.py               -> cleaned, enriched dataset (+ overview figure)
    task1_shot_stopping.py   -> Task 1: save percentage
    task2_workload.py        -> Task 2: shots on target faced per 90 minutes
    task3_age_profile.py     -> Task 3: goalkeeper age profile

Outputs
-------
    output/keepers_clean.csv      cleaned dataset
    output/task{1,2,3}_sample.csv the exact samples analysed (reproducible)
    output/results_summary.csv    every headline statistic in one table
    output/full_report.txt        the complete console output of the whole run
    figures/fig0..fig3 *.png      all figures used in the presentation

Run:  python run_all.py
"""

from __future__ import annotations

import io
import contextlib
import pandas as pd

import wrangle
import task1_shot_stopping as t1
import task2_workload as t2
import task3_age_profile as t3
from gk_common import OUT_DIR, banner, ensure_dirs, section


def build_summary(r1: dict, r2: dict, r3: dict) -> pd.DataFrame:
    rows = [
        {"Task": "1. Shot-stopping efficiency",
         "Variable": "Save percentage (%)",
         "Sampling": "Stratified random (by stage)",
         "n": r1["ci"]["n"],
         "Mean": round(r1["ci"]["mean"], 2),
         "SD": round(r1["ci"]["std_dev"], 2),
         "95% CI": f"[{r1['ci']['lower']:.2f}, {r1['ci']['upper']:.2f}]",
         "Test": "Two-sample Welch t-test (knockout vs group only)",
         "t": round(r1["ttest"]["t_stat"], 3),
         "p": round(r1["ttest"]["p_value"], 4),
         "Cohen's d": round(r1["ttest"]["cohens_d"], 2),
         "Decision at 5%": "Reject H0" if r1["ttest"]["reject_H0"] else "Fail to reject H0"},
        {"Task": "2. Defensive workload",
         "Variable": "Shots on target faced per 90",
         "Sampling": "Simple random",
         "n": r2["ci"]["n"],
         "Mean": round(r2["ci"]["mean"], 2),
         "SD": round(r2["ci"]["std_dev"], 2),
         "95% CI": f"[{r2['ci']['lower']:.2f}, {r2['ci']['upper']:.2f}]",
         "Test": "Two-sample Welch t-test (UEFA/CONMEBOL vs rest)",
         "t": round(r2["ttest"]["t_stat"], 3),
         "p": round(r2["ttest"]["p_value"], 4),
         "Cohen's d": round(r2["ttest"]["cohens_d"], 2),
         "Decision at 5%": "Reject H0" if r2["ttest"]["reject_H0"] else "Fail to reject H0"},
        {"Task": "3. Age / career-stage profile",
         "Variable": "Age (years)",
         "Sampling": "Systematic random",
         "n": r3["ci"]["n"],
         "Mean": round(r3["ci"]["mean"], 2),
         "SD": round(r3["ci"]["std_dev"], 2),
         "95% CI": f"[{r3['ci']['lower']:.2f}, {r3['ci']['upper']:.2f}]",
         "Test": "One-sample t-test (mu0 = 27 years)",
         "t": round(r3["ttest"]["t_stat"], 3),
         "p": round(r3["ttest"]["p_value"], 6),
         "Cohen's d": round(r3["ttest"]["cohens_d"], 2),
         "Decision at 5%": "Reject H0" if r3["ttest"]["reject_H0"] else "Fail to reject H0"},
    ]
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    buf = io.StringIO()
    with contextlib.redirect_stdout(_Tee(buf)):
        banner("FIFA WORLD CUP 2026 GOALKEEPER ANALYSIS - FULL RUN", "#")
        wrangle.main()
        r1 = t1.main()
        r2 = t2.main()
        r3 = t3.main()

        summary = build_summary(r1, r2, r3)
        banner("CONSOLIDATED RESULTS SUMMARY", "#")
        print(summary.to_string(index=False))

        section("OVERALL STORY")
        print("1. Goalkeepers save about 64% of the shots on target they face, and the")
        print("   keepers of teams that progressed were significantly better at it.")
        print("2. Yet those same teams did NOT significantly reduce the number of shots")
        print("   their keeper faced - progression is linked to shot-stopping quality,")
        print("   not to shielding the goalkeeper from work.")
        print("3. The men trusted with the job are significantly older than the")
        print("   27-year outfield benchmark: goalkeeping is an experience position.")

        section("LIMITATIONS")
        print("* Single tournament, 62 goalkeepers: the frame is small, so the samples")
        print("  represent a large fraction of it (finite-population correction would")
        print("  narrow the intervals slightly - our CIs are therefore conservative).")
        print("* Save% and shots faced are strongly influenced by team quality and by")
        print("  shot quality, which this dataset does not record (no xG / post-shot xG).")
        print("* Reaching the knockout stage is an outcome, not a randomised treatment,")
        print("  so Task 1 shows association, not causation.")
        print("* Task 2's random sample split 13 / 27 between the groups, which limits")
        print("  statistical power for detecting a small workload difference.")

    summary.to_csv(f"{OUT_DIR}/results_summary.csv", index=False)
    with open(f"{OUT_DIR}/full_report.txt", "w") as f:
        f.write(buf.getvalue())
    print(f"\nSaved -> {OUT_DIR}/results_summary.csv")
    print(f"Saved -> {OUT_DIR}/full_report.txt")


class _Tee(io.TextIOBase):
    """Write to the real console and to a buffer at the same time."""

    def __init__(self, buffer: io.StringIO):
        import sys
        self.buffer = buffer
        self.console = sys.__stdout__

    def write(self, s: str) -> int:
        self.console.write(s)
        self.buffer.write(s)
        return len(s)

    def flush(self) -> None:
        self.console.flush()


if __name__ == "__main__":
    main()
