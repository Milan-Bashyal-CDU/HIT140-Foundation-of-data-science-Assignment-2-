# FIFA World Cup 2026 — Goalkeeper Analysis

Three independent analytic tasks on the 62 goalkeepers who appeared at the 2026 World
Cup. All wrangling, statistics and visualisation are done in Python (pandas, NumPy,
SciPy, Matplotlib). No spreadsheet work is required — the raw CSV is read directly.

## How to run

```bash
pip install pandas numpy scipy matplotlib
python run_all.py            # wrangling + all three tasks + summary  (recommended)
```

Or run any part on its own:

```bash
python wrangle.py                 # must be run first — writes output/keepers_clean.csv
python task1_shot_stopping.py
python task2_workload.py
python task3_age_profile.py
```


## Files

| File | What it is |
|---|---|
| `keepers_2026.csv` | raw dataset (62 goalkeepers, 28 columns) |
| `gk_common.py` | shared statistics helpers — descriptives, CI, t-tests, effect sizes, plot style |
| `wrangle.py` | **Step 1**: audit, clean, validate, feature engineering |
| `task1_shot_stopping.py` | **Task 1** — save percentage |
| `task2_workload.py` | **Task 2** — shots on target faced per 90 minutes |
| `task3_age_profile.py` | **Task 3** — goalkeeper age |
| `run_all.py` | runs everything, writes the consolidated summary |

## The three analytic tasks

Each task has its **own** focal point, population, sampling frame, sampling technique
and hypothesis test, so no two tasks measure the same thing.

| | Task 1 | Task 2 | Task 3 |
|---|---|---|---|
| **Focal point** | how *well* a keeper performs | how *often* a keeper is tested | *who* nations pick |
| **Question** | What share of shots on target do keepers save, and do keepers of knockout-stage teams save more? | How many shots on target does a keeper face per 90 minutes, and do UEFA/CONMEBOL keepers face fewer? | Is the mean age of World Cup keepers different from the 27-year outfield peak age? |
| **Variable** | `Save_Pct` = (SoTA − GA) / SoTA × 100 | `SoTA90` = SoTA / 90s | `Age` (years) |
| **Frame** | 55 keepers (≥ 90 min **and** ≥ 5 shots on target faced) | 58 keepers (≥ 90 min) | all 62 keepers who appeared |
| **Sampling** | stratified random by stage, n = 36 | simple random, n = 40 | systematic (k = 2, random start), n = 31 |
| **Descriptives** | mean, median, mode, SD, variance, IQR, skew, kurtosis, SE, CV | same | same |
| **Confidence interval** | 95% t-interval: **63.64% [58.45, 68.84]** | **4.63 [3.94, 5.32]** | **31.13 yrs [29.52, 32.74]** |
| **Test** | two-sample Welch t-test | two-sample Welch t-test | one-sample t-test (μ₀ = 27) |
| **Result** | t = 2.94, p = 0.006, d = 0.98 → **reject H₀** | t = −1.42, p = 0.166, d = −0.46 → **fail to reject H₀** | t = 5.24, p < 0.0001, d = 0.94 → **reject H₀** |

## Method notes

- **Eligibility, not imputation.** Missing `Save_Pct`, `CS_Pct` and `PK_Save_Pct` are
  structurally undefined (division by zero), so the affected rows are excluded by an
  eligibility rule rather than filled in.
- **Rates, not totals.** Shots faced are divided by 90-minute units so keepers who
  played more matches are not mistaken for keepers who were busier.
- **Reproducibility.** One seed (`RANDOM_SEED = 2026`) drives all sampling; the exact
  samples analysed are saved to `output/task{1,2,3}_sample.csv`.
- **Assumption checks.** Shapiro–Wilk for normality and Levene's test for equal
  variances precede every t-test; Welch's version is used for the two-sample tests
  because it is robust to unequal variances.
- **Validation.** Because the frame is fully observed, each task prints the true frame
  mean next to the confidence interval — all three intervals contain it.

## Limitations

Single tournament and a small frame (the samples cover a large share of it, so the
intervals are conservative); no shot-quality data (xG / post-shot xG) to adjust for how
hard the chances were; reaching the knockout stage is an outcome rather than a
randomised treatment, so Task 1 shows association, not causation; and Task 2's random
split of 13 / 27 between groups limits statistical power.
