# FIFA World Cup 2026 Final Scorer Analysis

This project asks a specific pre-match question: **which player in the Spain vs Argentina final had the highest probability of scoring at least once?** It reconstructs the information set available at kickoff, predicts every player in the announced matchday squads, and then audits the forecast against the actual 1-0 result.

## Main finding

Run the analysis to regenerate the ranked probabilities, uncertainty intervals, model comparison, figures, and PDF report. The result is probabilistic: the top ranked player is the most likely among the available players, not a guaranteed scorer.

## Why the approach is suitable for high marks

- Non-trivial question at player-match level, including unused substitutes.
- Six-table relational data integration from all 104 FIFA Training Centre reports.
- Pinned and optionally automated acquisition with file hashes.
- Eleven structural and referential validation checks.
- Chronological feature engineering that prevents final-match leakage.
- Systematic predictors for opportunity, shot threat, form, team attack, opponent defence, position, and rest.
- Walk-forward comparison of a historical baseline, regularised logistic regression, and random forest.
- Brier score, log loss, ROC AUC, top-scorer hit rates, calibration, VIF, clustered bootstrap intervals, and 120-minute sensitivity.
- Honest post-match audit and explicit limitations.

## Run on Windows

```powershell
cd "D:\fifa data analyisi"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_analysis.py
```

If PowerShell blocks activation, use:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_analysis.py
```

The default run uses the bundled audited data snapshot and does not need internet access. To reacquire the same pinned source version:

```powershell
python run_analysis.py --refresh-data
```

## Outputs

- `outputs/World_Cup_2026_Final_Scorer_Report.pdf`
- `outputs/final_scorer_predictions.csv`
- `outputs/model_comparison.csv`
- `outputs/logistic_coefficients.csv`
- `outputs/model_diagnostics.json`
- `outputs/charts/*.png`
- `data/processed/player_match_features.csv`
- `data/processed/walk_forward_predictions.csv`

## Data source and licence

The data snapshot comes from [Alamyy/Worldcup26](https://github.com/Alamyy/Worldcup26), commit `7cc05cdc046f505939bb27f5097e22f9a2cb8a7d`. The repository organises public FIFA Training Centre post-match reports into CSV tables and includes a source URL for every match. Retain the upstream licence and cite the repository and FIFA reports in the presentation.

## Interpretation limits

This is an educational model, not betting advice. A single match cannot prove or disprove a probability forecast. The analysis has only one tournament, estimates substitute use, and lacks club-season xG, injury status, penalty duties, and tactical matchup data.

