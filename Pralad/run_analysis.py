"""Run the complete World Cup 2026 final scorer analysis.

Usage:
    python run_analysis.py
    python run_analysis.py --refresh-data

The default run is offline and uses the audited CSV snapshot bundled in data/raw.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data_pipeline import acquire_data, load_and_validate
from src.model_pipeline import run_modelling
from src.reporting import create_charts, create_pdf_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Download a fresh copy of the pinned source CSV files before analysis.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    output_dir = root / "outputs"
    chart_dir = output_dir / "charts"
    for directory in (raw_dir, processed_dir, output_dir, chart_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if args.refresh_data:
        acquire_data(raw_dir)

    tables, validation = load_and_validate(raw_dir)
    results = run_modelling(tables, processed_dir, output_dir)
    create_charts(results, chart_dir)
    create_pdf_report(results, validation, output_dir / "World_Cup_2026_Final_Scorer_Report.pdf")

    leader = results.final_predictions.iloc[0]
    print("Analysis completed successfully")
    print(f"Most likely scorer: {leader.player_name} ({leader.team})")
    print(f"Model probability: {leader.model_probability:.1%}")
    print(f"Report: {output_dir / 'World_Cup_2026_Final_Scorer_Report.pdf'}")


if __name__ == "__main__":
    main()

