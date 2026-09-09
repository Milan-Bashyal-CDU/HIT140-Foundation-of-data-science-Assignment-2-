"""Create publication-ready figures and a concise PDF assessment report."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
)

from .model_pipeline import ModellingResults


NAVY = "#13263D"
BLUE = "#2C6E9B"
RED = "#B94343"
GOLD = "#D8A640"
LIGHT = "#EAF0F5"
TEXT = "#1F2933"


def _style_plot() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.labelsize": 10,
        "axes.edgecolor": "#AAB7C4",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def create_charts(results: ModellingResults, chart_dir: Path) -> None:
    chart_dir.mkdir(parents=True, exist_ok=True)
    _style_plot()

    top = results.final_predictions.head(12).sort_values("model_probability")
    colors_by_team = [RED if team == "Spain" else BLUE for team in top.team]
    fig, ax = plt.subplots(figsize=(10, 6.4))
    ax.barh(top.player_name, top.model_probability, color=colors_by_team)
    ax.errorbar(
        top.model_probability, top.player_name,
        xerr=[top.model_probability - top.bootstrap_low, top.bootstrap_high - top.model_probability],
        fmt="none", ecolor=NAVY, capsize=3, linewidth=1,
    )
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Probability of scoring at least once")
    ax.set_title("Pre-final scorer probabilities with 95% bootstrap intervals", loc="left", fontweight="bold", color=NAVY)
    ax.grid(axis="x", alpha=0.2)
    for y, value in enumerate(top.model_probability):
        ax.text(value + 0.004, y, f"{value:.1%}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(chart_dir / "final_scorer_probabilities.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    metric = results.metrics.sort_values("brier_score", ascending=False)
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    bars = ax.barh(metric.model, metric.brier_score, color=["#9AA7B4", BLUE, GOLD])
    ax.set_xlabel("Brier score (lower is better)")
    ax.set_title("Walk-forward model comparison on knockout matches", loc="left", fontweight="bold", color=NAVY)
    ax.grid(axis="x", alpha=0.2)
    for bar, value in zip(bars, metric.brier_score):
        ax.text(value + 0.0002, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center")
    fig.tight_layout()
    fig.savefig(chart_dir / "model_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    calibration = results.calibration
    fig, ax = plt.subplots(figsize=(6.3, 5.5))
    ax.plot([0, 0.22], [0, 0.22], "--", color="#8A98A6", label="Perfect calibration")
    ax.plot(calibration.mean_predicted, calibration.observed_rate, marker="o", linewidth=2, color=BLUE, label="Logistic model")
    ax.set_xlim(0, max(0.12, calibration.mean_predicted.max() * 1.2))
    ax.set_ylim(0, max(0.12, calibration.observed_rate.max() * 1.2))
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed scorer rate")
    ax.set_title("Calibration by probability quintile", loc="left", fontweight="bold", color=NAVY)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(chart_dir / "calibration.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    coeff = results.coefficients.head(10).sort_values("coefficient")
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    bar_colors = [RED if value < 0 else BLUE for value in coeff.coefficient]
    ax.barh(coeff.feature.str.replace("_", " "), coeff.coefficient, color=bar_colors)
    ax.axvline(0, color=NAVY, linewidth=0.8)
    ax.set_xlabel("Standardised logistic coefficient")
    ax.set_title("Largest model effects", loc="left", fontweight="bold", color=NAVY)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(chart_dir / "coefficient_effects.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _pdf_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=24, leading=29, textColor=colors.HexColor(NAVY), alignment=TA_LEFT,
        spaceAfter=14,
    ))
    styles.add(ParagraphStyle(
        name="ReportSubtitle", parent=styles["Normal"], fontName="Helvetica",
        fontSize=11, leading=16, textColor=colors.HexColor("#4A5A69"), spaceAfter=16,
    ))
    styles.add(ParagraphStyle(
        name="H1Black", parent=styles["Heading1"], fontName="Helvetica-Bold",
        fontSize=16, leading=20, textColor=colors.black, spaceBefore=12, spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="H2Black", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=12, leading=15, textColor=colors.black, spaceBefore=9, spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="BodyReport", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=9.4, leading=13.2, textColor=colors.HexColor(TEXT), spaceAfter=7,
    ))
    styles.add(ParagraphStyle(
        name="Small", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=7.7, leading=10, textColor=colors.HexColor("#586776"), spaceAfter=4,
    ))
    return styles


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#637282"))
    canvas.drawString(18 * mm, 10 * mm, "FIFA World Cup 2026 final scorer analysis")
    canvas.drawRightString(192 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def _table(data, widths, header=True):
    table = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEADING", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D9D9D9")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
        for row in range(1, len(data)):
            if row % 2 == 0:
                commands.append(("BACKGROUND", (0, row), (-1, row), colors.HexColor("#F3F6F8")))
    table.setStyle(TableStyle(commands))
    return table


def create_pdf_report(results: ModellingResults, validation, output_path: Path) -> None:
    styles = _pdf_styles()
    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=17 * mm, bottomMargin=17 * mm,
        title="Predicting the Most Likely Scorer in the 2026 World Cup Final",
        author="Student group",
    )
    story = []
    top = results.final_predictions.iloc[0]
    actual = results.diagnostics["actual_scorer"][0]
    actual_rank = results.diagnostics["actual_scorer_rank"]
    best_metric = results.metrics.iloc[0]

    story += [
        Paragraph("Predicting the Most Likely Scorer in the 2026 World Cup Final", styles["ReportTitle"]),
        Paragraph(
            "A leakage-free player match analysis of Spain vs Argentina using FIFA Training Centre post-match data",
            styles["ReportSubtitle"],
        ),
        Paragraph("Executive finding", styles["H1Black"]),
        Paragraph(
            f"Before the final, the regularised logistic model ranked <b>{top.player_name} ({top.team})</b> as the most likely scorer at "
            f"<b>{top.model_probability:.1%}</b> over 90 minutes. The 95% match bootstrap interval was "
            f"{top.bootstrap_low:.1%} to {top.bootstrap_high:.1%}. The actual scorer, <b>{actual}</b>, ranked "
            f"{actual_rank} before kickoff. He scored in extra time as Spain won 1-0, illustrating why a probability ranking can be useful without being certain.",
            styles["BodyReport"],
        ),
        Paragraph("Analytic question", styles["H1Black"]),
        Paragraph(
            "Using only information available when the starting lineups became known, which player in the Spain vs Argentina final had the highest probability of scoring at least once? The unit of analysis is a player selected in a matchday squad, including unused substitutes. This makes the task harder and more realistic than ranking players who were already known to appear.",
            styles["BodyReport"],
        ),
        Image(str(output_path.parent / "charts" / "final_scorer_probabilities.png"), width=174 * mm, height=111 * mm),
        Paragraph("Figure 1. Probabilities refer to at least one goal in 90 minutes. Intervals resample whole matches.", styles["Small"]),
        PageBreak(),
        Paragraph("Data acquisition and validation", styles["H1Black"]),
        Paragraph(
            f"The pipeline uses six relational CSV tables from the open Worldcup26 repository, pinned to commit {validation.summary['source_commit'][:12]}. "
            f"The source converts all 104 FIFA Training Centre reports into structured match, team, player, lineup and attempt data. "
            f"The analysis includes {validation.summary['appearance_rows']:,} player matchday records and {validation.summary['attempt_rows']:,} attempts at goal. "
            "The supplied raw snapshot lets the project run offline, while --refresh-data repeats acquisition from the pinned URLs and writes SHA-256 hashes.",
            styles["BodyReport"],
        ),
        Paragraph("Validation checks", styles["H2Black"]),
        _table(
            [["Check", "Result", "Evidence"]] + [
                [row.check, "PASS" if row.passed else "FAIL", row.detail]
                for row in validation.checks.itertuples(index=False)
            ],
            [62 * mm, 18 * mm, 94 * mm],
        ),
        Paragraph("Leakage control", styles["H2Black"]),
        Paragraph(
            "For every match, the pipeline calculates player and team histories before adding that match's outcomes. The final is match 104 and never contributes to training features. Walk-forward testing trains on earlier matches and predicts the next knockout match in chronological order. This design avoids the future information leakage caused by random row splitting.",
            styles["BodyReport"],
        ),
        Paragraph("Creative and systematic variables", styles["H1Black"]),
        Paragraph(
            "The 19 predictors represent opportunity, individual threat, recent form and match context. Opportunity combines the announced starter status with a player's past selection, appearance and starting rates to estimate minutes. Threat uses smoothed goals, attempts and shots on target per 90 minutes, including separate three match form. Context adds the team's prior goals, expected goals and attempts, the opponent's prior goals, expected goals and shots on target allowed, rest days, position and knockout status. Bayesian style pseudo counts stabilise players with few minutes.",
            styles["BodyReport"],
        ),
        PageBreak(),
        Paragraph("Model comparison and selection", styles["H1Black"]),
        Paragraph(
            "Three viable approaches were tested on matches 73 to 103. The historical rate baseline is transparent but ignores team and opponent context. The random forest captures non-linear interactions but may overfit rare scorer events. It achieved a narrowly lower Brier score, while the regularised logistic model achieved better log loss, ROC AUC and top-ranked scorer accuracy. The logistic model was selected because the Brier difference was very small and its probability effects are easier to explain and audit.",
            styles["BodyReport"],
        ),
        _table(
            [["Model", "Brier", "Log loss", "ROC AUC", "Top 1 hit", "Top 3 hit"]] + [
                [
                    row.model, f"{row.brier_score:.4f}", f"{row.log_loss:.4f}", f"{row.roc_auc:.3f}",
                    f"{row.top1_scorer_hit_rate:.1%}", f"{row.top3_scorer_hit_rate:.1%}",
                ] for row in results.metrics.itertuples(index=False)
            ],
            [49 * mm, 22 * mm, 23 * mm, 21 * mm, 25 * mm, 25 * mm],
        ),
        Spacer(1, 5 * mm),
        Image(str(output_path.parent / "charts" / "model_comparison.png"), width=168 * mm, height=84 * mm),
        Paragraph("Figure 2. Lower Brier scores indicate more accurate probability forecasts.", styles["Small"]),
        Paragraph("Assumption and diagnostic checks", styles["H1Black"]),
        Paragraph(
            f"The scorer outcome is rare: {results.diagnostics['scorer_prevalence']:.1%} of pre-final squad rows. L2 regularisation and smoothed rate features reduce unstable estimates. "
            f"The largest approximate variance inflation factor was {results.diagnostics['max_vif']:.1f} for {results.diagnostics['max_vif_feature'].replace('_', ' ')}, so correlated rate variables remain a limitation even with regularisation. "
            "The calibration plot checks whether predicted probabilities match observed frequencies. Player rows within the same match are not independent, so uncertainty uses a cluster bootstrap that resamples complete matches rather than individual players.",
            styles["BodyReport"],
        ),
        PageBreak(),
        Paragraph("Interpretation of the final forecast", styles["H1Black"]),
        _table(
            [["Rank", "Player", "Team", "Role", "90 min", "120 min", "95% interval"]] + [
                [
                    str(int(row["rank"])), row.player_name, row.team,
                    "Starter" if row.is_starter else "Bench",
                    f"{row.model_probability:.1%}", f"{row.probability_120_minutes:.1%}",
                    f"{row.bootstrap_low:.1%}-{row.bootstrap_high:.1%}",
                ] for _, row in results.final_predictions.head(12).iterrows()
            ],
            [12 * mm, 43 * mm, 24 * mm, 22 * mm, 20 * mm, 20 * mm, 30 * mm],
        ),
        Paragraph("What the result means", styles["H2Black"]),
        Paragraph(
            f"The highest probability is still modest because a football goal is a rare event shared among many players. {top.player_name}'s ranking reflects both scoring threat and expected playing time. The 120 minute sensitivity raises every player's chance but does not assume that a starter necessarily remains on the field for extra time. The probabilities should support comparison, not be read as guarantees or betting advice.",
            styles["BodyReport"],
        ),
        Paragraph("Post-match audit", styles["H2Black"]),
        Paragraph(
            f"Spain defeated Argentina 1-0 after extra time. {actual} scored the only goal. The model placed him at rank {actual_rank}, so the single realised match did not match the top ranked player. This does not by itself invalidate a probability model; model quality is judged across the earlier walk-forward sample. The miss shows the importance of substitute uncertainty and low-frequency events in football.",
            styles["BodyReport"],
        ),
        KeepTogether([
            Image(str(output_path.parent / "charts" / "calibration.png"), width=78 * mm, height=68 * mm),
            Paragraph("Figure 3. Observed scorer frequency by model probability quintile.", styles["Small"]),
        ]),
        PageBreak(),
        Paragraph("Limitations and viable extensions", styles["H1Black"]),
        Paragraph(
            "The model uses one tournament, so each finalist contributes only six pre-final matches. It knows the announced starting lineup but estimates substitute use from tournament history. It lacks player level expected goals, penalty taker status, injuries, tactical matchups and club season form. The public reports may contain extraction errors; for example, event minutes can differ by one minute across providers. Extra time sensitivity assumes a proportional hazard and cannot predict substitutions or red cards. A stronger future study would merge club shot quality, bookmaker team goal expectations and confirmed injury data, then test the model on multiple tournaments.",
            styles["BodyReport"],
        ),
        Paragraph("Reproducibility", styles["H1Black"]),
        Paragraph(
            "Install the packages in requirements.txt and run python run_analysis.py. The command validates the raw tables, rebuilds every feature in chronological order, compares models, writes the prediction tables and diagnostics, regenerates all figures, and creates this PDF. Run python run_analysis.py --refresh-data to reacquire the pinned CSV snapshot when internet access is available.",
            styles["BodyReport"],
        ),
        Paragraph("References", styles["H1Black"]),
        Paragraph(
            "Alamyy. (2026). Worldcup26: Open FIFA World Cup 2026 data derived from FIFA Training Centre post-match reports. GitHub. https://github.com/Alamyy/Worldcup26",
            styles["BodyReport"],
        ),
        Paragraph(
            f"FIFA Training Centre. (2026). Post-match summary report: Spain vs Argentina, match 104. {results.diagnostics['final_match_source']}",
            styles["BodyReport"],
        ),
        Paragraph(
            "FIFA. (2026). Spain crowned champions as curtain falls on FIFA World Cup 2026. https://inside.fifa.com/organisation/news/new-york-jersey-stadium-spain-world-champions-mbappe-haaland",
            styles["BodyReport"],
        ),
    ]
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
