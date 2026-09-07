"""Leakage-safe feature engineering, model comparison, and final prediction."""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_pipeline import count_minute_events, count_regulation_events


FINAL_MATCH_NUMBER = 104
GROUP_STAGE_END = 72
SEED = 5752026

NUMERIC_FEATURES = [
    "is_starter",
    "prior_squad_selections",
    "prior_appearances",
    "appearance_rate",
    "starter_rate",
    "expected_minutes",
    "goals_per90_smoothed",
    "attempts_per90_smoothed",
    "on_target_per90_smoothed",
    "recent_attempts_per90",
    "recent_on_target_per90",
    "team_goals_per_match",
    "team_xg_per_match",
    "team_attempts_per_match",
    "opponent_goals_allowed_per_match",
    "opponent_xg_allowed_per_match",
    "opponent_sot_allowed_per_match",
    "days_rest",
    "knockout",
]
CATEGORICAL_FEATURES = ["position"]


@dataclass
class ModellingResults:
    feature_frame: pd.DataFrame
    evaluation_predictions: pd.DataFrame
    final_predictions: pd.DataFrame
    metrics: pd.DataFrame
    coefficients: pd.DataFrame
    calibration: pd.DataFrame
    uncertainty: pd.DataFrame
    diagnostics: dict


def _metric_table(key_stats: pd.DataFrame) -> pd.DataFrame:
    keep = key_stats[key_stats.metric.isin([
        "Goals", "xG (Expected Goals)", "Attempts at Goal (On Target)"
    ])].copy()
    keep["numeric"] = pd.to_numeric(keep.value, errors="coerce")
    attempt_mask = keep.metric.eq("Attempts at Goal (On Target)")
    extracted = keep.loc[attempt_mask, "raw"].astype(str).str.extract(r"(?P<attempts>\d+)\s*\((?P<sot>\d+)\)")
    keep.loc[attempt_mask, "numeric"] = pd.to_numeric(extracted.attempts, errors="coerce")
    sot_rows = keep.loc[attempt_mask, ["match_id", "team_id", "team"]].copy()
    sot_rows["metric"] = "Shots on Target"
    sot_rows["numeric"] = pd.to_numeric(extracted.sot, errors="coerce").to_numpy()
    keep = pd.concat([keep[["match_id", "team_id", "team", "metric", "numeric"]], sot_rows], ignore_index=True)
    pivot = keep.pivot_table(index=["match_id", "team_id"], columns="metric", values="numeric", aggfunc="first").reset_index()
    return pivot.rename(columns={
        "Goals": "goals",
        "xG (Expected Goals)": "xg",
        "Attempts at Goal (On Target)": "attempts",
        "Shots on Target": "sot",
    })


def _opponent_map(match_teams: pd.DataFrame) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for match_id, group in match_teams.groupby("match_id"):
        ids = group.team_id.tolist()
        result[(match_id, ids[0])] = ids[1]
        result[(match_id, ids[1])] = ids[0]
    return result


def _duration_by_match(appearances: pd.DataFrame, attempts: pd.DataFrame) -> dict[str, int]:
    maxima: dict[str, float] = defaultdict(float)
    for column in ("subbed_on_minutes", "subbed_off_minutes"):
        temp = appearances[["match_id", column]].copy()
        temp[column] = pd.to_numeric(temp[column], errors="coerce")
        for match_id, value in temp.groupby("match_id")[column].max().items():
            maxima[match_id] = max(maxima[match_id], 0 if pd.isna(value) else float(value))
    temp = attempts[["match_id", "minute"]].copy()
    temp.minute = pd.to_numeric(temp.minute, errors="coerce")
    for match_id, value in temp.groupby("match_id").minute.max().items():
        maxima[match_id] = max(maxima[match_id], 0 if pd.isna(value) else float(value))
    return {match_id: (120 if maximum > 105 else 90) for match_id, maximum in maxima.items()}


def _minutes_played(row: pd.Series, duration: int) -> float:
    if not bool(row.appeared):
        return 0.0
    on = pd.to_numeric(pd.Series([row.subbed_on_minutes]), errors="coerce").iloc[0]
    off = pd.to_numeric(pd.Series([row.subbed_off_minutes]), errors="coerce").iloc[0]
    regulation_end = 90.0
    start = 0.0 if bool(row.is_starter) else (float(on) if pd.notna(on) else regulation_end)
    end = float(off) if pd.notna(off) else regulation_end
    return float(np.clip(min(end, regulation_end) - min(start, regulation_end), 0, regulation_end))


def build_feature_frame(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    matches = tables["matches"].sort_values("match_number").copy()
    appearances = tables["match_appearances"].copy()
    attempts = tables["attempts_at_goal"].copy()
    metrics = _metric_table(tables["team_key_stats"])
    opponents = _opponent_map(tables["match_teams"])
    durations = _duration_by_match(appearances, attempts)

    attempts["match_duration"] = attempts.match_id.map(durations).fillna(90)
    attempts["minute_numeric"] = pd.to_numeric(attempts.minute, errors="coerce")
    regulation_attempts = attempts[
        attempts.match_duration.eq(90) | attempts.minute_numeric.le(90)
    ].copy()
    shot_agg = regulation_attempts.assign(
        on_target=regulation_attempts.outcome.astype(str).str.contains("On Target", case=False, na=False).astype(int)
    ).groupby("appearance_id").agg(attempts=("event_id", "size"), on_target=("on_target", "sum")).reset_index()
    appearances = appearances.merge(shot_agg, on="appearance_id", how="left")
    appearances[["attempts", "on_target"]] = appearances[["attempts", "on_target"]].fillna(0)
    appearances["goals"] = appearances.goal_minutes.map(count_regulation_events)
    appearances["goals_any"] = appearances.goal_minutes.map(count_minute_events)

    match_lookup = matches.set_index("match_id")
    metrics_lookup = metrics.set_index(["match_id", "team_id"])
    player_history: dict[str, dict] = defaultdict(lambda: {
        "squad": 0, "apps": 0, "starts": 0, "minutes": 0.0, "goals": 0,
        "attempts": 0, "sot": 0, "starter_minutes": [], "bench_minutes": [],
        "recent": deque(maxlen=3), "last_date": None,
    })
    team_history: dict[str, dict] = defaultdict(lambda: {
        "games": 0, "goals": 0.0, "xg": 0.0, "attempts": 0.0, "sot": 0.0,
        "goals_allowed": 0.0, "xg_allowed": 0.0, "sot_allowed": 0.0,
    })
    global_history = {"team_games": 0, "goals": 0.0, "xg": 0.0, "attempts": 0.0, "sot": 0.0}
    rows: list[dict] = []

    for match in matches.itertuples(index=False):
        match_date = pd.to_datetime(match.date, dayfirst=True)
        roster = appearances[appearances.match_id.eq(match.match_id)].copy()
        duration = durations.get(match.match_id, 90)

        for _, player in roster.iterrows():
            history = player_history[player.player_id]
            team = team_history[player.team_id]
            opponent_id = opponents[(match.match_id, player.team_id)]
            opponent = team_history[opponent_id]
            tournament_games = max(global_history["team_games"], 1)
            global_goal_rate = global_history["goals"] / tournament_games
            global_xg_rate = global_history["xg"] / tournament_games
            global_attempt_rate = global_history["attempts"] / tournament_games
            global_sot_rate = global_history["sot"] / tournament_games

            if bool(player.is_starter):
                expected_minutes = float(np.mean(history["starter_minutes"])) if history["starter_minutes"] else 80.0
            else:
                sub_use_rate = (len(history["bench_minutes"]) + 0.35) / (history["squad"] + 1.0)
                mean_sub_minutes = float(np.mean(history["bench_minutes"])) if history["bench_minutes"] else 22.0
                expected_minutes = sub_use_rate * mean_sub_minutes
            expected_minutes = float(np.clip(expected_minutes, 0, 90))

            recent_minutes = sum(item[0] for item in history["recent"])
            recent_attempts = sum(item[1] for item in history["recent"])
            recent_sot = sum(item[2] for item in history["recent"])
            days_rest = 7.0 if history["last_date"] is None else float((match_date - history["last_date"]).days)

            rows.append({
                "match_id": match.match_id,
                "match_number": int(match.match_number),
                "date": match_date,
                "group": match.group,
                "player_id": player.player_id,
                "player_name": player.player_name,
                "team_id": player.team_id,
                "team": player.team,
                "opponent_id": opponent_id,
                "position": player.position,
                "is_starter": int(bool(player.is_starter)),
                "prior_squad_selections": history["squad"],
                "prior_appearances": history["apps"],
                "appearance_rate": (history["apps"] + 0.5) / (history["squad"] + 1.0),
                "starter_rate": (history["starts"] + 0.5) / (history["squad"] + 1.0),
                "expected_minutes": expected_minutes,
                "goals_per90_smoothed": 90.0 * (history["goals"] + 0.20) / (history["minutes"] + 180.0),
                "attempts_per90_smoothed": 90.0 * (history["attempts"] + 1.0) / (history["minutes"] + 180.0),
                "on_target_per90_smoothed": 90.0 * (history["sot"] + 0.40) / (history["minutes"] + 180.0),
                "recent_attempts_per90": 90.0 * (recent_attempts + 0.5) / (recent_minutes + 90.0),
                "recent_on_target_per90": 90.0 * (recent_sot + 0.2) / (recent_minutes + 90.0),
                "team_goals_per_match": (team["goals"] + 2.0 * global_goal_rate) / (team["games"] + 2.0),
                "team_xg_per_match": (team["xg"] + 2.0 * global_xg_rate) / (team["games"] + 2.0),
                "team_attempts_per_match": (team["attempts"] + 2.0 * global_attempt_rate) / (team["games"] + 2.0),
                "opponent_goals_allowed_per_match": (opponent["goals_allowed"] + 2.0 * global_goal_rate) / (opponent["games"] + 2.0),
                "opponent_xg_allowed_per_match": (opponent["xg_allowed"] + 2.0 * global_xg_rate) / (opponent["games"] + 2.0),
                "opponent_sot_allowed_per_match": (opponent["sot_allowed"] + 2.0 * global_sot_rate) / (opponent["games"] + 2.0),
                "days_rest": float(np.clip(days_rest, 2, 14)),
                "knockout": int(match.match_number > GROUP_STAGE_END),
                "scored": int(player.goals > 0),
                "scored_any": int(player.goals_any > 0),
                "goals": int(player.goals),
                "appeared": int(bool(player.appeared)),
                "actual_minutes": _minutes_played(player, duration),
                "actual_attempts": int(player.attempts),
                "actual_on_target": int(player.on_target),
            })

        # Update histories only after every pre-match row for this match exists.
        for _, player in roster.iterrows():
            history = player_history[player.player_id]
            minutes = _minutes_played(player, duration)
            history["squad"] += 1
            history["apps"] += int(bool(player.appeared))
            history["starts"] += int(bool(player.is_starter))
            history["minutes"] += minutes
            history["goals"] += int(player.goals)
            history["attempts"] += int(player.attempts)
            history["sot"] += int(player.on_target)
            if bool(player.is_starter):
                history["starter_minutes"].append(minutes)
            elif bool(player.appeared):
                history["bench_minutes"].append(minutes)
            history["recent"].append((minutes, int(player.attempts), int(player.on_target)))
            history["last_date"] = match_date

        team_ids = roster.team_id.unique().tolist()
        for team_id in team_ids:
            opponent_id = opponents[(match.match_id, team_id)]
            own = metrics_lookup.loc[(match.match_id, team_id)]
            opp = metrics_lookup.loc[(match.match_id, opponent_id)]
            rate_scale = 90.0 / duration
            history = team_history[team_id]
            history["games"] += 1
            for field in ("goals", "xg", "attempts", "sot"):
                history[field] += rate_scale * float(own.get(field, 0) if pd.notna(own.get(field, 0)) else 0)
            history["goals_allowed"] += rate_scale * float(opp.get("goals", 0))
            history["xg_allowed"] += rate_scale * float(opp.get("xg", 0))
            history["sot_allowed"] += rate_scale * float(opp.get("sot", 0))
            global_history["team_games"] += 1
            for field in ("goals", "xg", "attempts", "sot"):
                global_history[field] += rate_scale * float(own.get(field, 0) if pd.notna(own.get(field, 0)) else 0)

    frame = pd.DataFrame(rows).sort_values(["match_number", "team", "is_starter"], ascending=[True, True, False])
    frame["baseline_probability"] = 1.0 - np.exp(
        -frame.goals_per90_smoothed * frame.expected_minutes / 90.0
    )
    return frame


def _logistic_pipeline() -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
    ])
    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ])),
        ("model", LogisticRegression(C=0.35, max_iter=2000, solver="lbfgs", random_state=SEED)),
    ])


def _forest_pipeline() -> Pipeline:
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("prep", prep),
        ("model", RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=12,
            max_features=0.65, random_state=SEED, n_jobs=-1,
        )),
    ])


def _walk_forward(frame: pd.DataFrame) -> pd.DataFrame:
    predictions: list[pd.DataFrame] = []
    for match_number in range(GROUP_STAGE_END + 1, FINAL_MATCH_NUMBER):
        train = frame[frame.match_number < match_number]
        test = frame[frame.match_number == match_number].copy()
        if test.empty:
            continue
        X_train, y_train = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train.scored
        X_test = test[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        logistic = _logistic_pipeline().fit(X_train, y_train)
        forest = _forest_pipeline().fit(X_train, y_train)
        test["logistic_probability"] = logistic.predict_proba(X_test)[:, 1]
        test["forest_probability"] = forest.predict_proba(X_test)[:, 1]
        predictions.append(test)
    return pd.concat(predictions, ignore_index=True)


def _metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, column in [
        ("Historical rate baseline", "baseline_probability"),
        ("Regularised logistic", "logistic_probability"),
        ("Random forest", "forest_probability"),
    ]:
        y = predictions.scored.to_numpy()
        p = np.clip(predictions[column].to_numpy(), 1e-6, 1 - 1e-6)
        top1_hits, top3_hits, match_count = 0, 0, 0
        for _, group in predictions.groupby("match_number"):
            actual = set(group.loc[group.scored.eq(1), "player_id"])
            if not actual:
                continue
            ranking = group.sort_values(column, ascending=False).player_id.tolist()
            top1_hits += int(ranking[0] in actual)
            top3_hits += int(bool(set(ranking[:3]) & actual))
            match_count += 1
        rows.append({
            "model": label,
            "brier_score": brier_score_loss(y, p),
            "log_loss": log_loss(y, p),
            "roc_auc": roc_auc_score(y, p),
            "top1_scorer_hit_rate": top1_hits / match_count,
            "top3_scorer_hit_rate": top3_hits / match_count,
            "evaluated_matches_with_scorer": match_count,
        })
    return pd.DataFrame(rows).sort_values("brier_score")


def _calibration(predictions: pd.DataFrame, column: str) -> pd.DataFrame:
    data = predictions[[column, "scored"]].copy()
    data["bin"] = pd.qcut(data[column].rank(method="first"), q=5, labels=False)
    return data.groupby("bin", as_index=False).agg(
        mean_predicted=(column, "mean"), observed_rate=("scored", "mean"), observations=("scored", "size")
    )


def _coefficient_table(model: Pipeline) -> pd.DataFrame:
    names = model.named_steps["prep"].get_feature_names_out()
    coefficients = model.named_steps["model"].coef_[0]
    frame = pd.DataFrame({"feature": names, "coefficient": coefficients})
    frame["feature"] = frame.feature.str.replace("num__", "", regex=False).str.replace("cat__", "", regex=False)
    frame["odds_ratio"] = np.exp(frame.coefficient)
    return frame.reindex(frame.coefficient.abs().sort_values(ascending=False).index)


def _bootstrap_uncertainty(train: pd.DataFrame, final: pd.DataFrame, repetitions: int = 250) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    match_numbers = train.match_number.unique()
    player_ids = final.player_id.tolist()
    draws = {player_id: [] for player_id in player_ids}
    X_final = final[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    for _ in range(repetitions):
        sampled = rng.choice(match_numbers, size=len(match_numbers), replace=True)
        boot = pd.concat([train[train.match_number.eq(number)] for number in sampled], ignore_index=True)
        if boot.scored.nunique() < 2:
            continue
        model = _logistic_pipeline().fit(boot[NUMERIC_FEATURES + CATEGORICAL_FEATURES], boot.scored)
        probabilities = model.predict_proba(X_final)[:, 1]
        for player_id, probability in zip(player_ids, probabilities):
            draws[player_id].append(float(probability))
    rows = []
    for player_id, values in draws.items():
        rows.append({
            "player_id": player_id,
            "bootstrap_low": np.quantile(values, 0.025),
            "bootstrap_high": np.quantile(values, 0.975),
            "bootstrap_median": np.median(values),
        })
    return pd.DataFrame(rows)


def _vif(frame: pd.DataFrame) -> tuple[float, str]:
    data = frame[NUMERIC_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(frame[NUMERIC_FEATURES].median())
    data = data.loc[:, data.nunique() > 1]
    corr = data.corr().to_numpy()
    inverse = np.linalg.pinv(corr)
    values = np.diag(inverse)
    index = int(np.argmax(values))
    return float(values[index]), str(data.columns[index])


def run_modelling(
    tables: dict[str, pd.DataFrame], processed_dir: Path, output_dir: Path
) -> ModellingResults:
    frame = build_feature_frame(tables)
    evaluation = _walk_forward(frame)
    metrics = _metrics(evaluation)

    train = frame[frame.match_number < FINAL_MATCH_NUMBER].copy()
    final = frame[frame.match_number.eq(FINAL_MATCH_NUMBER)].copy()
    model = _logistic_pipeline().fit(train[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train.scored)
    final["model_probability"] = model.predict_proba(final[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
    final["probability_120_minutes"] = 1 - np.power(1 - final.model_probability, 120 / 90)
    uncertainty = _bootstrap_uncertainty(train, final)
    final = final.merge(uncertainty, on="player_id", how="left")
    final = final.sort_values("model_probability", ascending=False).reset_index(drop=True)
    final["rank"] = np.arange(1, len(final) + 1)

    calibration = _calibration(evaluation, "logistic_probability")
    coefficients = _coefficient_table(model)
    max_vif, max_vif_feature = _vif(train)
    actual_scorer = final.loc[final.scored_any.eq(1), "player_name"].tolist()
    actual_scorer_rank = final.index[final.scored_any.eq(1)].tolist()
    diagnostics = {
        "training_rows": int(len(train)),
        "training_matches": int(train.match_number.nunique()),
        "scorer_rows": int(train.scored.sum()),
        "scorer_prevalence": float(train.scored.mean()),
        "evaluation_rows": int(len(evaluation)),
        "evaluation_matches": int(evaluation.match_number.nunique()),
        "max_vif": max_vif,
        "max_vif_feature": max_vif_feature,
        "actual_scorer": actual_scorer,
        "actual_scorer_rank": int(actual_scorer_rank[0] + 1) if actual_scorer_rank else None,
        "final_actual_score": "Spain 1-0 Argentina after extra time",
        "final_match_source": tables["matches"].loc[tables["matches"].match_number.eq(104), "source_url"].iloc[0],
    }

    frame.to_csv(processed_dir / "player_match_features.csv", index=False)
    evaluation.to_csv(processed_dir / "walk_forward_predictions.csv", index=False)
    final.to_csv(output_dir / "final_scorer_predictions.csv", index=False)
    metrics.to_csv(output_dir / "model_comparison.csv", index=False)
    coefficients.to_csv(output_dir / "logistic_coefficients.csv", index=False)
    (output_dir / "model_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    return ModellingResults(
        feature_frame=frame,
        evaluation_predictions=evaluation,
        final_predictions=final,
        metrics=metrics,
        coefficients=coefficients,
        calibration=calibration,
        uncertainty=uncertainty,
        diagnostics=diagnostics,
    )
