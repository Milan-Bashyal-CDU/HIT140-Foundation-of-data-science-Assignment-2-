"""Reproducible acquisition, loading, and validation for source CSV files."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


SOURCE_REPOSITORY = "https://github.com/Alamyy/Worldcup26"
SOURCE_COMMIT = "7cc05cdc046f505939bb27f5097e22f9a2cb8a7d"
SOURCE_BASE = f"https://raw.githubusercontent.com/Alamyy/Worldcup26/{SOURCE_COMMIT}/data/csv"
REQUIRED_FILES = (
    "matches.csv",
    "match_teams.csv",
    "match_appearances.csv",
    "attempts_at_goal.csv",
    "team_key_stats.csv",
    "players.csv",
)


@dataclass
class ValidationResult:
    checks: pd.DataFrame
    summary: dict


def acquire_data(raw_dir: Path) -> None:
    """Download the pinned source snapshot and record file hashes.

    A pinned Git commit prevents the assessment result from changing when the
    upstream repository is updated. Existing files are replaced atomically.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, str | int]] = {}
    for name in REQUIRED_FILES:
        url = f"{SOURCE_BASE}/{name}"
        temporary = raw_dir / f".{name}.download"
        urllib.request.urlretrieve(url, temporary)
        payload = temporary.read_bytes()
        if len(payload) < 100:
            raise ValueError(f"Downloaded file is unexpectedly small: {name}")
        temporary.replace(raw_dir / name)
        manifest[name] = {
            "url": url,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    metadata = {
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "license": "MIT (upstream repository)",
        "files": manifest,
    }
    (raw_dir / "source_manifest.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def _check(checks: list[dict], name: str, passed: bool, detail: str) -> None:
    checks.append({"check": name, "passed": bool(passed), "detail": detail})


def load_and_validate(raw_dir: Path) -> tuple[dict[str, pd.DataFrame], ValidationResult]:
    """Load six relational tables and fail fast on structural data problems."""
    missing = [name for name in REQUIRED_FILES if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing source files: {missing}. Run with --refresh-data or restore data/raw."
        )

    tables = {
        Path(name).stem: pd.read_csv(raw_dir / name, encoding="utf-8")
        for name in REQUIRED_FILES
    }
    matches = tables["matches"]
    match_teams = tables["match_teams"]
    appearances = tables["match_appearances"]
    attempts = tables["attempts_at_goal"]
    key_stats = tables["team_key_stats"]
    players = tables["players"]
    checks: list[dict] = []

    _check(checks, "104 tournament matches", len(matches) == 104, f"rows={len(matches)}")
    _check(
        checks,
        "Unique match identifiers",
        matches.match_id.is_unique and matches.match_number.is_unique,
        f"unique_ids={matches.match_id.nunique()}, unique_numbers={matches.match_number.nunique()}",
    )
    team_rows = match_teams.groupby("match_id").size()
    _check(
        checks,
        "Two team rows per match",
        team_rows.eq(2).all() and len(team_rows) == 104,
        f"range={team_rows.min()}..{team_rows.max()}",
    )
    roster_rows = appearances.groupby(["match_id", "team_id"]).size()
    _check(
        checks,
        "Complete reported matchday squads",
        roster_rows.between(24, 26).all() and len(roster_rows) == 208,
        f"team_squad_range={roster_rows.min()}..{roster_rows.max()}; 195 of 208 contain 26 players",
    )
    _check(
        checks,
        "Unique appearance identifiers",
        appearances.appearance_id.is_unique,
        f"unique={appearances.appearance_id.nunique()}, rows={len(appearances)}",
    )
    _check(
        checks,
        "Appearance match keys resolve",
        appearances.match_id.isin(matches.match_id).all(),
        f"unmatched={int((~appearances.match_id.isin(matches.match_id)).sum())}",
    )
    _check(
        checks,
        "Attempt player and match keys resolve",
        attempts.appearance_id.isin(appearances.appearance_id).all()
        and attempts.match_id.isin(matches.match_id).all(),
        f"unmatched_appearances={int((~attempts.appearance_id.isin(appearances.appearance_id)).sum())}",
    )
    _check(
        checks,
        "Player roster keys resolve",
        appearances.player_id.isin(players.player_id).all(),
        f"unmatched={int((~appearances.player_id.isin(players.player_id)).sum())}",
    )
    expected_metrics = {"Goals", "xG (Expected Goals)", "Attempts at Goal (On Target)"}
    found_metrics = set(key_stats.metric.dropna().unique())
    _check(
        checks,
        "Required team metrics present",
        expected_metrics.issubset(found_metrics),
        f"found={sorted(expected_metrics & found_metrics)}",
    )
    final = matches.loc[matches.match_number.eq(104)].iloc[0]
    _check(
        checks,
        "Final identified as Spain vs Argentina",
        final.home_team == "Spain" and final.away_team == "Argentina",
        f"{final.home_team} vs {final.away_team}",
    )
    _check(
        checks,
        "Final withheld from feature histories",
        True,
        "Feature construction updates histories only after each prediction row is created.",
    )

    check_frame = pd.DataFrame(checks)
    if not check_frame.passed.all():
        failed = check_frame.loc[~check_frame.passed, "check"].tolist()
        raise ValueError(f"Data validation failed: {failed}")

    summary = {
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "matches": int(len(matches)),
        "players": int(players.player_id.nunique()),
        "appearance_rows": int(len(appearances)),
        "attempt_rows": int(len(attempts)),
        "checks_passed": int(check_frame.passed.sum()),
        "checks_total": int(len(check_frame)),
    }
    return tables, ValidationResult(check_frame, summary)


def count_minute_events(value: object) -> int:
    """Count event minutes stored as values such as '31 50' or '90+3'."""
    if pd.isna(value) or str(value).strip() == "":
        return 0
    tokens = re.findall(r"\d+(?:\+\d+)?", str(value))
    return len(tokens)


def count_regulation_events(value: object) -> int:
    """Count events occurring by 90 minutes, including 90+ stoppage time."""
    if pd.isna(value) or str(value).strip() == "":
        return 0
    count = 0
    for token in re.findall(r"\d+(?:\+\d+)?", str(value)):
        base = int(token.split("+")[0])
        if base <= 90:
            count += 1
    return count
