from pathlib import Path

from src.data_pipeline import load_and_validate
from src.model_pipeline import build_feature_frame


ROOT = Path(__file__).resolve().parents[1]


def test_source_tables_pass_validation():
    _, result = load_and_validate(ROOT / "data" / "raw")
    assert result.checks.passed.all()


def test_final_never_enters_its_own_history():
    tables, _ = load_and_validate(ROOT / "data" / "raw")
    frame = build_feature_frame(tables)
    final = frame[frame.match_number.eq(104)]
    assert len(final) == 52
    assert set(final.team) == {"Spain", "Argentina"}
    assert final.prior_squad_selections.max() <= 7

