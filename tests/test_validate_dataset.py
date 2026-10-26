
import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    PROJECT_ROOT / "src" / "data" / "validate_dataset.py"
)

spec = importlib.util.spec_from_file_location(
    "validate_dataset", MODULE_PATH
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_valid_row():
    row = {
        field: "0"
        for field in module.EXPECTED_FIELDS
    }
    row["label"] = "HELLO"
    row["left_present"] = "1"
    row["right_present"] = "0"

    for i in range(module.LANDMARK_COUNT):
        row[f"left_{i}_x"] = "0.5"
        row[f"left_{i}_y"] = "0.5"
        row[f"left_{i}_z"] = "0.0"

    return row


def test_valid_row_passes():
    assert module.validate_row(make_valid_row()) == []


def test_missing_label_fails():
    row = make_valid_row()
    row["label"] = ""
    assert "missing label" in module.validate_row(row)


def test_invalid_presence_flag_fails():
    row = make_valid_row()
    row["left_present"] = "yes"
    errors = module.validate_row(row)
    assert "invalid left_present flag" in errors


def test_non_numeric_coordinate_fails():
    row = make_valid_row()
    row["left_0_x"] = "not-a-number"
    errors = module.validate_row(row)
    assert "left_0_x is not numeric" in errors


def test_no_hand_present_fails():
    row = make_valid_row()
    row["left_present"] = "0"
    errors = module.validate_row(row)
    assert "no hand marked as present" in errors
