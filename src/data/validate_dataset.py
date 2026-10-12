"""Validate SignBridge landmark CSV files."""

from __future__ import annotations

import csv
import math
from pathlib import Path

FEATURE_COUNT = 126
DATA_DIRECTORY = Path("data/raw")


def validate_file(path: Path) -> tuple[str, int]:
    label = path.stem

    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []

    expected_fields = [
        f"feature_{index}"
        for index in range(FEATURE_COUNT)
    ]

    if fieldnames != expected_fields:
        raise ValueError(
            f"{path}: expected {FEATURE_COUNT} feature columns, "
            f"found {len(fieldnames)}"
        )

    sample_count = 0

    for row_number, row in enumerate(reader, start=2):
        for field in expected_fields:
            value = row.get(field, "")

            if value == "":
                raise ValueError(
                    f"{path}, row {row_number}: missing {field}"
                )

            number = float(value)

            if not math.isfinite(number):
                raise ValueError(
                    f"{path}, row {row_number}: invalid {field}"
                )

        sample_count += 1

    return label, sample_count


def main() -> None:
    files = sorted(DATA_DIRECTORY.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {DATA_DIRECTORY}"
        )

    total_samples = 0
    labels: list[str] = []

    for path in files:
        label, sample_count = validate_file(path)
        labels.append(label)
        total_samples += sample_count
        print(f"PASS  {path}: {sample_count} samples")

    print()
    print(f"Labels: {len(labels)}")
    print(f"Total samples: {total_samples}")

    if len(labels) < 2:
        print("WARNING: Add at least one more label before training.")

    if total_samples < 20:
        print("WARNING: Dataset is currently too small for training.")


if __name__ == "__main__":
    main()
