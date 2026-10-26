
import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = PROJECT_ROOT / "data" / "landmarks.csv"

LANDMARK_COUNT = 21
COORDINATES = ("x", "y", "z")
SIDES = ("left", "right")

EXPECTED_FIELDS = (
    ["label", "left_present", "right_present"]
    + [
        f"{side}_{i}_{coordinate}"
        for side in SIDES
        for i in range(LANDMARK_COUNT)
        for coordinate in COORDINATES
    ]
)


def validate_row(row):
    errors = []

    label = (row.get("label") or "").strip()
    if not label:
        errors.append("missing label")

    presence = {}

    for side in SIDES:
        value = row.get(f"{side}_present", "")

        if value not in ("0", "1"):
            errors.append(f"invalid {side}_present flag")
        else:
            presence[side] = int(value)

    if len(presence) == 2 and not any(presence.values()):
        errors.append("no hand marked as present")

    for side in SIDES:
        for i in range(LANDMARK_COUNT):
            for coordinate in COORDINATES:
                field = f"{side}_{i}_{coordinate}"
                raw_value = row.get(field, "")

                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    errors.append(f"{field} is not numeric")
                    continue

                if not math.isfinite(value):
                    errors.append(f"{field} is not finite")
                    continue

                # MediaPipe x and y are normalized image coordinates.
                # Small tolerance accommodates floating-point rounding.
                if (
                    coordinate in ("x", "y")
                    and presence.get(side) == 1
                    and not -0.001 <= value <= 1.001
                ):
                    errors.append(f"{field} is outside the image")

    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Validate a SignBridge landmark CSV dataset."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to the landmark CSV file.",
    )
    args = parser.parse_args()

    csv_path = args.csv

    if not csv_path.is_file():
        print(f"ERROR: Dataset not found: {csv_path}")
        return 1

    valid_count = 0
    invalid_count = 0
    label_counts = Counter()

    try:
        with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)

            if reader.fieldnames != EXPECTED_FIELDS:
                print("ERROR: CSV header does not match the expected schema.")
                print(f"Expected {len(EXPECTED_FIELDS)} columns.")
                print(
                    f"Found {len(reader.fieldnames or [])} columns."
                )
                return 1

            for line_number, row in enumerate(reader, start=2):
                errors = validate_row(row)

                if errors:
                    invalid_count += 1
                    print(
                        f"Row {line_number}: "
                        + "; ".join(errors[:5])
                    )
                    if len(errors) > 5:
                        print(f"  ... and {len(errors) - 5} more issue(s)")
                else:
                    valid_count += 1
                    label_counts[row["label"].strip()] += 1

    except (OSError, UnicodeError, csv.Error) as error:
        print(f"ERROR: Could not read dataset: {error}")
        return 1

    total = valid_count + invalid_count

    print("\n--- SignBridge Dataset Report ---")
    print(f"File: {csv_path}")
    print(f"Total samples: {total}")
    print(f"Valid samples: {valid_count}")
    print(f"Invalid samples: {invalid_count}")
    print(f"Unique labels in valid samples: {len(label_counts)}")

    if label_counts:
        print("\nSamples per label:")
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count}")

    if total == 0:
        print("\nWARNING: Dataset is empty.")
        return 1

    if invalid_count:
        print("\nRESULT: Validation failed. Review the reported rows.")
        return 1

    print("\nRESULT: All samples passed structural validation.")
    print("Note: This does not verify that sign labels are correct.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
