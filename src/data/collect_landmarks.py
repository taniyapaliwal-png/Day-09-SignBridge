
import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CSV_PATH = DATA_DIR / "landmarks.csv"

LANDMARK_COUNT = 21
COORDINATES = ("x", "y", "z")

FIELDNAMES = (
    ["label", "left_present", "right_present"]
    + [
        f"{side}_{i}_{coordinate}"
        for side in ("left", "right")
        for i in range(LANDMARK_COUNT)
        for coordinate in COORDINATES
    ]
)


def create_row(label, results):
    row = {
        "label": label,
        "left_present": 0,
        "right_present": 0,
    }

    for side in ("left", "right"):
        for i in range(LANDMARK_COUNT):
            for coordinate in COORDINATES:
                row[f"{side}_{i}_{coordinate}"] = 0.0

    if not results.multi_hand_landmarks:
        return None

    for landmarks, handedness in zip(
        results.multi_hand_landmarks,
        results.multi_handedness,
    ):
        side = handedness.classification[0].label.lower()

        if side not in ("left", "right"):
            continue

        row[f"{side}_present"] = 1

        for i, point in enumerate(landmarks.landmark):
            row[f"{side}_{i}_x"] = point.x
            row[f"{side}_{i}_y"] = point.y
            row[f"{side}_{i}_z"] = point.z

    return row


def main():
    parser = argparse.ArgumentParser(
        description="Collect labeled SignBridge hand landmarks."
    )
    parser.add_argument(
        "label",
        help="Label for the sign being collected, e.g. HELLO",
    )
    args = parser.parse_args()

    label = args.label.strip().upper()

    if not label or not label.replace("_", "").isalnum():
        parser.error("Use letters, numbers, and underscores for labels.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = CSV_PATH.exists() and CSV_PATH.stat().st_size > 0

    if file_exists:
        with CSV_PATH.open("r", newline="", encoding="utf-8") as file:
            existing_header = next(csv.reader(file), [])

        if existing_header != FIELDNAMES:
            raise RuntimeError(
                f"{CSV_PATH} has a different schema. "
                "Back it up before changing the dataset format."
            )

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Could not open the webcam.")

    saved_count = 0

    try:
        with CSV_PATH.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDNAMES)

            if not file_exists:
                writer.writeheader()
                file.flush()

            with mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            ) as hands:

                print(f"Collecting label: {label}")
                print(f"Saving samples to: {CSV_PATH}")
                print("Press S to save a sample when hands are visible.")
                print("Press Q to quit.")

                while True:
                    success, frame = camera.read()

                    if not success:
                        print("Could not read webcam frame.")
                        break

                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)

                    if results.multi_hand_landmarks:
                        for hand in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                frame,
                                hand,
                                mp_hands.HAND_CONNECTIONS,
                            )

                    cv2.putText(
                        frame,
                        f"Label: {label} | Saved: {saved_count}",
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        frame,
                        "S = save sample | Q = quit",
                        (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                    cv2.imshow("SignBridge Data Collector", frame)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        break

                    if key == ord("s"):
                        row = create_row(label, results)

                        if row is None:
                            print("No hand detected. Sample not saved.")
                            continue

                        writer.writerow(row)
                        file.flush()
                        saved_count += 1
                        print(f"Saved sample {saved_count} for {label}.")

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"Session complete. Samples saved: {saved_count}")


if __name__ == "__main__":
    main()
