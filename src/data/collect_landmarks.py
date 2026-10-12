"""Collect hand-landmark samples from the webcam."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp

FEATURE_COUNT = 126


def open_camera() -> cv2.VideoCapture:
    """Return the first working camera."""
    for camera_index in (1, 0):
        camera = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
        if camera.isOpened():
            success, _ = camera.read()
            if success:
                print(f"Using camera index {camera_index}")
                return camera
        camera.release()

    raise RuntimeError(
        "No working webcam found. Close other camera apps and check permissions."
    )


def extract_features(results) -> list[float] | None:
    """Extract a fixed-size feature vector from detected hands."""
    if not results.multi_hand_landmarks:
        return None

    detected_hands = sorted(
        results.multi_hand_landmarks,
        key=lambda hand: hand.landmark[0].x,
    )

    features: list[float] = []

    for hand in detected_hands[:2]:
        for landmark in hand.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])

    if len(features) < FEATURE_COUNT:
        features.extend([0.0] * (FEATURE_COUNT - len(features)))

    return features[:FEATURE_COUNT]


def collect_samples(label: str, sample_count: int) -> None:
    output_directory = Path("data/raw")
    output_directory.mkdir(parents=True, exist_ok=True)

    output_path = output_directory / f"{label}.csv"

    camera = open_camera()

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    drawing = mp.solutions.drawing_utils
    hand_connections = mp.solutions.hands.HAND_CONNECTIONS

    fieldnames = [f"feature_{index}" for index in range(FEATURE_COUNT)]
    saved_samples = 0

    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        try:
            while saved_samples < sample_count:
                success, frame = camera.read()

                if not success:
                    print("Could not read a frame.")
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                results = hands.process(rgb_frame)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            hand_connections,
                        )

                features = extract_features(results)

                if features is not None:
                    row = {
                        f"feature_{index}": value
                        for index, value in enumerate(features)
                    }
                    writer.writerow(row)
                    csv_file.flush()
                    saved_samples += 1

                cv2.putText(
                    frame,
                    f"Label: {label}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Samples: {saved_samples}/{sample_count}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    "Press q to quit",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("SignBridge - Data Collection", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            hands.close()
            camera.release()
            cv2.destroyAllWindows()

    print(f"Saved {saved_samples} samples to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect hand-landmark training samples."
    )

    parser.add_argument(
        "--label",
        required=True,
        help="Class label, for example hello.",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=300,
        help="Number of samples to collect.",
    )

    args = parser.parse_args()

    if args.samples <= 0:
        raise ValueError("--samples must be greater than zero.")

    collect_samples(args.label, args.samples)


if __name__ == "__main__":
    main()
