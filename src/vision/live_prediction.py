
import cv2
import mediapipe as mp
import joblib
import numpy as np
from pathlib import Path
from collections import Counter, deque


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "signbridge_model.joblib"

SMOOTHING_FRAMES = 7
MIN_CONFIDENCE = 0.60


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. "
            "Run python3 models/train_model.py first."
        )

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    feature_columns = saved["feature_columns"]

    recent_predictions = deque(maxlen=SMOOTHING_FRAMES)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Could not open webcam.")

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        ) as hands:
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
                            frame, hand, mp_hands.HAND_CONNECTIONS
                        )

                    row = {column: 0.0 for column in feature_columns}

                    for landmarks, handedness in zip(
                        results.multi_hand_landmarks,
                        results.multi_handedness,
                    ):
                        side = handedness.classification[0].label.lower()
                        present_column = f"{side}_present"

                        if present_column in row:
                            row[present_column] = 1.0

                        for index, point in enumerate(landmarks.landmark):
                            for coordinate in ("x", "y", "z"):
                                name = f"{side}_{index}_{coordinate}"
                                if name in row:
                                    row[name] = getattr(point, coordinate)

                    features = np.array(
                        [[row[column] for column in feature_columns]],
                        dtype=float,
                    )

                    prediction = model.predict(features)[0]
                    confidence = None

                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(features)[0]
                        confidence = float(np.max(probabilities))

                    if (
                        confidence is None
                        or confidence >= MIN_CONFIDENCE
                    ):
                        recent_predictions.append(str(prediction))
                    else:
                        recent_predictions.clear()

                    if recent_predictions:
                        stable_prediction, count = Counter(
                            recent_predictions
                        ).most_common(1)[0]

                        if count >= 4:
                            display_text = f"Sign: {stable_prediction}"
                        else:
                            display_text = "Recognizing..."
                    else:
                        display_text = "Low confidence - try again"

                    cv2.putText(
                        frame,
                        display_text,
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0) if recent_predictions
                        else (0, 165, 255),
                        2,
                    )

                    if confidence is not None:
                        confidence_text = (
                            f"Model confidence: {confidence:.0%}"
                        )
                        cv2.putText(
                            frame,
                            confidence_text,
                            (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2,
                        )
                else:
                    recent_predictions.clear()
                    cv2.putText(
                        frame,
                        "Show your hand",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 165, 255),
                        2,
                    )

                cv2.putText(
                    frame,
                    "Press Q to quit",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow("SignBridge - Live Prediction", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
