"""Live webcam hand-landmark detector for SignBridge."""

from __future__ import annotations

import time

import cv2
import mediapipe as mp

def main() -> None:
    camera = cv2.VideoCapture(0)

if not camera.isOpened():
        raise RuntimeError(
            "Could not open the webcam. Check macOS camera permissions."
        )

hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

drawing = mp.solutions.drawing_utils
    hand_connections = mp.solutions.hands.HAND_CONNECTIONS

previous_time = time.perf_counter()

try:
        while True:
            success, frame = camera.read()

if not success:
                print("Could not read a frame from the webcam.")
                break

frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        hand_connections,
                    )

current_time = time.perf_counter()
            elapsed = current_time - previous_time
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            previous_time = current_time

hand_count = len(results.multi_hand_landmarks or [])

cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

cv2.putText(
                frame,
                f"Hands: {hand_count}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

cv2.imshow("SignBridge - Hand Landmarks", frame)

if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
        camera.release()
        hands.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
