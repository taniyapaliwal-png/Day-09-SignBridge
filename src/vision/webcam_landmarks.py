
import cv2
import mediapipe as mp


def main():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open the webcam. Check camera permissions."
        )

    print("SignBridge hand detection started.")
    print("Show your hand to the camera. Press Q to quit.")

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
                    print("Could not read a webcam frame.")
                    break

                # Mirror the image like a selfie camera.
                frame = cv2.flip(frame, 1)

                # MediaPipe expects RGB images.
                rgb_frame = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )
                rgb_frame.flags.writeable = False
                results = hands.process(rgb_frame)

                rgb_frame.flags.writeable = True

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style(),
                        )

                    hand_count = len(results.multi_hand_landmarks)
                    cv2.putText(
                        frame,
                        f"Hands detected: {hand_count}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )
                else:
                    cv2.putText(
                        frame,
                        "Show your hand",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 165, 255),
                        2,
                    )

                cv2.imshow("SignBridge - Hand Detection", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("SignBridge hand detection stopped.")


if __name__ == "__main__":
    main()
