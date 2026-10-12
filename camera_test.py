import cv2

for index in (1, 0):
    print(f"Trying camera index {index}...", flush=True)

    camera = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)

    if not camera.isOpened():
        print("  Could not open", flush=True)
        camera.release()
        continue

    success, frame = camera.read()
    print(
        f"  Opened: True | Frame received: {success}",
        flush=True,
    )

    if success:
        print(f"  Frame size: {frame.shape}", flush=True)
        cv2.imwrite("camera_test.jpg", frame)

        cv2.namedWindow("SignBridge Camera Test", cv2.WINDOW_NORMAL)
        cv2.imshow("SignBridge Camera Test", frame)

        print("Test image saved as camera_test.jpg", flush=True)
        print("Press any key while the window is focused.", flush=True)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        camera.release()
        break

    camera.release()
else:
    raise RuntimeError("Neither camera index returned a frame.")
