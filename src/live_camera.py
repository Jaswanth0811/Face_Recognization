import cv2
import face_recognition

def run_live(camera_index: int = 0, scale: float = 0.5, model: str = "hog"):
    """
    Live face detection (and optional recognition) from a webcam.

    Args:
        camera_index: Which camera to open (default 0).
        scale: Downscale factor for speed (0.5 = half size).
        model: "hog" (CPU, faster) or "cnn" (needs GPU/CUDA).
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Resize for speed
            small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            # Face detection (use face_encodings if you want recognition)
            locations = face_recognition.face_locations(rgb_small, model=model)
            # encodings = face_recognition.face_encodings(rgb_small, locations)

            # Draw boxes on the original frame
            inv = 1 / scale
            for (top, right, bottom, left) in locations:
                top = int(top * inv); right = int(right * inv)
                bottom = int(bottom * inv); left = int(left * inv)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            cv2.imshow("Live Camera - Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live()