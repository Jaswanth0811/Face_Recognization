import cv2
import face_recognition
import time
import argparse

def run_live(
    camera_index: int = 0,
    scale: float = 1.0,
    model: str = "hog",
    upsample: int = 1
):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")

    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                break

            small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            locations = face_recognition.face_locations(
                rgb_small,
                number_of_times_to_upsample=upsample,
                model=model
            )

            inv = 1 / scale
            for (top, right, bottom, left) in locations:
                top = int(top * inv); right = int(right * inv)
                bottom = int(bottom * inv); left = int(left * inv)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            fps = 1.0 / max(1e-6, (time.time() - t0))
            cv2.putText(frame, f"Faces: {len(locations)}  FPS: {fps:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Live Camera - Face Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="Camera index (0 = default)")
    parser.add_argument("--scale", type=float, default=1.0, help="Downscale factor (<=1.0)")
    parser.add_argument("--model", type=str, default="hog", choices=["hog", "cnn"], help="Detection model")
    parser.add_argument("--upsample", type=int, default=1, help="Upsample times (0,1,2...)")
    args = parser.parse_args()

    run_live(
        camera_index=args.camera,
        scale=args.scale,
        model=args.model,
        upsample=args.upsample
    )
