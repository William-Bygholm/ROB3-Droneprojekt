import argparse
import joblib
import cv2
import numpy as np
from skimage.feature import hog
from joblib import Parallel, delayed
import multiprocessing

# HOG window size
WINDOW_W, WINDOW_H = 64, 128

def extract_hog(img):
    """Extract HOG features from a grayscale image."""
    if img is None or img.size == 0:
        return None
    img = cv2.resize(img, (WINDOW_W, WINDOW_H))
    return hog(img,
               orientations=9,
               pixels_per_cell=(8, 8),
               cells_per_block=(2, 2),
               block_norm="L2-Hys")

def detect_window(pipe, patch, x, y, score_thresh):
    feat = extract_hog(patch)
    if feat is None:
        return None
    proba = pipe.predict_proba([feat])[0]
    if proba[1] >= score_thresh:
        return (x, y, x + WINDOW_W, y + WINDOW_H)
    return None

def detect_frame_parallel(pipe, frame, stride=32, score_thresh=0.6, n_jobs=-1):
    """Detect people in a single frame using parallelized sliding windows."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    tasks = []

    for y in range(0, H - WINDOW_H, stride):
        for x in range(0, W - WINDOW_W, stride):
            patch = gray[y:y+WINDOW_H, x:x+WINDOW_W]
            tasks.append((patch, x, y))

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(detect_window)(pipe, patch, x, y, score_thresh) for patch, x, y in tasks
    )

    # Filter out None results
    boxes = [r for r in results if r is not None]
    return boxes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="hog_rf_pipeline.pkl", help="Trained RandomForest model path")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--out", default=None, help="Optional output video path")
    parser.add_argument("--score", type=float, default=0.6, help="Confidence threshold")
    parser.add_argument("--resize", type=float, default=0.15, help="Resize factor for large videos (0-1)")
    parser.add_argument("--frame-skip", type=int, default=5, help="Process every Nth frame")
    parser.add_argument("--stride", type=int, default=None, help="Sliding window stride in pixels (optional)")
    parser.add_argument("--jobs", type=int, default=multiprocessing.cpu_count(), help="Number of CPU cores to use")
    args = parser.parse_args()

    print("Loading model:", args.model)
    pipe = joblib.load(args.model)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: " + args.video)

    # Video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    w_new = int(w * args.resize)
    h_new = int(h * args.resize)

    # Automatically set stride if not provided
    stride = args.stride
    if stride is None:
        stride = max(16, int(WINDOW_W * args.resize / 2))
        print(f"Auto stride set to {stride} based on resize factor.")

    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out, fourcc, fps, (w_new, h_new))

    print(f"Processing video ({w}x{h}) resized to ({w_new}x{h_new})...")
    print(f"Frame skip: {args.frame_skip}, stride: {stride}, parallel jobs: {args.jobs}")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached")
            break

        frame_count += 1
        if frame_count % args.frame_skip != 0:
            continue

        # Resize frame
        frame_resized = cv2.resize(frame, (w_new, h_new))

        # Detect people in parallel
        boxes = detect_frame_parallel(pipe, frame_resized, stride=stride, score_thresh=args.score, n_jobs=args.jobs)

        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Detections", frame_resized)
        if writer:
            writer.write(frame_resized)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Processing finished!")

if __name__ == "__main__":
    main()
