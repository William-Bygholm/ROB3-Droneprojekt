import argparse
import joblib
import cv2
from skimage.feature import hog

WINDOW_W, WINDOW_H = 64, 128

def extract_hog_frame(img):
    """Extract HOG features from a resized frame."""
    img_resized = cv2.resize(img, (WINDOW_W, WINDOW_H))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm="L2-Hys")
    return features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--model", required=True, help="Path to HOG+RandomForest model (.pkl)")
    parser.add_argument("--out", default=None, help="Optional output video path")
    parser.add_argument("--resize", type=float, default=0.25, help="Resize factor for large videos (0-1)")
    parser.add_argument("--frame-skip", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--no-display", action="store_true", help="Do not display video to speed up processing")
    args = parser.parse_args()

    print("Loading model:", args.model)
    pipe = joblib.load(args.model)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: " + args.video)

    # Original video properties
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Resize for display/output
    w_resized = int(w_orig * args.resize)
    h_resized = int(h_orig * args.resize)

    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out, fourcc, fps, (w_resized, h_resized))

    print(f"Processing video ({w_orig}x{h_orig}) resized to ({w_resized}x{h_resized})...")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % args.frame_skip != 0:
            continue

        # Resize frame first
        frame_small = cv2.resize(frame, (w_resized, h_resized))

        # Extract HOG and predict
        feat = extract_hog_frame(frame_small)
        pred = pipe.predict([feat])[0]

        # Draw border to indicate detection
        color = (0, 255, 0) if pred == 1 else (0, 0, 255)
        cv2.rectangle(frame_small, (0, 0), (w_resized-1, h_resized-1), color, 4)

        # Display video if enabled
        if not args.no_display:
            cv2.imshow("Detection", frame_small)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break

        # Write to output video
        if writer:
            writer.write(frame_small)

    cap.release()
    if writer:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    print("Processing finished!")

if __name__ == "__main__":
    main()
